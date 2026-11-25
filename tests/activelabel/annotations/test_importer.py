import io
import json
import zipfile

import pytest
from ann_helpers import make_project

from activelabel.annotations import apply, load_staged, parse, stage, validate
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.tasks import Task


def coco_file(task="seg"):
    ann = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25]}
    if task == "seg":
        ann["segmentation"] = [[0, 0, 50, 0, 50, 25, 0, 25]]
    if task == "pose":
        ann["keypoints"] = [10, 10, 2, 20, 20, 2]
    data = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 50},
        ],
        "categories": [{"id": 1, "name": "Cat"}],
        "annotations": [ann, {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    return ("labels.json", io.BytesIO(json.dumps(data).encode()))


@pytest.fixture
def proj(tmp_path):
    return make_project(tmp_path, ["a.png", "b.png", "c.png"], task="detection", classes=["cat"])


def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    buffer.seek(0)
    return buffer


def test_apply_import_replace_and_cleanup(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("det")])
    summary = apply(store, project.id, manifest, token, "import")
    assert summary["images_imported"] == 2 and summary["labels_imported"] == 2
    annotations = load_annotations(store.dir(project.id))
    assert annotations is not None and annotations.task is Task.DETECTION
    assert annotations.classes == ["cat"]  # existing casing kept
    assert store.get(project.id).classes == ["cat"]
    assert not (store.dir(project.id) / "staging" / token).exists()
    with pytest.raises(FileNotFoundError):
        load_staged(store.dir(project.id), token)


def test_apply_refuses_import_on_mismatch(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("seg")])
    with pytest.raises(ValueError, match="not possible"):
        apply(store, project.id, manifest, token, "import")


def test_apply_change_task_and_new_classes(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("pose")])
    store.update(project.id, classes=["dog"])
    summary = apply(store, project.id, manifest, token, "change_task_and_import")
    assert summary["task"] == "keypoints" and summary["classes_added"] == ["Cat"]
    assert store.get(project.id).task is Task.KEYPOINTS and store.get(project.id).classes == ["dog", "Cat"]


def test_apply_convert(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("seg")])
    summary = apply(store, project.id, manifest, token, "convert_and_import")
    assert summary["converted"] == 2 and store.get(project.id).task is Task.DETECTION
    labels = load_annotations(store.dir(project.id)).images  # type: ignore[union-attr]
    assert all(lb.polygons is None and lb.bbox for items in labels.values() for lb in items)


def test_apply_merge_keeps_existing(proj):
    store, project, manifest = proj
    ids = {r.name: r.id for r in manifest.all()}
    save_annotations(
        store.dir(project.id),
        AnnotationSet(Task.DETECTION, ["cat"], {ids["a.png"]: [Label("cat", bbox=[0, 0, 1, 1])]}, {}),
    )
    token = stage(store.dir(project.id), [coco_file("det")])
    summary = apply(store, project.id, manifest, token, "import", mode="merge")
    assert summary["images_kept"] == 1 and summary["images_added"] == 1
    images = load_annotations(store.dir(project.id)).images  # type: ignore[union-attr]
    assert images[ids["a.png"]][0].bbox == [0, 0, 1, 1]
    assert images[ids["b.png"]][0].bbox == pytest.approx([0, 0, 0.1, 0.2])


def test_apply_discard(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("det")])
    assert apply(store, project.id, manifest, token, "discard")["action"] == "discard"
    assert load_annotations(store.dir(project.id)) is None
    assert not (store.dir(project.id) / "staging" / token).exists()
    with pytest.raises(ValueError, match="Unknown action"):
        apply(store, project.id, manifest, token, "explode")


def _yolo_files(names, lines):
    return [
        ("data.yaml", io.BytesIO(f"names: [{names}]\n".encode())),
        ("labels/a.txt", io.BytesIO(lines.encode())),
    ]


def test_merge_adds_only_classes_that_received_labels(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png", "b.png"], task="detection", classes=["cão"])
    ids = {r.name: r.id for r in manifest.all()}
    save_annotations(
        store.dir(project.id),
        AnnotationSet(Task.DETECTION, ["cão"], {ids["a.png"]: [Label("cão", bbox=[0, 0, 1, 1])]}, {}),
    )
    # a.png already has labels, so the "cao" label of the file is skipped: no new class either
    token = stage(store.dir(project.id), _yolo_files("cao", "0 0.5 0.5 0.2 0.4\n"))
    summary = apply(store, project.id, manifest, token, "import", mode="merge")
    assert summary["labels_imported"] == 0 and summary["classes_added"] == []
    assert store.get(project.id).classes == ["cão"]
    assert load_annotations(store.dir(project.id)).classes == ["cão"]  # type: ignore[union-attr]


def test_merge_adds_the_class_of_a_label_that_was_imported(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png", "b.png"], task="detection", classes=["cão"])
    token = stage(store.dir(project.id), _yolo_files("cao", "0 0.5 0.5 0.2 0.4\n"))
    save_annotations(store.dir(project.id), AnnotationSet(Task.DETECTION, ["cão"], {}, {}))
    summary = apply(store, project.id, manifest, token, "import", mode="merge")
    assert summary["labels_imported"] == 1 and summary["classes_added"] == ["cao"]


def test_class_folders_zip_labels_a_classification_project(tmp_path):
    names = ["Ação/1.png", "Ação/2.png", "Não é gato/3.png", "Ônibus/4.png"]
    store, project, manifest = make_project(tmp_path, names, task="classification")
    folder = store.dir(project.id)
    archive = zip_bytes({f"fotos/{n}": "x" for n in names})
    token = stage(folder, [("fotos.zip", archive)])
    assert {p.parent.name for p in load_staged(folder, token)} == {"Ação", "Não é gato", "Ônibus"}
    report = validate(parse(load_staged(folder, token), "folders", manifest.all()), project)
    assert report.format == "folders" and report.can_import and report.task_matches
    summary = apply(store, project.id, manifest, token, "import", "replace")
    assert summary["format"] == "folders" and summary["images_imported"] == 4
    assert sorted(store.get(project.id).classes) == ["Ação", "Não é gato", "Ônibus"]
    labels = load_annotations(folder).images
    by_name = {r.name: labels[r.id][0].label for r in manifest.all()}
    assert by_name == {
        "Ação/1.png": "Ação",
        "Ação/2.png": "Ação",
        "Não é gato/3.png": "Não é gato",
        "Ônibus/4.png": "Ônibus",
    }


def test_replace_leaves_excluded_images_alone_and_copies_the_old_file(proj):
    store, project, manifest = proj
    apply(store, project.id, manifest, stage(store.dir(project.id), [coco_file("det")]), "import")
    first = load_annotations(store.dir(project.id))
    assert first is not None
    by_name = {r.name: r.id for r in manifest.all()}
    manifest.set_status([by_name["a.png"]], "excluded", "bad")
    token = stage(store.dir(project.id), [coco_file("det")])
    summary = apply(store, project.id, manifest, token, "import")
    assert summary["images_imported"] == 1  # only b.png: a.png is excluded
    assert "excluded_image_entries" in {p["code"] for p in summary["problems"]}
    saved = load_annotations(store.dir(project.id))
    assert saved is not None and saved.images[by_name["a.png"]] == first.images[by_name["a.png"]]
    assert summary["backup"].endswith("Z.replace.bak.json")
    assert (store.dir(project.id) / summary["backup"]).exists()


def test_replace_with_another_task_does_not_keep_labels_of_the_old_task(proj):
    store, project, manifest = proj
    apply(store, project.id, manifest, stage(store.dir(project.id), [coco_file("det")]), "import")
    manifest.set_status([manifest.all()[0].id], "excluded", "bad")
    polygon = [[0, 0, 50, 0, 50, 25, 0, 25]]
    data = {
        "images": [{"id": 1, "file_name": "b.png", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "Cat"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25], "segmentation": polygon}],
    }
    token = stage(store.dir(project.id), [("labels.json", io.BytesIO(json.dumps(data).encode()))])
    summary = apply(store, project.id, manifest, token, "change_task_and_import")
    saved = load_annotations(store.dir(project.id))
    assert saved is not None and saved.task is Task.SEGMENTATION
    assert all(lb.polygons for items in saved.images.values() for lb in items)  # no old boxes mixed in
    assert summary["backup"]  # they are in the copy
