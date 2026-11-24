import numpy as np
import pytest
from sample_images import make_image

from activelabel import preannotate
from activelabel.errors import CodedError
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.projects import Project
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore


def encode(mask):
    """Run-length encoding the way the predict worker writes it (alternating runs, zeros first)."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    counts, current, run = [], False, 0
    for value in flat:
        if bool(value) == current:
            run += 1
        else:
            counts.append(run)
            current, run = bool(value), 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


@pytest.fixture
def project_dir(tmp_path):
    """A project of 4 images in one task: p1 pre-annotated, p2 reviewed, p3 and p4 unlabeled."""
    manifest = Manifest(tmp_path)
    ids = ["p1", "p2", "p3", "p4"]
    for image_id in ids:
        make_image(tmp_path / "images" / f"{image_id}.png", seed=1)
    manifest.add([ImageRecord(i, f"{i}.png", f"images/{i}.png", 64, 48, 1, "2026-01-01") for i in ids])
    TaskStore(tmp_path).create("T", images=ids)
    images = {
        "p1": [Label("a", score=0.9, model="m1")],
        "p2": [Label("b")],
    }
    save_annotations(tmp_path, AnnotationSet(Task.CLASSIFICATION, ["a", "b"], images, {}))
    return tmp_path


def project(task=Task.CLASSIFICATION):
    return Project(1, "P", "", task, ["a", "b"])


def test_image_state():
    images = {"pre": [Label("a", score=0.5)], "mixed": [Label("a", score=0.5), Label("b")], "empty": []}
    assert preannotate.image_state(images, "absent") == "none"
    assert preannotate.image_state(images, "pre") == "pre"
    assert preannotate.image_state(images, "mixed") == "human"
    assert preannotate.image_state(images, "empty") == "human"  # reviewed as empty: never redone
    assert preannotate.is_preannotation(Label("a", score=0.1)) and not preannotate.is_preannotation(
        Label("a")
    )


def test_decode_mask_round_trip_and_size_check():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    assert (preannotate.decode_mask(encode(mask)) == mask).all()
    with pytest.raises(ValueError, match="does not fit"):
        preannotate.decode_mask({"size": [2, 2], "counts": [3, 3]})


def test_mask_polygons_drop_specks():
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 10:30] = True
    mask[0, 79] = True
    rings = preannotate.mask_polygons(mask)
    assert len(rings) == 1 and len(rings[0]) >= 4
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in rings[0])


def test_mask_labels_instance_and_semantic():
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 10:30] = True
    mask[5:35, 50:75] = True
    encoded = encode(mask)
    instance = {"label": "a", "score": 0.8, "model": "m1", "bbox": [0, 0, 1, 1], "mask": encoded}
    semantic = {"label": "b", "score": 0.7, "model": "m1", "mask": encoded}
    plain = {"label": "a", "score": 0.6, "model": "m1"}
    blank = {"label": "a", "score": 0.6, "model": "m1", "mask": encode(np.zeros((40, 80), dtype=bool))}
    out = preannotate.mask_labels([instance, semantic, plain, blank])
    assert [(lb.label, len(lb.polygons or [])) for lb in out] == [("a", 2), ("b", 1), ("b", 1), ("a", 0)]
    assert out[3].bbox is None  # the item without a mask is kept as it is
    assert out[0].bbox[0] < 0.2 and out[0].bbox[2] > 0.6  # the box covers both regions


def test_fits():
    assert preannotate.fits("classification", Task.CLASSIFICATION)
    assert preannotate.fits("semantic", Task.SEGMENTATION) and preannotate.fits("instance", Task.SEGMENTATION)
    assert not preannotate.fits("detection", Task.SEGMENTATION)
    assert not preannotate.fits("unknown", Task.DETECTION)


def test_targets_by_scope(project_dir):
    assert preannotate.targets(project_dir, 1, "unlabeled") == ["p3", "p4"]
    assert preannotate.targets(project_dir, 1, "unreviewed") == ["p1"]
    Manifest(project_dir).set_status(["p3"], EXCLUDED, "x")
    assert preannotate.targets(project_dir, 1, "unlabeled") == ["p4"]


def test_targets_errors(project_dir):
    with pytest.raises(CodedError) as bad_scope:
        preannotate.targets(project_dir, 1, "everything")
    assert bad_scope.value.code == "scope_invalid"
    with pytest.raises(CodedError) as no_task:
        preannotate.targets(project_dir, 99, "unlabeled")
    assert no_task.value.code == "task_not_found"


def test_task_counts_and_unreviewed(project_dir):
    counts = next(c for c in preannotate.task_counts(project_dir) if c["id"] == 1)
    assert counts["images"] == 4 and counts["unlabeled"] == 2 and counts["unreviewed"] == 1
    assert counts["models"] == ["m1"]
    assert preannotate.unreviewed_images(project_dir) == ["p1"]
    assert preannotate.unreviewed_images(project_dir, 1) == ["p1"]


def test_project_keypoints_counts_the_people_labels(tmp_path):
    assert preannotate.project_keypoints(tmp_path) is None
    points = [[0.1, 0.1, 2], [0.2, 0.2, 2], [0.3, 0.3, 2]]
    images = {
        "a": [Label("x", bbox=[0, 0, 1, 1], keypoints=points)],
        "b": [Label("x", bbox=[0, 0, 1, 1], keypoints=points[:2], score=0.9)],  # a model's: ignored
    }
    save_annotations(tmp_path, AnnotationSet(Task.KEYPOINTS, ["x"], images, {}))
    assert preannotate.project_keypoints(tmp_path) == 3


def test_apply_predictions_unlabeled_never_touches_people_work(project_dir):
    predictions = {
        "p2": [Label("a", score=0.9, model="m2")],  # a person's label: skipped
        "p3": [Label("a", score=0.9, model="m2"), Label("zzz", score=0.9, model="m2")],
        "p4": [],
        "gone": [Label("a", score=0.9, model="m2")],
    }
    counts = preannotate.apply_predictions(project_dir, project(), predictions, "unlabeled")
    assert counts == {
        "images": 1,
        "objects": 1,
        "skipped": 2,
        "empty": 1,
        "dropped_class": 1,
        "dropped_geometry": 0,
    }
    images = load_annotations(project_dir).images
    assert [lb.label for lb in images["p3"]] == ["a"] and "p4" not in images
    assert [lb.label for lb in images["p2"]] == ["b"]


def test_apply_predictions_drops_labels_without_the_shape_of_the_task(project_dir):
    boxless = Label("a", score=0.9, model="m2")
    counts = preannotate.apply_predictions(
        project_dir, project(Task.DETECTION), {"p3": [boxless]}, "unlabeled"
    )
    assert counts["dropped_geometry"] == 1 and counts["images"] == 0 and counts["empty"] == 1


def test_apply_predictions_unreviewed_replaces_old_pre_annotations(project_dir):
    counts = preannotate.apply_predictions(
        project_dir, project(), {"p1": [Label("b", score=0.6, model="m2")]}, "unreviewed"
    )
    assert counts["images"] == 1
    (label,) = load_annotations(project_dir).images["p1"]
    assert label.label == "b" and label.model == "m2"


def test_accept_marks_reviewed(project_dir):
    assert preannotate.accept(project_dir, ["p1", "p2", "p3"]) == 1  # p2 is a person's, p3 has nothing
    (label,) = load_annotations(project_dir).images["p1"]
    assert label.score is None and label.model is None
    assert preannotate.accept(project_dir, ["p1"]) == 0


def test_accept_without_annotations(tmp_path):
    assert preannotate.accept(tmp_path, ["x"]) == 0


def test_store_numbers_lists_and_updates(tmp_path):
    store = preannotate.PreannotationStore(tmp_path)
    assert store.list() == []
    first = store.create(task=1, scope="unlabeled")
    second = store.create(task=2, scope="unreviewed")
    assert (first["number"], second["number"]) == (1, 2) and first["status"] == "queued"
    assert [r["number"] for r in store.list()] == [2, 1]  # newest first
    updated = store.update(1, status="running", created="x")
    assert updated["status"] == "running" and updated["number"] == 1 and updated["created"] != "x"
    assert store.get(1)["status"] == "running"
    with pytest.raises(preannotate.RunNotFound):
        store.get(7)
    with pytest.raises(preannotate.RunNotFound):
        store.update(7, status="done")
