"""Import a classification project's images and classes from folders (#568)."""

import io
import zipfile

import pytest
from api_client import make_client
from sample_images import make_image

from activelabel.annotations.staging import STAGING
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.tasks import Task


def png(tmp_path, seed):
    """Bytes of a small image; ``seed`` makes each one different (equal images would be duplicates)."""
    path = make_image(tmp_path / f"src/{seed}.png", seed=seed)
    return path.read_bytes()


def zip_of(tmp_path, names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for seed, name in enumerate(names):
            archive.writestr(name, b"" if name.endswith("/") else png(tmp_path, seed))
    return buffer.getvalue()


def new_project(tmp_path, task="classification", classes=()):
    client, state = make_client(tmp_path)
    assert (
        client.post("/api/projects", json={"name": "P", "task": task, "classes": list(classes)}).status_code
        == 200
    )
    return client, state


def send_zip(client, content, name="set.zip"):
    response = client.post("/api/projects/1/images", files=[("files", (name, content))])
    assert response.status_code == 200, response.text
    return response.json()


def test_a_zip_with_class_folders_is_previewed_and_imported_in_one_go(tmp_path):
    client, _ = new_project(tmp_path)
    names = [
        "pets/train/cat/1.png",
        "pets/train/cat/2.png",
        "pets/val/cat/3.png",
        "pets/train/dog/4.png",
        "pets/test/dog/5.png",
        "pets/empty/",  # an empty class folder holds no image: no class
    ]
    assert send_zip(client, zip_of(tmp_path, names))["added"] == 5
    report = client.post("/api/projects/1/images/class-folders").json()
    assert report["found"] and report["format"] == "folders" and report["can_import"]
    assert report["classes_found"] == ["cat", "dog"]
    assert report["classes_new"] == []  # the project has no classes yet: nothing is "new" next to others
    assert report["counts"]["labels_per_class"] == {"cat": 3, "dog": 2}
    assert report["split_folders"] == ["test", "train", "val"]
    summary = client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import", "mode": "merge"}
    ).json()
    assert summary["images_added"] == 5 and summary["classes_added"] == ["cat", "dog"]
    assert client.get("/api/projects/1").json()["classes"] == ["cat", "dog"]
    assert not (tmp_path / "ws" / "projects" / "1" / STAGING / report["token"]).exists()


def test_loose_files_with_accents_and_spaces(tmp_path):
    client, _ = new_project(tmp_path)
    files = [
        ("files", ("dataset/Gato preto/a b.png", png(tmp_path, 1))),
        ("files", ("dataset/Gato preto/c.png", png(tmp_path, 2))),
        ("files", ("dataset/cão/d.png", png(tmp_path, 3))),
    ]
    assert client.post("/api/projects/1/images", files=files).json()["added"] == 3
    report = client.post("/api/projects/1/images/class-folders").json()
    assert report["classes_found"] == ["cão", "Gato preto"]
    token = report["token"]
    client.post(f"/api/projects/1/annotations/{token}/apply", json={"action": "import", "mode": "merge"})
    annotations = load_annotations(tmp_path / "ws" / "projects" / "1")
    assert sorted(annotations.classes) == ["Gato preto", "cão"]
    assert sorted(lb.label for items in annotations.images.values() for lb in items) == [
        "Gato preto",
        "Gato preto",
        "cão",
    ]


def test_existing_classes_are_reused_ignoring_case(tmp_path):
    client, _ = new_project(tmp_path, classes=["Cat"])
    send_zip(client, zip_of(tmp_path, ["cat/1.png", "cat/2.png", "dog/3.png"]))
    report = client.post("/api/projects/1/images/class-folders").json()
    assert report["classes_new"] == ["dog"]
    client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import", "mode": "merge"}
    )
    assert client.get("/api/projects/1").json()["classes"] == ["Cat", "dog"]
    annotations = load_annotations(tmp_path / "ws" / "projects" / "1")
    assert {lb.label for items in annotations.images.values() for lb in items} == {"Cat", "dog"}


def test_merge_keeps_the_class_an_image_already_has(tmp_path):
    client, state = new_project(tmp_path, classes=["bird"])
    send_zip(client, zip_of(tmp_path, ["cat/1.png", "cat/2.png", "dog/3.png"]))
    first = state.manifest(1).active()[0]
    save_annotations(
        state.store.dir(1), AnnotationSet(Task.CLASSIFICATION, ["bird"], {first.id: [Label("bird")]}, {})
    )
    report = client.post("/api/projects/1/images/class-folders").json()
    summary = client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import", "mode": "merge"}
    ).json()
    assert summary["images_added"] == 2 and summary["images_kept"] == 1
    annotations = load_annotations(state.store.dir(1))
    assert [lb.label for lb in annotations.images[first.id]] == ["bird"]


def test_images_outside_class_folders_give_nothing_to_offer(tmp_path):
    client, _ = new_project(tmp_path)
    send_zip(client, zip_of(tmp_path, ["1.png", "2.png", "train/3.png", "val/4.png"]))
    assert client.post("/api/projects/1/images/class-folders").json() == {"found": False}
    assert not list((tmp_path / "ws" / "projects" / "1" / STAGING).glob("*"))  # nothing was kept aside


@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints", "regression"])
def test_only_a_classification_project_can_use_folder_names(tmp_path, task):
    client, _ = new_project(tmp_path, task=task)
    send_zip(client, zip_of(tmp_path, ["cat/1.png", "dog/2.png"]))
    response = client.post("/api/projects/1/images/class-folders")
    assert response.status_code == 400
    assert response.json()["code"] == "class_folders_not_classification"


def test_the_check_can_be_discarded(tmp_path):
    client, _ = new_project(tmp_path)
    send_zip(client, zip_of(tmp_path, ["cat/1.png", "dog/2.png"]))
    token = client.post("/api/projects/1/images/class-folders").json()["token"]
    assert client.delete(f"/api/projects/1/annotations/{token}").json() == {"discarded": True}
    assert client.get("/api/projects/1").json()["classes"] == []
    assert client.get("/api/projects/1/annotations/pending").json()["report"] is None
