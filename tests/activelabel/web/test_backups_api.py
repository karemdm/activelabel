import json

from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import load_annotations


def project_with_labels(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["cat", "dog"]})
    assert upload(client, 1, [make_image(tmp_path / "in/a.png", seed=1)]).json()["added"] == 1
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 48}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 20, 10], "iscrowd": 0, "area": 200},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [30, 4, 20, 10], "iscrowd": 0, "area": 200},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    }
    files = [("files", ("instances.json", json.dumps(coco).encode()))]
    token = client.post("/api/projects/1/annotations", files=files).json()["token"]
    assert (
        client.post(f"/api/projects/1/annotations/{token}/apply", json={"action": "import"}).status_code
        == 200
    )
    return client, state


def test_removing_a_class_makes_a_copy_that_can_be_listed_and_restored(tmp_path):
    client, state = project_with_labels(tmp_path)
    assert client.get("/api/projects/1/annotations/backups").json() == {"items": []}
    done = client.patch("/api/projects/1", json={"classes": ["cat"], "classes_action": "remove_annotations"})
    assert done.status_code == 200, done.text
    items = client.get("/api/projects/1/annotations/backups").json()["items"]
    assert len(items) == 1
    copy = items[0]
    assert copy["name"].endswith("Z.classes.bak.json") and copy["created"].endswith("+00:00")
    assert (copy["images"], copy["labels"], copy["reason"], copy["valid"]) == (1, 2, "classes", True)
    assert sum(len(v) for v in load_annotations(state.store.dir(1)).images.values()) == 1
    restored = client.post(f"/api/projects/1/annotations/backups/{copy['name']}/restore")
    assert restored.status_code == 200, restored.text
    assert restored.json()["labels"] == 2 and restored.json()["backup"].endswith(".restore.bak.json")
    assert sum(len(v) for v in load_annotations(state.store.dir(1)).images.values()) == 2
    assert client.get("/api/projects/1").json()["classes"] == ["cat", "dog"]  # the removed class is back
    assert len(client.get("/api/projects/1/annotations/backups").json()["items"]) == 2  # + the replaced file


def test_restore_validates_the_name_and_the_project(tmp_path):
    client, _ = project_with_labels(tmp_path)
    r = client.post("/api/projects/1/annotations/backups/annotations.json/restore")
    assert r.status_code == 400 and r.json()["code"] == "backup_name_invalid"
    r = client.post("/api/projects/1/annotations/backups/..%2Fproject.json/restore")
    assert r.status_code in (400, 404)
    r = client.post("/api/projects/1/annotations/backups/annotations.20990101T000000Z.bak.json/restore")
    assert r.status_code == 404 and r.json()["code"] == "backup_not_found"
    assert client.get("/api/projects/9/annotations/backups").status_code == 404
    assert client.post("/api/projects/9/annotations/backups/x/restore").status_code == 404


def test_restore_of_a_damaged_copy_is_refused_and_changes_nothing(tmp_path):
    client, state = project_with_labels(tmp_path)
    folder = state.store.dir(1)
    (folder / "annotations.20990101T000000Z.bak.json").write_text("{broken", encoding="utf-8")
    r = client.post("/api/projects/1/annotations/backups/annotations.20990101T000000Z.bak.json/restore")
    assert r.status_code == 400 and r.json()["code"] == "backup_unreadable"
    assert not list(folder.glob("annotations.*.restore.bak.json"))


def test_restore_waits_for_a_running_activity(tmp_path):
    client, state = project_with_labels(tmp_path)
    client.patch("/api/projects/1", json={"classes": ["cat"], "classes_action": "remove_annotations"})
    name = client.get("/api/projects/1/annotations/backups").json()["items"][0]["name"]
    state.jobs.active = lambda project_id: True  # type: ignore[method-assign]
    r = client.post(f"/api/projects/1/annotations/backups/{name}/restore")
    assert r.status_code == 409 and r.json()["code"] == "busy"
