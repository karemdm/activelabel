import threading
import time

from api_client import make_client, upload
from sample_images import make_image

from activelabel.curation.issues import Thresholds, curation_path, load_thresholds, save_thresholds
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.storage import read_json, write_json
from activelabel.tasks import Task

POLY = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4]]


def setup(tmp_path, task="segmentation", with_labels=True):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": task})
    paths = [make_image(tmp_path / f"in/{i}.png", seed=i) for i in range(3)]
    upload(client, 1, paths)
    ids = [r.id for r in state.manifest(1).all()]
    if with_labels:
        images = {
            ids[0]: [Label("cat", bbox=[0.1, 0.1, 0.4, 0.3], polygons=[POLY]), Label("dog", polygons=[POLY])],
            ids[1]: [Label("cat", polygons=[POLY])],
            ids[2]: [],
        }
        gt = AnnotationSet(Task(task), ["cat", "dog"], images, {"format": "coco"})
        save_annotations(state.store.dir(1), gt)
    return client, state, ids


def test_task_change_without_decision_is_409(tmp_path):
    client, state, _ = setup(tmp_path)
    response = client.patch("/api/projects/1", json={"task": "detection"})
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "task_change_annotations",
        "message": (
            "This project has annotations for Segmentation (2 images). "
            "Changing the task to Detection needs a decision about them."
        ),
        "from_task": "segmentation",
        "to_task": "detection",
        "images": 2,
        "labels": 3,
        "options": ["convert", "discard"],
    }
    assert client.get("/api/projects/1").json()["task"] == "segmentation"
    assert load_annotations(state.store.dir(1)).task is Task.SEGMENTATION


def test_conversion_impossible_offers_only_discard(tmp_path):
    client, _, _ = setup(tmp_path)
    detail = client.patch("/api/projects/1", json={"task": "keypoints"}).json()["detail"]
    assert detail["options"] == ["discard"]
    bad = client.patch("/api/projects/1", json={"task": "keypoints", "annotations": "convert"})
    assert bad.status_code == 400
    assert client.get("/api/projects/1").json()["task"] == "segmentation"


def test_convert(tmp_path):
    client, state, ids = setup(tmp_path)
    response = client.patch("/api/projects/1", json={"task": "classification", "annotations": "convert"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "classification"
    assert body["annotations_result"] == {"action": "convert", "images": 2, "labels": 2}
    saved = load_annotations(state.store.dir(1))
    assert saved.task is Task.CLASSIFICATION and saved.source["converted_from"] == "segmentation"
    assert [lb.label for lb in saved.images[ids[0]]] == ["cat"]
    assert body["steps"]["annotations"] == "loaded"


def test_discard_keeps_a_backup(tmp_path):
    client, state, _ = setup(tmp_path)
    response = client.patch("/api/projects/1", json={"task": "keypoints", "annotations": "discard"})
    assert response.status_code == 200
    body = response.json()
    assert body["annotations_result"] == {"action": "discard", "images": 2, "labels": 3}
    assert body["steps"]["annotations"] == "none"
    folder = state.store.dir(1)
    assert load_annotations(folder) is None
    backups = list(folder.glob("annotations.*.bak.json"))
    assert len(backups) == 1 and read_json(backups[0])["task"] == "segmentation"


def test_no_decision_needed_without_ground_truth(tmp_path):
    client, _, _ = setup(tmp_path, with_labels=False)
    response = client.patch("/api/projects/1", json={"task": "detection"})
    assert response.status_code == 200 and "annotations_result" not in response.json()


def test_no_decision_needed_when_the_task_does_not_change(tmp_path):
    client, state, _ = setup(tmp_path)
    ok = client.patch("/api/projects/1", json={"name": "Renamed", "annotations": "discard"})
    assert ok.status_code == 200 and "annotations_result" not in ok.json()
    assert load_annotations(state.store.dir(1)) is not None
    same = client.patch("/api/projects/1", json={"task": "segmentation"})
    assert same.status_code == 200


def test_reset_curation(tmp_path):
    client, state, ids = setup(tmp_path, with_labels=False)
    folder = state.store.dir(1)
    manifest = state.manifest(1)
    manifest.update({i: {"metrics": {"brightness": 100.0}, "issues": ["dark"], "phash": "ff"} for i in ids})
    save_thresholds(folder, Thresholds(dark=50.0))
    write_json(curation_path(folder), {**read_json(curation_path(folder)), "duplicate_of": {"a": "b"}})
    assert client.get("/api/projects/1").json()["steps"]["curation"] == "analyzed"
    assert client.delete("/api/projects/1/curation").json() == {"reset": 3}
    overview = client.get("/api/projects/1").json()
    assert overview["steps"]["curation"] == "not_analyzed" and overview["counts"]["with_issues"] == 0
    assert all(not r.phash and not r.metrics and not r.issues for r in manifest.all())
    assert load_thresholds(folder).dark == 50.0
    assert read_json(curation_path(folder))["duplicate_of"] == {}
    assert client.delete("/api/projects/9/curation").status_code == 404


def test_reset_curation_is_refused_while_a_job_runs(tmp_path):
    client, state, _ = setup(tmp_path, with_labels=False)
    release = threading.Event()
    state.jobs.submit(1, "slow", lambda progress: release.wait(5) and {})
    try:
        assert client.delete("/api/projects/1/curation").status_code == 409
    finally:
        release.set()


def test_clear_jobs(tmp_path):
    client, state, _ = setup(tmp_path, with_labels=False)
    job = state.jobs.submit(1, "x", lambda progress: {})
    while job.status not in ("done", "failed"):
        time.sleep(0.01)
    assert client.delete("/api/projects/1/jobs").json() == {"removed": 1}
    assert client.get("/api/projects/1/jobs").json() == []
    assert client.delete("/api/projects/9/jobs").status_code == 404


def test_removing_used_classes_needs_a_decision(tmp_path):
    client, state, _ = setup(tmp_path)
    response = client.patch("/api/projects/1", json={"classes": ["Cat"]})
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "classes_in_use",
        "message": "The class dog is used by 1 annotations in 1 images.",
        "classes": [{"name": "dog", "labels": 1, "images": 1}],
        "options": ["remove_annotations"],
    }
    assert load_annotations(state.store.dir(1)).images and not list(state.store.dir(1).glob("*.bak.json"))
    assert client.get("/api/projects/1").json()["classes"] == []


def test_remove_class_annotations(tmp_path):
    client, state, ids = setup(tmp_path)
    response = client.patch(
        "/api/projects/1", json={"classes": ["cat"], "classes_action": "remove_annotations"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["classes"] == ["cat"] and body["classes_result"] == {"removed_labels": 1, "images": 1}
    saved = load_annotations(state.store.dir(1))
    assert [lb.label for lb in saved.images[ids[0]]] == ["cat"]
    assert len(list(state.store.dir(1).glob("annotations.*.bak.json"))) == 1


def test_adding_or_reordering_classes_needs_no_decision(tmp_path):
    client, _, _ = setup(tmp_path)
    ok = client.patch("/api/projects/1", json={"classes": ["dog", "cat", "bird"]})
    assert ok.status_code == 200 and "classes_result" not in ok.json()


def test_task_decision_comes_before_classes(tmp_path):
    client, _, _ = setup(tmp_path)
    detail = client.patch("/api/projects/1", json={"task": "detection", "classes": ["cat"]}).json()["detail"]
    assert detail["code"] == "task_change_annotations"
