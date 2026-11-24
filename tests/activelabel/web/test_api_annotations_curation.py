import json
import time

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task


def wait_job(client, project_id, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        job = next(j for j in client.get(f"/api/projects/{project_id}/jobs").json() if j["id"] == job_id)
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def coco(names, task_segmentation=False):
    images = [{"id": i + 1, "file_name": n, "width": 64, "height": 48} for i, n in enumerate(names)]
    annotation = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 20, 10], "iscrowd": 0, "area": 200}
    if task_segmentation:
        annotation["segmentation"] = [[4, 4, 24, 4, 24, 14, 4, 14]]
    return json.dumps(
        {"images": images, "annotations": [annotation], "categories": [{"id": 1, "name": "cat"}]}
    )


def setup_project(tmp_path, task="detection"):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": task})
    paths = [make_image(tmp_path / f"in/{i}.png", seed=i) for i in range(2)]
    assert upload(client, 1, paths).json()["added"] == 2
    return client, state


def test_annotations_check_and_import(tmp_path):
    client, _ = setup_project(tmp_path)
    assert client.get("/api/projects/1/annotations").json() == {"loaded": False}
    files = [("files", ("instances.json", coco(["0.png", "1.png"]).encode()))]
    report = client.post("/api/projects/1/annotations", files=files).json()
    assert report["format"] == "coco" and report["task_matches"] and "import" in report["options"]
    summary = client.post(f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"})
    assert summary.status_code == 200, summary.text
    loaded = client.get("/api/projects/1/annotations").json()
    assert loaded["loaded"] and loaded["labels_per_class"] == {"cat": 1} and loaded["task_matches"]
    project = client.get("/api/projects/1").json()
    assert project["steps"]["annotations"] == "loaded" and project["classes"] == ["cat"]


def test_annotations_task_mismatch_offers_choices(tmp_path):
    client, _ = setup_project(tmp_path, task="classification")
    files = [("files", ("instances.json", coco(["0.png"], task_segmentation=True).encode()))]
    report = client.post("/api/projects/1/annotations", files=files).json()
    assert not report["task_matches"]
    assert "import" not in report["options"] and "change_task_and_import" in report["options"]
    bad = client.post(f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"})
    assert bad.status_code == 400
    ok = client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "change_task_and_import"}
    )
    assert ok.status_code == 200, ok.text
    assert client.get("/api/projects/1").json()["task"] == "segmentation"


def test_unknown_annotation_format(tmp_path):
    client, _ = setup_project(tmp_path)
    response = client.post("/api/projects/1/annotations", files=[("files", ("notes.txt", b"hello"))])
    assert response.status_code == 400 and "not recognised" in response.json()["detail"]


def test_curation_analyze_thresholds_and_suggestions(tmp_path):
    client, _ = setup_project(tmp_path)
    job = client.post("/api/projects/1/curation/analyze", json={}).json()
    done = wait_job(client, 1, job["id"])
    assert done["status"] == "done", done["error"]
    assert done["result"]["measured"] == 2
    overview = client.get("/api/projects/1/curation").json()
    assert overview["totals"]["measured"] == 2 and "blur_relative" in overview["thresholds"]
    assert client.get("/api/projects/1").json()["steps"]["curation"] == "analyzed"
    changed = client.put("/api/projects/1/curation/thresholds", json={"min_side": 1000}).json()
    assert changed["issues"]["small"] == 2 and changed["thresholds"]["min_side"] == 1000
    assert client.put("/api/projects/1/curation/thresholds", json={"nope": 1}).status_code == 400
    assert isinstance(client.get("/api/projects/1/curation/suggested").json()["ids"], list)
    small = client.get("/api/projects/1/images", params={"issue": "small"}).json()
    assert small["total"] == 2


@pytest.mark.integration
def test_pull_exclusions_needs_dataset(tmp_path):
    client, _ = setup_project(tmp_path)
    response = client.post("/api/projects/1/fiftyone/pull-exclusions")
    assert response.status_code in (200, 400)


def staged(state):
    folder = state.store.dir(1) / "staging"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_files_that_cannot_be_imported_are_not_kept_aside(tmp_path):
    client, state = setup_project(tmp_path)
    nothing = [("files", ("instances.json", coco(["nope1.png", "nope2.png"]).encode()))]  # matches no image
    report = client.post("/api/projects/1/annotations", files=nothing).json()
    assert report["options"] == ["discard"] and staged(state) == []
    assert (
        client.post(
            f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "discard"}
        ).status_code
        == 200
    )
    good = [("files", ("instances.json", coco(["0.png", "1.png"]).encode()))]
    kept = client.post("/api/projects/1/annotations", files=good).json()
    assert staged(state) == [kept["token"]]


def test_old_staging_is_cleaned_at_the_next_check_and_at_start(tmp_path):
    import os

    from activelabel.annotations import cleanup_staging

    client, state = setup_project(tmp_path)
    good = [("files", ("instances.json", coco(["0.png", "1.png"]).encode()))]
    old = client.post("/api/projects/1/annotations", files=good).json()["token"]
    client.post("/api/projects/1/annotations", files=good)
    assert len(staged(state)) == 2
    folder = state.store.dir(1) / "staging"
    long_ago = time.time() - 25 * 3600
    os.utime(folder / old, (long_ago, long_ago))
    cleanup_staging(state.store.dir(1))
    assert old not in staged(state) and len(staged(state)) == 1


def test_ground_truth_class_names_follow_the_project_classes(tmp_path):
    client, _ = setup_project(tmp_path)
    files = [("files", ("instances.json", coco(["0.png", "1.png"]).encode()))]
    token = client.post("/api/projects/1/annotations", files=files).json()["token"]
    client.post(f"/api/projects/1/annotations/{token}/apply", json={"action": "import"})
    assert client.get("/api/projects/1/annotations").json()["class_names"] == ["cat"]
    client.patch("/api/projects/1", json={"classes": ["cat", "dog"]})
    assert client.get("/api/projects/1/annotations").json()["class_names"] == ["cat", "dog"]
    client.patch("/api/projects/1", json={"classes": ["dog"], "classes_action": "remove_annotations"})
    assert client.get("/api/projects/1/annotations").json()["class_names"] == ["dog"]


def test_a_cut_off_json_is_reported_as_unreadable_not_as_unknown_format(tmp_path):
    client, state = setup_project(tmp_path)
    cut = coco(["0.png", "1.png"])[:40].encode()
    response = client.post("/api/projects/1/annotations", files=[("files", ("instances.json", cut))])
    body = response.json()
    assert response.status_code == 400 and body["code"] == "annotation_unreadable"
    assert body["params"]["file"] == "instances.json" and body["params"]["reason"]
    assert "instances.json" in body["detail"] and staged(state) == []
    other = client.post("/api/projects/1/annotations", files=[("files", ("x.json", b'{"a": 1}'))])
    assert other.json()["code"] == "annotation_format_unknown"
    old = b'[{"id": 1, "data": {"image": "/data/upload/1/a-0.png"}, "annotations": [{"result": []}]}]'
    r = client.post("/api/projects/1/annotations", files=[("files", ("export.json", old))])
    assert r.status_code == 400 and r.json()["code"] == "annotation_format_unknown"
    assert "Label Studio" not in r.json()["detail"] and staged(state) == []


def coco_with(names, annotations):
    images = [{"id": i + 1, "file_name": n, "width": 100, "height": 100} for i, n in enumerate(names)]
    categories = [{"id": 1, "name": "cat"}]
    return json.dumps({"images": images, "annotations": annotations, "categories": categories})


def test_import_reports_and_drops_labels_without_the_shape_of_the_task(tmp_path):
    from activelabel.labels import load_annotations

    client, state = setup_project(tmp_path, task="segmentation")
    polygon = [[0, 0, 50, 0, 50, 25, 0, 25]]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25], "segmentation": polygon},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},  # a box, no polygon
    ]
    files = [("files", ("labels.json", coco_with(["0.png", "1.png"], anns).encode()))]
    report = client.post("/api/projects/1/annotations", files=files).json()
    found = {p["code"]: p for p in report["problems"]}
    assert (
        report["task_matches"]
        and found["no_geometry"]["count"] == 1
        and found["no_geometry"]["level"] == "warning"
    )
    summary = client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"}
    ).json()
    assert summary["labels_dropped"] == {"no_geometry": 1} and summary["labels_imported"] == 1
    stored = load_annotations(state.store.dir(1))
    assert sum(len(v) for v in stored.images.values()) == 1
    assert all(lb.polygons for v in stored.images.values() for lb in v)


def test_import_drops_poses_with_another_number_of_points(tmp_path):
    from activelabel.labels import load_annotations

    client, state = setup_project(tmp_path, task="keypoints")
    three = [10, 10, 2, 20, 20, 2, 30, 30, 2]
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25], "keypoints": three},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "keypoints": three[:3]},
    ]
    files = [("files", ("labels.json", coco_with(["0.png", "1.png"], anns).encode()))]
    report = client.post("/api/projects/1/annotations", files=files).json()
    assert "keypoints_count_mismatch" in {p["code"] for p in report["problems"]}
    summary = client.post(
        f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"}
    ).json()
    assert sum(len(v) for v in load_annotations(state.store.dir(1)).images.values()) == 1
    assert summary["labels_dropped"] == {"keypoints_count_mismatch": 1}


def test_changing_the_task_of_a_project_without_labels_rewrites_the_saved_task(tmp_path):
    from activelabel.labels import AnnotationSet, load_annotations, save_annotations
    from activelabel.tasks import Task

    client, state = setup_project(tmp_path)  # detection
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["cat"], {}, {}))  # no images in it
    assert client.patch("/api/projects/1", json={"task": "classification"}).status_code == 200
    assert load_annotations(state.store.dir(1)).task is Task.CLASSIFICATION


def test_issue_any_lists_every_flagged_image_whatever_the_count(tmp_path):
    client, state = setup_project(tmp_path)
    first = state.manifest(1).all()[0]
    state.manifest(1).update({first.id: {"issues": ["blurry", "dark"]}})
    page = client.get("/api/projects/1/images", params={"issue": "any"}).json()
    assert [item["id"] for item in page["items"]] == [first.id]
    assert page["total"] == 1


def test_a_classification_image_item_names_its_classes(tmp_path):
    client, state = setup_project(tmp_path, task="classification")
    first, second = (r.id for r in state.manifest(1).all())
    labels = {first: [Label("cat"), Label("dog")], second: []}
    save_annotations(state.store.dir(1), AnnotationSet(Task.CLASSIFICATION, ["cat", "dog"], labels, {}))
    items = {i["id"]: i for i in client.get("/api/projects/1/images").json()["items"]}
    assert items[first]["classes"] == ["cat", "dog"]
    assert items[second]["classes"] == []


def test_a_detection_image_item_has_no_class_list(tmp_path):
    client, state = setup_project(tmp_path)
    first = state.manifest(1).all()[0].id
    labels = {first: [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]}
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["cat"], labels, {}))
    assert "classes" not in client.get("/api/projects/1/images").json()["items"][0]
