import json
import time

import pytest
from api_client import make_client, upload
from make_video import make_video
from sample_images import make_image

from activelabel.labels import load_annotations
from activelabel.tasks_store import TaskStore

# video of 60 frames at 64x48; the tasks below take frames 10, 20, 30, 40, 50
MOT = "11,3,6.4,4.8,32,24,1,1,1\n21,3,12.8,4.8,32,24,1,1,1\n31,4,0,0,32,24,1,1,1\n6,9,0,0,32,24,1,1,1\n"


def _wait(client, job):
    for _ in range(300):
        current = next(j for j in client.get("/api/projects/1/jobs").json() if j["id"] == job["id"])
        if current["status"] in ("done", "failed", "cancelled"):
            return current
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _setup(tmp_path, tasks=1, task="detection"):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": task})
    path = make_video(tmp_path / "clip.mp4")
    video = client.post("/api/projects/1/videos", files={"file": (path.name, path.read_bytes())}).json()["id"]
    for i in range(tasks):
        job = client.post(
            "/api/projects/1/tasks/from-video",
            json={"video": video, "name": f"T{i}", "start": 10, "stop": 50, "step": 10},
        )
        assert _wait(client, job.json())["status"] == "done"
    return client, state


def _check(client, name, content, **data):
    return client.post("/api/projects/1/annotations", files=[("files", (name, content))], data=data)


def _by_frame(state, task_id=1):
    stored = TaskStore(state.store.dir(1)).get(task_id)
    records = state.manifest(1).load()
    labels = load_annotations(state.store.dir(1)).images
    return {records[i].frame: labels.get(i) for i in stored.images}


def test_mot_default_task_and_apply(tmp_path):
    client, state = _setup(tmp_path)
    response = _check(client, "gt.txt", MOT)
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["format"] == "mot" and report["task"] == 1 and report["task_matches"]
    assert report["counts"]["images_matched"] == 3 and report["counts"]["labels_total"] == 3
    assert report["counts"]["images_without_labels"] == 2
    problem = next(p for p in report["problems"] if p["code"] == "frames_not_in_task")
    assert problem["count"] == 1 and problem["examples"] == ["frame 6 (MOT)"]
    assert report["can_import"] and "import" in report["options"]
    done = client.post(f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"})
    assert done.status_code == 200, done.text
    got = _by_frame(state)
    assert [(lb.label, lb.track) for lb in got[10]] == [("object", 3)]
    assert got[10][0].bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert got[30][0].track == 4 and got[40] is None and got[50] is None
    assert client.get("/api/projects/1/annotations").json()["labels_total"] == 3


def test_cvat_video_with_chosen_task_and_class_name(tmp_path):
    client, state = _setup(tmp_path, tasks=2)
    xml = (
        "<annotations><meta><task><start_frame>10</start_frame><stop_frame>50</stop_frame>"
        "<frame_filter>step=10</frame_filter></task></meta>"
        '<track id="2" label="car">'
        '<box frame="1" outside="0" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="32" ybr="24"/>'
        '<box frame="2" outside="1" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="32" ybr="24"/>'
        "</track></annotations>"
    )
    assert _check(client, "v.xml", xml).status_code == 400  # two video tasks: choose one
    report = _check(client, "v.xml", xml, task="2").json()
    assert report["format"] == "cvatvideo" and report["task"] == 2
    client.post(f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"})
    got = _by_frame(state, 2)
    assert got[20][0].track == 2 and got[20][0].label == "car"
    assert got[30] == [] and got[10] == []  # inside the range, nothing visible
    assert all(v is None for v in _by_frame(state, 1).values())


def test_yolo_frames_and_scoped_replace(tmp_path):
    client, state = _setup(tmp_path, tasks=2)
    files = [
        ("files", ("frame_000010.txt", "0 0.5 0.5 0.5 0.5\n")),
        ("files", ("frame_000099.txt", "0 0.5 0.5 0.5 0.5\n")),
        ("files", ("classes.txt", "car\n")),
    ]
    report = client.post("/api/projects/1/annotations", files=files, data={"task": "1"}).json()
    assert report["format"] == "yolo" and report["counts"]["labels_total"] == 1
    assert report["problems"][0]["code"] == "frames_not_in_task"
    assert (
        client.post(
            f"/api/projects/1/annotations/{report['token']}/apply", json={"action": "import"}
        ).status_code
        == 200
    )
    assert [lb.label for lb in _by_frame(state, 1)[10]] == ["car"]
    # replace for task 2 keeps the labels of task 1
    files = [("files", ("frame_20.txt", "0 0.5 0.5 0.2 0.2\n")), ("files", ("classes.txt", "car\n"))]
    report = client.post("/api/projects/1/annotations", files=files, data={"task": "2"}).json()
    body = {"action": "import", "mode": "replace"}
    assert client.post(f"/api/projects/1/annotations/{report['token']}/apply", json=body).status_code == 200
    assert len(_by_frame(state, 1)[10]) == 1 and len(_by_frame(state, 2)[20]) == 1


def test_image_formats_unchanged_without_task(tmp_path):
    client, state = _setup(tmp_path)
    names = [r.name for r in state.manifest(1).load().values()]
    coco = json.dumps(
        {
            "images": [{"id": 1, "file_name": names[0], "width": 64, "height": 48}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            "categories": [{"id": 1, "name": "cat"}],
        }
    )
    report = _check(client, "c.json", coco).json()
    assert report["format"] == "coco" and report["task"] is None and report["counts"]["images_matched"] == 1


def test_errors(tmp_path):
    client, _ = _setup(tmp_path)
    assert _check(client, "gt.txt", MOT, task="9").status_code == 400  # no such task
    plain, _ = make_client(tmp_path / "plain")
    plain.post("/api/projects", json={"name": "P", "task": "detection"})
    upload(plain, 1, [make_image(tmp_path / "in/a.png")])
    response = _check(plain, "gt.txt", MOT)
    assert response.status_code == 400 and "video" in response.json()["detail"]
