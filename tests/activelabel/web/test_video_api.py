import collections
import threading
import time
from io import BytesIO

from api_client import make_client
from make_video import make_video
from PIL import Image

import activelabel.images.video as video_module
from activelabel.images import ingest
from activelabel.tasks_store import TaskStore


def _setup(tmp_path, **kw):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification"})
    path = make_video(tmp_path / "clip.mp4", **kw)
    return client, state, path


def _upload(client, path):
    return client.post("/api/projects/1/videos", files={"file": (path.name, path.read_bytes())})


def _wait(client, job):
    for _ in range(300):
        current = next(j for j in client.get("/api/projects/1/jobs").json() if j["id"] == job["id"])
        if current["status"] in ("done", "failed", "cancelled"):
            return current
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _make_task(client, **body):
    response = client.post("/api/projects/1/tasks/from-video", json=body)
    assert response.status_code == 200, response.text
    return _wait(client, response.json())


def _delete_task(client, task_id):
    return client.request("DELETE", f"/api/projects/1/tasks/{task_id}", json={"confirm": True})


def test_upload_probe_and_same_video_twice(tmp_path):
    client, state, path = _setup(tmp_path)
    first = _upload(client, path)
    assert first.status_code == 200
    info = first.json()
    assert (round(info["fps"]), info["frames"], info["width"], info["height"]) == (30, 60, 64, 48)
    assert info["name"] == path.name and info["size_bytes"] == path.stat().st_size
    assert abs(info["duration"] - 2.0) < 0.1 and len(info["id"]) == 16
    assert _upload(client, path).json() == info
    assert client.get("/api/projects/1/videos").json() == {"videos": [info]}
    files = [p for p in (state.store.dir(1) / "videos").iterdir() if not p.name.startswith(".")]
    assert len(files) == 1 and files[0].stem == info["id"]


def test_invalid_files(tmp_path):
    client, state, _ = _setup(tmp_path)
    bad = client.post("/api/projects/1/videos", files={"file": ("a.mp4", b"not a video at all")})
    assert bad.status_code == 400 and bad.json()["detail"]
    wrong = client.post("/api/projects/1/videos", files={"file": ("a.txt", b"hello")})
    assert wrong.status_code == 400
    assert client.get("/api/projects/1/videos").json() == {"videos": []}
    assert list((state.store.dir(1) / "videos").iterdir()) == []


def test_plan(tmp_path):
    client, _, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    url = f"/api/projects/1/videos/{vid}/plan"
    assert client.post(url, json={"step": 7}).json() == {"frames": 9, "step": 7, "first": 0, "last": 56}
    got = client.post(url, json={"every_seconds": 0.5, "start": 10, "stop": 50}).json()
    assert got == {"frames": 3, "step": 15, "first": 10, "last": 40}
    for bad in [{"start": 60}, {"start": 5, "stop": 4}, {"step": 0}, {"every_seconds": 0}]:
        assert client.post(url, json=bad).status_code == 400
    assert client.post("/api/projects/1/videos/nope/plan", json={}).status_code == 404


def test_task_from_video(tmp_path):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    job = _make_task(client, video=vid, name="Clip", job_size=4, start=10, stop=50, step=10)
    assert job["status"] == "done", job
    assert job["kind"] == "video_task"
    result = job["result"]
    assert (result["frames"], result["step"]) == (5, 10)
    task = result["task"]
    assert (task["name"], task["job_size"], task["open"], task["images"]) == ("Clip", 4, False, 5)
    assert task["source"]["video"] == vid and task["source"]["step"] == 10
    assert (task["source"]["start"], task["source"]["stop"]) == (10, 50)
    assert round(task["source"]["fps"]) == 30
    assert [j["images"] for j in task["jobs"]] == [4, 1]
    stored = TaskStore(state.store.dir(1)).get(task["id"])
    records = state.manifest(1).load()
    ordered = [records[i] for i in stored.images]
    assert [r.name for r in ordered] == [f"clip/frame_{n:06d}.jpg" for n in (10, 20, 30, 40, 50)]
    assert [r.frame for r in ordered] == [10, 20, 30, 40, 50]
    assert all(r.video == vid and abs(r.time - r.frame / 30) < 0.01 for r in ordered)
    assert all((state.store.dir(1) / r.file).exists() for r in ordered)
    assert [p for p in (state.store.dir(1) / "videos").iterdir() if p.name.startswith(".")] == []
    listed = client.get("/api/projects/1/tasks").json()["tasks"][0]
    assert listed["source"] == task["source"]


def test_default_name_and_every_seconds(tmp_path):
    client, _, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    job = _make_task(client, video=vid, every_seconds=1)
    assert job["result"]["step"] == 30 and job["result"]["frames"] == 2
    assert job["result"]["task"]["name"] == "clip"
    assert job["result"]["task"]["source"]["step"] == 30


def test_bad_task_requests(tmp_path):
    client, _, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    url = "/api/projects/1/tasks/from-video"
    assert client.post(url, json={"video": vid, "start": 99}).status_code == 400
    assert client.post(url, json={"video": vid, "job_size": 0}).status_code == 400
    assert client.post(url, json={"video": "nope"}).status_code == 404


def test_identical_frames_are_all_stored(tmp_path):
    client, state, path = _setup(tmp_path, frames=12, same=True)
    vid = _upload(client, path).json()["id"]
    job = _make_task(client, video=vid, step=2)
    assert job["status"] == "done", job
    assert job["result"]["frames"] == 6
    assert _make_task(client, video=vid, step=2)["status"] == "done"  # same frames, another task
    assert len(state.manifest(1).load()) == 12
    tasks = TaskStore(state.store.dir(1)).load()
    assert [len(t.images) for t in tasks] == [6, 6] and not set(tasks[0].images) & set(tasks[1].images)


def test_cancel_leaves_nothing(tmp_path, monkeypatch):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    real = ingest._Ingester.tick
    calls = []

    def tick(self, message):
        calls.append(message)
        if len(calls) == 6:
            state.jobs.cancel(state.jobs.list(1)[0].id)
        real(self, message)

    monkeypatch.setattr(ingest._Ingester, "tick", tick)
    job = _make_task(client, video=vid, step=2)
    assert job["status"] == "cancelled"
    assert TaskStore(state.store.dir(1)).load() == []
    assert state.manifest(1).load() == {}
    images = state.store.dir(1) / "images"
    assert not images.exists() or not list(images.iterdir())
    assert [p for p in (state.store.dir(1) / "videos").iterdir() if p.name.startswith(".frames")] == []


def test_cancel_during_extraction(tmp_path, monkeypatch):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    real = video_module.extract_frames

    def extract(video, start, stop, step, target, progress=None):
        state.jobs.cancel(state.jobs.list(1)[0].id)
        return real(video, start, stop, step, target, progress)

    monkeypatch.setattr("activelabel.web.video_api.extract_frames", extract)
    assert _make_task(client, video=vid)["status"] == "cancelled"
    assert TaskStore(state.store.dir(1)).load() == []
    assert state.manifest(1).load() == {}


def test_frame_preview(tmp_path):
    client, _, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    response = client.get(f"/api/projects/1/videos/{vid}/frame", params={"n": 5, "w": 32})
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(response.content)).size == (32, 24)
    big = client.get(f"/api/projects/1/videos/{vid}/frame", params={"n": 5})
    assert Image.open(BytesIO(big.content)).size == (64, 48)  # never enlarged
    assert client.get(f"/api/projects/1/videos/{vid}/frame", params={"n": 60}).status_code == 400
    assert client.get("/api/projects/1/videos/nope/frame", params={"n": 0}).status_code == 404


def test_delete_rules(tmp_path):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    video_file = next((state.store.dir(1) / "videos").glob(f"{vid}.*"))
    task_a = _make_task(client, video=vid, step=20)["result"]["task"]
    task_b = _make_task(client, video=vid, step=30)["result"]["task"]
    assert client.delete(f"/api/projects/1/videos/{vid}").status_code == 409
    assert _delete_task(client, task_a["id"]).status_code == 200
    assert video_file.exists()  # task b still uses it
    assert _delete_task(client, task_b["id"]).status_code == 200
    assert not video_file.exists()
    assert client.get("/api/projects/1/videos").json() == {"videos": []}


def test_delete_unused_video(tmp_path):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    assert client.delete(f"/api/projects/1/videos/{vid}").status_code == 200
    assert not list((state.store.dir(1) / "videos").glob(f"{vid}.*"))
    assert client.delete(f"/api/projects/1/videos/{vid}").status_code == 404


def test_early_end_stores_the_real_last_frame(tmp_path, monkeypatch):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    real = video_module.extract_frames

    def short(video, start, stop, step, target, progress=None):
        return real(video, start, stop, step, target, progress)[:3]  # the file ended early

    monkeypatch.setattr("activelabel.web.video_api.extract_frames", short)
    job = _make_task(client, video=vid, start=0, stop=50, step=10)
    assert job["status"] == "done", job
    assert (job["result"]["frames"], job["result"]["planned"], job["result"]["stopped_early"]) == (3, 6, True)
    assert job["result"]["task"]["source"]["stop"] == 20


def test_delete_video_refused_while_a_video_task_runs(tmp_path):
    client, state, path = _setup(tmp_path)
    vid = _upload(client, path).json()["id"]
    release = threading.Event()
    state.jobs.submit(1, "video_task", lambda progress: release.wait(5) and {})
    try:
        assert client.delete(f"/api/projects/1/videos/{vid}").status_code == 409
    finally:
        release.set()


def test_selection_too_big_or_without_disk_space_is_refused(tmp_path, monkeypatch):
    client, state, path = _setup(tmp_path)
    video = _upload(client, path).json()["id"]
    monkeypatch.setattr(video_module, "MAX_TASK_FRAMES", 10)
    big = client.post("/api/projects/1/tasks/from-video", json={"video": video, "step": 1})
    assert big.status_code == 400 and "at most 10" in big.json()["detail"]
    monkeypatch.setattr(video_module, "MAX_TASK_FRAMES", 20_000)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(video_module.shutil, "disk_usage", lambda _: usage(10**9, 10**9 - 1000, 1000))
    full = client.post("/api/projects/1/tasks/from-video", json={"video": video, "step": 1})
    assert full.status_code == 400 and "free disk space" in full.json()["detail"]
    assert not TaskStore(state.store.dir(1)).load()


def test_video_errors_have_codes_and_one_file_per_upload(tmp_path):
    client, state, path = _setup(tmp_path)
    data = path.read_bytes()
    two = client.post("/api/projects/1/videos", files=[("file", ("a.mp4", data)), ("file", ("b.mp4", data))])
    assert two.status_code == 400 and two.json()["code"] == "one_file_only"
    assert client.get("/api/projects/1/videos").json() == {"videos": []}  # nothing was kept
    wrong = client.post("/api/projects/1/videos", files={"file": ("a.txt", b"hello")})
    assert wrong.json()["code"] == "video_type_unsupported" and ".mp4" in wrong.json()["params"]["allowed"]
    bad = client.post("/api/projects/1/videos", files={"file": ("a.mp4", b"not a video at all")})
    assert bad.json()["code"] == "video_unreadable"
    video = _upload(client, path).json()["id"]
    assert (
        client.get("/api/projects/1/videos/nope/frame", params={"n": 0}).json()["code"] == "video_not_found"
    )
    far = client.get(f"/api/projects/1/videos/{video}/frame", params={"n": 9999})
    assert far.json()["code"] == "frame_out_of_range"
    plan = client.post(f"/api/projects/1/videos/{video}/plan", json={"start": 5, "stop": 2})
    assert plan.json()["code"] == "video_range_invalid" and plan.json()["params"]["frames"] == 60


def test_two_tasks_from_one_video_get_different_names(tmp_path):
    client, state, path = _setup(tmp_path)
    video = _upload(client, path).json()["id"]
    first = _make_task(client, video=video, step=30)["result"]["task"]["name"]
    second = _make_task(client, video=video, step=30)["result"]["task"]["name"]
    assert first == "clip" and second == "clip (2)"
