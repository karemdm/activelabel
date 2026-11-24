import time

from api_client import make_client, upload
from sample_images import make_image

from activelabel.tasks_store import TaskStore


def _setup(tmp_path, n=5):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification"})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(n)]
    return client, state, paths


def _files(paths):
    return [("files", (p.name, p.read_bytes())) for p in paths]


def _names(client, **params):
    return [i["name"] for i in client.get("/api/projects/1/images", params=params).json()["items"]]


def test_create_task_with_files(tmp_path):
    client, _, paths = _setup(tmp_path, 5)
    response = client.post(
        "/api/projects/1/tasks", files=_files(paths), data={"name": "Field", "job_size": "2", "open": "false"}
    )
    assert response.status_code == 200
    task = response.json()["task"]
    assert response.json()["report"]["added"] == 5
    assert (task["id"], task["name"], task["job_size"], task["open"], task["images"]) == (
        1,
        "Field",
        2,
        False,
        5,
    )
    jobs = [(j["n"], j["first"], j["last"], j["images"]) for j in task["jobs"]]
    assert jobs == [(1, 1, 2, 2), (2, 3, 4, 2), (3, 5, 5, 1)]
    assert client.get("/api/projects/1/tasks").json()["tasks"][0]["name"] == "Field"
    assert client.get("/api/projects/1").json()["counts"]["tasks"] == 1
    bad = client.post("/api/projects/1/tasks", files=_files(paths), data={"job_size": "0"})
    assert bad.status_code == 400


def test_chunked_append_then_close(tmp_path):
    client, _, paths = _setup(tmp_path, 4)
    first = client.post("/api/projects/1/tasks", files=_files(paths[:2]), data={"name": "Big"}).json()
    tid = first["task"]["id"]
    assert first["task"]["open"] is True
    appended = client.post(f"/api/projects/1/tasks/{tid}/images", files=_files(paths[2:]))
    assert appended.status_code == 200 and appended.json()["added"] == 2
    again = client.post(f"/api/projects/1/tasks/{tid}/images", files=_files(paths[:1]))
    assert again.json()["added"] == 0 and again.json()["duplicates_count"] == 1
    closed = client.patch(f"/api/projects/1/tasks/{tid}", json={"open": False}).json()
    assert closed["open"] is False and closed["images"] == 4
    assert _names(client, task=tid) == [p.name for p in paths]  # upload order
    refused = client.post(f"/api/projects/1/tasks/{tid}/images", files=_files(paths[:1]))
    assert refused.status_code == 409
    assert client.post("/api/projects/1/tasks/9/images", files=_files(paths[:1])).status_code == 404


def test_rename_and_delete_with_confirm(tmp_path):
    client, _, paths = _setup(tmp_path, 3)
    client.post("/api/projects/1/tasks", files=_files(paths[:2]), data={"name": "A", "open": "false"})
    client.post("/api/projects/1/tasks", files=_files(paths[2:]), data={"name": "B", "open": "false"})
    assert client.patch("/api/projects/1/tasks/1", json={"name": "Renamed"}).json()["name"] == "Renamed"
    assert client.patch("/api/projects/1/tasks/1", json={"name": " "}).status_code == 400
    assert client.patch("/api/projects/1/tasks/1", json={"open": True}).status_code == 409
    assert client.request("DELETE", "/api/projects/1/tasks/1", json={}).status_code == 400
    assert client.get("/api/projects/1/images", params={"status": "all"}).json()["total"] == 3
    result = client.request("DELETE", "/api/projects/1/tasks/1", json={"confirm": True})
    assert result.json() == {"deleted": 2}
    assert [t["id"] for t in client.get("/api/projects/1/tasks").json()["tasks"]] == [2]
    assert client.get("/api/projects/1/images", params={"status": "all"}).json()["total"] == 1
    assert client.request("DELETE", "/api/projects/1/tasks/1", json={"confirm": True}).status_code == 404


def test_task_and_job_filters(tmp_path):
    client, _, paths = _setup(tmp_path, 5)
    client.post("/api/projects/1/tasks", files=_files(paths[:1]), data={"open": "false"})
    client.post("/api/projects/1/tasks", files=_files(list(reversed(paths[1:]))), data={"job_size": "3"})
    names = [p.name for p in paths]
    reverse = list(reversed(names[1:]))
    assert _names(client, task=1) == names[:1]
    assert _names(client, task=2) == reverse
    assert _names(client, task=2, job=1) == reverse[:3]
    assert _names(client, task=2, job=2) == reverse[3:]
    assert client.get("/api/projects/1/images", params={"job": 1}).status_code == 400
    assert client.get("/api/projects/1/images", params={"task": 7}).status_code == 404
    assert client.get("/api/projects/1/images", params={"task": 2, "job": 5}).status_code == 404


def test_view_counts_annotated_and_active(tmp_path):
    client, _, paths = _setup(tmp_path, 3)
    client.post("/api/projects/1/tasks", files=_files(paths), data={"job_size": "2"})
    items = client.get("/api/projects/1/images", params={"task": 1}).json()["items"]
    client.post("/api/projects/1/images/exclude", json={"ids": [items[2]["id"]]})
    task = client.get("/api/projects/1/tasks").json()["tasks"][0]
    assert (task["images"], task["active"], task["annotated"]) == (3, 2, 0)
    assert [j["active"] for j in task["jobs"]] == [2, 0]


def test_old_upload_creates_a_task(tmp_path):
    client, _, paths = _setup(tmp_path, 3)
    first = upload(client, 1, paths[:2])
    assert first.status_code == 200 and first.json()["added"] == 2 and first.json()["task"] == 1
    second = upload(client, 1, paths[1:])  # one duplicate, one new
    assert second.json()["added"] == 1 and second.json()["duplicates_count"] == 1
    tasks = client.get("/api/projects/1/tasks").json()["tasks"]
    assert [(t["name"], t["images"], t["open"]) for t in tasks] == [
        ("Task 1", 2, False),
        ("Task 2", 1, False),
    ]
    nothing = upload(client, 1, paths)
    assert nothing.json()["added"] == 0 and "task" not in nothing.json()
    assert len(client.get("/api/projects/1/tasks").json()["tasks"]) == 2


def test_upload_to_a_project_without_tasks_json_migrates_first(tmp_path):
    client, state, paths = _setup(tmp_path, 3)
    upload(client, 1, paths[:2])
    (state.store.dir(1) / "tasks.json").unlink()
    upload(client, 1, paths[2:])
    tasks = client.get("/api/projects/1/tasks").json()["tasks"]
    assert [t["images"] for t in tasks] == [2, 1]


def test_image_delete_removes_it_from_its_task(tmp_path):
    client, state, paths = _setup(tmp_path, 3)
    upload(client, 1, paths[:2])
    upload(client, 1, paths[2:])
    in_task_1 = client.get("/api/projects/1/images", params={"task": 1}).json()["items"]
    in_task_2 = client.get("/api/projects/1/images", params={"task": 2}).json()["items"]
    client.post("/api/projects/1/images/delete", json={"ids": [in_task_1[0]["id"]], "confirm": True})
    assert [len(t.images) for t in TaskStore(state.store.dir(1)).load()] == [1, 1]
    client.post("/api/projects/1/images/delete", json={"ids": [in_task_2[0]["id"]], "confirm": True})
    assert [t.id for t in TaskStore(state.store.dir(1)).load()] == [1]  # task 2 became empty


def test_folder_import_creates_a_task(tmp_path):
    root = tmp_path / "root"
    for k in range(2):
        make_image(root / "set" / f"{k}.png", seed=k, fmt="PNG")
    client, _ = make_client(tmp_path, import_root=root)
    client.post("/api/projects", json={"name": "P", "task": "classification"})
    job = client.post("/api/projects/1/images/folder", json={"path": str(root / "set")}).json()
    end = time.time() + 10
    status = {"status": "running"}
    while time.time() < end and status["status"] not in ("done", "failed"):
        status = next(j for j in client.get("/api/projects/1/jobs").json() if j["id"] == job["id"])
        time.sleep(0.05)
    assert status["status"] == "done"
    tasks = client.get("/api/projects/1/tasks").json()["tasks"]
    assert [(t["name"], t["images"]) for t in tasks] == [("Task 1", 2)]


def test_invalid_name_or_job_size_stores_no_image(tmp_path):
    client, state, paths = _setup(tmp_path, 2)
    assert (
        client.post("/api/projects/1/tasks", files=_files(paths), data={"name": "x" * 101}).status_code == 400
    )
    assert (
        client.post("/api/projects/1/tasks", files=_files(paths), data={"job_size": "-3"}).status_code == 400
    )
    assert client.get("/api/projects/1/images", params={"status": "all"}).json()["total"] == 0
    assert TaskStore(state.store.dir(1)).load() == []


def test_append_to_closed_or_unknown_task_stores_nothing(tmp_path):
    client, _, paths = _setup(tmp_path, 3)
    tid = client.post("/api/projects/1/tasks", files=_files(paths[:1])).json()["task"]["id"]
    client.patch(f"/api/projects/1/tasks/{tid}", json={"open": False})
    assert client.post(f"/api/projects/1/tasks/{tid}/images", files=_files(paths[1:])).status_code == 409
    assert client.post("/api/projects/1/tasks/9/images", files=_files(paths[1:])).status_code == 404
    assert client.get("/api/projects/1/images", params={"status": "all"}).json()["total"] == 1


def test_patch_open_true_is_idempotent_on_open_task_only(tmp_path):
    client, _, paths = _setup(tmp_path, 1)
    tid = client.post("/api/projects/1/tasks", files=_files(paths)).json()["task"]["id"]
    again = client.patch(f"/api/projects/1/tasks/{tid}", json={"open": True})
    assert again.status_code == 200 and again.json()["open"] is True
    client.patch(f"/api/projects/1/tasks/{tid}", json={"open": False})
    assert client.patch(f"/api/projects/1/tasks/{tid}", json={"open": True}).status_code == 409


def test_orphan_image_goes_to_a_new_task(tmp_path):
    client, state, paths = _setup(tmp_path, 2)
    upload(client, 1, paths[:1])
    from activelabel.images.ingest import ingest_files

    manifest = state.manifest(1)
    report = ingest_files(manifest, [(paths[1].name, paths[1].open("rb"))])
    tasks = client.get("/api/projects/1/tasks").json()["tasks"]
    assert [(t["name"], t["images"], t["open"]) for t in tasks] == [
        ("Task 1", 1, False),
        ("Task 2", 1, False),
    ]
    assert TaskStore(state.store.dir(1)).get(2).images == report.ids


def test_tasks_show_cvat_jobs_when_linked(tmp_path, monkeypatch, remotes, fake_keyring):
    from sync_fakes import FakeRemote

    from activelabel.sync import links, registry
    from activelabel.web import tasks_api

    client, state, paths = _setup(tmp_path, 4)
    client.post(
        "/api/projects/1/tasks", files=_files(paths), data={"name": "A", "job_size": "2", "open": "false"}
    )
    assert client.get("/api/projects/1/tasks").json()["tasks"][0].get("remote") is None  # not linked
    fake = FakeRemote("cvat", tasks=True)
    monkeypatch.setitem(registry.OVERRIDES, "cvat", fake)
    monkeypatch.setattr(tasks_api, "_remote_cache", {})
    rid = fake.seed("P").id
    tid = fake.new_task(rid, "A", [(p.name, []) for p in paths], job_size=2)
    link = links.new_link("http://cvat", rid, "P")
    link["tasks"] = {"1": {"remote_task": tid}}
    links.save_link(state.store.dir(1), "cvat", link)
    body = {"url": "http://cvat", "username": "u", "secret": "pw"}
    assert client.put("/api/annotation-tools/cvat", json=body).status_code == 200
    remote = client.get("/api/projects/1/tasks").json()["tasks"][0]["remote"]
    assert remote["task"] == tid and [j["n"] for j in remote["jobs"]] == [1, 2]
    fake.fail = "Cannot reach CVAT"  # cached for a minute; and a failure never breaks the list
    monkeypatch.setattr(tasks_api, "_remote_cache", {})
    assert client.get("/api/projects/1/tasks").status_code == 200


def test_bad_job_size_is_a_readable_400(tmp_path):
    client, _, paths = _setup(tmp_path, 2)
    r = client.post("/api/projects/1/tasks", files=_files(paths), data={"job_size": "two"})
    assert r.status_code == 400 and r.json()["detail"] == "The job size must be a whole number."


def _link_cvat(client, state, fake):
    from activelabel.sync import links

    link = links.new_link("http://cvat", fake.seed("P").id, "P")
    link["tasks"] = {"1": {"remote_task": 1}}
    links.save_link(state.store.dir(1), "cvat", link)
    body = {"url": "http://cvat", "username": "u", "secret": "pw"}
    assert client.put("/api/annotation-tools/cvat", json=body).status_code == 200


def test_unreachable_cvat_is_cached_and_never_breaks_the_list(tmp_path, monkeypatch, remotes, fake_keyring):
    from sync_fakes import FakeRemote

    from activelabel.sync import registry
    from activelabel.web import tasks_api

    client, state, paths = _setup(tmp_path, 2)
    client.post("/api/projects/1/tasks", files=_files(paths), data={"name": "A", "open": "false"})
    calls = []

    class Broken(FakeRemote):
        def tasks(self, settings, remote_id):
            calls.append(1)
            raise RuntimeError("connection reset")

    fake = Broken("cvat", tasks=True)
    monkeypatch.setitem(registry.OVERRIDES, "cvat", fake)
    monkeypatch.setattr(tasks_api, "_remote_cache", {})
    _link_cvat(client, state, fake)
    for _ in range(3):
        r = client.get("/api/projects/1/tasks")
        assert r.status_code == 200 and r.json()["tasks"][0].get("remote") is None
    assert len(calls) == 1  # the failure is remembered, CVAT is not asked on every refresh


def test_slow_cvat_does_not_hang_the_list(tmp_path, monkeypatch, remotes, fake_keyring):
    import threading

    from sync_fakes import FakeRemote

    from activelabel.sync import registry
    from activelabel.web import tasks_api

    client, state, paths = _setup(tmp_path, 2)
    client.post("/api/projects/1/tasks", files=_files(paths), data={"name": "A", "open": "false"})
    release = threading.Event()

    class Slow(FakeRemote):
        def tasks(self, settings, remote_id):
            release.wait(5)
            return []

    fake = Slow("cvat", tasks=True)
    monkeypatch.setitem(registry.OVERRIDES, "cvat", fake)
    monkeypatch.setattr(tasks_api, "_remote_cache", {})
    monkeypatch.setattr(tasks_api, "REMOTE_TIMEOUT", 0.2)
    _link_cvat(client, state, fake)
    started = time.monotonic()
    try:
        assert client.get("/api/projects/1/tasks").status_code == 200
        assert time.monotonic() - started < 3
    finally:
        release.set()
