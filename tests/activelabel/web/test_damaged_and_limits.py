from api_client import make_client, upload
from sample_images import make_image

from activelabel.projects import clean_classes


def make_projects(tmp_path, count=2):
    client, state = make_client(tmp_path)
    for n in range(count):
        assert client.post("/api/projects", json={"name": f"P{n}", "task": "detection"}).status_code == 200
    return client, state


def test_one_damaged_project_file_does_not_hide_the_others(tmp_path):
    client, state = make_projects(tmp_path)
    (state.store.dir(1) / "project.json").write_text("{cut", encoding="utf-8")
    listed = client.get("/api/projects")
    assert listed.status_code == 200
    by_id = {p["id"]: p for p in listed.json()}
    assert "project.json is damaged" in by_id[1]["damaged"] and by_id[1]["counts"]["images"] == 0
    assert "damaged" not in by_id[2]
    detail = client.get("/api/projects/1")
    assert detail.status_code == 500 and "project.json is damaged" in detail.json()["detail"]


def test_damaged_images_file_flags_only_that_project(tmp_path):
    client, state = make_projects(tmp_path)
    (state.store.dir(2) / "images.json").write_text("[", encoding="utf-8")
    by_id = {p["id"]: p for p in client.get("/api/projects").json()}
    assert "images.json is damaged" in by_id[2]["damaged"] and by_id[2]["name"] == "P1"
    assert "damaged" not in by_id[1]


def test_creating_a_project_next_to_a_damaged_one_never_reuses_its_id(tmp_path):
    client, state = make_projects(tmp_path)
    (state.store.dir(2) / "project.json").write_text("{cut", encoding="utf-8")
    (state.store.root / "next_id.json").unlink()
    created = client.post("/api/projects", json={"name": "New", "task": "detection"})
    assert created.status_code == 200 and created.json()["id"] == 3


def test_a_damaged_counter_is_not_a_bad_request(tmp_path):
    client, state = make_projects(tmp_path, 1)
    (state.store.root / "next_id.json").write_text("{", encoding="utf-8")
    response = client.post("/api/projects", json={"name": "New", "task": "detection"})
    assert response.status_code == 500 and "next_id.json is damaged" in response.json()["detail"]


def test_control_characters_and_huge_job_sizes_are_refused(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    bad = client.post("/api/projects", json={"name": "a\x00b", "task": "detection"})
    assert bad.status_code == 400 and "control characters" in bad.json()["detail"]
    assert (
        client.post(
            "/api/projects", json={"name": "ok", "description": "x\x07", "task": "detection"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/projects", json={"name": "ok", "description": "l1\nl2", "task": "detection"}
        ).status_code
        == 200
    )
    assert client.post("/api/projects", json={"name": "x" * 101, "task": "detection"}).status_code == 400
    path = tmp_path / "a.png"
    make_image(path)
    files = [("files", ("a.png", path.read_bytes()))]
    assert client.post("/api/projects/1/tasks", files=files, data={"name": "t\x00"}).status_code == 400
    assert (
        client.post("/api/projects/1/tasks", files=files, data={"job_size": "999999999999"}).status_code
        == 400
    )
    assert client.post("/api/projects/1/tasks", files=files, data={"job_size": "10"}).status_code == 200
    assert upload(client, 1, [path]).status_code == 200


def test_clean_classes_keeps_first_spelling_and_order():
    assert clean_classes([" Cat", "cat", "", "Dog", "DOG", "bird"]) == ["Cat", "Dog", "bird"]
    many = [f"c{i}" for i in range(20000)]
    assert clean_classes(many + many) == many


def test_reading_projects_and_tasks_does_not_wait_for_a_running_upload(tmp_path):
    import threading

    from activelabel.tasks_store import TaskStore

    client, state = make_projects(tmp_path, 1)
    path = tmp_path / "a.png"
    make_image(path)
    assert upload(client, 1, [path]).status_code == 200
    holding, release = threading.Event(), threading.Event()

    def hold_the_lock():  # what ingest does for a whole upload
        with TaskStore(state.store.dir(1)).locked():
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    assert holding.wait(5)
    results = {}

    def read():
        for url in ("/api/projects", "/api/projects/1", "/api/projects/1/tasks"):
            results[url] = client.get(url).status_code

    reader = threading.Thread(target=read)
    reader.start()
    reader.join(5)
    still_waiting = reader.is_alive()
    release.set()
    holder.join()
    reader.join()
    assert not still_waiting and set(results.values()) == {200}


def test_an_upload_that_adds_nothing_leaves_no_empty_task(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    path = tmp_path / "a.png"
    make_image(path)
    files = [("files", ("a.png", path.read_bytes()))]
    assert client.post("/api/projects/1/tasks", files=files).status_code == 200  # task 1, open
    again = client.post("/api/projects/1/tasks", files=files)  # a duplicate: nothing added
    assert again.status_code == 200 and again.json()["task"]["images"] == 0
    closed = client.patch(f"/api/projects/1/tasks/{again.json()['task']['id']}", json={"open": False})
    assert closed.status_code == 200 and closed.json()["removed"] is True
    assert [t["images"] for t in client.get("/api/projects/1/tasks").json()["tasks"]] == [1]
    assert upload(client, 1, [path]).json().get("task") is None  # POST /images makes none either


def test_reopening_a_closed_task_is_a_conflict_and_checks_come_before_settings(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    path = tmp_path / "a.png"
    make_image(path)
    assert (
        client.post("/api/projects/1/tasks", files=[("files", ("a.png", path.read_bytes()))]).status_code
        == 200
    )
    client.patch("/api/projects/1/tasks/1", json={"open": False})
    assert client.patch("/api/projects/1/tasks/1", json={"open": True}).status_code == 409
    # nothing is configured: a wrong direction or scope is still reported as such
    wrong = client.post("/api/projects/1/sync/cvat", json={"direction": "sideways"})
    assert wrong.status_code == 400 and "direction" in wrong.json()["detail"]
    wrong = client.post("/api/projects/1/annotate", json={"tool": "cvat", "scope": "nope"})
    assert wrong.status_code == 400 and "scope" in wrong.json()["detail"]
