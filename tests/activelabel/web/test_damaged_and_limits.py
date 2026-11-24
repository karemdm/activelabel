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


def test_class_names_are_limited(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    long_name = client.patch("/api/projects/1", json={"classes": ["x" * 101]})
    assert long_name.status_code == 400 and long_name.json()["code"] == "class_name_too_long"
    assert long_name.json()["params"]["max"] == 100
    many = client.patch("/api/projects/1", json={"classes": [f"c{i}" for i in range(1001)]})
    assert many.status_code == 400 and many.json()["code"] == "too_many_classes"
    assert (
        client.patch("/api/projects/1", json={"classes": ["x" * 100, "b\x00"]}).json()["code"]
        == "class_name_invalid"
    )
    ok = client.patch("/api/projects/1", json={"classes": ["x" * 100, *[f"c{i}" for i in range(999)]]})
    assert ok.status_code == 200
    created = client.post("/api/projects", json={"name": "N", "task": "detection", "classes": ["y" * 101]})
    assert created.status_code == 400 and created.json()["code"] == "class_name_too_long"


def test_task_names_are_unique_per_project_ignoring_case(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    path = tmp_path / "a.png"
    make_image(path)

    def new_task(name, seed):
        make_image(path, seed=seed)
        return client.post(
            "/api/projects/1/tasks", files=[("files", ("a.png", path.read_bytes()))], data={"name": name}
        )

    assert new_task("T A", 1).status_code == 200
    dup = new_task("t a", 2)
    assert dup.status_code == 400 and dup.json()["code"] == "task_name_taken"
    assert dup.json()["params"] == {"name": "T A"}
    assert len(client.get("/api/projects/1/images").json()["items"]) == 1  # nothing stored by the refused one
    assert new_task("T B", 3).status_code == 200
    renamed = client.patch("/api/projects/1/tasks/2", json={"name": "T a"})
    assert renamed.status_code == 400 and renamed.json()["code"] == "task_name_taken"
    assert client.patch("/api/projects/1/tasks/2", json={"name": "t b"}).status_code == 200  # its own name


def test_errors_carry_codes_next_to_the_english_detail(tmp_path):
    client, _ = make_projects(tmp_path, 1)
    taken = client.post("/api/projects", json={"name": "p0", "task": "detection"}).json()
    assert taken["detail"] == "A project named 'P0' already exists." and taken["code"] == "project_name_taken"
    missing = client.get("/api/projects/99").json()
    assert missing["code"] == "project_not_found" and "detail" in missing
    delete = client.post("/api/projects/1/images/delete", json={"ids": []})
    assert delete.json()["code"] == "confirm_required"
    gone = client.get("/api/jobs/nope").json()
    assert gone["code"] == "activity_not_found"


def test_task_and_classes_wait_for_a_run_or_pre_annotation_but_a_rename_does_not(tmp_path):
    client, state = make_projects(tmp_path, 1)
    state.runner.has_active = lambda project_id: True  # a pre-annotation is running
    task = client.patch("/api/projects/1", json={"task": "classification"})
    assert task.status_code == 409 and task.json()["code"] == "busy_training"
    classes = client.patch("/api/projects/1", json={"classes": ["cat"]})
    assert classes.status_code == 409 and classes.json()["code"] == "busy_training"
    assert (
        client.patch("/api/projects/1", json={"classes": [], "task": "detection"}).status_code == 200
    )  # no change
    assert client.patch("/api/projects/1", json={"name": "Renamed"}).status_code == 200
    state.runner.has_active = lambda project_id: False
    assert client.patch("/api/projects/1", json={"classes": ["cat"]}).status_code == 200
