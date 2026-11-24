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
