import time

import pytest
import requests
from api_client import make_client
from import_fakes import FakeKeyring, FakeResp, FakeSession, image_resp, make_images

from activelabel.annotate import credentials
from activelabel.importing import cvat_import, labelstudio_import

BASE = "http://ls.local:8080"
CONFIG = (
    '<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image">'
    '<Label value="cat"/></RectangleLabels></View>'
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "keyring", FakeKeyring())
    client, state = make_client(tmp_path)
    return client, state, tmp_path


def configure_ls(client):
    body = {"url": BASE, "username": "", "secret": "pat-secret"}
    assert client.put("/api/annotation-tools/labelstudio", json=body).status_code == 200


def fake_ls(monkeypatch, tmp_path):
    images = make_images(tmp_path / "remote", 2)
    rect = {
        "type": "rectanglelabels",
        "original_width": 64,
        "original_height": 48,
        "value": {"x": 10, "y": 10, "width": 20, "height": 20, "rotation": 0, "rectanglelabels": ["cat"]},
    }
    tasks = [
        {
            "id": i + 1,
            "data": {"image": f"/data/upload/1/abcd123{i}-img{i}.png"},
            "annotations": [{"result": [rect], "was_cancelled": False}],
        }
        for i in range(2)
    ]
    page = {
        "next": None,
        "results": [
            {"id": 7, "title": "Pets", "task_number": 2, "created_at": "2026-05-01", "label_config": CONFIG}
        ],
    }
    routes = {
        ("POST", "/api/token/refresh/"): FakeResp(200, {"access": "A"}),
        ("GET", "/api/projects/7/export"): FakeResp(200, tasks),
        ("GET", "/api/projects/7"): FakeResp(200, page["results"][0]),
        ("GET", "/api/projects?"): FakeResp(200, page),
        ("GET", "/data/upload/1/abcd1230-img0.png"): image_resp(images[0]),
        ("GET", "/data/upload/1/abcd1231-img1.png"): image_resp(images[1]),
    }
    fake = FakeSession(routes, BASE)
    monkeypatch.setattr(requests, "Session", lambda: fake)
    return fake


def wait(client, job_id):
    for _ in range(200):
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_not_configured_is_400(env):
    client, _, _ = env
    r = client.get("/api/import/cvat/projects")
    assert r.status_code == 400 and r.json()["detail"] == "Configure cvat first"
    r = client.post("/api/import", json={"tool": "labelstudio", "remote_id": 7})
    assert r.status_code == 400 and "Configure labelstudio first" in r.json()["detail"]


def test_unknown_tool_and_kind(env):
    client, _, _ = env
    assert client.get("/api/import/fiftyone/projects").status_code == 404
    assert client.post("/api/import", json={"tool": "fiftyone", "remote_id": 1}).status_code == 404
    configure_ls(client)
    r = client.post("/api/import", json={"tool": "labelstudio", "remote_id": 7, "kind": "task"})
    assert r.status_code == 400


def test_list_remote(env, monkeypatch):
    client, _, tmp_path = env
    configure_ls(client)
    fake_ls(monkeypatch, tmp_path)
    r = client.get("/api/import/labelstudio/projects", params={"search": "pet"})
    assert r.status_code == 200
    assert r.json() == {
        "items": [
            {
                "id": 7,
                "kind": "project",
                "name": "Pets",
                "images": 2,
                "created": "2026-05-01",
                "classes": ["cat"],
            }
        ]
    }
    assert "pat-secret" not in r.text


def test_list_remote_tool_error_is_400(env, monkeypatch):
    client, _, _ = env
    configure_ls(client)

    def fail(settings, search):
        raise labelstudio_import.ToolError("Cannot reach Label Studio at x")

    monkeypatch.setattr(labelstudio_import, "list_remote", fail)
    r = client.get("/api/import/labelstudio/projects")
    assert r.status_code == 400 and "Cannot reach" in r.json()["detail"]


def test_unknown_remote_is_404(env, monkeypatch):
    client, _, tmp_path = env
    configure_ls(client)
    fake_ls(monkeypatch, tmp_path)
    r = client.post("/api/import", json={"tool": "labelstudio", "remote_id": 99})
    assert r.status_code == 404


def test_cvat_unknown_remote_and_login(env, monkeypatch):
    client, _, _ = env
    body = {"url": "http://cvat", "username": "u", "secret": "pw"}
    assert client.put("/api/annotation-tools/cvat", json=body).status_code == 200
    routes = {("GET", "/api/tasks/5"): FakeResp(404), ("GET", "/api/projects/6"): FakeResp(401)}
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(routes, "http://cvat"))
    assert (
        client.post("/api/import", json={"tool": "cvat", "remote_id": "5", "kind": "task"}).status_code == 404
    )
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 6, "kind": "project"})
    assert r.status_code == 400 and "login failed" in r.json()["detail"]
    assert "pw" not in r.json()["detail"].replace("password", "")


def test_import_job_and_origin(env, monkeypatch):
    client, state, tmp_path = env
    configure_ls(client)
    fake_ls(monkeypatch, tmp_path)
    r = client.post("/api/import", json={"tool": "labelstudio", "remote_id": 7, "kind": "project"})
    assert r.status_code == 200
    job = r.json()
    assert job["kind"] == "import_project" and job["project_id"] == 0
    done = wait(client, job["id"])
    assert done["status"] == "done", done
    result = done["result"]
    assert result["name"] == "Pets" and result["task"] == "detection"
    assert (result["images_added"], result["images_rejected"], result["labels"]) == (2, 0, 2)
    assert result["classes"] == ["cat"] and result["problems"] == []
    project = client.get(f"/api/projects/{result['project_id']}").json()
    assert project["origin"]["tool"] == "labelstudio" and project["origin"]["remote_id"] == "7"
    assert project["counts"]["annotated"] == 2
    assert "pat-secret" not in str(done) and "pat-secret" not in str(project)
    # imported projects are normal ones: a second import of the same remote gets a new name
    second = client.post("/api/import", json={"tool": "labelstudio", "remote_id": "7"}).json()
    assert wait(client, second["id"])["result"]["name"] == "Pets (2)"
    created = client.post("/api/projects", json={"name": "Plain", "task": "detection"}).json()
    assert client.get(f"/api/projects/{created['id']}").json()["origin"] is None


def test_failed_import_reports_error_and_leaves_no_project(env, monkeypatch):
    client, _, tmp_path = env
    configure_ls(client)
    fake = fake_ls(monkeypatch, tmp_path)
    fake.routes[("GET", "/api/projects/7/export")] = FakeResp(200, [])
    job = client.post("/api/import", json={"tool": "labelstudio", "remote_id": 7}).json()
    done = wait(client, job["id"])
    assert done["status"] == "failed" and "no tasks" in done["error"]
    assert client.get("/api/projects").json() == []


def test_cvat_import_uses_module_functions(env, monkeypatch):
    client, _, _ = env
    client.put("/api/annotation-tools/cvat", json={"url": "http://cvat", "username": "u", "secret": "pw"})
    seen = {}
    monkeypatch.setattr(cvat_import, "check_remote", lambda s, kind, rid: {"name": "X", "classes": []})

    def fake_import(store, settings, workspace, **kw):
        seen.update(kw)
        progress = kw["progress"]
        progress.update(message="Listing CVAT data")
        return {"project_id": 0}

    monkeypatch.setattr(cvat_import, "import_project", fake_import)
    body = {
        "tool": "cvat",
        "remote_id": 3,
        "kind": "task",
        "name": "N",
        "description": "D",
        "task": "keypoints",
    }
    job = client.post("/api/import", json=body).json()
    assert wait(client, job["id"])["status"] == "done"
    assert (seen["kind"], seen["remote_id"], seen["name"], seen["description"]) == ("task", "3", "N", "D")
    assert seen["task"].value == "keypoints"


def test_get_job_any_project(env):
    client, state, _ = env
    assert client.get("/api/jobs/nope").status_code == 404
    pid = client.post("/api/projects", json={"name": "p", "task": "detection"}).json()["id"]
    job = state.jobs.submit(pid, "x", lambda progress: {"ok": 1})
    assert wait(client, job.id)["result"] == {"ok": 1}
    zero = state.jobs.submit(0, "import_project", lambda progress: {"ok": 2})
    assert wait(client, zero.id)["project_id"] == 0
