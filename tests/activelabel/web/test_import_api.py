import json
import time

import pytest
import requests
from api_client import make_client
from import_fakes import FakeKeyring, FakeResp, FakeSession

from activelabel import optional
from activelabel.annotate import credentials
from activelabel.annotate.base import ToolError
from activelabel.importing import cvat_import
from activelabel.sync import links
from activelabel.tasks import Task


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "keyring", FakeKeyring())
    client, state = make_client(tmp_path)
    return client, state, tmp_path


def configure_cvat(client):
    body = {"url": "http://cvat", "username": "u", "secret": "pw-123"}
    assert client.put("/api/annotation-tools/cvat", json=body).status_code == 200


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
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 7})
    assert r.status_code == 400 and "Configure cvat first" in r.json()["detail"]


def test_only_cvat_can_be_imported_from_and_only_known_kinds(env):
    client, _, _ = env
    for tool in ("fiftyone", "labelstudio"):  # the second one was supported before
        assert client.get(f"/api/import/{tool}/projects").status_code == 404
        r = client.post("/api/import", json={"tool": tool, "remote_id": 1})
        assert r.status_code == 404 and r.json()["code"] == "tool_unknown"
    configure_cvat(client)
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 7, "kind": "folder"})
    assert r.status_code == 400


def test_list_remote(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)
    items = [
        {"id": 7, "kind": "project", "name": "Pets", "images": 2, "created": "2026-05-01", "classes": ["cat"]}
    ]
    seen = {}

    def fake_list(settings, search):
        seen["search"], seen["url"] = search, settings.url
        return items

    monkeypatch.setattr(cvat_import, "list_remote", fake_list)
    r = client.get("/api/import/cvat/projects", params={"search": "pet"})
    assert r.status_code == 200 and r.json() == {"items": items}
    assert seen == {"search": "pet", "url": "http://cvat"} and "pw-123" not in r.text


def test_list_remote_tool_error_is_400(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)

    def fail(settings, search):
        raise ToolError("Cannot reach CVAT at x")

    monkeypatch.setattr(cvat_import, "list_remote", fail)
    r = client.get("/api/import/cvat/projects")
    assert r.status_code == 400 and "Cannot reach" in r.json()["detail"]


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


def test_a_failed_import_reports_the_error_and_a_plain_project_has_no_origin(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)
    monkeypatch.setattr(cvat_import, "check_remote", lambda s, kind, rid: {"name": "X", "classes": []})

    def fail(store, settings, workspace, **kw):
        raise ToolError("CVAT returned no images for this project.")

    monkeypatch.setattr(cvat_import, "import_project", fail)
    job = client.post("/api/import", json={"tool": "cvat", "remote_id": 7}).json()
    done = wait(client, job["id"])
    assert done["status"] == "failed" and "no images" in done["error"]
    assert client.get("/api/projects").json() == []
    created = client.post("/api/projects", json={"name": "Plain", "task": "detection"}).json()
    assert client.get(f"/api/projects/{created['id']}").json()["origin"] is None


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


def test_a_project_imported_before_from_a_removed_tool_still_opens(env):
    client, state, _ = env
    pid = client.post("/api/projects", json={"name": "Old", "task": "detection"}).json()["id"]
    origin = {"tool": "labelstudio", "url": "http://ls", "remote_id": "7", "kind": "project", "name": "Pets"}
    (state.store.dir(pid) / "origin.json").write_text(json.dumps(origin), encoding="utf-8")
    shown = client.get(f"/api/projects/{pid}")
    assert shown.status_code == 200 and shown.json()["origin"]["tool"] == "labelstudio"
    assert client.get("/api/projects").status_code == 200


def test_remote_failures_answer_with_the_code_of_the_error(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)

    def down(settings, search):
        raise ToolError("Cannot reach CVAT at http://cvat", "cvat_unreachable", url="http://cvat")

    monkeypatch.setattr(cvat_import, "list_remote", down)
    r = client.get("/api/import/cvat/projects")
    assert (r.status_code, r.json()["code"], r.json()["params"]) == (
        400,
        "cvat_unreachable",
        {"url": "http://cvat"},
    )
    monkeypatch.setattr(cvat_import, "list_remote", lambda s, q: (_ for _ in ()).throw(ToolError("odd")))
    plain = client.get("/api/import/cvat/projects").json()
    assert (plain["code"], plain["params"]) == ("tool_error", {"tool": "cvat", "reason": "odd"})


def test_a_missing_remote_item_is_404_with_its_code(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)
    routes = {("GET", "/api/tasks/5"): FakeResp(404)}
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(routes, "http://cvat"))
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": "5", "kind": "task"})
    assert (r.status_code, r.json()["code"]) == (404, "cvat_not_found")


def test_an_unknown_kind_has_a_code(env):
    client, _, _ = env
    configure_cvat(client)
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 7, "kind": "folder"})
    assert (r.status_code, r.json()["code"]) == (400, "import_kind_invalid")


def test_a_failed_import_job_carries_the_code_and_no_traceback(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)
    monkeypatch.setattr(cvat_import, "check_remote", lambda s, kind, rid: {"name": "X", "classes": []})

    def fail(store, settings, workspace, **kw):
        raise ToolError("CVAT returned no images for this project.", "cvat_no_images", what="project")

    monkeypatch.setattr(cvat_import, "import_project", fail)
    job = client.post("/api/import", json={"tool": "cvat", "remote_id": 7}).json()
    done = wait(client, job["id"])
    assert (done["error_code"], done["error_params"]) == ("cvat_no_images", {"what": "project"})
    assert done["result"] == {}


def test_starting_an_import_without_fiftyone_answers_501(env, monkeypatch):
    client, _, _ = env
    configure_cvat(client)
    monkeypatch.setattr(optional, "FIFTYONE", False)
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 7})
    assert r.status_code == 501
    assert r.json()["code"] == "fiftyone_not_installed"
    assert r.json()["params"] == {"install": optional.FIFTYONE_INSTALL}


def link_local_project(state, name, remote_id, base="http://cvat"):
    project = state.store.create(name, "", Task.DETECTION)
    links.save_link(
        state.store.dir(project.id), "cvat", links.new_link(f"{base}/projects/{remote_id}", remote_id, "Pets")
    )
    return project


def test_a_remote_project_that_is_already_linked_is_marked_and_not_imported_again(env, monkeypatch):
    client, state, _ = env
    configure_cvat(client)
    project = link_local_project(state, "Pets", "7")
    link_local_project(state, "Other server", "8", base="http://elsewhere")  # another server: not this one
    items = [
        {"id": 7, "kind": "project", "name": "Pets", "images": 2, "classes": []},
        {"id": 8, "kind": "project", "name": "Birds", "images": 2, "classes": []},
        {"id": 9, "kind": "task", "name": "Loose", "images": 1, "classes": []},
    ]
    monkeypatch.setattr(cvat_import, "list_remote", lambda settings, search: items)
    listed = client.get("/api/import/cvat/projects").json()["items"]
    assert listed[0]["linked_to"] == {"id": project.id, "name": "Pets"}
    assert "linked_to" not in listed[1] and "linked_to" not in listed[2]
    r = client.post("/api/import", json={"tool": "cvat", "remote_id": 7, "kind": "project"})
    assert r.status_code == 409 and r.json()["code"] == "cvat_already_imported"
    assert r.json()["params"] == {"project_id": project.id, "name": "Pets"}
    monkeypatch.setattr(cvat_import, "check_remote", lambda s, kind, rid: {"name": "Birds", "classes": []})
    monkeypatch.setattr(
        cvat_import, "import_project", lambda store, settings, workspace, **kw: {"project_id": 0}
    )
    assert (
        client.post("/api/import", json={"tool": "cvat", "remote_id": 8, "kind": "project"}).status_code
        == 200
    )
