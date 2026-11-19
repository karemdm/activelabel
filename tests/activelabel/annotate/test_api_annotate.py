import time
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.annotate import registry
from activelabel.annotate.base import Sent, ToolError
from activelabel.labels import Label, load_annotations
from activelabel.web import annotate_api


class FakeTool:
    name = "cvat"

    def __init__(self):
        self.pulled: dict[str, list[Label]] = {}
        self.closed: list[tuple[dict, bool]] = []
        self.fail = ""

    def check(self, settings):
        if self.fail:
            raise ToolError(self.fail)
        return f"connected to {settings.url}"

    def send(self, dataset, image_ids, project, settings, session_id):
        if self.fail:
            raise ToolError(self.fail)
        return Sent([{"label": "Task", "url": "http://cvat/tasks/1"}], {"anno_key": session_id})

    def pull(self, dataset, image_ids, project, settings, ref):
        return {i: self.pulled.get(i, []) for i in image_ids}

    def close(self, dataset, settings, ref, delete_remote):
        self.closed.append((ref, delete_remote))


@pytest.fixture
def env(tmp_path, monkeypatch):
    client, state = make_client(tmp_path)
    tool = FakeTool()
    monkeypatch.setitem(registry.OVERRIDES, "cvat", tool)
    monkeypatch.setattr(annotate_api, "prepare_dataset", lambda project, state: SimpleNamespace())
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project: SimpleNamespace())
    body = {"name": "p", "task": "detection", "classes": ["cat"]}
    pid = client.post("/api/projects", json=body).json()["id"]
    paths = []
    for i in range(2):
        paths.append(tmp_path / "img" / f"i{i}.png")
        make_image(paths[-1], seed=i)
    assert upload(client, pid, paths).status_code == 200
    ids = [i["id"] for i in client.get(f"/api/projects/{pid}/images").json()["items"]]
    return SimpleNamespace(client=client, tool=tool, pid=pid, ids=ids, state=state)


def wait(env, job):
    for _ in range(200):
        jobs = env.client.get(f"/api/projects/{env.pid}/jobs").json()
        data = next(j for j in jobs if j["id"] == job["id"])
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def configure(env):
    body = {"url": "http://cvat", "username": "u", "secret": "pw-123"}
    r = env.client.put("/api/annotation-tools/cvat", json=body)
    assert r.status_code == 200
    return r.json()


def start(env, **body):
    return env.client.post(f"/api/projects/{env.pid}/annotate", json={"tool": "cvat", **body})


def test_tools_settings_never_return_secret(env):
    assert "pw-123" not in str(configure(env))
    tools = env.client.get("/api/annotation-tools").json()
    assert [t["tool"] for t in tools] == ["fiftyone", "cvat", "labelstudio"]
    assert "pw-123" not in str(tools) and next(t for t in tools if t["tool"] == "cvat")["configured"]
    assert "pw-123" not in (env.state.settings.workspace / "tools.json").read_text()
    ok = env.client.post("/api/annotation-tools/cvat/test").json()
    assert ok == {"ok": True, "message": "connected to http://cvat"}
    env.tool.fail = "Bad login."
    assert env.client.post("/api/annotation-tools/cvat/test").json() == {"ok": False, "message": "Bad login."}
    assert env.client.delete("/api/annotation-tools/cvat").json() == {"removed": True}
    after = next(t for t in env.client.get("/api/annotation-tools").json() if t["tool"] == "cvat")
    assert not after["configured"]
    assert env.client.put("/api/annotation-tools/cvat", json={"url": ""}).status_code == 400
    assert env.client.put("/api/annotation-tools/nope", json={}).status_code == 404


def test_send_requires_configuration_and_selection(env):
    r = start(env, scope="all")
    assert r.status_code == 400
    assert r.json()["detail"] == "Configure cvat first (URL, user, password/token)."
    configure(env)
    r = start(env, scope="selected", ids=["zz"])
    assert r.status_code == 400 and "No images" in r.json()["detail"]
    assert env.client.post("/api/projects/99/annotate", json={"tool": "cvat"}).status_code == 404


def test_full_flow(env):
    configure(env)
    job = wait(env, start(env, scope="all").json())
    assert job["status"] == "done", job
    session = job["result"]["session"]
    assert job["result"]["links"][0]["url"] == "http://cvat/tasks/1"
    assert session["status"] == "open" and session["images"] == 2 and "image_ids" not in session

    base = f"/api/projects/{env.pid}/annotate/{session['id']}"
    assert env.client.post(f"{base}/apply").status_code == 400  # not pulled yet
    first = [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3]), Label("Bird", bbox=[0.5, 0.5, 0.2, 0.2])]
    env.tool.pulled = {env.ids[0]: first}
    pulled = wait(env, env.client.post(f"{base}/pull").json())
    assert pulled["status"] == "done", pulled
    diff = pulled["result"]["diff"]
    assert diff["added"] == 2 and diff["unknown_classes"] == ["Bird"]
    assert load_annotations(env.state.store.dir(env.pid)) is None  # nothing applied yet

    env.tool.pulled[env.ids[1]] = [Label("cat", bbox=[0, 0, 0.5, 0.5])]  # people keep editing: pull again
    again = wait(env, env.client.post(f"{base}/pull").json())
    assert again["result"]["diff"]["added"] == 3
    listed = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"]
    assert listed[0]["status"] == "pulled" and listed[0]["diff"]["added"] == 3

    applied = env.client.post(f"{base}/apply").json()["applied"]
    assert applied["added"] == 3 and applied["new_classes"] == ["Bird"]
    saved = load_annotations(env.state.store.dir(env.pid))
    assert len(saved.images[env.ids[0]]) == 2 and env.state.store.get(env.pid).classes == ["cat", "Bird"]
    assert env.client.post(f"{base}/pull").status_code == 409

    r = env.client.request("DELETE", base, json={"delete_remote": True})
    assert r.json() == {"closed": True} and env.tool.closed[0][1] is True
    assert env.client.get(f"/api/projects/{env.pid}/annotate").json() == {"sessions": []}
    assert env.client.delete(base).status_code == 404


def test_discard_and_tool_errors(env):
    configure(env)
    env.tool.fail = "Cannot reach the server."
    job = wait(env, start(env).json())
    assert job["status"] == "failed" and "Cannot reach the server." in job["error"]
    env.tool.fail = ""
    job = wait(env, start(env).json())
    base = f"/api/projects/{env.pid}/annotate/{job['result']['session']['id']}"
    assert env.client.post(f"{base}/discard").json() == {"discarded": True}
    assert env.client.post(f"{base}/pull").status_code == 409
    assert env.client.post(f"/api/projects/{env.pid}/annotate/nope/pull").status_code == 404
