import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.annotate import registry, sessions
from activelabel.annotate.base import Sent, ToolError
from activelabel.labels import AnnotationSet, Label, annotations_lock, load_annotations, save_annotations
from activelabel.tasks import Task
from activelabel.web import annotate_api


class FakeTool:
    name = "fiftyone"

    def __init__(self):
        self.pulled: dict[str, list[Label]] = {}
        self.closed: list[dict] = []
        self.fail = ""
        self.told: tuple[str, list[str]] | None = None

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

    def applied(self, dataset, image_ids, project, labels):
        self.told = ("applied", list(image_ids))

    def discarded(self, dataset, image_ids, project, labels):
        self.told = ("discarded", list(image_ids))

    def close(self, dataset, settings, ref):
        self.closed.append(ref)


@pytest.fixture
def env(tmp_path, monkeypatch):
    client, state = make_client(tmp_path)
    tool = FakeTool()
    monkeypatch.setitem(registry.OVERRIDES, "fiftyone", tool)
    monkeypatch.setattr(annotate_api, "prepare_dataset", lambda project, state: SimpleNamespace())
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: SimpleNamespace())
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
    return env.client.post(f"/api/projects/{env.pid}/annotate", json={"tool": "fiftyone", **body})


def test_tools_settings_never_return_secret(env, remotes):
    assert "pw-123" not in str(configure(env))
    tools = env.client.get("/api/annotation-tools").json()
    assert [t["tool"] for t in tools] == ["fiftyone", "cvat"]
    assert "pw-123" not in str(tools) and next(t for t in tools if t["tool"] == "cvat")["configured"]
    assert "pw-123" not in (env.state.settings.workspace / "tools.json").read_text()
    ok = env.client.post("/api/annotation-tools/cvat/test").json()
    assert (ok["ok"], ok["message"], ok["code"]) == (True, "connected to http://cvat", "test_connected")
    assert ok["params"] == {"url": "http://cvat", "user": "u"}
    remotes["cvat"].fail = "Bad login."
    bad = env.client.post("/api/annotation-tools/cvat/test").json()
    assert (bad["ok"], bad["message"], bad["code"]) == (False, "Bad login.", "tool_error")
    assert env.client.delete("/api/annotation-tools/cvat").json() == {"removed": True}
    after = next(t for t in env.client.get("/api/annotation-tools").json() if t["tool"] == "cvat")
    assert not after["configured"]
    assert env.client.put("/api/annotation-tools/cvat", json={"url": ""}).status_code == 400
    assert env.client.put("/api/annotation-tools/nope", json={}).status_code == 404
    old = env.client.put("/api/annotation-tools/labelstudio", json={"url": "http://ls", "secret": "x"})
    assert old.status_code == 404 and old.json()["code"] == "tool_unknown"  # a tool that was removed


def test_send_requires_a_selection(env):
    r = start(env, scope="selected", ids=["zz"])
    assert r.status_code == 400 and "No images" in r.json()["detail"]
    assert env.client.post("/api/projects/99/annotate", json={"tool": "fiftyone"}).status_code == 404


def test_full_flow(env):
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
    assert env.client.post(f"{base}/discard").status_code == 409  # applied: nothing to discard
    assert env.client.post(f"{base}/apply").status_code in (400, 409)  # a second click changes nothing
    listed = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"]
    assert listed[0]["status"] == "applied"

    r = env.client.delete(base)
    assert r.json() == {"closed": True} and env.tool.closed == [{"anno_key": session["id"]}]
    assert env.client.get(f"/api/projects/{env.pid}/annotate").json() == {"sessions": []}
    assert env.client.delete(base).status_code == 404


def test_discard_and_tool_errors(env):
    env.tool.fail = "Cannot reach the server."
    job = wait(env, start(env).json())
    assert job["status"] == "failed" and "Cannot reach the server." in job["error"]
    env.tool.fail = ""
    job = wait(env, start(env).json())
    base = f"/api/projects/{env.pid}/annotate/{job['result']['session']['id']}"
    assert env.client.post(f"{base}/discard").json() == {"discarded": True}
    assert env.client.post(f"{base}/pull").status_code == 409
    assert env.client.post(f"/api/projects/{env.pid}/annotate/nope/pull").status_code == 404


def test_a_task_or_job_can_be_sent_for_editing(env):
    """Scope ``task`` sends exactly the active images of one task (or of one of its jobs)."""
    task = env.client.get(f"/api/projects/{env.pid}/tasks").json()["tasks"][0]
    assert task["images"] == 2
    job = wait(env, start(env, scope="task", task=task["id"]).json())
    assert job["status"] == "done", job
    session = job["result"]["session"]
    assert (session["scope"], session["task"], session["images"]) == ("task", task["id"], 2)
    assert "job" not in session
    whole = start(env, scope="task", task=task["id"], job=1).json()  # a single job holds the whole task
    assert wait(env, whole)["result"]["session"]["job"] == 1


def test_sending_a_task_that_does_not_exist_or_without_one_is_refused(env):
    assert start(env, scope="task", task=99).status_code == 404
    assert start(env, scope="task", task=1, job=9).json()["code"] == "job_not_found"
    assert start(env, scope="task").json()["code"] == "task_required"


def _pulled_session(env):
    session = wait(env, start(env, scope="all").json())["result"]["session"]
    base = f"/api/projects/{env.pid}/annotate/{session['id']}"
    env.tool.pulled = {env.ids[0]: [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3])]}
    assert wait(env, env.client.post(f"{base}/pull").json())["status"] == "done"
    return base


def test_apply_refused_when_local_labels_changed_since_pull(env):
    base = _pulled_session(env)
    other = {env.ids[1]: [Label("cat", bbox=[0.2, 0.2, 0.2, 0.2])]}  # e.g. a sync or an import
    save_annotations(env.state.store.dir(env.pid), AnnotationSet(Task.DETECTION, ["cat"], other, {}))
    r = env.client.post(f"{base}/apply")
    assert r.status_code == 409
    assert r.json()["detail"] == (
        "The annotations of 1 images changed since you brought them back: bring them back again."
    )
    listed = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"][0]
    assert listed["status"] == "pulled" and "base" not in listed
    wait(env, env.client.post(f"{base}/pull").json())  # pulling again records the new state
    assert env.client.post(f"{base}/apply").status_code == 200


def test_apply_and_discard_of_the_same_session_cannot_both_succeed(env):
    """Both wait for the annotations lock; the one that gets it first wins and the other is refused (409)."""
    base = _pulled_session(env)
    with ThreadPoolExecutor(max_workers=2) as pool, annotations_lock(env.state.store.dir(env.pid)):
        apply = pool.submit(env.client.post, f"{base}/apply")
        discard = pool.submit(env.client.post, f"{base}/discard")
        time.sleep(0.5)  # both requests are past their first checks and wait for the lock
    codes = sorted([apply.result().status_code, discard.result().status_code])
    assert codes == [200, 409]
    final = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"][0]["status"]
    assert final == ("applied" if apply.result().status_code == 200 else "discarded")
    saved = load_annotations(env.state.store.dir(env.pid))
    assert bool(saved and saved.images.get(env.ids[0])) == (final == "applied")


def test_apply_after_discard_is_refused(env):
    base = _pulled_session(env)
    assert env.client.post(f"{base}/discard").status_code == 200
    r = env.client.post(f"{base}/apply")
    assert r.status_code in (400, 409)
    assert load_annotations(env.state.store.dir(env.pid)) is None


def _legacy_session(env, tool):
    """A session an old workspace may hold: made when ActiveLabel still had edit sessions in other tools."""

    return sessions.create(
        env.state.store.dir(env.pid),
        tool,
        "all",
        env.ids,
        [{"label": "Task", "url": "http://cvat/tasks/1"}],
        {"anno_key": "al_old"},
    )


@pytest.mark.parametrize(
    ("tool", "status", "code"), [("cvat", 400, "tool_unsupported"), ("labelstudio", 404, "tool_unknown")]
)
def test_only_fiftyone_opens_edit_sessions(env, tool, status, code):
    assert start(env).status_code == 200  # the fiftyone fake still works
    r = env.client.post(f"/api/projects/{env.pid}/annotate", json={"tool": tool})
    assert r.status_code == status and r.json()["code"] == code
    assert env.client.post(f"/api/projects/{env.pid}/annotate", json={"tool": "nope"}).status_code == 404


@pytest.mark.parametrize("tool", ["cvat", "labelstudio"])
def test_old_session_of_a_removed_tool_is_listed_and_closed_without_contacting_anything(env, remotes, tool):
    old = _legacy_session(env, tool)
    listed = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"]
    assert [x["id"] for x in listed] == [old["id"]] and listed[0]["tool"] == tool
    base = f"/api/projects/{env.pid}/annotate/{old['id']}"
    assert env.client.post(f"{base}/pull").json()["code"] == "tool_unsupported"
    r = env.client.request("DELETE", base, json={"delete_remote": True})  # an old client may still send this
    assert r.status_code == 200 and r.json() == {"closed": True}
    assert env.client.get(f"/api/projects/{env.pid}/annotate").json() == {"sessions": []}
    assert not env.tool.closed and not remotes["cvat"].calls  # no tool was contacted


def test_a_tool_url_with_spaces_or_without_a_server_is_refused_and_spaces_around_it_are_dropped(env, remotes):
    for bad in ("  http://127.0.0.1:8113  /x y", "cvat.local:8080", "http://my host"):
        r = env.client.put("/api/annotation-tools/cvat", json={"url": bad, "username": "u", "secret": "pw"})
        assert r.status_code == 400 and r.json()["code"] == "url_invalid", bad
    cvat = next(t for t in env.client.get("/api/annotation-tools").json() if t["tool"] == "cvat")
    assert cvat["url"] == ""  # nothing was saved
    body = {"url": "  http://cvat.local:8080/  ", "username": " u ", "secret": "pw"}
    saved = env.client.put("/api/annotation-tools/cvat", json=body)
    assert saved.status_code == 200 and saved.json()["url"] == "http://cvat.local:8080"


def test_apply_and_discard_tell_the_tool_and_a_failing_tool_does_not_undo_them(env):
    job = wait(env, start(env, scope="all").json())
    session = job["result"]["session"]
    base = f"/api/projects/{env.pid}/annotate/{session['id']}"
    assert env.client.post(f"{base}/discard").json() == {"discarded": True}
    assert env.tool.told == ("discarded", env.ids)

    second = wait(env, start(env, scope="all").json())["result"]["session"]
    base = f"/api/projects/{env.pid}/annotate/{second['id']}"
    env.tool.pulled = {env.ids[0]: [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3])]}
    assert wait(env, env.client.post(f"{base}/pull").json())["status"] == "done"

    def boom(*args):
        raise RuntimeError("FiftyOne is down")

    env.tool.applied = boom
    assert env.client.post(f"{base}/apply").status_code == 200  # the apply already happened
    listed = env.client.get(f"/api/projects/{env.pid}/annotate").json()["sessions"]
    assert next(s for s in listed if s["id"] == second["id"])["status"] == "applied"


def test_a_keypoints_project_cannot_open_a_fiftyone_session(env):
    """#458: the App cannot edit keypoints, so the API refuses instead of only the interface."""
    body = {"name": "k", "task": "keypoints", "classes": ["person"]}
    pid = env.client.post("/api/projects", json=body).json()["id"]
    for scope in ("all", "task"):
        reply = env.client.post(f"/api/projects/{pid}/annotate", json={"tool": "fiftyone", "scope": scope})
        assert reply.status_code == 400 and reply.json()["code"] == "fiftyone_keypoints_not_editable", (
            reply.text
        )
    assert env.client.get(f"/api/projects/{pid}/annotate").json() == {"sessions": []}


def test_summary_source_says_the_labels_were_last_sessiond_by_a_fiftyone_session(env):
    """#459: after an apply the annotations summary keeps the original source and names the session."""
    save_annotations(
        env.state.store.dir(env.pid),
        AnnotationSet(Task.DETECTION, ["cat"], {}, {"format": "coco", "files": ["a.json"]}),
    )
    session = wait(env, start(env, scope="all").json())["result"]["session"]
    base = f"/api/projects/{env.pid}/annotate/{session['id']}"
    env.tool.pulled = {env.ids[0]: [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3])]}
    assert wait(env, env.client.post(f"{base}/pull").json())["status"] == "done"
    assert env.client.post(f"{base}/apply").status_code == 200
    source = env.client.get(f"/api/projects/{env.pid}/annotations").json()["source"]
    assert source["format"] == "coco" and source["files"] == ["a.json"]  # the import is still told
    assert source["last_session"]["tool"] == "fiftyone" and source["last_session"]["session"] == session["id"]
    assert source["last_session"]["at"]
