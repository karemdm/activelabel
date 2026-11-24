import time
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image
from sync_fakes import box

from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, load_annotations, save_annotations
from activelabel.sync import links
from activelabel.tasks_store import TaskStore

CAT = [box("cat")]
CAT2 = [box("cat", 0.5, 0.5)]


@pytest.fixture
def env(tmp_path, remotes, fake_keyring):
    client, state = make_client(tmp_path)
    body = {"name": "Pets", "task": "detection", "classes": ["cat"]}
    pid = client.post("/api/projects", json=body).json()["id"]
    paths = []
    for i in range(2):
        paths.append(tmp_path / "img" / f"img{i}.png")
        make_image(paths[-1], seed=i + 1, fmt="PNG")
    assert upload(client, pid, paths).status_code == 200
    items = client.get(f"/api/projects/{pid}/images").json()["items"]
    ids = {i["name"]: i["id"] for i in items}
    project_dir = state.store.dir(pid)
    save_annotations(
        project_dir,
        AnnotationSet(state.store.get(pid).task, ["cat"], {ids["img0.png"]: CAT, ids["img1.png"]: CAT}, {}),
    )
    remote = remotes["cvat"]
    info = remote.seed("Pets remote", [("img0.png", CAT), ("img1.png", CAT)])
    return SimpleNamespace(client=client, state=state, pid=pid, ids=ids, remote=remote, rid=info.id)


def configure(env, tool="cvat"):
    body = {"url": "http://remote", "username": "u", "secret": "pw-123"}
    assert env.client.put(f"/api/annotation-tools/{tool}", json=body).status_code == 200


def wait(env, job):
    for _ in range(200):
        jobs = env.client.get(f"/api/projects/{env.pid}/jobs").json()
        data = next(j for j in jobs if j["id"] == job["id"])
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def plan(env, **body):
    configure(env)
    r = env.client.post(f"/api/projects/{env.pid}/sync/cvat", json=body)
    assert r.status_code == 200
    assert r.json()["kind"] == "sync_plan"
    return wait(env, r.json())


def apply(env, plan_id, **body):
    r = env.client.post(f"/api/projects/{env.pid}/sync/cvat/{plan_id}/apply", json=body)
    assert r.status_code == 200
    assert r.json()["kind"] == "sync_apply"
    return wait(env, r.json())


def test_plan_and_apply_through_the_api(env):
    job = plan(env, remote_id=env.rid)
    assert job["status"] == "done" and job["message"] == "Comparing"
    assert job["result"]["unchanged"] == 2 and job["result"]["remote"]["id"] == env.rid
    done = apply(env, job["result"]["plan_id"])
    assert done["status"] == "done" and done["message"] == "Updating the project"
    shown = env.client.get(f"/api/projects/{env.pid}/links").json()["links"]
    assert [(x["tool"], x["remote_id"], x["images"]) for x in shown] == [("cvat", env.rid, 2)]
    key = env.remote.key_of(env.rid, "img0.png")
    env.remote.projects[env.rid]["items"][key].labels = CAT2
    second = plan(env)
    assert second["result"]["pull"] == 1
    done = apply(env, second["result"]["plan_id"])
    assert done["result"]["pulled"] == 1
    annotations = load_annotations(env.state.store.dir(env.pid))
    assert annotations.images[env.ids["img0.png"]][0].bbox == CAT2[0].bbox


def test_conflict_resolutions_through_the_api(env):
    apply(env, plan(env, remote_id=env.rid)["result"]["plan_id"])
    directory = env.state.store.dir(env.pid)
    annotations = load_annotations(directory)
    annotations.images[env.ids["img0.png"]] = CAT2
    save_annotations(directory, annotations)
    env.remote.projects[env.rid]["items"][env.remote.key_of(env.rid, "img0.png")].labels = [
        box("cat", 0.2, 0.6)
    ]
    job = plan(env)
    (conflict,) = job["result"]["conflicts"]
    done = apply(env, job["result"]["plan_id"], resolutions={conflict["image_id"]: "local"})
    assert done["result"]["pushed"] == 1
    assert env.remote.labels(env.rid, "img0.png")[0].bbox == CAT2[0].bbox


def test_apply_refused_when_remote_changed(env):
    job = plan(env, remote_id=env.rid)
    apply(env, job["result"]["plan_id"])
    env.remote.projects[env.rid]["items"][env.remote.key_of(env.rid, "img0.png")].labels = CAT2
    job = plan(env)
    env.remote.projects[env.rid]["items"][env.remote.key_of(env.rid, "img0.png")].labels = [
        box("cat", 0.3, 0.3)
    ]
    done = apply(env, job["result"]["plan_id"])
    assert done["status"] == "failed" and "Synchronise again" in done["error"]


def test_create_through_the_api(env):
    configure(env)
    env.remote.projects.clear()
    job = plan(env, create=True)
    assert job["result"]["create"] is True and job["result"]["add"] == 2
    done = apply(env, job["result"]["plan_id"])
    assert done["result"]["added"] == 2 and len(env.remote.projects) == 1


def test_unlink(env):
    apply(env, plan(env, remote_id=env.rid)["result"]["plan_id"])
    assert env.client.delete(f"/api/projects/{env.pid}/links/cvat").json() == {"unlinked": True}
    assert links.get_link(env.state.store.dir(env.pid), "cvat") is None
    assert env.remote.projects  # the remote project is untouched
    assert env.client.delete(f"/api/projects/{env.pid}/links/cvat").status_code == 404
    assert env.client.get(f"/api/projects/{env.pid}/links").json() == {"links": []}


def test_errors(env):
    base = f"/api/projects/{env.pid}"
    assert env.client.post(f"{base}/sync/cvat", json={}).status_code == 400  # not configured
    assert env.client.post(f"{base}/sync/fiftyone", json={}).status_code == 404
    assert env.client.post("/api/projects/99/sync/cvat", json={}).status_code == 404
    assert env.client.get("/api/projects/99/links").status_code == 404
    assert env.client.delete(f"{base}/links/fiftyone").status_code == 404
    configure(env)
    assert env.client.post(f"{base}/sync/cvat/abcdef12/apply", json={}).status_code == 404
    assert env.client.post(f"{base}/sync/cvat/..%2Fx/apply", json={}).status_code == 404
    plan_id = plan(env, remote_id=env.rid)["result"]["plan_id"]
    bad = env.client.post(f"{base}/sync/cvat/{plan_id}/apply", json={"default": "both"})
    assert bad.status_code == 400


def test_plan_job_fails_readably_without_remote(env):
    env.remote.projects.clear()
    configure(env)
    job = wait(env, env.client.post(f"/api/projects/{env.pid}/sync/cvat", json={}).json())
    assert job["status"] == "failed" and "no remote project" in job["error"]
    env.remote.fail = "CVAT login failed"
    job = wait(env, env.client.post(f"/api/projects/{env.pid}/sync/cvat", json={"remote_id": "1"}).json())
    assert job["status"] == "failed" and job["error"].endswith("CVAT login failed")


def test_scope_validation(env):
    configure(env)
    url = f"/api/projects/{env.pid}/sync/cvat"
    assert env.client.post(url, json={"scope": "task"}).status_code == 400
    assert env.client.post(url, json={"scope": "job", "task": 1}).status_code == 400
    assert env.client.post(url, json={"scope": "images"}).status_code == 400
    assert env.client.post(url, json={"scope": "job", "job": 1}).status_code == 400


def test_scopes_and_delete_tasks_through_the_api(env):
    env.remote.supports_tasks = True
    job = plan(env, create=True)
    assert job["status"] == "done" and len(job["result"]["tasks_create"]) == 1
    done = apply(env, job["result"]["plan_id"])
    assert done["result"]["tasks_created"] == 1
    remote_id = list(env.remote.task_meta)[-1]  # the project the sync created
    assert [t.name for t in env.remote.tasks(None, remote_id)] == ["Task 1"]
    scoped = plan(env, scope="task", task=1)
    assert scoped["result"]["scope"] == {"scope": "task", "task": 1, "job": None}
    assert scoped["result"]["unchanged"] == 2
    one_job = plan(env, scope="job", task=1, job=1)
    assert one_job["result"]["scope"]["job"] == 1 and one_job["result"]["unchanged"] == 2
    failed = plan(env, scope="task", task=99)
    assert failed["status"] == "failed"
    # delete the task here: only a confirmed apply deletes it in the tool
    directory = env.state.store.dir(env.pid)
    ids = TaskStore(directory).delete(1)
    Manifest(directory).delete(ids)
    review = plan(env)
    assert len(review["result"]["tasks_delete"]) == 1
    done = apply(env, review["result"]["plan_id"], delete_tasks=True)
    assert done["result"]["tasks_deleted"] == 1
    assert env.remote.tasks(None, remote_id) == []


def test_unlinking_is_refused_while_a_job_runs(env):
    import threading

    links.save_link(
        env.state.store.dir(env.pid), "cvat", links.new_link("http://remote", env.rid, "Pets remote")
    )
    release = threading.Event()
    env.state.jobs.submit(env.pid, "sync_apply", lambda progress: release.wait(10) and None)
    busy = env.client.delete(f"/api/projects/{env.pid}/links/cvat")
    assert busy.status_code == 409 and busy.json()["code"] == "busy"
    assert links.get_link(env.state.store.dir(env.pid), "cvat") is not None
    release.set()
    for _ in range(100):
        if not env.state.jobs.active(env.pid):
            break
        time.sleep(0.05)
    assert env.client.delete(f"/api/projects/{env.pid}/links/cvat").status_code == 200


def test_audit_lists_the_server_read_only_and_says_which_project_is_linked(env):
    r = env.client.get("/api/cvat/audit")
    assert r.status_code == 400 and r.json()["code"] == "tool_not_configured"
    configure(env)
    other = env.remote.seed("Somebody else's", [])
    r = env.client.get("/api/cvat/audit")
    assert r.status_code == 200 and r.json()["url"] == "http://remote"
    assert [(p["name"], p["linked_to"]) for p in r.json()["projects"]] == [
        ("Pets remote", None),
        ("Somebody else's", None),
    ]
    links.save_link(
        env.state.store.dir(env.pid), "cvat", links.new_link("http://remote", env.rid, "Pets remote")
    )
    found = {p["id"]: p["linked_to"] for p in env.client.get("/api/cvat/audit").json()["projects"]}
    assert found == {env.rid: {"id": env.pid, "name": "Pets"}, other.id: None}
    assert not [c for c in env.remote.calls if c[0] != "audit"]  # nothing was changed in the server


def test_audit_reports_a_server_problem(env):
    configure(env)
    env.remote.fail = "Cannot reach CVAT."
    r = env.client.get("/api/cvat/audit")
    assert r.status_code == 400 and r.json()["detail"] == "Cannot reach CVAT."


def test_an_old_link_of_a_tool_that_cannot_be_synchronised_is_listed_and_can_be_forgotten(env):
    project_dir = env.state.store.dir(env.pid)
    links.save_link(project_dir, "labelstudio", links.new_link("http://ls", "3", "Pets LS"))
    links.save_link(project_dir, "cvat", links.new_link("http://remote", env.rid, "Pets remote"))
    found = env.client.get(f"/api/projects/{env.pid}/links").json()["links"]
    assert {x["tool"]: x["supported"] for x in found} == {"labelstudio": False, "cvat": True}
    r = env.client.post(f"/api/projects/{env.pid}/sync/labelstudio", json={})
    assert r.status_code == 404 and r.json()["code"] == "tool_unknown"
    assert env.client.delete(f"/api/projects/{env.pid}/links/labelstudio").json() == {"unlinked": True}
    assert set(links.load_links(project_dir)) == {"cvat"}
    assert env.client.get(f"/api/projects/{env.pid}").status_code == 200  # the project opens as usual
