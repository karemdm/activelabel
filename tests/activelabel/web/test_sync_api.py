import time
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image
from sync_fakes import box

from activelabel.labels import AnnotationSet, load_annotations, save_annotations
from activelabel.sync import links

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
