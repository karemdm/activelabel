import time
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image
from sync_fakes import box

from activelabel.annotate import registry
from activelabel.annotate.base import Sent
from activelabel.labels import AnnotationSet, save_annotations
from activelabel.sync import links
from activelabel.web import annotate_api


class FakeTool:
    """Like the Label Studio tool: creates the remote project and tells its name in the ref."""

    name = "labelstudio"

    def __init__(self, remote, project_name):
        self.remote, self.project_name = remote, project_name

    def send(self, dataset, image_ids, project, settings, session_id):
        names = {r.id: r.name for r in dataset.records}
        self.remote.seed(self.project_name, [(names[i], dataset.labels.get(i, [])) for i in image_ids])
        return Sent([{"label": "Task", "url": "http://ls/tasks/1"}], {"project_name": self.project_name})


@pytest.fixture
def env(tmp_path, monkeypatch, remotes):
    client, state = make_client(tmp_path)
    pid = client.post("/api/projects", json={"name": "p", "task": "detection", "classes": ["cat"]}).json()[
        "id"
    ]
    paths = [make_image(tmp_path / "img" / f"i{i}.png", seed=i + 1, fmt="PNG") for i in range(2)]
    assert upload(client, pid, paths).status_code == 200
    items = client.get(f"/api/projects/{pid}/images").json()["items"]
    ids = {i["name"]: i["id"] for i in items}
    save_annotations(
        state.store.dir(pid), AnnotationSet(state.store.get(pid).task, ["cat"], {ids["i0.png"]: [box()]}, {})
    )
    dataset = SimpleNamespace(records=state.manifest(pid).all(), labels={ids["i0.png"]: [box()]})
    tool = FakeTool(remotes["labelstudio"], f"ActiveLabel {pid} - p")
    monkeypatch.setitem(registry.OVERRIDES, "labelstudio", tool)
    monkeypatch.setattr(annotate_api, "prepare_dataset", lambda project, state: dataset)
    body = {"url": "http://ls", "username": "u", "secret": "pw-123"}
    assert client.put("/api/annotation-tools/labelstudio", json=body).status_code == 200
    return SimpleNamespace(
        client=client, state=state, pid=pid, ids=ids, remote=remotes["labelstudio"], tool=tool
    )


def send(env):
    return env.client.post(f"/api/projects/{env.pid}/annotate", json={"tool": "labelstudio"})


def wait(env, job):
    for _ in range(200):
        jobs = env.client.get(f"/api/projects/{env.pid}/jobs").json()
        data = next(j for j in jobs if j["id"] == job["id"])
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_send_conflicts_with_remote_project_of_the_same_name(env):
    info = env.remote.seed(env.tool.project_name, [("i0.png", [])])
    r = send(env)
    assert r.status_code == 409
    assert r.json()["detail"] == {
        "code": "remote_exists",
        "tool": "labelstudio",
        "remote": {"id": info.id, "name": env.tool.project_name, "url": info.url, "images": 1},
        "linked": False,
        "options": ["cancel", "sync"],
    }
    assert len(env.remote.projects) == 1  # nothing was sent


def test_send_conflicts_with_link(env):
    info = env.remote.seed("Another name", [])
    links.save_link(
        env.state.store.dir(env.pid), "labelstudio", links.new_link("http://ls", info.id, "Another name")
    )
    r = send(env)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert (
        detail["code"] == "remote_exists" and detail["linked"] is True and detail["remote"]["id"] == info.id
    )


def test_link_created_after_send_then_second_send_conflicts(env):
    job = wait(env, send(env).json())
    assert job["status"] == "done"
    link = links.get_link(env.state.store.dir(env.pid), "labelstudio")
    (rid,) = env.remote.projects
    assert link["remote_id"] == rid and set(link["items"]) == set(env.ids.values())
    assert link["items"][env.ids["i0.png"]]["base"] == links.label_hash([box()])
    assert link["items"][env.ids["i1.png"]]["base"] == links.label_hash([])
    again = send(env)
    assert again.status_code == 409 and again.json()["detail"]["linked"] is True


def test_send_still_works_when_linking_fails(env):
    env.tool.send = lambda *a, **k: Sent([], {"project_name": "missing"})
    job = wait(env, send(env).json())
    assert job["status"] == "done"
    assert links.get_link(env.state.store.dir(env.pid), "labelstudio") is None


def test_link_to_a_deleted_remote_project_is_dropped_and_send_goes_on(env):
    links.save_link(
        env.state.store.dir(env.pid), "labelstudio", links.new_link("http://ls", "999", "Deleted")
    )
    job = wait(env, send(env).json())
    assert job["status"] == "done"
    link = links.get_link(env.state.store.dir(env.pid), "labelstudio")
    assert link is not None and link["remote_id"] != "999"  # the new project is linked instead
