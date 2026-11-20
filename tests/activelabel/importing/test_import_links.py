import time
from types import SimpleNamespace

import pytest
from api_client import make_client
from sample_images import make_image
from sync_fakes import box

from activelabel.annotate.base import ToolSettings
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.importing import common
from activelabel.importing.common import Origin
from activelabel.sync import links
from activelabel.web import import_api


def fake_importer(folder):
    """An importer that makes a project of the images in ``folder`` with one box on the first."""

    def check_remote(settings, kind, remote_id):
        return {"name": "Remote pets", "classes": ["cat"]}

    def import_project(store, settings, workspace, *, kind, remote_id, name, description, task, progress):
        origin = Origin("cvat", settings.url, str(remote_id), kind, "Remote pets")
        with common.new_project(store, origin, name, description, task) as project:
            manifest = Manifest(store.dir(project.id))
            report = ingest_folder(manifest, folder, folder)
            first = sorted(manifest.all(), key=lambda r: r.name)[0]
            return common.finish(store, project, {first.id: [box()]}, ["cat"], origin, report, [], task)

    return SimpleNamespace(check_remote=check_remote, import_project=import_project)


@pytest.fixture
def env(tmp_path, monkeypatch, fake_keyring, remotes):
    folder = tmp_path / "remote_files"
    for i in range(2):
        make_image(folder / f"img{i}.png", seed=i + 1, fmt="PNG")
    monkeypatch.setitem(import_api.IMPORTERS, "cvat", fake_importer(folder))
    client, state = make_client(tmp_path)
    body = {"url": "http://cvat", "username": "u", "secret": "pw-123"}
    assert client.put("/api/annotation-tools/cvat", json=body).status_code == 200
    return SimpleNamespace(client=client, state=state, remote=remotes["cvat"], settings=ToolSettings("cvat"))


def run_import(env):
    r = env.client.post("/api/import", json={"tool": "cvat", "remote_id": "5", "task": "detection"})
    assert r.status_code == 200
    for _ in range(200):
        data = env.client.get(f"/api/jobs/{r.json()['id']}").json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_import_creates_link_mapping_every_image(env):
    env.remote.projects["5"] = {"name": "Remote pets", "items": {}, "classes": ["cat"]}
    env.remote.put("5", "img0.png", [box()])
    env.remote.put("5", "img1.png", [])
    job = run_import(env)
    assert job["status"] == "done" and job["result"]["linked"] is True
    directory = env.state.store.dir(job["result"]["project_id"])
    link = links.get_link(directory, "cvat")
    assert link["remote_id"] == "5" and link["remote_name"] == "Remote pets"
    assert link["url"] == "http://remote/projects/5"
    records = {r.name: r.id for r in Manifest(directory).all()}
    assert link["items"][records["img0.png"]] == links.base_entry(env.remote.key_of("5", "img0.png"), [box()])
    assert link["items"][records["img1.png"]]["base"] == links.label_hash([])


def test_import_without_reachable_remote_still_succeeds(env):
    env.remote.fail = "server down"
    job = run_import(env)
    assert job["status"] == "done" and job["result"]["linked"] is False
    assert links.load_links(env.state.store.dir(job["result"]["project_id"])) == {}
