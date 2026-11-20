"""An in-memory remote project (RemoteProject) and project builders for the synchronisation tests."""

import copy
import threading
from pathlib import Path

from sample_images import make_image

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.jobs import Job, Progress
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.projects import ProjectStore
from activelabel.sync.remote import RemoteInfo, RemoteItem

SETTINGS = ToolSettings("cvat", "http://remote", "u", "pw")


def progress() -> Progress:
    return Progress(Job("t", 1, "sync"), threading.Event())


def box(name="cat", x=0.1, y=0.1, w=0.3, h=0.3) -> Label:
    return Label(name, bbox=[x, y, w, h])


class FakeRemote:
    """Remote projects kept in memory: ``projects[id] = {name, items: {key: RemoteItem}, classes}``."""

    def __init__(self, tool="cvat"):
        self.tool = tool
        self.projects: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.fail = ""
        self._n = 1

    # -- test helpers
    def seed(self, name, files=(), classes=("cat",)) -> RemoteInfo:
        rid = str(self._n)
        self._n += 1
        self.projects[rid] = {"name": name, "items": {}, "classes": list(classes)}
        for file_name, labels in files:
            self.put(rid, file_name, labels)
        return self.info(rid)

    def put(self, rid, file_name, labels, key=None) -> str:
        items = self.projects[rid]["items"]
        key = key or f"t{len(items) + 1}:0"
        items[key] = RemoteItem(key, file_name, 64, 48, copy.deepcopy(list(labels)))
        return key

    def key_of(self, rid, file_name) -> str:
        return next(k for k, it in self.projects[rid]["items"].items() if it.file_name == file_name)

    def labels(self, rid, file_name) -> list[Label]:
        return self.projects[rid]["items"][self.key_of(rid, file_name)].labels

    def info(self, rid) -> RemoteInfo:
        p = self.projects[rid]
        return RemoteInfo(rid, p["name"], f"http://remote/projects/{rid}", len(p["items"]))

    def _check(self):
        if self.fail:
            raise ToolError(self.fail)

    # -- RemoteProject
    def find(self, settings, name):
        self._check()
        return next((self.info(i) for i, p in self.projects.items() if p["name"] == name), None)

    def get(self, settings, remote_id):
        self._check()
        if remote_id not in self.projects:
            raise ToolError(f"The remote project {remote_id} does not exist.")
        return self.info(remote_id)

    def create(self, settings, project):
        self._check()
        self.calls.append(("create", project.name))
        return self.seed(f"ActiveLabel {project.id} - {project.name}", classes=project.classes)

    def items(self, settings, remote_id, task):
        self._check()
        return copy.deepcopy(list(self.projects[remote_id]["items"].values()))

    def add(self, settings, remote_id, project, images):
        self._check()
        self.calls.append(("add", [r.name for r, _, _ in images]))
        return {r.id: self.put(remote_id, r.name, labels) for r, _, labels in images}

    def write(self, settings, remote_id, project, updates):
        self._check()
        self.calls.append(("write", sorted(updates)))
        for key, labels in updates.items():
            self.projects[remote_id]["items"][key].labels = copy.deepcopy(labels)

    def ensure_classes(self, settings, remote_id, classes):
        self._check()
        self.calls.append(("classes", list(classes)))
        have = self.projects[remote_id]["classes"]
        have += [c for c in classes if c not in have]


def make_project(tmp_path: Path, count=3, labels=None, classes=("cat", "dog"), store=None):
    """A project with ``count`` images named img0.png...; ``labels``: ``{index: [Label]}`` ground truth.

    Returns ``(store, project, ids)`` with ``ids`` in name order.
    """
    store = store or ProjectStore(tmp_path / "projects")
    project = store.create("Pets", "", "detection", list(classes))
    folder = tmp_path / "src"
    for i in range(count):
        make_image(folder / f"img{i}.png", seed=i + 1, fmt="PNG")
    manifest = Manifest(store.dir(project.id))
    ingest_folder(manifest, folder, folder)
    by_name = {r.name: r.id for r in manifest.all()}
    ids = [by_name[f"img{i}.png"] for i in range(count)]
    if labels:
        save_annotations(
            store.dir(project.id),
            AnnotationSet(project.task, list(classes), {ids[i]: v for i, v in labels.items()}, {}),
        )
    return store, project, ids
