"""An in-memory remote project (RemoteProject) and project builders for the synchronisation tests."""

import copy
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from sample_images import make_image

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.jobs import Job, Progress
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.projects import Project, ProjectStore
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteJob, RemoteTask

SETTINGS = ToolSettings("cvat", "http://remote", "u", "pw")


def progress() -> Progress:
    return Progress(Job("t", 1, "sync"), threading.Event())


def box(name="cat", x=0.1, y=0.1, w=0.3, h=0.3) -> Label:
    return Label(name, bbox=[x, y, w, h])


class FakeRemote:
    """Remote projects kept in memory: ``projects[id] = {name, items: {key: RemoteItem}, classes}``.

    By default a flat list of items (``supports_tasks`` False, like Label Studio). With ``tasks=True``
    it behaves like CVAT: tasks with jobs (``task_meta[rid][task_id]``), deleted frames hidden from
    ``items()``, ``frame()`` bytes, ``add_task`` / ``set_frames`` / ``delete_task``.
    """

    def __init__(self, tool="cvat", tasks=False):
        self.tool = tool
        self.supports_tasks = tasks
        self.task_meta: dict[str, dict[str, dict]] = {}
        self.frame_bytes: dict[tuple[str, int], bytes] = {}
        self._png_seed = 1000
        self._task_n = 100
        self.projects: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.item_calls: list[list[str] | None] = []
        self.fail = ""
        self._n = 1

    # -- test helpers
    def seed(self, name, files=(), classes=("cat",)) -> RemoteInfo:
        rid = str(self._n)
        self._n += 1
        self.projects[rid] = {"name": name, "items": {}, "classes": list(classes)}
        self.task_meta[rid] = {}
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
        shown = [k for k in p["items"] if not self._hidden(rid, k)]
        return RemoteInfo(rid, p["name"], f"http://remote/projects/{rid}", len(shown))

    def _hidden(self, rid, key) -> bool:
        task, _, frame = key.partition(":")
        meta = self.task_meta.get(rid, {}).get(task)
        return bool(meta) and int(frame or 0) in meta["deleted"]

    def _png(self) -> bytes:
        self._png_seed += 1
        rng = np.random.default_rng(self._png_seed)
        data = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        out = io.BytesIO()
        Image.fromarray(data).save(out, format="PNG")
        return out.getvalue()

    def new_task(self, rid, name, files, job_size=None, task_id=None) -> str:
        """Test helper: someone adds a whole task in the tool. ``files``: ``[(file_name, labels)]``."""
        self._task_n += 1
        tid = task_id or str(self._task_n)
        self.task_meta[rid][tid] = {"name": name, "job_size": job_size, "deleted": set(), "size": len(files)}
        for frame, (file_name, labels) in enumerate(files):
            self.put(rid, file_name, labels, key=f"{tid}:{frame}")
            self.frame_bytes[(tid, frame)] = self._png()
        return tid

    def task_keys(self, rid, task_id) -> list[str]:
        """Keys of the frames of a remote task (deleted ones included), in frame order."""
        keys = [k for k in self.projects[rid]["items"] if k.partition(":")[0] == str(task_id)]
        return sorted(keys, key=lambda k: int(k.partition(":")[2]))

    def frames(self, settings, task_id):
        """``(file name, deleted)`` of every frame of a task, in order."""
        self._check()
        rid = next(r for r, tasks in self.task_meta.items() if str(task_id) in tasks)
        deleted = self.task_meta[rid][str(task_id)]["deleted"]
        return [
            (self.projects[rid]["items"][k].file_name, int(k.partition(":")[2]) in deleted)
            for k in self.task_keys(rid, task_id)
        ]

    def deleted_frames(self, rid, task_id) -> list[int]:
        return sorted(self.task_meta[rid][str(task_id)]["deleted"])

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
            raise ToolError(f"The remote project {remote_id} was not found (deleted?).")
        return self.info(remote_id)

    def create(self, settings, project):
        self._check()
        self.calls.append(("create", project.name))
        return self.seed(f"ActiveLabel {project.id} - {project.name}", classes=project.classes)

    def items(self, settings, remote_id, task, task_ids=None):
        self._check()
        self.item_calls.append(None if task_ids is None else sorted(task_ids))
        wanted = None if task_ids is None else {str(t) for t in task_ids}
        found = [
            it
            for k, it in self.projects[remote_id]["items"].items()
            if (wanted is None or k.partition(":")[0] in wanted) and not self._hidden(remote_id, k)
        ]
        return copy.deepcopy(found)

    def tasks(self, settings, remote_id):
        self._check()
        if not self.supports_tasks:
            return []
        out = []
        for tid, meta in self.task_meta[remote_id].items():
            size = meta["size"]
            step = meta["job_size"] or size or 1
            jobs = [
                RemoteJob(f"{tid}{n + 1:02d}", n + 1, start, min(start + step, size) - 1)
                for n, start in enumerate(range(0, size, step))
            ]
            out.append(
                RemoteTask(tid, meta["name"], size, meta["job_size"], f"http://remote/tasks/{tid}", jobs)
            )
        return out

    def add_task(self, settings, remote_id, project: Project, name, job_size, images):
        self._check()
        self.calls.append(("add_task", name, job_size, [r.name for r, _, _ in images]))
        files = [(r.name, labels) for r, _, labels in images]
        tid = self.new_task(remote_id, name, files, job_size)
        task = next(t for t in self.tasks(settings, remote_id) if t.id == tid)
        keys = {r.id: f"{tid}:{n}" for n, (r, _, _) in enumerate(images)}
        return task, keys

    def set_frames(self, settings, task_id, delete, restore):
        self._check()
        self.calls.append(("set_frames", str(task_id), sorted(delete), sorted(restore)))
        for meta in (m[str(task_id)] for m in self.task_meta.values() if str(task_id) in m):
            meta["deleted"] |= set(delete)
            meta["deleted"] -= set(restore)

    def delete_task(self, settings, task_id):
        self._check()
        self.calls.append(("delete_task", str(task_id)))
        for rid, metas in self.task_meta.items():
            if str(task_id) in metas:
                del metas[str(task_id)]
                items = self.projects[rid]["items"]
                for key in [k for k in items if k.partition(":")[0] == str(task_id)]:
                    del items[key]

    def frame(self, settings, task_id, frame):
        self._check()
        self.calls.append(("frame", str(task_id), frame))
        return self.frame_bytes[(str(task_id), frame)]

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
