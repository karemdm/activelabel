"""Label Studio as a remote project of the synchronisation (docs/SYNC.md), over its REST API.

One ActiveLabel project is one Label Studio project; the key of an image is its task id. Login and
error handling are those of :class:`activelabel.annotate.labelstudio_tool._Client` (PAT -> Bearer).

* Labels are read from the latest non-cancelled annotation of each task and written back to it (or a
  new annotation when the task has none), with the labeling config of the project (the control and
  image names are taken from it, so projects not created by ActiveLabel work too).
* Image size: from ``original_width/height`` of the annotation results; for tasks without any, a
  relative ``/data/...`` image is downloaded from the same server when a size is needed to write
  (``items`` reports 0x0 for such tasks).
* Uploaded files are renamed by Label Studio (``<hash>-<name>``); tasks are matched on the suffix.
"""

from __future__ import annotations

import io
import mimetypes
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.annotate.labelstudio_tool import (
    BATCH,
    FROM,
    TO,
    UPLOAD_PREFIX,
    UPLOAD_TIMEOUT,
    _Client,
    from_results,
    labeling_config,
    to_results,
)
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteTask
from activelabel.tasks import Task

PAGE = 100
CONTROLS = {
    "Choices": "Choice",
    "RectangleLabels": "Label",
    "PolygonLabels": "Label",
    "KeyPointLabels": "Label",
}


def _page(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return list(data.get(key) or data.get("results") or [])
    return list(data or [])


def _control(config: str) -> tuple[ET.Element, ET.Element]:
    """``(root, labeling control)`` of a labeling config; ToolError when it has none we understand."""
    try:
        root = ET.fromstring(config)
    except ET.ParseError:
        raise ToolError(
            "The Label Studio labeling config is not valid XML: fix it in the project settings."
        ) from None
    for el in root.iter():
        if el.tag in CONTROLS:
            return root, el
    raise ToolError(
        "The Label Studio project has no label control we can synchronise (boxes, polygons, points, choices)."
    )


def _needed_names(labels: list[Label], task: Task) -> list[str]:
    """Names a config must offer for these labels (``"<class> <n>"`` for keypoints)."""
    names: list[str] = []
    for lb in labels:
        if task is Task.KEYPOINTS:
            names += [
                f"{lb.label} {i + 1}" for i, kp in enumerate(lb.keypoints or []) if len(kp) < 3 or kp[2] > 0
            ]
        else:
            names.append(lb.label)
    return names


def _task_name(item: dict[str, Any]) -> str:
    data = item.get("data") or {}
    value = ""
    if isinstance(data, dict):
        value = str(data.get("image") or next((v for v in data.values() if isinstance(v, str)), ""))
    return UPLOAD_PREFIX.sub("", value.split("?", 1)[0].rsplit("/", 1)[-1])


def _latest(item: dict[str, Any]) -> dict[str, Any] | None:
    done = [a for a in item.get("annotations") or [] if not a.get("was_cancelled")]
    if not done:
        return None
    return max(
        done, key=lambda a: (str(a.get("updated_at") or a.get("created_at") or ""), int(a.get("id") or 0))
    )


def _size(results: list[dict[str, Any]]) -> tuple[int, int]:
    for r in results:
        if r.get("original_width") and r.get("original_height"):
            return int(r["original_width"]), int(r["original_height"])
    return 0, 0


class LabelStudioRemote:
    """A Label Studio project, through the REST API."""

    tool = "labelstudio"
    supports_tasks = False  # Label Studio has no tasks/jobs: one flat list of items

    # -- tasks (not in Label Studio) --
    def tasks(self, settings: ToolSettings, remote_id: str) -> list[RemoteTask]:
        """Label Studio has no tasks in the CVAT sense: always empty."""
        return []

    def add_task(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        name: str,
        job_size: int | None,
        images: list[tuple[ImageRecord, Path, list[Label]]],
        video: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[RemoteTask, dict[str, str]]:
        """Add the images to the project (Label Studio has no tasks/jobs; a video goes as its frames)."""
        keys = self.add(settings, remote_id, project, images)
        return RemoteTask(remote_id, name, len(keys)), keys

    def set_frames(self, settings: ToolSettings, task_id: str, delete: list[int], restore: list[int]) -> None:
        """Not available in Label Studio."""
        raise ToolError("Label Studio has no deleted frames.")

    def frames(self, settings: ToolSettings, task_id: str) -> list[tuple[str, bool]]:
        """Not available in Label Studio."""
        raise ToolError("Label Studio has no task frames.")

    def delete_task(self, settings: ToolSettings, task_id: str) -> None:
        """Not available in Label Studio."""
        raise ToolError("Label Studio has no tasks to delete.")

    def video_source(
        self, settings: ToolSettings, task_id: str, dest_dir: Path
    ) -> tuple[Path, dict[str, int]] | None:
        """Label Studio has no video tasks here."""
        return None

    def frame(self, settings: ToolSettings, task_id: str, frame: int) -> bytes:
        """Not available in Label Studio."""
        raise ToolError("Label Studio has no task frames.")

    # -- helpers --
    @staticmethod
    def _info(client: _Client, p: dict[str, Any]) -> RemoteInfo:
        images = int(p.get("task_number") or p.get("total_tasks_number") or 0)
        return RemoteInfo(
            str(p["id"]), str(p.get("title", "")), f"{client.url}/projects/{p['id']}/data", images
        )

    @staticmethod
    def _config(client: _Client, rid: str) -> tuple[str, dict[str, Any]]:
        p = client.request("GET", f"/api/projects/{rid}").json()
        return str(p.get("label_config") or ""), p

    def find(self, settings: ToolSettings, name: str) -> RemoteInfo | None:
        """The project with exactly this title (the oldest when there are several)."""
        client = _Client(settings)
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            data = client.request(
                "GET", "/api/projects", params={"title": name, "page": page, "page_size": PAGE}
            ).json()
            rows = _page(data, "results")
            found += [p for p in rows if p.get("title") == name]
            if not rows or not (isinstance(data, dict) and data.get("next")):
                break
            page += 1
        return self._info(client, min(found, key=lambda p: int(p["id"]))) if found else None

    def get(self, settings: ToolSettings, remote_id: str) -> RemoteInfo:
        """One project (ToolError when it is gone)."""
        client = _Client(settings)
        return self._info(client, client.request("GET", f"/api/projects/{remote_id}").json())

    def create(self, settings: ToolSettings, project: Project) -> RemoteInfo:
        """Create the project with the labeling config of the task and the project classes."""
        client = _Client(settings)
        body = {
            "title": f"ActiveLabel {project.id} - {project.name}",
            "label_config": labeling_config(project.task, list(project.classes), 1),
        }
        return self._info(client, client.request("POST", "/api/projects", json=body).json())

    # -- config --
    def ensure_classes(self, settings: ToolSettings, remote_id: str, classes: list[str]) -> None:
        """Add the classes the labeling config does not offer yet (keypoints: ``"<class> 1"``)."""
        client = _Client(settings)
        _, control = _control(self._config(client, remote_id)[0])
        keypoints = control.tag == "KeyPointLabels"
        have = [str(c.get("value", "")) for c in control]
        names = [
            f"{c} 1" if keypoints else c
            for c in classes
            if not (any(h.startswith(f"{c} ") for h in have) if keypoints else c in have)
        ]
        self._ensure_names(client, remote_id, names)

    def _ensure_names(self, client: _Client, rid: str, names: list[str]) -> tuple[str, str]:
        """Add missing label names to the config; return the ``(from_name, to_name)`` of the control."""
        config, _ = self._config(client, rid)
        root, control = _control(config)
        have = {str(c.get("value", "")) for c in control}
        missing = [n for n in dict.fromkeys(names) if n not in have]
        if missing:
            for n in missing:
                ET.SubElement(control, CONTROLS[control.tag], {"value": n})
            client.request(
                "PATCH", f"/api/projects/{rid}", json={"label_config": ET.tostring(root, encoding="unicode")}
            )
        return str(control.get("name") or FROM), str(control.get("toName") or TO)

    # -- read --
    def _tasks(self, client: _Client, rid: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            data = client.request(
                "GET", "/api/tasks", params={"project": rid, "fields": "all", "page": page, "page_size": PAGE}
            ).json()
            rows = _page(data, "tasks")
            out += rows
            total = int(data.get("total") or 0) if isinstance(data, dict) else 0
            if len(rows) < PAGE or (total and len(out) >= total):
                return out
            page += 1

    def items(
        self, settings: ToolSettings, remote_id: str, task: Task, task_ids: list[str] | None = None
    ) -> list[RemoteItem]:
        """Every task with the labels of its latest non-cancelled annotation."""
        client = _Client(settings)
        out: list[RemoteItem] = []
        for t in self._tasks(client, remote_id):
            ann = _latest(t)
            results = list((ann or {}).get("result") or [])
            w, h = _size(results)
            out.append(RemoteItem(str(t["id"]), _task_name(t), w, h, from_results(results, task, 0)))
        return out

    # -- write --
    def _image_size(self, client: _Client, item: dict[str, Any]) -> tuple[int, int]:
        """Size of the task image downloaded from the same server (0x0 when it cannot be read)."""
        data = item.get("data") or {}
        value = str(data.get("image") or "") if isinstance(data, dict) else ""
        if not value.startswith("/"):
            return 0, 0  # external URL: never send the token there
        try:
            from PIL import Image

            with Image.open(io.BytesIO(client.request("GET", value, timeout=UPLOAD_TIMEOUT).content)) as im:
                return int(im.width), int(im.height)
        except Exception:  # noqa: BLE001 - the size is best effort
            return 0, 0

    @staticmethod
    def _retarget(results: list[dict[str, Any]], names: tuple[str, str]) -> list[dict[str, Any]]:
        for r in results:
            r["from_name"], r["to_name"] = names
        return results

    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Replace the latest annotation of each task (a new one when it has none)."""
        client = _Client(settings)
        wanted = [n for labels in updates.values() for n in _needed_names(labels, project.task)]
        names = self._ensure_names(client, remote_id, wanted)
        for key, labels in updates.items():
            if not str(key).isdigit():
                raise ToolError(f"'{key}' is not a Label Studio item key (expected a task id).")
            item = client.request("GET", f"/api/tasks/{key}").json()
            ann = _latest(item)
            size = _size(list((ann or {}).get("result") or []))
            if not all(size) and labels:
                size = self._image_size(client, item)
            results = self._retarget(to_results(labels, project.task, size), names)
            if ann is not None:
                client.request(
                    "PATCH", f"/api/annotations/{ann['id']}", json={"result": results, "was_cancelled": False}
                )
            elif results:
                client.request("POST", f"/api/tasks/{key}/annotations", json={"result": results})

    # -- add --
    def add(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> dict[str, str]:
        """Import the images, find their new tasks by file name and create their annotations."""
        client = _Client(settings)
        wanted = [n for _, _, labels in images for n in _needed_names(labels, project.task)]
        names = self._ensure_names(client, remote_id, wanted)
        before = {int(t["id"]) for t in self._tasks(client, remote_id)}
        upload: dict[str, str] = {}  # image id -> file name sent
        for rec, _, _ in images:
            base = Path(rec.name).name
            upload[rec.id] = base if base not in upload.values() else f"{rec.id}_{base}"
        for start in range(0, len(images), BATCH):
            handles = []
            try:
                files = []
                for rec, path, _ in images[start : start + BATCH]:
                    fh = open(path, "rb")  # noqa: SIM115 - closed in the finally below
                    handles.append(fh)
                    mime = mimetypes.guess_type(upload[rec.id])[0] or "application/octet-stream"
                    files.append(("file", (upload[rec.id], fh, mime)))
                client.request(
                    "POST",
                    f"/api/projects/{remote_id}/import?commit_to_project=true",
                    files=files,
                    timeout=UPLOAD_TIMEOUT,
                )
            finally:
                for fh in handles:
                    fh.close()
        new = {
            _task_name(t): int(t["id"]) for t in self._tasks(client, remote_id) if int(t["id"]) not in before
        }
        keys: dict[str, str] = {}
        for rec, _, labels in images:
            tid = new.get(upload[rec.id])
            if tid is None:
                raise ToolError(f"Label Studio did not create a task for '{rec.name}': upload again.")
            keys[rec.id] = str(tid)
            results = self._retarget(to_results(labels, project.task, (rec.width, rec.height)), names)
            if results:
                client.request("POST", f"/api/tasks/{tid}/annotations", json={"result": results})
        return keys
