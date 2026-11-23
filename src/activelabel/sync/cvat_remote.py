"""CVAT as a remote project of the synchronisation (docs/SYNC.md), over the plain REST API v2.

One ActiveLabel project is one CVAT *project*; its images live in CVAT tasks. The key of an image is
``"<task_id>:<frame>"``. Basic auth (url, username, password) like the other CVAT code; the password
never appears in logs or messages.

Conversions (relative 0..1 <-> pixels of the frame, from ``GET /api/tasks/{id}/data/meta``):

* rectangle -> bbox; polygon -> polygon ring + bbox; points -> one keypoints object (all the points
  of the shape; a point at the top-left corner is "not labeled", as in ``cvat_tool``) + bbox; tag ->
  image class (classification projects only).
* Other shape types (polyline, ellipse, cuboid, mask, skeleton) and *tracks* are not converted: they
  are neither read nor touched when writing (tracks are kept as they are).
* Attributes are never sent (CVAT rejects attribute specs of another label).
"""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import ImageRecord
from activelabel.importing.cvat_import import _Http
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteJob, RemoteTask
from activelabel.tasks import Task

log = logging.getLogger(__name__)

TIMEOUT = 60
UPLOAD_TIMEOUT = 600
BATCH = 100  # images per CVAT task created by ``add``
POLL_INTERVAL = 2.0  # seconds between checks of a task's data
MAX_POLLS = 300  # checks before giving up (about ten minutes)
ALL = 1_000_000  # "no limit" for paginated lists
MISSING_CORNER = (0.0, 0.0)  # position of a point that is not labeled (as in cvat_tool)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


class _Api(_Http):
    """CVAT REST client: the read side of the importer plus writes, with readable errors."""

    def request(self, method: str, path: str, *, timeout: int = TIMEOUT, **kw: Any) -> Any:
        """Call the API and return the decoded JSON (None when the answer has no body)."""
        try:
            r = self._http.request(method, f"{self.url}{path}", auth=self._auth, timeout=timeout, **kw)
        except Exception:  # noqa: BLE001 - network/timeouts: never echo the exception text
            raise ToolError(f"Cannot reach CVAT at {self.url}: check the address and your network.") from None
        status = r.status_code
        where = path.split("?", 1)[0]
        if status in (401, 403):
            raise ToolError(f"CVAT login failed at {self.url}: check the username and password.")
        if status == 404:
            from activelabel.importing.common import RemoteNotFound

            raise RemoteNotFound(f"CVAT answered 'not found' for {where}: it may have been deleted.")
        if status == 429:
            raise ToolError("CVAT asked to slow down (too many requests): wait a minute and try again.")
        if status >= 400:
            # CVAT explains validation problems in the body (never contains the login)
            reason = " ".join((getattr(r, "text", "") or "").split())[:300]
            raise ToolError(f"CVAT error {status} on {where}." + (f" CVAT says: {reason}" if reason else ""))
        if not getattr(r, "content", b"x"):
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON document."""
        return self.request("GET", path, params=params)


# -- conversions ---------------------------------------------------------------------------------
def _box(xs: list[float], ys: list[float]) -> list[float]:
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def shape_to_label(shape: dict[str, Any], name: str, w: int, h: int) -> Label | None:
    """One CVAT shape -> internal label (None for unsupported types)."""
    kind, pts = shape.get("type"), [float(p) for p in shape.get("points") or []]
    w, h = max(w, 1), max(h, 1)
    xs = [_clamp(x / w) for x in pts[0::2]]
    ys = [_clamp(y / h) for y in pts[1::2]]
    if kind == "rectangle" and len(pts) >= 4:
        x0, x1 = sorted(xs[:2])
        y0, y1 = sorted(ys[:2])
        return Label(name, bbox=[x0, y0, x1 - x0, y1 - y0])
    if kind == "polygon" and len(pts) >= 6:
        return Label(name, bbox=_box(xs, ys), polygons=[[[x, y] for x, y in zip(xs, ys, strict=True)]])
    if kind == "points" and len(pts) >= 2:
        vis = 1.0 if shape.get("occluded") else 2.0
        kps = [
            [0.0, 0.0, 0.0] if (px, py) == MISSING_CORNER else [x, y, vis]
            for px, py, x, y in zip(pts[0::2], pts[1::2], xs, ys, strict=True)
        ]
        seen = [k for k in kps if k[2]]
        box = _box([k[0] for k in seen], [k[1] for k in seen]) if seen else [0.0, 0.0, 0.0, 0.0]
        return Label(name, bbox=box, keypoints=kps)
    return None


def labels_to_cvat(
    labels: list[Label], ids: dict[str, int], frame: int, w: int, h: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Internal labels -> ``(shapes, tags)`` of one frame, in pixels (unknown classes are skipped)."""
    shapes: list[dict[str, Any]] = []
    tags: list[dict[str, Any]] = []

    def shape(kind: str, label_id: int, points: list[float]) -> dict[str, Any]:
        return {
            "type": kind,
            "frame": frame,
            "label_id": label_id,
            "points": points,
            "rotation": 0,
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "group": 0,
            "source": "manual",
            "attributes": [],
        }

    for lb in labels:
        lid = ids.get(lb.label)
        if lid is None:
            continue
        if lb.polygons:
            for ring in lb.polygons:
                if len(ring) >= 3:
                    shapes.append(shape("polygon", lid, [v for p in ring for v in (p[0] * w, p[1] * h)]))
        elif lb.keypoints:
            flat: list[float] = []
            for kp in lb.keypoints:
                missing = len(kp) > 2 and kp[2] <= 0
                flat += list(MISSING_CORNER) if missing else [kp[0] * w, kp[1] * h]
            shapes.append(shape("points", lid, flat))
        elif lb.bbox:
            x, y, bw, bh = lb.bbox
            shapes.append(shape("rectangle", lid, [x * w, y * h, (x + bw) * w, (y + bh) * h]))
        else:
            tags.append({"frame": frame, "label_id": lid, "group": 0, "source": "manual", "attributes": []})
    return shapes, tags


def _split_key(key: str) -> tuple[int, int]:
    task, _, frame = str(key).partition(":")
    if not (task.isdigit() and frame.isdigit()):
        raise ToolError(f"'{key}' is not a CVAT item key (expected '<task>:<frame>').")
    return int(task), int(frame)


def _label_payload(name: str) -> dict[str, Any]:
    return {"name": name, "type": "any", "attributes": []}


class CvatRemote:
    """A CVAT project, through the REST API."""

    tool = "cvat"
    supports_tasks = True  # tasks/jobs mirror the project's (docs/TASKS.md)

    # -- tasks, frames --
    def set_frames(self, settings: ToolSettings, task_id: str, delete: list[int], restore: list[int]) -> None:
        """Mark frames of a task as deleted (or not deleted any more); annotations are kept by CVAT."""
        if not delete and not restore:
            return
        api = self._api(settings)
        meta = api.get(f"/api/tasks/{task_id}/data/meta") or {}
        deleted = ({int(d) for d in meta.get("deleted_frames") or []} | set(delete)) - set(restore)
        api.request("PATCH", f"/api/tasks/{task_id}/data/meta", json={"deleted_frames": sorted(deleted)})

    def delete_task(self, settings: ToolSettings, task_id: str) -> None:
        """Delete a CVAT task with its frames and annotations (nothing when it is already gone)."""
        api = self._api(settings)
        try:
            api.request("DELETE", f"/api/tasks/{task_id}")
        except ToolError as err:
            if "not found" not in str(err).lower():
                raise

    def frame(self, settings: ToolSettings, task_id: str, frame: int) -> bytes:
        """The original image of one frame."""
        api = self._api(settings)
        try:
            r = api._http.get(  # noqa: SLF001 - raw bytes, not JSON
                f"{api.url}/api/tasks/{task_id}/data",
                params={"type": "frame", "number": str(int(frame)), "quality": "original"},
                auth=api._auth,  # noqa: SLF001
                timeout=UPLOAD_TIMEOUT,
            )
        except Exception:  # noqa: BLE001 - never echo the exception text
            raise ToolError(f"Cannot reach CVAT at {api.url}: check the address and your network.") from None
        if r.status_code >= 400:
            raise ToolError(f"CVAT error {r.status_code} while downloading frame {frame} of task {task_id}.")
        return bytes(r.content)

    # -- helpers --
    @staticmethod
    def _api(settings: ToolSettings) -> _Api:
        return _Api(settings)

    @staticmethod
    def _info(api: _Api, item: dict[str, Any]) -> RemoteInfo:
        tasks = api.paged("/api/tasks", {"project_id": item["id"]}, ALL)
        images = sum(int(t.get("size") or 0) for t in tasks)
        return RemoteInfo(
            str(item["id"]), str(item.get("name", "")), f"{api.url}/projects/{item['id']}", images
        )

    @staticmethod
    def _labels(api: _Api, rid: str) -> dict[str, int]:
        found = api.paged("/api/labels", {"project_id": rid}, ALL)
        out: dict[str, int] = {}
        for x in found:
            if x.get("name") and not x.get("parent_id"):
                out.setdefault(str(x["name"]), int(x["id"]))
        return out

    def find(self, settings: ToolSettings, name: str) -> RemoteInfo | None:
        """The project with exactly this name (the oldest when there are several)."""
        api = self._api(settings)
        found = [
            p for p in api.paged("/api/projects", {"name": name, "sort": "id"}, ALL) if p.get("name") == name
        ]
        return self._info(api, found[0]) if found else None

    def get(self, settings: ToolSettings, remote_id: str) -> RemoteInfo:
        """One project (ToolError when it is gone)."""
        api = self._api(settings)
        return self._info(api, api.get(f"/api/projects/{remote_id}"))

    def create(self, settings: ToolSettings, project: Project) -> RemoteInfo:
        """Create the project with the classes of the ActiveLabel project (labels of type "any")."""
        api = self._api(settings)
        body = {
            "name": f"ActiveLabel {project.id} - {project.name}",
            "labels": [_label_payload(c) for c in project.classes],
        }
        return self._info(api, api.request("POST", "/api/projects", json=body))

    def ensure_classes(self, settings: ToolSettings, remote_id: str, classes: list[str]) -> None:
        """Add the labels the project does not have yet (existing ones are not touched)."""
        self._ensure(self._api(settings), remote_id, classes)

    def _ensure(self, api: _Api, rid: str, names: list[str]) -> dict[str, int]:
        ids = self._labels(api, rid)
        missing = [n for n in dict.fromkeys(names) if n not in ids]
        if missing:
            api.request(
                "PATCH", f"/api/projects/{rid}", json={"labels": [_label_payload(n) for n in missing]}
            )
            ids = self._labels(api, rid)
        return ids

    # -- read --
    def tasks(self, settings: ToolSettings, remote_id: str) -> list[RemoteTask]:
        """The tasks of the project (oldest first) with their jobs in frame order."""
        api = self._api(settings)
        out: list[RemoteTask] = []
        for t in api.paged("/api/tasks", {"project_id": remote_id, "sort": "id"}, ALL):
            out.append(self._task(api, t))
        return out

    @staticmethod
    def _task(api: _Api, t: dict[str, Any]) -> RemoteTask:
        tid = t["id"]
        jobs = sorted(
            (  # annotation jobs only (not ground-truth jobs)
                j
                for j in api.paged("/api/jobs", {"task_id": tid}, ALL)
                if str(j.get("type") or "annotation") == "annotation"
            ),
            key=lambda j: int(j.get("start_frame") or 0),
        )
        segment = int(t.get("segment_size") or 0)
        size = int(t.get("size") or 0)
        return RemoteTask(
            str(tid),
            str(t.get("name", "")),
            size,
            segment if segment and segment < size else None,
            f"{api.url}/tasks/{tid}",
            [
                RemoteJob(
                    str(j["id"]),
                    n,
                    int(j.get("start_frame") or 0),
                    int(j.get("stop_frame") or 0),
                    f"{api.url}/tasks/{tid}/jobs/{j['id']}",
                    str(j.get("state") or ""),
                    str(j.get("stage") or ""),
                    str((j.get("assignee") or {}).get("username") or ""),
                )
                for n, j in enumerate(jobs, 1)
            ],
        )

    def items(
        self, settings: ToolSettings, remote_id: str, task: Task, task_ids: list[str] | None = None
    ) -> list[RemoteItem]:
        """Every frame of every task of the project (or of ``task_ids``) with its shapes/tags converted.

        Tracks are skipped.
        """
        api = self._api(settings)
        names = {i: n for n, i in self._labels(api, remote_id).items()}
        out: list[RemoteItem] = []
        for t in api.paged("/api/tasks", {"project_id": remote_id, "sort": "id"}, ALL):
            tid = t["id"]
            if task_ids is not None and str(tid) not in task_ids:
                continue
            meta = api.get(f"/api/tasks/{tid}/data/meta") or {}
            frames = list(meta.get("frames") or [])
            ann = api.get(f"/api/tasks/{tid}/annotations") or {}
            if ann.get("tracks"):
                log.warning("CVAT task %s has tracks: they are not synchronised.", tid)
            per_frame: dict[int, list[Label]] = {}
            for s in ann.get("shapes") or []:
                f, name = int(s.get("frame", -1)), names.get(s.get("label_id"))
                if task is Task.CLASSIFICATION or name is None or not 0 <= f < len(frames):
                    continue
                label = shape_to_label(s, name, int(frames[f]["width"]), int(frames[f]["height"]))
                if label:
                    per_frame.setdefault(f, []).append(label)
            if task is Task.CLASSIFICATION:
                for tag in ann.get("tags") or []:
                    f, name = int(tag.get("frame", -1)), names.get(tag.get("label_id"))
                    if name and 0 <= f < len(frames) and f not in per_frame:
                        per_frame[f] = [Label(name)]
            deleted = {int(d) for d in meta.get("deleted_frames") or []}
            for f, fr in enumerate(frames):
                if f in deleted:
                    continue
                out.append(
                    RemoteItem(
                        f"{tid}:{f}",
                        str(fr["name"]),
                        int(fr["width"]),
                        int(fr["height"]),
                        per_frame.get(f, []),
                    )
                )
        return out

    @staticmethod
    def _frames(api: _Api, tid: Any) -> list[dict[str, Any]]:
        meta = api.get(f"/api/tasks/{tid}/data/meta") or {}
        return list(meta.get("frames") or [])

    # -- write --
    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Replace the shapes/tags of the updated frames; other frames and the tracks stay as they are."""
        api = self._api(settings)
        by_task: dict[int, dict[int, list[Label]]] = {}
        for key, labels in updates.items():
            tid, frame = _split_key(key)
            by_task.setdefault(tid, {})[frame] = labels
        wanted = [lb.label for per in by_task.values() for labels in per.values() for lb in labels]
        ids = self._ensure(api, remote_id, wanted)
        for tid, per in by_task.items():
            frames = self._frames(api, tid)
            for frame in per:
                if not 0 <= frame < len(frames):
                    raise ToolError(f"CVAT task {tid} has no frame {frame}: synchronise again.")
            ann = api.get(f"/api/tasks/{tid}/annotations") or {}
            shapes = [s for s in ann.get("shapes") or [] if s.get("frame") not in per]
            tags = [t for t in ann.get("tags") or [] if t.get("frame") not in per]
            for frame, labels in per.items():
                new_shapes, new_tags = labels_to_cvat(
                    labels, ids, frame, int(frames[frame]["width"]), int(frames[frame]["height"])
                )
                shapes += new_shapes
                tags += new_tags
            body = {
                "version": ann.get("version", 0),
                "tags": [_without_ids(t) for t in tags],
                "shapes": [_without_ids(sh) for sh in shapes],
                "tracks": [_without_ids(tr) for tr in ann.get("tracks") or []],
            }
            api.request("PUT", f"/api/tasks/{tid}/annotations", json=body)

    # -- add --
    def add(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> dict[str, str]:
        """Upload the images in tasks of ``BATCH``, then write their labels; return ``{image_id: key}``."""
        api = self._api(settings)
        ids = self._ensure(api, remote_id, [lb.label for _, _, labels in images for lb in labels])
        keys: dict[str, str] = {}
        for n, start in enumerate(range(0, len(images), BATCH), 1):
            name = f"ActiveLabel {project.id} added {n} ({time.strftime('%Y-%m-%d')})"
            _, added = self._create_task(api, remote_id, name, None, images[start : start + BATCH], ids)
            keys.update(added)
        return keys

    def add_task(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        name: str,
        job_size: int | None,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> tuple[RemoteTask, dict[str, str]]:
        """One CVAT task with these images in this order, ``segment_size = job_size``, and their labels."""
        api = self._api(settings)
        ids = self._ensure(api, remote_id, [lb.label for _, _, labels in images for lb in labels])
        tid, keys = self._create_task(api, remote_id, name, job_size, images, ids)
        return self._task(api, api.get(f"/api/tasks/{tid}")), keys

    def _create_task(
        self,
        api: _Api,
        rid: str,
        name: str,
        job_size: int | None,
        batch: list[tuple[ImageRecord, Path, list[Label]]],
        ids: dict[str, int],
    ) -> tuple[str, dict[str, str]]:
        body: dict[str, Any] = {"name": name, "project_id": int(rid)}
        if job_size:
            body["segment_size"] = int(job_size)
        task = api.request("POST", "/api/tasks", json=body)
        tid = task["id"]
        names: dict[str, str] = {}  # image id -> file name in CVAT (unique inside the task)
        for rec, _, _ in batch:
            name = rec.name if rec.name not in names.values() else f"{rec.id}_{Path(rec.name).name}"
            names[rec.id] = name
        handles = []
        try:
            files = []
            for i, (rec, path, _) in enumerate(batch):
                fh = open(path, "rb")  # noqa: SIM115 - closed in the finally below
                handles.append(fh)
                mime = mimetypes.guess_type(names[rec.id])[0] or "application/octet-stream"
                files.append((f"client_files[{i}]", (names[rec.id], fh, mime)))
            answer = api.request(
                "POST",
                f"/api/tasks/{tid}/data",
                data={"image_quality": 95, "sorting_method": "predefined"},
                files=files,
                timeout=UPLOAD_TIMEOUT,
            )
        finally:
            for fh in handles:
                fh.close()
        self._wait_ready(api, tid, (answer or {}).get("rq_id") if isinstance(answer, dict) else None)
        frames = self._frames(api, tid)
        by_name = {str(fr["name"]): i for i, fr in enumerate(frames)}
        keys: dict[str, str] = {}
        shapes: list[dict[str, Any]] = []
        tags: list[dict[str, Any]] = []
        for position, (rec, _, labels) in enumerate(batch):
            frame = by_name.get(names[rec.id])
            if frame is None:
                raise ToolError(f"CVAT did not register the image '{rec.name}': upload again.")
            if job_size and frame != position:  # jobs only match when the order is kept
                raise ToolError(
                    f"CVAT changed the order of the images of task {tid}: its jobs would not match."
                )
            keys[rec.id] = f"{tid}:{frame}"
            s, t = labels_to_cvat(
                labels, ids, frame, int(frames[frame]["width"]), int(frames[frame]["height"])
            )
            shapes += s
            tags += t
        if shapes or tags:
            api.request(
                "PUT",
                f"/api/tasks/{tid}/annotations",
                json={"version": 0, "tags": tags, "shapes": shapes, "tracks": []},
            )
        return str(tid), keys

    @staticmethod
    def _wait_ready(api: _Api, tid: Any, rq_id: str | None) -> None:
        """Poll until CVAT has processed the uploaded data (``/api/requests`` or the task status)."""
        for _ in range(MAX_POLLS):
            done = failed = False
            message = ""
            data: dict[str, Any] | None = None
            if rq_id:
                try:
                    data = api.get(f"/api/requests/{rq_id}") or {}
                except ToolError as err:
                    if "not found" not in str(err):
                        raise
                    rq_id = None  # older CVAT: use the task status
                else:
                    state = str(data.get("status", "")).lower()
                    done, failed, message = (
                        state == "finished",
                        state == "failed",
                        str(data.get("message") or ""),
                    )
            if not rq_id:
                data = api.get(f"/api/tasks/{tid}/status") or {}
                state = str(data.get("state", "")).lower()
                done, failed, message = state == "finished", state == "failed", str(data.get("message") or "")
            if failed:
                raise ToolError(
                    f"CVAT could not process the uploaded images: {message[:200] or 'unknown error'}"
                )
            if done:
                return
            _sleep(POLL_INTERVAL)
        raise ToolError("CVAT took too long to process the uploaded images: try again later.")


def _without_ids(item: dict[str, Any]) -> dict[str, Any]:
    """Copy of a CVAT annotation without ids (a PUT replaces everything and rejects existing ids)."""
    out = {k: v for k, v in item.items() if k != "id"}
    if isinstance(out.get("shapes"), list):  # the shapes of a track
        out["shapes"] = [{k: v for k, v in sh.items() if k != "id"} for sh in out["shapes"]]
    if isinstance(out.get("elements"), list):  # skeleton parts
        out["elements"] = [_without_ids(el) for el in out["elements"]]
    return out
