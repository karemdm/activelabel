"""CVAT as a remote project of the synchronisation (docs/SYNC.md), over the plain REST API v2.

One ActiveLabel project is one CVAT *project*; its images live in CVAT tasks. The key of an image is
``"<task_id>:<frame>"``. Basic auth (url, username, password) like the other CVAT code; the password
never appears in logs or messages.

Conversions (relative 0..1 <-> pixels of the frame, from ``GET /api/tasks/{id}/data/meta``):

* rectangle -> bbox; polygon -> polygon ring + bbox; points -> one keypoints object (all the points
  of the shape; a point at the top-left corner is "not labeled", as in ``cvat_tool``) + bbox; tag ->
  image class (classification projects only).
* *Tracks* (interpolation mode) become one label per visible frame with ``Label.track`` = the CVAT
  track id (see :func:`track_frames` for how the frames between keyframes are filled, and
  :func:`_write_tracks` for how edits go back into the tracks).
* Other shape types (polyline, ellipse, cuboid, mask, skeleton) are not converted: they are neither
  read nor touched when writing (tracks of those types are kept as they are).
* Attributes are never sent (CVAT rejects attribute specs of another label).
"""

from __future__ import annotations

import copy
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


def _meta_frames(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per frame of a task (``/data/meta``).

    For a task made from a video CVAT lists a single entry (the video) while the task has ``size``
    frames: every frame then gets that entry's size and a name ``<video stem>_frame_<index>.jpg``
    (index in the video, from ``start_frame`` and the ``step=N`` frame filter).
    """
    frames = list(meta.get("frames") or [])
    size = int(meta.get("size") or len(frames))
    if len(frames) != 1 or size <= 1:
        return frames
    start = int(meta.get("start_frame") or 0)
    filt = str(meta.get("frame_filter") or "")
    step = int(filt.split("=", 1)[1]) if filt.startswith("step=") and filt.split("=", 1)[1].isdigit() else 1
    stem = Path(str(frames[0].get("name") or "video")).stem
    return [{**frames[0], "name": f"{stem}_frame_{start + i * step:06d}.jpg"} for i in range(size)]


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
        if status == 401:
            raise ToolError(f"CVAT login failed at {self.url}: check the username and password.")
        if status == 403:  # logged in, not allowed (plan limits, permissions): say what CVAT says
            reason = " ".join((getattr(r, "text", "") or "").split())[:300]
            raise ToolError("CVAT refused the request." + (f" CVAT says: {reason}" if reason else ""))
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


TRACKABLE = ("rectangle", "polygon", "points")  # shape types that tracks read/write
INTERPOLATED = ("rectangle", "points")  # types CVAT interpolates linearly between keyframes
_NOT_TRACK_FIELDS = ("label_id", "group", "source")  # of a plain shape, not kept in a track keyframe
_SAME = 1e-6  # relative tolerance to call two geometries equal when writing tracks


def _between(a: dict[str, Any], b: dict[str, Any] | None, t: float) -> list[float]:
    """Points at fraction ``t`` (0..1) from keyframe ``a`` to the next keyframe ``b``.

    Rectangles and points are interpolated linearly, like CVAT; every other type (polygons) keeps the
    points of ``a`` (approximation: CVAT also interpolates polygons, but only when both keyframes
    have matching points, which is not guaranteed here).
    """
    pa = [float(v) for v in a.get("points") or []]
    if b is None or b.get("outside") or a.get("type") not in INTERPOLATED or b.get("type") != a.get("type"):
        return pa
    pb = [float(v) for v in b.get("points") or []]
    if len(pa) != len(pb):
        return pa
    return [x + (y - x) * t for x, y in zip(pa, pb, strict=True)]


def track_frames(track: dict[str, Any], count: int) -> dict[int, tuple[dict[str, Any], bool]]:
    """Where a CVAT track is visible: ``{frame: (shape, is_keyframe)}`` for the frames ``0..count-1``.

    * a keyframe is the shape as it is; frames between two keyframes are interpolated (rectangles and
      points linearly; polygons use the previous keyframe's points);
    * a keyframe with ``outside`` hides the object from that frame until the next keyframe that is not
      outside, and a keyframe followed by an outside one holds its shape until then;
    * after the last keyframe the object continues with the last shape, unless it is outside.
    """
    by_frame = {int(s["frame"]): s for s in track.get("shapes") or [] if int(s.get("frame", -1)) >= 0}
    keys = [by_frame[f] for f in sorted(by_frame)]
    out: dict[int, tuple[dict[str, Any], bool]] = {}
    for i, a in enumerate(keys):
        b = keys[i + 1] if i + 1 < len(keys) else None
        first = int(a["frame"])
        end = int(b["frame"]) if b is not None else count
        if a.get("outside"):
            continue
        for f in range(first, min(end, count)):
            if f == first:
                out[f] = (a, True)
            else:
                points = _between(a, b, (f - first) / (end - first))
                out[f] = ({**a, "frame": f, "points": points, "outside": False}, False)
    return out


def _trackable(track: dict[str, Any], ids: dict[str, int]) -> str | None:
    """Shape type of a track this module can read and write (None: skeleton, other types, no id)."""
    if track.get("id") is None or track.get("elements") or track.get("label_id") not in ids.values():
        return None
    kinds = {s.get("type") for s in track.get("shapes") or []}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    return kind if kind in TRACKABLE else None


def _same_geometry(a: Label, b: Label) -> bool:
    def close(x: Any, y: Any) -> bool:
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(close(p, q) for p, q in zip(x, y, strict=True))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(float(x) - float(y)) <= _SAME
        return bool(x == y)

    return all(close(getattr(a, f), getattr(b, f)) for f in ("bbox", "polygons", "keypoints"))


def _write_tracks(
    tracks: list[dict[str, Any]],
    per: dict[int, list[Label]],
    ids: dict[str, int],
    frames: list[dict[str, Any]],
) -> dict[int, list[Label]]:
    """Write the tracked labels of the updated frames into ``tracks`` (in place).

    Returns the labels of every updated frame that are left to write as plain shapes (no track, a track
    that is gone or cannot take them: another class or geometry type).

    * A tracked label that differs from what the track shows on that frame becomes a keyframe there
      (replacing the keyframe of that frame or inserted between the others). Frames between keyframes
      are interpolated again by CVAT, so editing one frame moves its neighbours as in CVAT's editor.
    * A track visible on a frame but missing from the new labels of that frame gets an ``outside``
      keyframe there (and a keyframe on the frame before when that one was interpolated).
    * When such an edit changes whether the next frame shows the object, a keyframe at the next frame
      (outside, or the shape it had) keeps the rest of the track as it was.
    """
    count = len(frames)
    names = {v: k for k, v in ids.items()}
    info: dict[Any, tuple[dict[str, Any], str, dict[int, tuple[dict[str, Any], bool]]]] = {}
    for tr in tracks:
        kind = _trackable(tr, ids)
        if kind:
            info[tr["id"]] = (tr, kind, track_frames(tr, count))
    edits: dict[Any, dict[int, dict[str, Any] | None]] = {tid: {} for tid in info}
    plain: dict[int, list[Label]] = {}
    for frame, labels in per.items():
        w, h = int(frames[frame]["width"]), int(frames[frame]["height"])
        attached: set[Any] = set()
        rest: list[Label] = []
        for lb in labels:
            found = info.get(lb.track) if lb.track is not None and lb.track not in attached else None
            shapes = labels_to_cvat([lb], ids, frame, w, h)[0] if found else []
            if found is None or not shapes:
                rest.append(lb)
                continue
            tr, kind, shown = found
            if shapes[0]["type"] != kind or lb.label != names[tr["label_id"]]:
                rest.append(lb)
                continue
            attached.add(lb.track)
            current = shown.get(frame)
            old = shape_to_label(current[0], lb.label, w, h) if current else None
            if old is None or not _same_geometry(old, lb):
                edits[lb.track][frame] = {k: v for k, v in shapes[0].items() if k not in _NOT_TRACK_FIELDS}
        plain[frame] = rest
        for tid, (_, _, shown) in info.items():
            if tid not in attached and frame in shown:
                edits[tid][frame] = None
    for tid, edit in edits.items():
        if edit:
            _apply_edits(info[tid][0], info[tid][2], edit, count)
    return plain


def _apply_edits(
    track: dict[str, Any],
    old: dict[int, tuple[dict[str, Any], bool]],
    edit: dict[int, dict[str, Any] | None],
    count: int,
) -> None:
    keys = {int(s["frame"]): s for s in track.get("shapes") or [] if int(s.get("frame", -1)) >= 0}
    for f, shape in edit.items():
        keys[f] = shape if shape is not None else {**old[f][0], "frame": f, "outside": True}
    for f, shape in edit.items():  # the frame before keeps its shape instead of holding the one before it
        if shape is None and f - 1 in old and f - 1 not in keys:
            keys[f - 1] = {**old[f - 1][0], "frame": f - 1, "outside": False}
    new = track_frames({"shapes": list(keys.values())}, count)
    for f in sorted(edit):
        n = f + 1
        if n >= count or n in keys or (n in new) == (n in old):
            continue
        keys[n] = {**old[n][0], "frame": n, "outside": False} if n in old else {**new[n][0], "outside": True}
    track["shapes"] = [keys[f] for f in sorted(keys)]
    track["frame"] = min(keys)


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

    def frames(self, settings: ToolSettings, task_id: str) -> list[tuple[str, bool]]:
        """``(file name, deleted)`` of every frame of a task, in order."""
        meta = self._api(settings).get(f"/api/tasks/{task_id}/data/meta") or {}
        deleted = {int(d) for d in meta.get("deleted_frames") or []}
        return [(str(fr.get("name", "")), i in deleted) for i, fr in enumerate(_meta_frames(meta))]

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

        Tracks give one label per frame where the object is visible (see :func:`track_frames`).
        """
        api = self._api(settings)
        names = {i: n for n, i in self._labels(api, remote_id).items()}
        out: list[RemoteItem] = []
        for t in api.paged("/api/tasks", {"project_id": remote_id, "sort": "id"}, ALL):
            tid = t["id"]
            if task_ids is not None and str(tid) not in task_ids:
                continue
            meta = api.get(f"/api/tasks/{tid}/data/meta") or {}
            frames = _meta_frames(meta)
            ann = api.get(f"/api/tasks/{tid}/annotations") or {}
            per_frame: dict[int, list[Label]] = {}
            for s in ann.get("shapes") or []:
                f, name = int(s.get("frame", -1)), names.get(s.get("label_id"))
                if task is Task.CLASSIFICATION or name is None or not 0 <= f < len(frames):
                    continue
                label = shape_to_label(s, name, int(frames[f]["width"]), int(frames[f]["height"]))
                if label:
                    per_frame.setdefault(f, []).append(label)
            ids = {n: i for i, n in names.items()}
            for tr in ann.get("tracks") or []:
                name = names.get(tr.get("label_id"))
                if task is Task.CLASSIFICATION or name is None or _trackable(tr, ids) is None:
                    continue
                for f, (s, _) in sorted(track_frames(tr, len(frames)).items()):
                    label = shape_to_label(s, name, int(frames[f]["width"]), int(frames[f]["height"]))
                    if label:
                        label.track = int(tr["id"])
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
        return _meta_frames(meta)

    # -- write --
    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Replace the shapes/tags of the updated frames; other frames stay as they are.

        Labels with the id of a track of the task go into that track (see :func:`_write_tracks`);
        the other tracks are kept as they are.
        """
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
            tracks = copy.deepcopy(ann.get("tracks") or [])
            if project.task is not Task.CLASSIFICATION:
                per = _write_tracks(tracks, per, ids, frames)
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
                "tracks": [_without_ids(tr) for tr in tracks],
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
        video: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[RemoteTask, dict[str, str]]:
        """One CVAT task with these images in this order, ``segment_size = job_size``, and their labels.

        ``video``: ``(video file, {start, stop, step})`` for a task made from a video (docs/VIDEO.md):
        the original video is sent with the same selection, so frame *i* of the CVAT task is image *i*.
        """
        api = self._api(settings)
        ids = self._ensure(api, remote_id, [lb.label for _, _, labels in images for lb in labels])
        tid, keys = self._create_task(api, remote_id, name, job_size, images, ids, video)
        return self._task(api, api.get(f"/api/tasks/{tid}")), keys

    def _create_task(
        self,
        api: _Api,
        rid: str,
        name: str,
        job_size: int | None,
        batch: list[tuple[ImageRecord, Path, list[Label]]],
        ids: dict[str, int],
        video: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, str]]:
        body: dict[str, Any] = {"name": name, "project_id": int(rid)}
        if job_size:
            body["segment_size"] = int(job_size)
            body["overlap"] = 0  # CVAT repeats frames between jobs of a video by default
        task = api.request("POST", "/api/tasks", json=body)
        tid = task["id"]
        if video is not None:
            return self._fill_from_video(api, tid, video, job_size, batch, ids)
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

    def _fill_from_video(
        self,
        api: _Api,
        tid: Any,
        video: tuple[Path, dict[str, Any]],
        job_size: int | None,
        batch: list[tuple[ImageRecord, Path, list[Label]]],
        ids: dict[str, int],
    ) -> tuple[str, dict[str, str]]:
        """Send the original video with the task's selection; map frame *i* to image *i*."""
        path, sel = video
        data = {
            "image_quality": 95,
            "start_frame": int(sel.get("start") or 0),
            "stop_frame": int(sel["stop"]),
            "frame_filter": f"step={int(sel.get('step') or 1)}",
        }
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        with path.open("rb") as fh:
            answer = api.request(
                "POST",
                f"/api/tasks/{tid}/data",
                data=data,
                files=[("client_files[0]", (path.name, fh, mime))],
                timeout=UPLOAD_TIMEOUT,
            )
        self._wait_ready(api, tid, (answer or {}).get("rq_id") if isinstance(answer, dict) else None)
        frames = self._frames(api, tid)
        if len(frames) != len(batch):
            raise ToolError(
                f"CVAT made {len(frames)} frames from the video of task {tid}, the task has {len(batch)}: "
                "its jobs would not match."
            )
        keys: dict[str, str] = {}
        shapes: list[dict[str, Any]] = []
        tags: list[dict[str, Any]] = []
        for frame, (rec, _, labels) in enumerate(batch):
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
