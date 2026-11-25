"""CVAT as a remote project of the synchronisation (docs/SYNC.md), over the plain REST API v2.

One ActiveLabel project is one CVAT *project*; its images live in CVAT tasks. The key of an image is
``"<task_id>:<frame>"``. Basic auth (url, username, password) like the other CVAT code; the password
never appears in logs or messages.

Conversions (relative 0..1 <-> pixels of the frame, from ``GET /api/tasks/{id}/data/meta``):

* rectangle -> bbox (a rectangle with ``rotation`` -> the axis-aligned box that encloses it, see
  :func:`enclosing_box`); polygon -> polygon ring + bbox; points -> one keypoints object (all the points
  of the shape; a point at the top-left corner is "not labeled", as in ``cvat_tool``) + bbox; tag ->
  image class (classification projects only).
* Visibility of keypoints: CVAT has one ``occluded`` flag per points shape. A pose whose labeled points
  are all "labeled but not visible" (1) is an occluded shape; a pose that mixes 1 and 2 keeps the
  visibility of every point in the text attribute :data:`VISIBILITY` of its label (``"2,1,0"``), added
  to that CVAT label the first time it is needed (see :func:`visibility_of`).
* *Tracks* (interpolation mode) become one label per visible frame with ``Label.track`` = the CVAT
  track id (see :func:`track_frames` for how the frames between keyframes are filled, and
  :func:`_write_tracks` for how edits go back into the tracks).
* skeleton (keypoints projects) -> one keypoints object: one point per sublabel of the skeleton label,
  in the order of the sublabels (their ids); an element that is missing or ``outside`` is "not labeled"
  (0), an ``occluded`` one is 1 (see :func:`skeleton_parts`). A pose of a skeleton class is written back
  as a skeleton.
* Other shape types (polyline, ellipse, cuboid, mask) are not converted: they are neither read nor
  touched when writing (tracks of those types, and skeleton tracks, are kept as they are). Each item
  counts what it could not convert in :attr:`RemoteItem.not_converted`, by type (plus ``tracks`` and
  ``rotated_rectangle``, see :func:`_not_converted`), so the plan can say it.
* Attributes are never invented (CVAT rejects attribute specs of another label): a new shape has none.
* Class names are matched ignoring case, like the project's classes (:func:`class_id`): a shape of the
  CVAT label ``Cat`` is the class ``cat`` and stays on its own label when it is written back.

What :meth:`CvatRemote.write` touches, exactly. Only the frames it is given, and only through
differences (``PATCH .../annotations?action=delete|update|create``, never a ``PUT`` of the task): every
object it does not change keeps its id, and only objects it has just read can be deleted. On each frame:

* shapes of the converted types (rectangle, polygon, points; skeletons in keypoints projects) of a known
  class are compared with the new labels by type, class and geometry (:data:`SAME_SHAPE` of the image
  side; points and skeletons also by visibility). One that matches is kept
  as CVAT has it, with its attributes, occluded, rotation, group, z_order and source. A label without
  such a match takes the old shape of the same class and type it overlaps most (IoU above
  :data:`MATCH_IOU`): that shape is updated with the new points and keeps its id and everything else.
  An old shape left without a label is deleted; a label left without a shape is created as a fresh
  shape. In a classification project no shape is converted, so none is touched;
* shapes of any other type (or of a class the project does not know) are kept as they are;
* tags: in a classification project the tags of a known class are matched by class in the same way;
  in the other projects every tag is kept (a label without geometry only adds a tag that is missing);
* tracks are handled by :func:`_write_tracks` (a changed track is updated, keeping its id); frames
  that are not given are not touched.
"""

from __future__ import annotations

import copy
import logging
import math
import mimetypes
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import ImageRecord
from activelabel.images.video import VIDEO_EXTENSIONS
from activelabel.importing import common
from activelabel.importing.common import RemoteNotFound
from activelabel.importing.cvat_import import _Http
from activelabel.labels import MISSING_CORNER, Label, clamp01
from activelabel.projects import Project
from activelabel.sync import links
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteJob, RemoteTask
from activelabel.tasks import Task

log = logging.getLogger(__name__)

TIMEOUT = 60
UPLOAD_TIMEOUT = 600
BATCH = 100  # images per CVAT task created by ``add``
POLL_INTERVAL = 2.0  # seconds between checks of a task's data
MAX_POLLS = 300  # checks before giving up (about ten minutes)
ALL = 1_000_000  # "no limit" for paginated lists


def _meta_selection(meta: dict[str, Any]) -> tuple[int, int, int]:
    """``(start, stop, step)`` of a video task from its meta (``start_frame``, ``stop_frame``, ``step=N``).

    Args:
        meta: The JSON of ``GET /api/tasks/{id}/data/meta``.

    Returns:
        The first video frame, the last video frame and the step between selected frames (at least 1).
    """
    start = int(meta.get("start_frame") or 0)
    filt = str(meta.get("frame_filter") or "")
    step = int(filt.split("=", 1)[1]) if filt.startswith("step=") and filt.split("=", 1)[1].isdigit() else 1
    step = max(1, step)
    last = start + (int(meta.get("size") or 1) - 1) * step
    return start, max(int(meta.get("stop_frame") or 0), last), step


def _meta_frames(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per frame of a task (``/data/meta``).

    For a task made from a video CVAT lists a single entry (the video) while the task has ``size``
    frames: every frame then gets that entry's size and a name ``<video stem>_frame_<index>.jpg``
    (index in the video, from ``start_frame`` and the ``step=N`` frame filter).

    Args:
        meta: The JSON of ``GET /api/tasks/{id}/data/meta``.

    Returns:
        One dict per frame, in order, with at least ``name``, ``width`` and ``height``.
    """
    frames = list(meta.get("frames") or [])
    size = int(meta.get("size") or len(frames))
    if len(frames) != 1 or size <= 1:
        return frames
    start, _, step = _meta_selection(meta)
    stem = Path(str(frames[0].get("name") or "video")).stem
    return [{**frames[0], "name": f"{stem}_frame_{start + i * step:06d}.jpg"} for i in range(size)]


def task_pattern(name: str) -> str:
    """The name of a task with every run of digits replaced by ``#`` (``ds_round_3`` -> ``ds_round_#``).

    Args:
        name: The task name as CVAT has it.

    Returns:
        The name with each run of digits replaced by ``#``, or ``(no name)`` when nothing is left.
    """
    return re.sub(r"\d+", "#", name.strip()) or "(no name)"


def _sleep(seconds: float) -> None:
    """Wait ``seconds`` (a function of its own so tests can replace it).

    Args:
        seconds: How long to wait, in seconds.
    """
    time.sleep(seconds)


class _Api(_Http):
    """CVAT REST client: the read side of the importer plus writes, with readable errors."""

    def request(self, method: str, path: str, *, timeout: int = TIMEOUT, **kw: Any) -> Any:
        """Call the API and return the decoded JSON (None when the answer has no body).

        A task is only created inside a project (review in CVAT goes through the sync with the linked
        project): ``POST /api/tasks`` without ``project_id`` is refused before anything is sent.

        Args:
            method: The HTTP method (``GET``, ``POST``, ``PUT``, ``PATCH``, ``DELETE``).
            path: The path on the server, starting with ``/api/`` (it may carry a query string).
            timeout: Seconds to wait for the answer.
            **kw: Extra arguments of the HTTP call (``params``, ``json``, ``data``, ``files``).

        Returns:
            The decoded JSON of the answer, or None when it has no body or the body is not JSON.

        Raises:
            ToolError: Creating a task without a project, no connection, a 401, 403, 429 or other
                error status of CVAT.
            RemoteNotFound: CVAT answered 404.
        """
        if method.upper() == "POST" and path.split("?", 1)[0].rstrip("/") == "/api/tasks":
            if not (kw.get("json") or {}).get("project_id"):
                raise ToolError(
                    "Refusing to create a CVAT task outside a project: the task needs a project_id."
                )
        try:
            r = self._http.request(method, f"{self.url}{path}", auth=self._auth, timeout=timeout, **kw)
        except Exception:  # noqa: BLE001 - network/timeouts: never echo the exception text
            raise common.unreachable(self.url) from None
        status = r.status_code
        where = path.split("?", 1)[0]
        if status == 401:
            raise common.login_failed(self.url)
        if status == 403:  # logged in, not allowed (plan limits, permissions)
            raise common.refused(_forbidden_reason(r))
        if status == 404:
            raise common.not_found(where)
        if status == 429:
            raise common.rate_limited()
        if status >= 400:
            # CVAT explains validation problems in the body (never contains the login)
            raise common.http_error(status, where, getattr(r, "text", "") or "")
        if not getattr(r, "content", b"x"):
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON document.

        Args:
            path: The path on the server, starting with ``/api/``.
            params: Query parameters, if any.

        Returns:
            The decoded JSON of the answer (None when it has no body).
        """
        return self.request("GET", path, params=params)


def _forbidden_reason(r: Any) -> str:
    """What CVAT says in a 403: the ``detail`` / ``message`` of a JSON body as one sentence, else the text.

    Args:
        r: The HTTP response (anything with ``json()`` and ``text``).

    Returns:
        At most 300 characters on one line, ending in punctuation; empty when CVAT said nothing.
    """
    try:
        body = r.json()
    except Exception:  # noqa: BLE001 - not JSON (or a fake without a body)
        body = None
    found: Any = None
    if isinstance(body, dict):
        found = body.get("detail") or body.get("message")
    elif isinstance(body, list) and body:
        found = body
    if isinstance(found, (list, tuple)):
        found = " ".join(str(x) for x in found)
    if found:
        text = " ".join(str(found).split())[:300]
    else:
        text = " ".join((getattr(r, "text", "") or "").split())[:300]
    return text if not text or text.endswith((".", "!", "?")) else text + "."


# -- conversions ---------------------------------------------------------------------------------
def _box(xs: list[float], ys: list[float]) -> list[float]:
    """The bounding box of a set of points.

    Args:
        xs: The x coordinates (relative 0..1).
        ys: The y coordinates (relative 0..1), same length as ``xs``.

    Returns:
        ``[x, y, width, height]``, relative 0..1.
    """
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def enclosing_box(points: list[float], rotation: float) -> list[float]:
    """The axis-aligned box ``[x0, y0, x1, y1]`` around a rectangle turned by ``rotation`` about its centre.

    CVAT keeps the corners of the rectangle before the turn (``xtl, ytl, xbr, ybr``) and the angle apart,
    in degrees. Detection works on axis-aligned boxes, so a turned rectangle becomes the smallest box that
    holds its four corners (the same for either direction of the turn).

    Args:
        points: ``[xtl, ytl, xbr, ybr]`` of the rectangle before the turn, in pixels.
        rotation: The angle in degrees.

    Returns:
        The enclosing box in pixels; ``points`` itself when there is no turn.
    """
    x0, y0, x1, y1 = (float(v) for v in points[:4])
    turn = math.radians(float(rotation or 0) % 360)
    if not turn:
        return [x0, y0, x1, y1]
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    ex = hw * abs(math.cos(turn)) + hh * abs(math.sin(turn))
    ey = hw * abs(math.sin(turn)) + hh * abs(math.cos(turn))
    return [cx - ex, cy - ey, cx + ex, cy + ey]


def _seen_points(shape: dict[str, Any]) -> list[float]:
    """The pixel points of a shape as the project sees them.

    A turned rectangle gives its enclosing box; a skeleton its labeled points.

    Args:
        shape: A CVAT shape.

    Returns:
        Flat ``[x0, y0, x1, y1, ...]`` in pixels.
    """
    if shape.get("type") == "skeleton":
        return [c for x, y, v in _elements(shape).values() if v > 0 for c in (x, y)]
    pts = [float(p) for p in shape.get("points") or []]
    if shape.get("type") == "rectangle" and len(pts) >= 4 and shape.get("rotation"):
        return enclosing_box(pts, float(shape["rotation"]))
    return pts


VISIBILITY = "visibility"  # text attribute of a CVAT label: the visibility of each point of a pose


def _visibility_spec() -> dict[str, Any]:
    """The attribute spec :data:`VISIBILITY` added to a CVAT label that needs it (a mutable text).

    Returns:
        The attribute as CVAT takes it in the ``attributes`` of a label.
    """
    return {"name": VISIBILITY, "mutable": True, "input_type": "text", "default_value": "", "values": [""]}


def visibility_specs(rows: list[dict[str, Any]]) -> dict[int, int]:
    """The id of the :data:`VISIBILITY` attribute of each CVAT label that has one.

    Args:
        rows: Labels as ``GET /api/labels`` lists them (with their ``attributes``).

    Returns:
        ``{label id: attribute spec id}``.
    """
    out: dict[int, int] = {}
    for row in rows:
        for attr in row.get("attributes") or []:
            if attr.get("name") == VISIBILITY and attr.get("id") is not None:
                out[int(row["id"])] = int(attr["id"])
    return out


def _mixed(label: Label) -> bool:
    """Whether a pose has labeled points of both visibilities (1 and 2): only an attribute keeps that.

    Args:
        label: A label.

    Returns:
        True when its labeled points are not all of the same visibility.
    """
    seen = {kp[2] if len(kp) > 2 else 2.0 for kp in label.keypoints or [] if len(kp) <= 2 or kp[2] > 0}
    return len(seen) > 1


def visibility_of(shape: dict[str, Any], count: int, spec: int | None) -> list[float]:
    """Visibility of each point of a CVAT points shape (before "not labeled" corners are applied).

    ``occluded`` makes every point 1. Otherwise the :data:`VISIBILITY` attribute gives one value per
    point (any other text, or a value for another number of points, is ignored); without it every
    point is 2.

    Args:
        shape: A CVAT points shape (or track keyframe).
        count: Number of points of the shape.
        spec: Id of the :data:`VISIBILITY` attribute of the shape's label, or None when it has none.

    Returns:
        ``count`` values, each 1 or 2.
    """
    if shape.get("occluded"):
        return [1.0] * count
    text = next(
        (str(a.get("value", "")) for a in shape.get("attributes") or [] if a.get("spec_id") == spec), ""
    )
    parts = [x.strip() for x in text.split(",")] if spec is not None and text else []
    if len(parts) == count and all(x in ("0", "1", "2") for x in parts):
        return [1.0 if x == "1" else 2.0 for x in parts]
    return [2.0] * count


def _visibility_attrs(label: Label, spec: int | None) -> tuple[bool, list[dict[str, Any]]]:
    """``(occluded, attributes)`` of the CVAT points shape of a pose label.

    Args:
        label: A pose label.
        spec: Id of the :data:`VISIBILITY` attribute of its CVAT label, or None when it has none.

    Returns:
        Occluded when every labeled point is 1; the :data:`VISIBILITY` value when they are mixed (and the
        label has the attribute); else not occluded and no attribute.
    """
    kps = label.keypoints or []
    labeled = [kp[2] for kp in kps if len(kp) > 2 and kp[2] > 0]
    if labeled and all(v == 1 for v in labeled):
        return True, []
    if _mixed(label) and spec is not None:
        value = ",".join(str(int(kp[2])) if len(kp) > 2 else "2" for kp in kps)
        return False, [{"spec_id": spec, "value": value}]
    return False, []


def skeleton_parts(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    """The sublabels of each skeleton label, in order (their ids, ascending: the order they were made in).

    Args:
        rows: Labels as ``GET /api/labels`` lists them (sublabels have a ``parent_id``).

    Returns:
        ``{skeleton label id: [sublabel id, ...]}``.
    """
    skeletons = {int(r["id"]) for r in rows if r.get("type") == "skeleton" and not r.get("parent_id")}
    out: dict[int, list[int]] = {sid: [] for sid in skeletons}
    for r in sorted(rows, key=lambda r: int(r["id"])):
        if r.get("parent_id") in skeletons:
            out[int(r["parent_id"])].append(int(r["id"]))
    return out


def _elements(shape: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
    """The points of a skeleton shape by sublabel: ``(x, y, visibility)`` in pixels.

    Args:
        shape: A CVAT skeleton shape (its ``elements`` are points shapes of one point each).

    Returns:
        ``{sublabel id: (x, y, v)}`` with v 0 (outside or no point), 1 (occluded) or 2.
    """
    out: dict[int, tuple[float, float, float]] = {}
    for el in shape.get("elements") or []:
        pts = [float(v) for v in el.get("points") or []]
        if el.get("label_id") is None:
            continue
        if el.get("outside") or len(pts) < 2:
            out[int(el["label_id"])] = (0.0, 0.0, 0.0)
        else:
            out[int(el["label_id"])] = (pts[0], pts[1], 1.0 if el.get("occluded") else 2.0)
    return out


ELEMENT_FIELDS = (  # what a part of a skeleton shape has (no rotation, no parts of its own)
    "id",
    "type",
    "frame",
    "label_id",
    "points",
    "occluded",
    "outside",
    "z_order",
    "source",
    "attributes",
)


def _skeleton_payload(
    frame: int, label_id: int, parts: list[int], points: list[list[float]]
) -> dict[str, Any]:
    """A new CVAT skeleton shape: one element per sublabel (a missing point is an ``outside`` element).

    Args:
        frame: Index of the frame in the task.
        label_id: The CVAT id of the skeleton label.
        parts: The sublabel ids, in order.
        points: ``[x, y, v]`` per point, in pixels (v 0: not labeled, 1: occluded, 2: visible).

    Returns:
        The shape as the annotations endpoint takes it.
    """
    elements = []
    for n, sub in enumerate(parts):
        x, y, v = (points[n] + [0.0, 0.0, 0.0])[:3] if n < len(points) else (0.0, 0.0, 0.0)
        part = {**_shape_payload("points", frame, sub, [x, y]), "occluded": v == 1, "outside": v <= 0}
        elements.append({k: part[k] for k in ELEMENT_FIELDS if k in part})
    return {**_shape_payload("skeleton", frame, label_id, []), "elements": elements}


def shape_to_label(
    shape: dict[str, Any],
    name: str,
    w: int,
    h: int,
    spec: int | None = None,
    parts: list[int] | None = None,
) -> Label | None:
    """One CVAT shape -> internal label (None for unsupported types).

    Args:
        shape: A CVAT shape (``type``, ``points`` in pixels, ``occluded``, ``attributes``).
        name: The class name to give the label.
        w: Width of the frame in pixels.
        h: Height of the frame in pixels.
        spec: Id of the :data:`VISIBILITY` attribute of the shape's label (points), or None.
        parts: The sublabel ids of the shape's label when it is a skeleton (:func:`skeleton_parts`).

    Returns:
        The label with relative 0..1 geometry, or None for a type that is not converted or too few points.
    """
    if shape.get("type") == "skeleton":
        return _skeleton_to_label(shape, name, w, h, parts or [])
    kind, pts = shape.get("type"), _seen_points(shape)
    w, h = max(w, 1), max(h, 1)
    xs = [clamp01(x / w) for x in pts[0::2]]
    ys = [clamp01(y / h) for y in pts[1::2]]
    if kind == "rectangle" and len(pts) >= 4:
        x0, x1 = sorted(xs[:2])
        y0, y1 = sorted(ys[:2])
        return Label(name, bbox=[x0, y0, x1 - x0, y1 - y0])
    if kind == "polygon" and len(pts) >= 6:
        return Label(name, bbox=_box(xs, ys), polygons=[[[x, y] for x, y in zip(xs, ys, strict=True)]])
    if kind == "points" and len(pts) >= 2:
        vis = visibility_of(shape, len(xs), spec)
        kps = [
            [0.0, 0.0, 0.0] if (px, py) == MISSING_CORNER else [x, y, v]
            for px, py, x, y, v in zip(pts[0::2], pts[1::2], xs, ys, vis, strict=True)
        ]
        seen = [k for k in kps if k[2]]
        box = _box([k[0] for k in seen], [k[1] for k in seen]) if seen else [0.0, 0.0, 0.0, 0.0]
        return Label(name, bbox=box, keypoints=kps)
    return None


def _skeleton_to_label(shape: dict[str, Any], name: str, w: int, h: int, parts: list[int]) -> Label | None:
    """A CVAT skeleton -> one pose label, its points in the order of ``parts``.

    Args:
        shape: A CVAT skeleton shape.
        name: The class name to give the label.
        w: Width of the frame in pixels.
        h: Height of the frame in pixels.
        parts: The sublabel ids of the skeleton label, in order.

    Returns:
        The pose (a box around its labeled points), or None when the skeleton has no part.
    """
    found = _elements(shape)
    if not parts or not found:
        return None
    w, h = max(w, 1), max(h, 1)
    kps: list[list[float]] = []
    for sub in parts:
        x, y, v = found.get(sub, (0.0, 0.0, 0.0))
        kps.append([clamp01(x / w), clamp01(y / h), v] if v > 0 else [0.0, 0.0, 0.0])
    seen = [k for k in kps if k[2]]
    box = _box([k[0] for k in seen], [k[1] for k in seen]) if seen else [0.0, 0.0, 0.0, 0.0]
    return Label(name, bbox=box, keypoints=kps)


def _shape_payload(kind: str, frame: int, label_id: int, points: list[float]) -> dict[str, Any]:
    """A new CVAT shape with the default fields (no attributes, not occluded, manual source).

    Args:
        kind: The CVAT shape type (``rectangle``, ``polygon`` or ``points``).
        frame: Index of the frame in the task.
        label_id: The CVAT id of the label (class).
        points: Flat ``[x0, y0, x1, y1, ...]`` in pixels.

    Returns:
        The shape as the annotations endpoint takes it.
    """
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


def class_id(ids: dict[str, int], name: str) -> int | None:
    """The CVAT label of a class: the one with exactly that name, else the first that differs only in case.

    Args:
        ids: CVAT label id of each label name of the project.
        name: The class name.

    Returns:
        The label id, or None when the project has no such label.
    """
    if name in ids:
        return ids[name]
    folded = name.casefold()
    return next((i for n, i in ids.items() if n.casefold() == folded), None)


def class_aliases(ids: dict[str, int]) -> dict[int, int]:
    """Every CVAT label id -> the id of the label of its class (labels differing only in case are one class).

    Args:
        ids: CVAT label id of each label name of the project.

    Returns:
        ``{label id: id of the first label with the same name ignoring case}``.
    """
    first: dict[str, int] = {}
    for name, lid in ids.items():
        first.setdefault(name.casefold(), lid)
    return {lid: first[name.casefold()] for name, lid in ids.items()}


def _same_class(a: Any, b: Any, aliases: dict[int, int] | None) -> bool:
    """Whether two CVAT label ids are the same class (see :func:`class_aliases`).

    Args:
        a: A label id.
        b: Another label id.
        aliases: The class of each label id, or None to compare the ids themselves.

    Returns:
        True when both are the same class.
    """
    if aliases is None:
        return bool(a == b)
    return aliases.get(a, a) == aliases.get(b, b)


def labels_to_cvat(
    labels: list[Label],
    ids: dict[str, int],
    frame: int,
    w: int,
    h: int,
    specs: dict[int, int] | None = None,
    skeletons: dict[int, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Internal labels -> ``(shapes, tags)`` of one frame, in pixels (unknown classes are skipped).

    Args:
        labels: The labels of the frame (relative 0..1 geometry).
        ids: CVAT label id of each class name.
        frame: Index of the frame in the task.
        w: Width of the frame in pixels.
        h: Height of the frame in pixels.
        specs: Id of the :data:`VISIBILITY` attribute of each CVAT label that has one (poses that mix
            visibilities 1 and 2 need it, see :func:`_visibility_attrs`).
        skeletons: The sublabels of each skeleton label (:func:`skeleton_parts`): a pose of such a class
            is written as a skeleton.

    Returns:
        The CVAT shapes and the CVAT tags (labels without geometry) of the frame.
    """
    shapes: list[dict[str, Any]] = []
    tags: list[dict[str, Any]] = []

    for lb in labels:
        lid = class_id(ids, lb.label)
        if lid is None:
            continue
        if lb.polygons:
            for ring in lb.polygons:
                if len(ring) >= 3:
                    shapes.append(
                        _shape_payload("polygon", frame, lid, [v for p in ring for v in (p[0] * w, p[1] * h)])
                    )
        elif lb.keypoints and lid in (skeletons or {}):
            pixels = [[kp[0] * w, kp[1] * h, kp[2] if len(kp) > 2 else 2.0] for kp in lb.keypoints]
            shapes.append(_skeleton_payload(frame, lid, (skeletons or {})[lid], pixels))
        elif lb.keypoints:
            flat: list[float] = []
            for kp in lb.keypoints:
                missing = len(kp) > 2 and kp[2] <= 0
                flat += list(MISSING_CORNER) if missing else [kp[0] * w, kp[1] * h]
            occluded, attributes = _visibility_attrs(lb, (specs or {}).get(lid))
            shapes.append(
                {**_shape_payload("points", frame, lid, flat), "occluded": occluded, "attributes": attributes}
            )
        elif lb.bbox:
            x, y, bw, bh = lb.bbox
            shapes.append(_shape_payload("rectangle", frame, lid, [x * w, y * h, (x + bw) * w, (y + bh) * h]))
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
    have matching points, which is not guaranteed here). The ``rotation`` of rectangles is interpolated
    the short way round, see :func:`_turn_between`.

    Args:
        a: The keyframe the interval starts at (CVAT shape, points in pixels).
        b: The next keyframe, or None when there is none.
        t: How far between ``a`` and ``b``, from 0 (at ``a``) to 1 (at ``b``).

    Returns:
        The flat list of points, in pixels.
    """
    pa = [float(v) for v in a.get("points") or []]
    if b is None or b.get("outside") or a.get("type") not in INTERPOLATED or b.get("type") != a.get("type"):
        return pa
    pb = [float(v) for v in b.get("points") or []]
    if len(pa) != len(pb):
        return pa
    return [x + (y - x) * t for x, y in zip(pa, pb, strict=True)]


def _turn_between(a: dict[str, Any], b: dict[str, Any] | None, t: float) -> float:
    """``rotation`` at fraction ``t`` from keyframe ``a`` to keyframe ``b``, the short way round.

    Args:
        a: The keyframe the interval starts at.
        b: The next keyframe, or None when there is none.
        t: How far between ``a`` and ``b``, from 0 to 1.

    Returns:
        The angle in degrees, in ``[0, 360)``.
    """
    ra = float(a.get("rotation") or 0)
    if b is None or b.get("outside") or a.get("type") != "rectangle" or b.get("type") != "rectangle":
        return ra
    step = (float(b.get("rotation") or 0) - ra + 180) % 360 - 180
    return (ra + step * t) % 360


def track_frames(track: dict[str, Any], count: int) -> dict[int, tuple[dict[str, Any], bool]]:
    """Where a CVAT track is visible: ``{frame: (shape, is_keyframe)}`` for the frames ``0..count-1``.

    * a keyframe is the shape as it is; frames between two keyframes are interpolated (rectangles and
      points linearly; polygons use the previous keyframe's points);
    * a keyframe with ``outside`` hides the object from that frame until the next keyframe that is not
      outside, and a keyframe followed by an outside one holds its shape until then;
    * after the last keyframe the object continues with the last shape, unless it is outside.

    Args:
        track: A CVAT track (``shapes`` are its keyframes).
        count: Number of frames of the task.

    Returns:
        For each frame where the object is visible, its shape (points in pixels) and whether the
        frame is a keyframe.
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
                t = (f - first) / (end - first)
                shape = {**a, "frame": f, "points": _between(a, b, t), "outside": False}
                if a.get("rotation"):
                    shape["rotation"] = _turn_between(a, b, t)
                out[f] = (shape, False)
    return out


def _trackable(track: dict[str, Any], ids: dict[str, int]) -> str | None:
    """Shape type of a track this module can read and write (None: skeleton, other types, no id).

    Args:
        track: A CVAT track.
        ids: CVAT label id of each class name (the classes the project knows).

    Returns:
        ``rectangle``, ``polygon`` or ``points``; None when the track cannot be handled.
    """
    if track.get("id") is None or track.get("elements") or track.get("label_id") not in ids.values():
        return None
    kinds = {s.get("type") for s in track.get("shapes") or []}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    return kind if kind in TRACKABLE else None


def _close(x: Any, y: Any) -> bool:
    """Whether two values (numbers or nested lists of numbers) are equal within ``_SAME``.

    Args:
        x: A number, a list or any other value.
        y: The value to compare with.

    Returns:
        True when equal (numbers closer than ``_SAME``, lists element by element).
    """
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_close(p, q) for p, q in zip(x, y, strict=True))
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= _SAME
    return bool(x == y)


def _same_geometry(a: Label, b: Label) -> bool:
    """Whether two labels have the same box, polygons and keypoints (within ``_SAME``).

    Args:
        a: The first label.
        b: The second label.

    Returns:
        True when the geometry is the same.
    """
    return all(_close(getattr(a, f), getattr(b, f)) for f in ("bbox", "polygons", "keypoints"))


def _write_tracks(
    tracks: list[dict[str, Any]],
    per: dict[int, list[Label]],
    ids: dict[str, int],
    frames: list[dict[str, Any]],
    specs: dict[int, int] | None = None,
) -> dict[int, list[Label]]:
    """Write the tracked labels of the updated frames into ``tracks`` (in place).

    Args:
        tracks: The CVAT tracks of the task; changed in place.
        per: The new labels of each updated frame, by frame index.
        ids: CVAT label id of each class name.
        frames: The frames of the task (``width`` and ``height`` in pixels), see :func:`_meta_frames`.
        specs: Id of the :data:`VISIBILITY` attribute of each CVAT label that has one.

    Returns:
        The labels of every updated frame that are left to write as plain shapes (no track, a track
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
            shapes = labels_to_cvat([lb], ids, frame, w, h, specs)[0] if found else []
            if found is None or not shapes:
                rest.append(lb)
                continue
            tr, kind, shown = found
            if shapes[0]["type"] != kind or lb.label.casefold() != names[tr["label_id"]].casefold():
                rest.append(lb)
                continue
            attached.add(lb.track)
            current = shown.get(frame)
            spec = (specs or {}).get(tr["label_id"])
            old = shape_to_label(current[0], lb.label, w, h, spec) if current else None
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
    """Apply keyframe edits to a track (in place) and keep the rest of the track as it was.

    Args:
        track: The CVAT track; its ``shapes`` and ``frame`` are replaced.
        old: What the track showed before, from :func:`track_frames`.
        edit: New keyframe shape (points in pixels) per edited frame, or None to hide the object there.
        count: Number of frames of the task.
    """
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


SAME_SHAPE = links.TOLERANCE  # relative to the image side: shapes closer than this are the same shape
MATCH_IOU = 0.5  # an edited shape is the old one of the same class and type that overlaps it more than this
ACTIONS = ("delete", "update", "create")  # order of the PATCH requests of a write
KINDS = ("tags", "shapes", "tracks")


def _same_shape(
    old: dict[str, Any],
    new: dict[str, Any],
    w: int,
    h: int,
    spec: int | None = None,
    aliases: dict[int, int] | None = None,
) -> bool:
    """Whether a shape CVAT has and a shape built from a label are the same (type, class, geometry).

    Args:
        old: A shape CVAT has (points in pixels).
        new: A shape built from a label (points in pixels).
        w: Width of the frame in pixels.
        h: Height of the frame in pixels.
        spec: Id of the :data:`VISIBILITY` attribute of the label, or None.
        aliases: The class of each label id (:func:`class_aliases`); None compares the label ids.

    Returns:
        True when type and class match and every coordinate differs by at most ``SAME_SHAPE`` of the side
        (a turned rectangle is compared by its enclosing box, which is what the project has of it); points
        must also have the same visibility (:func:`visibility_of`), skeletons the same parts.
    """
    if old.get("type") != new.get("type") or not _same_class(
        old.get("label_id"), new.get("label_id"), aliases
    ):
        return False
    sides = (max(w, 1), max(h, 1))
    if old.get("type") == "skeleton":
        ea, eb = _elements(old), _elements(new)
        return set(ea) == set(eb) and all(
            ea[k][2] == eb[k][2]
            and (
                ea[k][2] == 0
                or max(abs(ea[k][0] - eb[k][0]) / sides[0], abs(ea[k][1] - eb[k][1]) / sides[1]) <= SAME_SHAPE
            )
            for k in ea
        )
    a, b = _seen_points(old), [float(v) for v in new.get("points") or []]
    if len(a) != len(b):
        return False
    if not all(abs(p - q) / sides[i % 2] <= SAME_SHAPE for i, (p, q) in enumerate(zip(a, b, strict=True))):
        return False
    count = len(a) // 2
    return old.get("type") != "points" or visibility_of(old, count, spec) == visibility_of(new, count, spec)


def _extent(points: list[float]) -> tuple[float, float, float, float]:
    """The box ``(x0, y0, x1, y1)`` around flat pixel points, one pixel larger on every side.

    The margin gives a single point or a flat line an area, so it can still overlap its edited self.

    Args:
        points: Flat ``[x0, y0, x1, y1, ...]`` in pixels (at least one point).

    Returns:
        The left, top, right and bottom of the box, in pixels.
    """
    xs, ys = points[0::2], points[1::2]
    return min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1


def _overlap(old: dict[str, Any], new: dict[str, Any], aliases: dict[int, int] | None = None) -> float:
    """Intersection over union of the boxes around two shapes of the same type and class (0 otherwise).

    A turned rectangle counts with its enclosing box.

    Args:
        old: A shape CVAT has (points in pixels).
        new: A shape built from a label (points in pixels).
        aliases: The class of each label id (:func:`class_aliases`); None compares the label ids.

    Returns:
        The IoU, from 0 to 1.
    """
    if old.get("type") != new.get("type") or not _same_class(
        old.get("label_id"), new.get("label_id"), aliases
    ):
        return 0.0
    a, b = _seen_points(old), _seen_points(new)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ax0, ay0, ax1, ay1 = _extent(a)
    bx0, by0, bx1, by1 = _extent(b)
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def _edited(
    old: dict[str, Any], new: dict[str, Any], w: int, h: int, spec: int | None = None
) -> dict[str, Any]:
    """The old shape with the geometry of the new one: id, attributes, occluded, z_order... are kept.

    A skeleton takes the points of the new one part by part (:func:`_edited_skeleton`). Points also take
    the visibility of the new shape (``occluded`` and the :data:`VISIBILITY` attribute;
    the other attributes are kept). A turned rectangle whose enclosing box was only moved (same size
    within :data:`SAME_SHAPE`) is moved and keeps its turn. When the box was resized there is no honest
    way to turn it back: the rectangle becomes the new axis-aligned box (``rotation`` 0), which is what
    the person drew.

    Args:
        old: A shape CVAT has.
        new: The shape built from the edited label.
        w: Width of the frame in pixels.
        h: Height of the frame in pixels.
        spec: Id of the :data:`VISIBILITY` attribute of the label, or None.

    Returns:
        A copy of ``old`` with the geometry of ``new``.
    """
    if old.get("type") == "skeleton":
        return _edited_skeleton(old, new)
    if old.get("type") == "points":
        others = [a for a in old.get("attributes") or [] if spec is None or a.get("spec_id") != spec]
        mine = [a for a in new.get("attributes") or [] if spec is not None and a.get("spec_id") == spec]
        return {
            **old,
            "points": new["points"],
            "occluded": bool(new.get("occluded")),
            "attributes": others + mine,
        }
    if old.get("type") != "rectangle" or not old.get("rotation"):
        return {**old, "points": new["points"]}
    seen, box = _seen_points(old), [float(v) for v in new["points"]]
    same_w = abs((seen[2] - seen[0]) - (box[2] - box[0])) / max(w, 1) <= SAME_SHAPE
    same_h = abs((seen[3] - seen[1]) - (box[3] - box[1])) / max(h, 1) <= SAME_SHAPE
    if not (same_w and same_h):
        return {**old, "points": box, "rotation": 0}
    dx, dy = box[0] - seen[0], box[1] - seen[1]
    pts = [float(v) for v in old["points"]]
    return {**old, "points": [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(pts)]}


def _edited_skeleton(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """The old skeleton with the points of the new one: each element keeps its id and attributes.

    Args:
        old: A skeleton CVAT has.
        new: The skeleton built from the edited label.

    Returns:
        A copy of ``old`` whose elements have the points, ``occluded`` and ``outside`` of ``new`` (and only
        the fields of a skeleton element, :data:`ELEMENT_FIELDS`).
    """
    fresh = {el.get("label_id"): el for el in new.get("elements") or []}
    elements = []
    for el in old.get("elements") or []:
        src = fresh.pop(el.get("label_id"), None)
        changed = {**el, **{k: src[k] for k in ("points", "occluded", "outside")}} if src is not None else el
        elements.append({k: v for k, v in changed.items() if k in ELEMENT_FIELDS})
    return {**old, "elements": elements + list(fresh.values())}


def _visible(task: Task, label: Label) -> bool:
    """Whether a label read from CVAT is one the project keeps (the same rule as annotate.diff).

    Args:
        task: The task type of the project.
        label: A label converted from a CVAT shape.

    Returns:
        True when it has the geometry the task type uses (box, polygons or keypoints).
    """
    if task is Task.DETECTION:
        return bool(label.bbox)
    if task is Task.SEGMENTATION:
        return bool(label.polygons)
    if task is Task.KEYPOINTS:
        return bool(label.keypoints)
    return False


def _merge_frame(
    old_shapes: list[dict[str, Any]],
    old_tags: list[dict[str, Any]],
    labels: list[Label],
    ids: dict[str, int],
    frame: int,
    size: tuple[int, int],
    task: Task,
    specs: dict[int, int] | None = None,
    skeletons: dict[int, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shapes and tags of one frame after writing ``labels``: what CVAT has, changed only where needed.

    See the module docstring for what is converted. Unchanged shapes and tags are the old objects (the
    same Python objects). A label is matched first with a shape of the same geometry, else with the
    old shape of the same class and type it overlaps most (IoU above :data:`MATCH_IOU`): that shape is
    returned as an edited copy that keeps its id and everything else CVAT has for it. Only shapes the
    project can see (``task``: boxes for detection, polygons for segmentation, points for keypoints)
    are matched or removed; the others stay in CVAT as they are.

    Args:
        old_shapes: The shapes CVAT has on the frame.
        old_tags: The tags CVAT has on the frame.
        labels: The new labels of the frame.
        ids: CVAT label id of each class name.
        frame: Index of the frame in the task.
        size: ``(width, height)`` of the frame in pixels.
        task: The task type of the project.
        specs: Id of the :data:`VISIBILITY` attribute of each CVAT label that has one.
        skeletons: The sublabels of each skeleton label (:func:`skeleton_parts`); skeletons of those labels
            are converted (keypoints projects).

    Returns:
        The shapes and the tags the frame has after the write.
    """
    classification = task is Task.CLASSIFICATION
    w, h = size
    parts = skeletons or {}
    new_shapes, new_tags = labels_to_cvat(labels, ids, frame, w, h, specs, parts)
    vis = specs or {}
    aliases = class_aliases(ids)
    known = set(ids.values())
    kept_shapes: list[dict[str, Any]] = []
    convertible: list[dict[str, Any]] = []
    for s in old_shapes:
        lid = int(s.get("label_id") or -1)  # -1: no label (never a CVAT id)
        kind_ok = s.get("type") in TRACKABLE or (s.get("type") == "skeleton" and lid in parts)
        mine = not classification and kind_ok and lid in known
        converted = shape_to_label(s, "x", w, h, vis.get(lid), parts.get(lid)) if mine else None
        mine = converted is not None and _visible(task, converted)
        (convertible if mine else kept_shapes).append(s)
    unmatched: list[dict[str, Any]] = []
    for shape in new_shapes:
        spec = vis.get(shape["label_id"])
        match = next(
            (i for i, o in enumerate(convertible) if _same_shape(o, shape, w, h, spec, aliases)), None
        )
        if match is None:
            unmatched.append(shape)
        else:
            kept_shapes.append(convertible.pop(match))
    for shape in unmatched:
        scores = [_overlap(o, shape, aliases) for o in convertible]
        best = max(range(len(scores)), key=scores.__getitem__, default=None)
        if best is not None and scores[best] > MATCH_IOU:
            kept_shapes.append(_edited(convertible.pop(best), shape, w, h, vis.get(shape["label_id"])))
        else:
            kept_shapes.append(shape)
    # what is left in ``convertible`` is gone from the labels: it is deleted
    old_known = [t for t in old_tags if classification and t.get("label_id") in known]
    kept_tags = [t for t in old_tags if not any(t is o for o in old_known)]
    for tag in new_tags:
        if classification:
            match = next(
                (
                    i
                    for i, o in enumerate(old_known)
                    if _same_class(o.get("label_id"), tag["label_id"], aliases)
                ),
                None,
            )
            kept_tags.append(old_known.pop(match) if match is not None else tag)
        elif not any(_same_class(o.get("label_id"), tag["label_id"], aliases) for o in kept_tags):
            kept_tags.append(tag)
    return kept_shapes, kept_tags


def _empty_patch() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """An empty set of changes: ``{action: {kind: []}}`` for every action and kind.

    Returns:
        The lists of objects to delete, update and create, per kind (tags, shapes, tracks).
    """
    return {action: {kind: [] for kind in KINDS} for action in ACTIONS}


def _add_delta(
    patch: dict[str, dict[str, list[dict[str, Any]]]],
    kind: str,
    old: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> None:
    """Add to ``patch`` what turns ``old`` into ``new`` (objects of one kind on one frame).

    An object of ``new`` that is one of ``old`` (the same Python object) is unchanged; one with the id
    of an old object is an update; any other is created. Old objects that are neither kept nor updated
    are deleted: only objects that were read (``old``) can be deleted.

    Args:
        patch: The changes, see :func:`_empty_patch`; changed in place.
        kind: ``tags``, ``shapes`` or ``tracks``.
        old: The objects CVAT has (as read).
        new: The objects after the write.
    """
    kept = {id(x) for x in new}
    old_ids = {o.get("id") for o in old if o.get("id") is not None}
    updated: set[Any] = set()
    for item in new:
        if any(item is o for o in old):
            continue
        if item.get("id") is not None and item["id"] in old_ids:
            patch["update"][kind].append(item)
            updated.add(item["id"])
        else:
            patch["create"][kind].append(item)
    for o in old:
        if id(o) not in kept and (o.get("id") is None or o["id"] not in updated):
            patch["delete"][kind].append(o)


def _not_converted(shape: dict[str, Any], task: Task, parts: dict[int, list[int]]) -> str | None:
    """Why a CVAT shape does not reach the project as it is, as a key of :attr:`RemoteItem.not_converted`.

    Args:
        shape: A CVAT shape of a known label.
        task: The task type of the project.
        parts: The sublabels of each skeleton label.

    Returns:
        The shape type when ActiveLabel does not convert it (``ellipse``, ``mask``, ``polyline``,
        ``cuboid``, or ``skeleton`` outside keypoints projects / without parts), ``rotated_rectangle`` for
        a turned rectangle read as its enclosing box, else None.
    """
    kind = str(shape.get("type"))
    if kind == "skeleton":
        return None if task is Task.KEYPOINTS and shape.get("label_id") in parts else kind
    if kind not in TRACKABLE:
        return kind
    return "rotated_rectangle" if kind == "rectangle" and shape.get("rotation") else None


def _sendable(item: dict[str, Any]) -> dict[str, Any]:
    """An annotation object as it is sent back: the parts of a skeleton keep only their own fields.

    Args:
        item: A tag, shape or track (as read, or built here).

    Returns:
        ``item`` itself, or for a skeleton shape a copy whose ``elements`` keep only
        :data:`ELEMENT_FIELDS` (tracks are returned as they are).
    """
    if not item.get("elements") or item.get("shapes"):
        return item
    return {
        **item,
        "elements": [{k: v for k, v in el.items() if k in ELEMENT_FIELDS} for el in item["elements"]],
    }


def _audit_task(task: dict[str, Any], base_url: str) -> dict[str, Any]:
    """One CVAT task as the audit lists it: id, name, number of images and its address on the server.

    Args:
        task: A task as ``GET /api/tasks`` returns it.
        base_url: The address of the CVAT server, without a trailing slash.

    Returns:
        ``{id, name, size, url}``.
    """
    return {
        "id": str(task["id"]),
        "name": str(task.get("name", "")),
        "size": int(task.get("size") or 0),
        "url": f"{base_url}/tasks/{task['id']}",
    }


def _split_key(key: str) -> tuple[int, int]:
    """Split an item key ``"<task_id>:<frame>"``.

    Args:
        key: The key of an image on the remote.

    Returns:
        The task id and the frame index.

    Raises:
        ToolError: The key is not of that form.
    """
    task, _, frame = str(key).partition(":")
    if not (task.isdigit() and frame.isdigit()):
        raise ToolError(f"'{key}' is not a CVAT item key (expected '<task>:<frame>').")
    return int(task), int(frame)


def _label_payload(name: str) -> dict[str, Any]:
    """A CVAT label of type ``any`` without attributes.

    Args:
        name: The class name.

    Returns:
        The label as CVAT takes it when creating or patching a project.
    """
    return {"name": name, "type": "any", "attributes": []}


class CvatRemote:
    """A CVAT project, through the REST API."""

    tool = "cvat"

    # -- tasks, frames --
    def set_frames(self, settings: ToolSettings, task_id: str, delete: list[int], restore: list[int]) -> None:
        """Mark frames of a task as deleted (or not deleted any more); annotations are kept by CVAT.

        Args:
            settings: Address and login of the CVAT server.
            task_id: The CVAT task id.
            delete: Frame indices to mark as deleted.
            restore: Frame indices to mark as not deleted (wins over ``delete``).
        """
        if not delete and not restore:
            return
        api = self._api(settings)
        meta = api.get(f"/api/tasks/{task_id}/data/meta") or {}
        deleted = ({int(d) for d in meta.get("deleted_frames") or []} | set(delete)) - set(restore)
        api.request("PATCH", f"/api/tasks/{task_id}/data/meta", json={"deleted_frames": sorted(deleted)})

    def frames(self, settings: ToolSettings, task_id: str) -> list[tuple[str, bool]]:
        """``(file name, deleted)`` of every frame of a task, in order.

        Args:
            settings: Address and login of the CVAT server.
            task_id: The CVAT task id.

        Returns:
            The file name and whether the frame is deleted, one entry per frame.
        """
        meta = self._api(settings).get(f"/api/tasks/{task_id}/data/meta") or {}
        deleted = {int(d) for d in meta.get("deleted_frames") or []}
        return [(str(fr.get("name", "")), i in deleted) for i, fr in enumerate(_meta_frames(meta))]

    def check(self, settings: ToolSettings) -> str:
        """Log in to the server; return ``Connected to <url> as <user>`` (raises ToolError).

        Args:
            settings: Address and login of the CVAT server.

        Returns:
            The message ``Connected to <url> as <user>``.
        """
        api = self._api(settings)
        who = api.get("/api/users/self") or {}
        return f"Connected to {api.url} as {who.get('username') or settings.username}"

    def audit(self, settings: ToolSettings) -> dict[str, Any]:
        """Read-only inventory of the tasks of the server, per project (nothing is changed or deleted).

        Args:
            settings: Address and login of the CVAT server.

        Returns:
            ``{"projects": [{id, name, url, tasks: [...]}], "loose": [{pattern, tasks: [...]}]}``; a task is
            ``{id, name, size, url}``. ``loose`` holds the tasks that are in no project, grouped by
            :func:`task_pattern` of their name (the biggest group first).
        """
        api = self._api(settings)

        projects: dict[str, dict[str, Any]] = {
            str(p["id"]): {
                "id": str(p["id"]),
                "name": str(p.get("name", "")),
                "url": f"{api.url}/projects/{p['id']}",
                "tasks": [],
            }
            for p in api.paged("/api/projects", {"sort": "id"}, ALL)
        }
        groups: dict[str, list[dict[str, Any]]] = {}
        for t in api.paged("/api/tasks", {"sort": "id"}, ALL):
            if t.get("project_id"):
                # a project the account cannot list still owns the task: show it by its number
                pid = str(t["project_id"])
                projects.setdefault(
                    pid, {"id": pid, "name": "", "url": f"{api.url}/projects/{pid}", "tasks": []}
                )
                projects[pid]["tasks"].append(_audit_task(t, api.url))
            else:
                groups.setdefault(task_pattern(str(t.get("name", ""))), []).append(_audit_task(t, api.url))
        loose = [{"pattern": pattern, "tasks": found} for pattern, found in groups.items()]
        loose.sort(key=lambda g: (-len(g["tasks"]), g["pattern"]))
        return {"projects": list(projects.values()), "loose": loose}

    def delete_task(self, settings: ToolSettings, task_id: str) -> None:
        """Delete a CVAT task with its frames and annotations (nothing when it is already gone).

        Args:
            settings: Address and login of the CVAT server.
            task_id: The CVAT task id.

        Raises:
            ToolError: CVAT refuses the request or cannot be reached (a missing task is not an error).
        """
        api = self._api(settings)
        try:
            api.request("DELETE", f"/api/tasks/{task_id}")
        except ToolError as err:
            if not isinstance(err, RemoteNotFound):
                raise

    def frame(self, settings: ToolSettings, task_id: str, frame: int) -> bytes:
        """The original image of one frame.

        Args:
            settings: Address and login of the CVAT server.
            task_id: The CVAT task id.
            frame: Index of the frame in the task.

        Returns:
            The bytes of the image in its original quality.

        Raises:
            ToolError: CVAT cannot be reached or answers with an error.
        """
        api = self._api(settings)
        try:
            r = api._http.get(  # noqa: SLF001 - raw bytes, not JSON
                f"{api.url}/api/tasks/{task_id}/data",
                params={"type": "frame", "number": str(int(frame)), "quality": "original"},
                auth=api._auth,  # noqa: SLF001
                timeout=UPLOAD_TIMEOUT,
            )
        except Exception:  # noqa: BLE001 - never echo the exception text
            raise common.unreachable(api.url) from None
        if r.status_code >= 400:
            raise ToolError(
                f"CVAT error {r.status_code} while downloading frame {frame} of task {task_id}.",
                "cvat_download_failed",
                http=r.status_code,
            )
        return bytes(r.content)

    def video_source(
        self, settings: ToolSettings, task_id: str, dest_dir: Path
    ) -> tuple[Path, dict[str, int]] | None:
        """The original video of a video task, taken from the task backup (None for an image task).

        Args:
            settings: Address and login of the CVAT server.
            task_id: The CVAT task id.
            dest_dir: Folder where the video is written (created when missing).

        Returns:
            The path of the video file and its selection ``{start, stop, step}`` (frame indices of the
            video); None when the task is not made from a video.

        Raises:
            ToolError: the backup could not be made, downloaded or does not contain the video.
        """
        api = self._api(settings)
        meta = api.get(f"/api/tasks/{task_id}/data/meta") or {}
        frames = list(meta.get("frames") or [])
        if len(frames) != 1 or int(meta.get("size") or 0) <= 1:
            return None
        start, stop, step = _meta_selection(meta)
        answer = api.request("POST", f"/api/tasks/{task_id}/backup/export", timeout=UPLOAD_TIMEOUT)
        rq_id = str((answer or {}).get("rq_id") or "") if isinstance(answer, dict) else ""
        if not rq_id:
            raise ToolError("CVAT did not start the task backup.", "cvat_backup_failed")
        url = ""
        for _ in range(MAX_POLLS):
            data = api.get(f"/api/requests/{rq_id}") or {}
            state = str(data.get("status", "")).lower()
            if state == "failed":
                raise ToolError(
                    f"CVAT could not make the task backup: {str(data.get('message') or '')[:200]}",
                    "cvat_backup_failed",
                )
            if state == "finished":
                url = str(data.get("result_url") or "")
                break
            _sleep(POLL_INTERVAL)
        if not url:
            raise ToolError("CVAT did not finish the task backup in time.", "cvat_backup_timeout")
        if url.startswith("/"):
            url = f"{api.url}{url}"
        try:
            r = api._http.request("GET", url, auth=api._auth, timeout=UPLOAD_TIMEOUT)  # noqa: SLF001
        except Exception:  # noqa: BLE001 - never echo the exception text
            raise common.unreachable(api.url) from None
        if r.status_code >= 400:
            raise ToolError(
                f"CVAT error {r.status_code} while downloading the backup of task {task_id}.",
                "cvat_download_failed",
                http=r.status_code,
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / "backup.zip"
        archive.write_bytes(bytes(r.content))
        try:
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if n.startswith("data/") and not n.endswith("/")]
                wanted = str(frames[0].get("name") or "")
                pick = next((n for n in names if n == f"data/{wanted}"), None) or next(
                    (n for n in names if Path(n).suffix.lower() in VIDEO_EXTENSIONS), None
                )
                if pick is None:
                    raise ToolError(
                        "The task backup does not contain the original video.", "cvat_backup_invalid"
                    )
                target = dest_dir / Path(pick).name  # the name only: nothing is written outside dest_dir
                with zf.open(pick) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
        except zipfile.BadZipFile:
            raise ToolError(
                "The task backup from CVAT is not a valid archive.", "cvat_backup_invalid"
            ) from None
        finally:
            archive.unlink(missing_ok=True)
        return target, {"start": start, "stop": stop, "step": step}

    # -- helpers --
    @staticmethod
    def _api(settings: ToolSettings) -> _Api:
        """A REST client for the server in ``settings``.

        Args:
            settings: Address and login of the CVAT server.

        Returns:
            The client.
        """
        return _Api(settings)

    @staticmethod
    def _info(api: _Api, item: dict[str, Any]) -> RemoteInfo:
        """A CVAT project as a :class:`RemoteInfo`, counting the images of its tasks.

        Args:
            api: The REST client.
            item: A project as CVAT returns it (``id``, ``name``).

        Returns:
            Id, name, address and number of images of the project.
        """
        tasks = api.paged("/api/tasks", {"project_id": item["id"]}, ALL)
        images = sum(int(t.get("size") or 0) for t in tasks)
        return RemoteInfo(
            str(item["id"]), str(item.get("name", "")), f"{api.url}/projects/{item['id']}", images
        )

    @staticmethod
    def _label_rows(api: _Api, rid: str) -> list[dict[str, Any]]:
        """Every label of a project as CVAT lists it (sublabels of skeletons included).

        Args:
            api: The REST client.
            rid: The CVAT project id.

        Returns:
            The labels (``id``, ``name``, ``type``, ``attributes``, ``parent_id`` ...).
        """
        return list(api.paged("/api/labels", {"project_id": rid}, ALL))

    def _pose_specs(self, api: _Api, rid: str, ids: dict[str, int], labels: list[Label]) -> dict[int, int]:
        """The :data:`VISIBILITY` attribute of each label, added first to the labels that need it.

        A label needs it when one of ``labels`` of its class mixes visibilities 1 and 2 (the attribute is
        added to that CVAT label with ``PATCH /api/projects/{id}``; the others are not touched).

        Args:
            api: The REST client.
            rid: The CVAT project id.
            ids: CVAT label id of each class name.
            labels: The labels about to be written.

        Returns:
            ``{label id: attribute spec id}``; empty when no label is a pose.
        """
        if not any(lb.keypoints for lb in labels):
            return {}
        rows = self._label_rows(api, rid)
        specs = visibility_specs(rows)
        names = {int(r["id"]): str(r.get("name", "")) for r in rows}
        wanted = {class_id(ids, lb.label) for lb in labels if _mixed(lb)}
        need = sorted({lid for lid in wanted if lid is not None} - set(specs))
        if need:
            body = [
                {"id": lid, "name": names.get(lid, ""), "attributes": [_visibility_spec()]} for lid in need
            ]
            api.request("PATCH", f"/api/projects/{rid}", json={"labels": body})
            specs = visibility_specs(self._label_rows(api, rid))
        return specs

    @classmethod
    def _labels(cls, api: _Api, rid: str) -> dict[str, int]:
        """The top-level labels of a project.

        Args:
            api: The REST client.
            rid: The CVAT project id.

        Returns:
            The CVAT label id of each class name.
        """
        found = cls._label_rows(api, rid)
        out: dict[str, int] = {}
        for x in found:
            if x.get("name") and not x.get("parent_id"):
                out.setdefault(str(x["name"]), int(x["id"]))
        return out

    def find(self, settings: ToolSettings, name: str) -> RemoteInfo | None:
        """The project with exactly this name (the oldest when there are several).

        Args:
            settings: Address and login of the CVAT server.
            name: The exact project name.

        Returns:
            The project, or None when there is none with that name.
        """
        api = self._api(settings)
        found = [
            p for p in api.paged("/api/projects", {"name": name, "sort": "id"}, ALL) if p.get("name") == name
        ]
        return self._info(api, found[0]) if found else None

    def get(self, settings: ToolSettings, remote_id: str) -> RemoteInfo:
        """One project (ToolError when it is gone).

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.

        Returns:
            The project.
        """
        api = self._api(settings)
        return self._info(api, api.get(f"/api/projects/{remote_id}"))

    def create(self, settings: ToolSettings, project: Project) -> RemoteInfo:
        """Create the project with the classes of the ActiveLabel project (labels of type "any").

        Args:
            settings: Address and login of the CVAT server.
            project: The ActiveLabel project (its id, name and classes).

        Returns:
            The CVAT project that was created.
        """
        api = self._api(settings)
        body = {
            "name": f"ActiveLabel {project.id} - {project.name}",
            "labels": [_label_payload(c) for c in project.classes],
        }
        return self._info(api, api.request("POST", "/api/projects", json=body))

    def classes(self, settings: ToolSettings, remote_id: str) -> dict[str, str]:
        """The top-level labels of the project by id, to notice a label renamed in CVAT (its id stays).

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.

        Returns:
            ``{label id: name}``.
        """
        return {str(i): n for n, i in self._labels(self._api(settings), remote_id).items()}

    def ensure_classes(self, settings: ToolSettings, remote_id: str, classes: list[str]) -> None:
        """Add the labels the project does not have yet (existing ones are not touched).

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.
            classes: The class names the project must have.
        """
        self._ensure(self._api(settings), remote_id, classes)

    def _ensure(self, api: _Api, rid: str, names: list[str]) -> dict[str, int]:
        """Make sure the project has a label for each name (a label that differs only in case counts).

        Args:
            api: The REST client.
            rid: The CVAT project id.
            names: The class names that must exist.

        Returns:
            The CVAT label id of each label name of the project (not only of ``names``).
        """
        ids = self._labels(api, rid)
        missing = [n for n in dict.fromkeys(names) if class_id(ids, n) is None]
        if missing:
            api.request(
                "PATCH", f"/api/projects/{rid}", json={"labels": [_label_payload(n) for n in missing]}
            )
            ids = self._labels(api, rid)
        return ids

    # -- read --
    def tasks(self, settings: ToolSettings, remote_id: str) -> list[RemoteTask]:
        """The tasks of the project (oldest first) with their jobs in frame order.

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.

        Returns:
            The tasks, each with its annotation jobs.
        """
        api = self._api(settings)
        out: list[RemoteTask] = []
        for t in api.paged("/api/tasks", {"project_id": remote_id, "sort": "id"}, ALL):
            out.append(self._task(api, t))
        return out

    @staticmethod
    def _task(api: _Api, t: dict[str, Any]) -> RemoteTask:
        """A CVAT task as a :class:`RemoteTask` with its annotation jobs in frame order.

        Args:
            api: The REST client.
            t: A task as CVAT returns it.

        Returns:
            The task; ``job_size`` is None when it has a single job.
        """
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

        Tracks give one label per frame where the object is visible (see :func:`track_frames`). What
        cannot be converted is counted per item in ``not_converted`` (see :func:`_not_converted`; tracks
        that are not read count once, on their first frame).

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.
            task: The task type of the project (classification reads tags, the others shapes).
            task_ids: Only these CVAT task ids; None for every task of the project.

        Returns:
            One item per frame that is not deleted, keyed ``"<task_id>:<frame>"``, relative 0..1 labels.
        """
        api = self._api(settings)
        rows = self._label_rows(api, remote_id)
        names = {int(r["id"]): str(r["name"]) for r in rows if r.get("name") and not r.get("parent_id")}
        specs = visibility_specs(rows)
        parts = skeleton_parts(rows)
        out: list[RemoteItem] = []
        for t in api.paged("/api/tasks", {"project_id": remote_id, "sort": "id"}, ALL):
            tid = t["id"]
            if task_ids is not None and str(tid) not in task_ids:
                continue
            meta = api.get(f"/api/tasks/{tid}/data/meta") or {}
            frames = _meta_frames(meta)
            ann = api.get(f"/api/tasks/{tid}/annotations") or {}
            per_frame: dict[int, list[Label]] = {}
            skipped: dict[int, dict[str, int]] = {}
            for s in ann.get("shapes") or []:
                f, lid = int(s.get("frame", -1)), s.get("label_id")
                name = names.get(lid)
                if task is Task.CLASSIFICATION or name is None or not 0 <= f < len(frames):
                    continue
                why = _not_converted(s, task, parts)
                if why:
                    skipped.setdefault(f, {})[why] = skipped.get(f, {}).get(why, 0) + 1
                size = int(frames[f]["width"]), int(frames[f]["height"])
                label = shape_to_label(s, name, *size, specs.get(lid), parts.get(lid))
                if label:
                    per_frame.setdefault(f, []).append(label)
            ids = {n: i for i, n in names.items()}
            for tr in ann.get("tracks") or []:
                name = names.get(tr.get("label_id"))
                if task is Task.CLASSIFICATION or name is None:
                    continue
                if _trackable(tr, ids) is None:  # skeleton tracks, other types: kept, not read
                    f = int(tr.get("frame") or 0)
                    skipped.setdefault(f, {})["tracks"] = skipped.get(f, {}).get("tracks", 0) + 1
                    continue
                for f, (s, _) in sorted(track_frames(tr, len(frames)).items()):
                    size = int(frames[f]["width"]), int(frames[f]["height"])
                    label = shape_to_label(s, name, *size, specs.get(tr.get("label_id")))
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
                        skipped.get(f, {}),
                    )
                )
        return out

    @staticmethod
    def _frames(api: _Api, tid: Any) -> list[dict[str, Any]]:
        """The frames of a task (see :func:`_meta_frames`).

        Args:
            api: The REST client.
            tid: The CVAT task id.

        Returns:
            One dict per frame with ``name``, ``width`` and ``height``.
        """
        meta = api.get(f"/api/tasks/{tid}/data/meta") or {}
        return _meta_frames(meta)

    # -- write --
    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Write the labels of the updated frames; other frames stay as they are.

        Only what changed is sent, as differences (``PATCH .../annotations?action=delete|update|create``):
        a shape matched with a label keeps its id and everything CVAT has for it (an edited one gets
        only the new points), shapes or tags ActiveLabel does not convert are kept, and only objects
        read just before can be deleted, so what someone draws in CVAT meanwhile stays (module
        docstring). Labels with the id of a track of the task go into that track (see
        :func:`_write_tracks`), which is updated in place; the other tracks are not sent.

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.
            project: The ActiveLabel project (its task type decides what is converted).
            updates: The new labels (relative 0..1) of each updated image, keyed ``"<task_id>:<frame>"``.

        Raises:
            ToolError: A key is not valid or names a frame the task does not have.
        """
        api = self._api(settings)
        by_task: dict[int, dict[int, list[Label]]] = {}
        for key, labels in updates.items():
            tid, frame = _split_key(key)
            by_task.setdefault(tid, {})[frame] = labels
        wanted = [lb.label for per in by_task.values() for labels in per.values() for lb in labels]
        ids = self._ensure(api, remote_id, wanted)
        every = [lb for per in by_task.values() for labels in per.values() for lb in labels]
        pose = project.task is Task.KEYPOINTS
        specs = self._pose_specs(api, remote_id, ids, every) if pose else {}
        skeletons = skeleton_parts(self._label_rows(api, remote_id)) if pose else {}
        for tid, per in by_task.items():
            frames = self._frames(api, tid)
            for frame in per:
                if not 0 <= frame < len(frames):
                    raise ToolError(
                        f"CVAT task {tid} has no frame {frame}: synchronise again.", "sync_remote_changed"
                    )
            ann = api.get(f"/api/tasks/{tid}/annotations") or {}
            patch = _empty_patch()
            old_tracks = ann.get("tracks") or []
            tracks = copy.deepcopy(old_tracks)
            if project.task is not Task.CLASSIFICATION:
                per = _write_tracks(tracks, per, ids, frames, specs)
            patch["update"]["tracks"] = [
                new for new, old in zip(tracks, old_tracks, strict=True) if new != old
            ]
            for frame, labels in per.items():
                size = (int(frames[frame]["width"]), int(frames[frame]["height"]))
                old_shapes = [s for s in ann.get("shapes") or [] if s.get("frame") == frame]
                old_tags = [t for t in ann.get("tags") or [] if t.get("frame") == frame]
                new_shapes, new_tags = _merge_frame(
                    old_shapes, old_tags, labels, ids, frame, size, project.task, specs, skeletons
                )
                _add_delta(patch, "shapes", old_shapes, new_shapes)
                _add_delta(patch, "tags", old_tags, new_tags)
            self._send_patch(api, tid, patch, int(ann.get("version") or 0))

    @staticmethod
    def _send_patch(
        api: _Api, tid: int, patch: dict[str, dict[str, list[dict[str, Any]]]], version: int
    ) -> None:
        """Send the changes of a task: one ``PATCH`` per action that has something (delete, update, create).

        Args:
            api: The REST client.
            tid: The CVAT task id.
            patch: The objects to delete, update and create, see :func:`_empty_patch`.
            version: The ``version`` of the annotations as read.
        """
        for action in ACTIONS:
            body = {kind: [_sendable(x) for x in patch[action][kind]] for kind in KINDS}
            if any(body[kind] for kind in KINDS):
                api.request(
                    "PATCH",
                    f"/api/tasks/{tid}/annotations",
                    params={"action": action},
                    json={"version": version, **body},
                )

    # -- add --
    def add(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> dict[str, str]:
        """Upload the images in tasks of ``BATCH``, then write their labels; return ``{image_id: key}``.

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.
            project: The ActiveLabel project (its id is part of the task names).
            images: For each image its record, the file to upload and its labels (relative 0..1).

        Returns:
            The key ``"<task_id>:<frame>"`` of each image id.
        """
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

        Args:
            settings: Address and login of the CVAT server.
            remote_id: The CVAT project id.
            project: The ActiveLabel project.
            name: The name of the new task.
            job_size: Frames per job (``segment_size``); None for a single job.
            images: For each image its record, the file to upload and its labels (relative 0..1).
            video: The video file and its selection, or None for a task made of images.

        Returns:
            The task as CVAT has it and the key ``"<task_id>:<frame>"`` of each image id.
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
        """Create a task in the project and fill it; delete it again when filling fails.

        Args:
            api: The REST client.
            rid: The CVAT project id.
            name: The name of the new task.
            job_size: Frames per job (``segment_size``); None for a single job.
            batch: For each image its record, the file to upload and its labels (relative 0..1).
            ids: CVAT label id of each class name.
            video: The video file and its selection, or None for a task made of images.

        Returns:
            The id of the new task and the key ``"<task_id>:<frame>"`` of each image id.
        """
        body: dict[str, Any] = {"name": name, "project_id": int(rid)}
        if job_size:
            body["segment_size"] = int(job_size)
            body["overlap"] = 0  # CVAT repeats frames between jobs of a video by default
        every = [lb for _, _, labels in batch for lb in labels]
        specs = self._pose_specs(api, rid, ids, every)
        skeletons = skeleton_parts(self._label_rows(api, rid)) if any(lb.keypoints for lb in every) else {}
        task = api.request("POST", "/api/tasks", json=body)
        tid = task["id"]
        try:
            return self._populate_task(api, tid, job_size, batch, ids, video, specs, skeletons)
        except Exception:
            self._discard_task(api, tid)  # a retry must not leave empty tasks (they count in the plan)
            raise

    @staticmethod
    def _discard_task(api: _Api, tid: Any) -> None:
        """Delete a task whose creation failed half way (best effort: the original error is what matters).

        Args:
            api: The REST client.
            tid: The CVAT task id.
        """
        try:
            api.request("DELETE", f"/api/tasks/{tid}")
        except Exception:  # noqa: BLE001 - never hide the real failure
            log.warning("could not delete the half-made CVAT task %s", tid)

    def _populate_task(
        self,
        api: _Api,
        tid: Any,
        job_size: int | None,
        batch: list[tuple[ImageRecord, Path, list[Label]]],
        ids: dict[str, int],
        video: tuple[Path, dict[str, Any]] | None,
        specs: dict[int, int] | None = None,
        skeletons: dict[int, list[int]] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Upload the images (or the video) to an empty task and write the labels.

        Args:
            api: The REST client.
            tid: The CVAT task id.
            job_size: Frames per job; when set, CVAT must keep the order of the images.
            batch: For each image its record, the file to upload and its labels (relative 0..1).
            ids: CVAT label id of each class name.
            video: The video file and its selection, or None to upload the images.
            specs: Id of the :data:`VISIBILITY` attribute of each CVAT label that has one.
            skeletons: The sublabels of each skeleton label (poses of those classes become skeletons).

        Returns:
            The task id as text and the key ``"<task_id>:<frame>"`` of each image id.

        Raises:
            ToolError: CVAT did not register an image or changed the order of the images.
        """
        if video is not None:
            return self._fill_from_video(api, tid, video, job_size, batch, ids, specs, skeletons)
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
                raise ToolError(
                    f"CVAT did not register the image '{rec.name}': upload again.",
                    "cvat_image_not_registered",
                    name=rec.name,
                )
            if job_size and frame != position:  # jobs only match when the order is kept
                raise ToolError(
                    f"CVAT changed the order of the images of task {tid}: its jobs would not match.",
                    "cvat_order_changed",
                )
            keys[rec.id] = f"{tid}:{frame}"
            s, t = labels_to_cvat(
                labels,
                ids,
                frame,
                int(frames[frame]["width"]),
                int(frames[frame]["height"]),
                specs,
                skeletons,
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
        specs: dict[int, int] | None = None,
        skeletons: dict[int, list[int]] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Send the original video with the task's selection; map frame *i* to image *i*.

        Args:
            api: The REST client.
            tid: The CVAT task id.
            video: The video file and its selection ``{start, stop, step[, name]}`` (video frame indices).
            job_size: Frames per job (``segment_size``); None for a single job.
            batch: For each image its record, the file (not used here) and its labels (relative 0..1).
            ids: CVAT label id of each class name.
            specs: Id of the :data:`VISIBILITY` attribute of each CVAT label that has one.
            skeletons: The sublabels of each skeleton label (poses of those classes become skeletons).

        Returns:
            The task id as text and the key ``"<task_id>:<frame>"`` of each image id.

        Raises:
            ToolError: CVAT made a different number of frames than the task has images.
        """
        path, sel = video
        data = {
            "image_quality": 95,
            "start_frame": int(sel.get("start") or 0),
            "stop_frame": int(sel["stop"]),
            "frame_filter": f"step={int(sel.get('step') or 1)}",
        }
        # the original file name (the project stores videos under their content id)
        name = Path(str(sel.get("name") or path.name)).name
        mime = mimetypes.guess_type(name)[0] or "video/mp4"
        with path.open("rb") as fh:
            answer = api.request(
                "POST",
                f"/api/tasks/{tid}/data",
                data=data,
                files=[("client_files[0]", (name, fh, mime))],
                timeout=UPLOAD_TIMEOUT,
            )
        self._wait_ready(api, tid, (answer or {}).get("rq_id") if isinstance(answer, dict) else None)
        frames = self._frames(api, tid)
        if len(frames) != len(batch):
            raise ToolError(
                f"CVAT made {len(frames)} frames from the video of task {tid}, the task has {len(batch)}: "
                "its jobs would not match.",
                "cvat_frames_mismatch",
                frames=len(frames),
                expected=len(batch),
            )
        keys: dict[str, str] = {}
        shapes: list[dict[str, Any]] = []
        tags: list[dict[str, Any]] = []
        for frame, (rec, _, labels) in enumerate(batch):
            keys[rec.id] = f"{tid}:{frame}"
            s, t = labels_to_cvat(
                labels,
                ids,
                frame,
                int(frames[frame]["width"]),
                int(frames[frame]["height"]),
                specs,
                skeletons,
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
        """Poll until CVAT has processed the uploaded data (``/api/requests`` or the task status).

        Args:
            api: The REST client.
            tid: The CVAT task id.
            rq_id: The id of the CVAT request that processes the data; None to follow the task status.

        Raises:
            ToolError: CVAT reports a failure or does not finish within ``MAX_POLLS`` checks.
        """
        for _ in range(MAX_POLLS):
            done = failed = False
            message = ""
            data: dict[str, Any] | None = None
            if rq_id:
                try:
                    data = api.get(f"/api/requests/{rq_id}") or {}
                except ToolError as err:
                    if not isinstance(err, RemoteNotFound):
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
                    f"CVAT could not process the uploaded images: {message[:200] or 'unknown error'}",
                    "cvat_processing_failed",
                )
            if done:
                return
            _sleep(POLL_INTERVAL)
        raise ToolError(
            "CVAT took too long to process the uploaded images: try again later.", "cvat_processing_timeout"
        )
