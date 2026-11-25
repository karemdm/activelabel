"""A strict in-memory CVAT server (REST API v2) for end-to-end synchronisation tests.

``FakeCvatServer`` stands in for ``requests.Session``: install it with :meth:`FakeCvatServer.install` and
every call of :class:`activelabel.sync.cvat_remote.CvatRemote` is answered from memory (nothing leaves the
process). It follows the formats of CVAT's REST API:

* projects with labels ``{id, name, type, attributes, sublabels, parent_id}`` (types ``any``,
  ``rectangle``, ``polygon``, ``points``, ``skeleton``, ``tag`` ...);
* tasks (``project_id``, ``segment_size``, ``size``, ``jobs``), their data meta (``frames``,
  ``start_frame``, ``stop_frame``, ``frame_filter``, ``deleted_frames``) and jobs whose ``start_frame`` /
  ``stop_frame`` are TASK frame numbers;
* annotations of a task and of a job (``GET``, ``PUT`` replace, ``PATCH ?action=create|update|delete``,
  ``DELETE``) with ids assigned by the server and strict validation: unknown label ids, unknown fields,
  wrong point counts, frames outside the task (or the job), attribute specs of another label and shape
  types a label does not allow are answered with a 400 and a text, like CVAT.

Every request is recorded in :attr:`FakeCvatServer.requests`. Test helpers (``add_project``, ``add_task``,
``edit_shape`` ...) change the server state directly, as another person working in CVAT would.

Assumptions that only a real CVAT can confirm are listed in the report of the test module (for example that a
``PUT`` with shape ids is refused here, which the adapter never does).
"""

from __future__ import annotations

import copy
import io
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import requests
from PIL import Image

from activelabel.sync import cvat_remote

HOST = "cvat.test"
SHAPE_TYPES = ("rectangle", "polygon", "polyline", "points", "ellipse", "cuboid", "mask", "skeleton")
LABEL_TYPES = (
    "any",
    "rectangle",
    "polygon",
    "polyline",
    "points",
    "ellipse",
    "cuboid",
    "mask",
    "skeleton",
    "tag",
)
SOURCES = ("manual", "auto", "semi-auto", "file")
SHAPE_FIELDS = {
    "type",
    "frame",
    "label_id",
    "points",
    "rotation",
    "occluded",
    "outside",
    "z_order",
    "group",
    "source",
    "attributes",
    "elements",
}
TAG_FIELDS = {"frame", "label_id", "group", "source", "attributes"}
TRACK_FIELDS = {"frame", "label_id", "group", "source", "shapes", "attributes", "elements"}
TRACKED_FIELDS = {
    "type",
    "frame",
    "points",
    "rotation",
    "occluded",
    "outside",
    "z_order",
    "attributes",
    "keyframe",
}
ELEMENT_FIELDS = {
    "type",
    "frame",
    "label_id",
    "points",
    "occluded",
    "outside",
    "z_order",
    "source",
    "attributes",
}
ANNOTATION_KEYS = {"version", "tags", "shapes", "tracks"}
POINT_COUNT = {"rectangle": 4, "ellipse": 4, "cuboid": 16}
POINT_MIN = {"polygon": 6, "polyline": 4, "points": 2, "mask": 5}


class BadRequest(Exception):
    """A validation problem: the server answers 400 with this text."""


@dataclass
class Request:
    """One request the server received.

    Attributes:
        method: HTTP method.
        path: the path without the query string.
        params: query parameters (from ``params=`` and from the URL).
        body: the JSON body (None when there was none).
    """

    method: str
    path: str
    params: dict[str, Any]
    body: Any = None


class Response:
    """The part of ``requests.Response`` the adapter reads."""

    def __init__(self, status: int = 200, body: Any = None, raw: bytes | None = None) -> None:
        self.status_code = status
        self._body = body
        self.content = raw if raw is not None else (b"" if body is None else b"x")
        self.text = "" if body is None else json.dumps(body)

    def json(self) -> Any:
        """The JSON body.

        Returns:
            The decoded body.

        Raises:
            ValueError: the answer has no JSON body.
        """
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


def _number(value: Any, what: str) -> float:
    """A finite number (not a bool or a string).

    Args:
        value: the value to check.
        what: where it is, for the error text.

    Returns:
        The value as a float.

    Raises:
        BadRequest: not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BadRequest(f"{what}: expected a finite number, got {value!r}")
    return float(value)


def _unknown(raw: dict[str, Any], allowed: set[str], what: str) -> None:
    """Refuse fields CVAT's serializer does not have.

    Args:
        raw: the object received.
        allowed: its legal field names.
        what: where it is, for the error text.

    Raises:
        BadRequest: there are unknown fields.
    """
    extra = sorted(set(raw) - allowed)
    if extra:
        raise BadRequest(f"{what}: unknown field(s) {extra}")


def _check_points(kind: str, points: Any, what: str) -> list[float]:
    """Validate the flat point list of a shape of ``kind``.

    Args:
        kind: the shape type.
        points: the ``points`` value.
        what: where it is, for the error text.

    Returns:
        The points as floats.

    Raises:
        BadRequest: wrong type, wrong count or non-numeric.
    """
    if not isinstance(points, list):
        raise BadRequest(f"{what}: points must be a list")
    pts = [_number(p, f"{what}.points") for p in points]
    if kind == "skeleton":
        if pts:
            raise BadRequest(f"{what}: a skeleton has no points of its own (its parts are in elements)")
        return pts
    if kind in POINT_COUNT and len(pts) != POINT_COUNT[kind]:
        raise BadRequest(f"{what}: a {kind} needs {POINT_COUNT[kind]} numbers, got {len(pts)}")
    if kind in POINT_MIN:
        if len(pts) < POINT_MIN[kind]:
            raise BadRequest(f"{what}: a {kind} needs at least {POINT_MIN[kind]} numbers, got {len(pts)}")
        if kind != "mask" and len(pts) % 2:
            raise BadRequest(f"{what}: a {kind} needs an even number of coordinates, got {len(pts)}")
    return pts


class Validator:
    """Checks and normalises annotation objects against one task (and optionally one job's frames)."""

    def __init__(self, labels: dict[int, dict[str, Any]], size: int, frames: range, ids_ok: bool) -> None:
        """Create the validator.

        Args:
            labels: the labels of the task by id (sublabels included).
            size: number of frames of the task.
            frames: the task frames the request may touch (the whole task or one job).
            ids_ok: whether ``id`` is a legal field (PATCH update / delete).
        """
        self.labels = labels
        self.size = size
        self.frames = frames
        self.ids_ok = ids_ok

    def _fields(self, raw: dict[str, Any], allowed: set[str], what: str) -> None:
        _unknown(raw, allowed | ({"id"} if self.ids_ok else set()), what)

    def frame(self, value: Any, what: str) -> int:
        """Check a frame number.

        Args:
            value: the frame.
            what: where it is, for the error text.

        Returns:
            The frame.

        Raises:
            BadRequest: not an integer or outside the task (or the job).
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise BadRequest(f"{what}: frame must be an integer")
        if not 0 <= value < self.size:
            raise BadRequest(f"{what}: frame {value} is outside the task (0..{self.size - 1})")
        if value not in self.frames:
            raise BadRequest(
                f"{what}: frame {value} is outside the job ({self.frames.start}..{self.frames.stop - 1})"
            )
        return value

    def label(self, value: Any, what: str, kind: str | None) -> dict[str, Any]:
        """Check a label id (and that the label allows ``kind``).

        Args:
            value: the label id.
            what: where it is, for the error text.
            kind: the shape type to place with it, ``"tag"`` for a tag, None to skip the type check.

        Returns:
            The label.

        Raises:
            BadRequest: unknown label or a type the label does not allow.
        """
        label = self.labels.get(value) if isinstance(value, int) and not isinstance(value, bool) else None
        if label is None:
            raise BadRequest(f"{what}: label_id {value!r} is not a label of this task")
        if kind is not None and label["type"] != "any" and label["type"] != kind:
            raise BadRequest(f"{what}: a '{label['type']}' label cannot have a '{kind}'")
        return label

    def attributes(self, value: Any, label: dict[str, Any], what: str) -> list[dict[str, Any]]:
        """Check the attribute values against the specs of the label.

        Args:
            value: the ``attributes`` list.
            label: the label of the object.
            what: where it is, for the error text.

        Returns:
            The attribute values.

        Raises:
            BadRequest: bad shape of the list or a spec that belongs to another label.
        """
        if not isinstance(value, list):
            raise BadRequest(f"{what}: attributes must be a list")
        specs = {a["id"] for a in label.get("attributes", [])}
        for item in value:
            if not isinstance(item, dict) or set(item) != {"spec_id", "value"}:
                raise BadRequest(f"{what}: an attribute is {{spec_id, value}}")
            if item["spec_id"] not in specs:
                raise BadRequest(
                    f"{what}: spec_id {item['spec_id']!r} is not an attribute of label {label['id']}"
                )
            if not isinstance(item["value"], str):
                raise BadRequest(f"{what}: attribute values are strings")
        return copy.deepcopy(value)

    def _common(self, raw: dict[str, Any], what: str) -> tuple[bool, int, int, str]:
        occluded = raw.get("occluded", False)
        z_order = raw.get("z_order", 0)
        group = raw.get("group", 0)
        source = raw.get("source", "manual")
        if not isinstance(occluded, bool):
            raise BadRequest(f"{what}: occluded must be a boolean")
        if isinstance(z_order, bool) or not isinstance(z_order, int):
            raise BadRequest(f"{what}: z_order must be an integer")
        if isinstance(group, bool) or not isinstance(group, int) or group < 0:
            raise BadRequest(f"{what}: group must be a non-negative integer")
        if source not in SOURCES:
            raise BadRequest(f"{what}: source must be one of {SOURCES}")
        return occluded, z_order, group, source

    def _rotation(self, raw: dict[str, Any], kind: str, what: str) -> float:
        rotation = _number(raw.get("rotation", 0), f"{what}.rotation")
        if not 0 <= rotation < 360:
            raise BadRequest(f"{what}: rotation must be in [0, 360)")
        if rotation and kind not in ("rectangle", "ellipse"):
            raise BadRequest(f"{what}: only rectangles and ellipses can be rotated")
        return rotation

    def shape(self, raw: Any, what: str) -> dict[str, Any]:
        """Validate and normalise one shape (with defaults filled in, as CVAT returns it).

        Args:
            raw: the shape received.
            what: where it is, for the error text.

        Returns:
            The normalised shape.

        Raises:
            BadRequest: anything CVAT would refuse.
        """
        if not isinstance(raw, dict):
            raise BadRequest(f"{what}: a shape is an object")
        self._fields(raw, SHAPE_FIELDS, what)
        for needed in ("type", "frame", "label_id", "points"):
            if needed not in raw:
                raise BadRequest(f"{what}: missing {needed}")
        kind = raw["type"]
        if kind not in SHAPE_TYPES:
            raise BadRequest(f"{what}: unknown shape type {kind!r}")
        frame = self.frame(raw["frame"], what)
        label = self.label(raw["label_id"], what, kind)
        occluded, z_order, group, source = self._common(raw, what)
        outside = raw.get("outside", False)
        if not isinstance(outside, bool):
            raise BadRequest(f"{what}: outside must be a boolean")
        out: dict[str, Any] = {
            "type": kind,
            "frame": frame,
            "label_id": label["id"],
            "points": _check_points(kind, raw["points"], what),
            "rotation": self._rotation(raw, kind, what),
            "occluded": occluded,
            "outside": outside,
            "z_order": z_order,
            "group": group,
            "source": source,
            "attributes": self.attributes(raw.get("attributes", []), label, what),
            "elements": self.elements(raw.get("elements", []), label, kind, frame, what),
        }
        if "id" in raw:
            out["id"] = raw["id"]
        return out

    def elements(
        self, raw: Any, label: dict[str, Any], kind: str, frame: int | None, what: str
    ) -> list[dict[str, Any]]:
        """Validate the parts of a skeleton (and refuse parts on any other shape).

        Args:
            raw: the ``elements`` list.
            label: the label of the skeleton.
            kind: the type of the parent shape.
            frame: the frame of the parent shape (None for a track).
            what: where it is, for the error text.

        Returns:
            The normalised parts.

        Raises:
            BadRequest: parts on a non-skeleton, or parts that do not fit the sublabels.
        """
        if not isinstance(raw, list):
            raise BadRequest(f"{what}: elements must be a list")
        if kind != "skeleton":
            if raw:
                raise BadRequest(f"{what}: only a skeleton has elements")
            return []
        if not raw:
            raise BadRequest(f"{what}: a skeleton needs its elements")
        subs = {s["id"] for s in label.get("sublabels", [])}
        parts: list[dict[str, Any]] = []
        for n, part in enumerate(raw):
            where = f"{what}.elements[{n}]"
            if not isinstance(part, dict):
                raise BadRequest(f"{where}: an element is an object")
            self._fields(part, ELEMENT_FIELDS | {"shapes", "group"}, where)
            if part.get("label_id") not in subs:
                raise BadRequest(
                    f"{where}: label_id {part.get('label_id')!r} is not a sublabel of the skeleton"
                )
            if "shapes" in part:  # an element of a track
                parts.append(self._track_element(part, where))
                continue
            sub = self.labels[part["label_id"]]
            shape = self.shape(
                {"type": "points", "frame": frame, **{k: v for k, v in part.items() if k != "type"}}, where
            )
            shape["label_id"] = sub["id"]
            parts.append(shape)
        return parts

    def _track_element(self, part: dict[str, Any], where: str) -> dict[str, Any]:
        sub = self.labels[part["label_id"]]
        shapes = self.tracked_shapes(part["shapes"], sub, "points", where)
        return {"label_id": sub["id"], "shapes": shapes, "attributes": [], "group": 0, "source": "manual"}

    def tag(self, raw: Any, what: str) -> dict[str, Any]:
        """Validate and normalise one tag.

        Args:
            raw: the tag received.
            what: where it is, for the error text.

        Returns:
            The normalised tag.

        Raises:
            BadRequest: anything CVAT would refuse.
        """
        if not isinstance(raw, dict):
            raise BadRequest(f"{what}: a tag is an object")
        self._fields(raw, TAG_FIELDS, what)
        for needed in ("frame", "label_id"):
            if needed not in raw:
                raise BadRequest(f"{what}: missing {needed}")
        frame = self.frame(raw["frame"], what)
        label = self.label(raw["label_id"], what, "tag")
        _, _, group, source = self._common(raw, what)
        out = {
            "frame": frame,
            "label_id": label["id"],
            "group": group,
            "source": source,
            "attributes": self.attributes(raw.get("attributes", []), label, what),
        }
        if "id" in raw:
            out["id"] = raw["id"]
        return out

    def tracked_shapes(
        self, raw: Any, label: dict[str, Any], kind: str | None, what: str
    ) -> list[dict[str, Any]]:
        """Validate the keyframes of a track.

        Args:
            raw: the ``shapes`` list of the track.
            label: the label of the track.
            kind: the shape type all keyframes must have (None: the type of the first one).
            what: where it is, for the error text.

        Returns:
            The normalised keyframes, in frame order.

        Raises:
            BadRequest: empty, mixed types, repeated or out-of-range frames, bad geometry.
        """
        if not isinstance(raw, list) or not raw:
            raise BadRequest(f"{what}: a track needs at least one shape")
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for n, item in enumerate(raw):
            where = f"{what}.shapes[{n}]"
            if not isinstance(item, dict):
                raise BadRequest(f"{where}: a tracked shape is an object")
            self._fields(item, TRACKED_FIELDS, where)
            for needed in ("type", "frame", "points"):
                if needed not in item:
                    raise BadRequest(f"{where}: missing {needed}")
            kind = kind or item["type"]
            if item["type"] != kind or kind not in SHAPE_TYPES:
                raise BadRequest(f"{where}: every shape of a track has the same type ({kind})")
            frame = self.frame(item["frame"], where)
            if frame in seen:
                raise BadRequest(f"{where}: two keyframes on frame {frame}")
            seen.add(frame)
            occluded, z_order, _, _ = self._common(item, where)
            outside = item.get("outside", False)
            if not isinstance(outside, bool):
                raise BadRequest(f"{where}: outside must be a boolean")
            shape = {
                "type": kind,
                "frame": frame,
                "points": _check_points(kind, item["points"], where),
                "rotation": self._rotation(item, kind, where),
                "occluded": occluded,
                "outside": outside,
                "z_order": z_order,
                "attributes": self.attributes(item.get("attributes", []), label, where),
            }
            if "keyframe" in item:
                shape["keyframe"] = bool(item["keyframe"])
            if "id" in item:
                shape["id"] = item["id"]
            out.append(shape)
        return sorted(out, key=lambda s: s["frame"])

    def track(self, raw: Any, what: str) -> dict[str, Any]:
        """Validate and normalise one track.

        Args:
            raw: the track received.
            what: where it is, for the error text.

        Returns:
            The normalised track.

        Raises:
            BadRequest: anything CVAT would refuse.
        """
        if not isinstance(raw, dict):
            raise BadRequest(f"{what}: a track is an object")
        self._fields(raw, TRACK_FIELDS, what)
        for needed in ("frame", "label_id", "shapes"):
            if needed not in raw:
                raise BadRequest(f"{what}: missing {needed}")
        first = self.frame(raw["frame"], what)
        label = self.label(raw["label_id"], what, None)
        shapes = self.tracked_shapes(raw["shapes"], label, None, what)
        if label["type"] not in ("any", shapes[0]["type"]):
            raise BadRequest(f"{what}: a '{label['type']}' label cannot have a '{shapes[0]['type']}' track")
        if first != shapes[0]["frame"]:
            raise BadRequest(
                f"{what}: frame {first} is not the frame of the first keyframe {shapes[0]['frame']}"
            )
        _, _, group, source = self._common(raw, what)
        out = {
            "frame": first,
            "label_id": label["id"],
            "group": group,
            "source": source,
            "shapes": shapes,
            "attributes": self.attributes(raw.get("attributes", []), label, what),
            "elements": [],
        }
        if shapes[0]["type"] == "skeleton":
            out["elements"] = self.elements(raw.get("elements", []), label, "skeleton", None, what)
        elif raw.get("elements"):
            raise BadRequest(f"{what}: only a skeleton has elements")
        if "id" in raw:
            out["id"] = raw["id"]
        return out


@dataclass
class _Task:
    id: int
    name: str
    project_id: int
    segment_size: int
    overlap: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    size: int = 0
    media: dict[int, bytes] = field(default_factory=dict)
    deleted: list[int] = field(default_factory=list)
    start_frame: int = 0
    stop_frame: int = 0
    frame_filter: str = ""
    video: bool = False
    has_data: bool = False
    jobs: list[int] = field(default_factory=list)


class FakeCvatServer:
    """A stateful CVAT server in memory, usable as ``requests.Session``."""

    def __init__(self) -> None:
        self.projects: dict[int, dict[str, Any]] = {}
        self.labels: dict[int, dict[str, Any]] = {}
        self.tasks: dict[int, _Task] = {}
        self.jobs: dict[int, dict[str, Any]] = {}
        self.anns: dict[int, dict[str, Any]] = {}
        self.requests: list[Request] = []
        self._ids = {"object": 100, "shape": 0, "track": 0, "tag": 0, "attr": 0}
        self.auth: Any = None
        self.polls: dict[str, int] = {}
        self.fail: dict[tuple[str, str], int] = {}
        self.after: dict[tuple[str, str], Callable[[], None]] = {}

    # -- plumbing -------------------------------------------------------------------------------
    def install(self, monkeypatch: Any) -> FakeCvatServer:
        """Make the adapter talk to this server (``requests.Session`` and its sleeps are replaced).

        Args:
            monkeypatch: the pytest ``monkeypatch`` fixture.

        Returns:
            This server.
        """
        monkeypatch.setattr(requests, "Session", lambda: self)
        monkeypatch.setattr(cvat_remote, "_sleep", lambda seconds: None)
        return self

    def get(self, url: str, **kw: Any) -> Response:
        """``requests.Session.get``.

        Args:
            url: the full URL.
            **kw: ``params``, ``auth``, ``timeout`` as ``requests`` takes them.

        Returns:
            The answer.
        """
        return self.request("GET", url, **kw)

    def request(self, method: str, url: str, **kw: Any) -> Response:
        """``requests.Session.request``: route the call to the handler of the API.

        Args:
            method: HTTP method.
            url: the full URL (``https://cvat.test/api/...``).
            **kw: ``params``, ``json``, ``data``, ``files``, ``auth``, ``timeout``.

        Returns:
            The answer (400 for a validation problem, 404 for an unknown path).
        """
        parts = urlsplit(url)
        params: dict[str, Any] = {**dict(parse_qsl(parts.query)), **(kw.get("params") or {})}
        body = kw.get("json")
        if body is None and kw.get("data") is not None:
            body = dict(kw["data"])
        if kw.get("files"):
            body = {**(body or {}), "_files": [f[1][0] for f in kw["files"]]}
        self.auth = kw.get("auth")
        method = method.upper()
        self.requests.append(Request(method, parts.path, params, copy.deepcopy(body)))
        for (m, prefix), status in self.fail.items():
            if m == method and parts.path.startswith(prefix):
                return Response(status, {})
        for (m, pattern), handler in self._routes():
            hit = re.fullmatch(pattern, parts.path) if m == method else None
            if hit:
                try:
                    answer = handler(params, kw, *hit.groups())
                except BadRequest as err:
                    return Response(400, [str(err)])
                for (hm, prefix), hook in list(self.after.items()):
                    if hm == method and parts.path.startswith(prefix):
                        hook()  # another person acts right after this answer (a concurrent edit)
                return answer
        return Response(404, {"detail": "Not found."})

    def calls(self, method: str, prefix: str) -> list[Request]:
        """The recorded requests of one method whose path starts with ``prefix``.

        Args:
            method: HTTP method.
            prefix: start of the path.

        Returns:
            The matching requests, oldest first.
        """
        return [r for r in self.requests if r.method == method and r.path.startswith(prefix)]

    def _routes(self) -> list[tuple[tuple[str, str], Callable[..., Response]]]:
        return [
            (("GET", r"/api/users/self"), lambda p, kw: Response(200, {"id": 1, "username": "ana"})),
            (("GET", r"/api/projects"), self._list_projects),
            (("POST", r"/api/projects"), self._post_project),
            (("GET", r"/api/projects/(\d+)"), self._get_project),
            (("PATCH", r"/api/projects/(\d+)"), self._patch_project),
            (("GET", r"/api/labels"), self._list_labels),
            (("GET", r"/api/tasks"), self._list_tasks),
            (("POST", r"/api/tasks"), self._post_task),
            (("GET", r"/api/tasks/(\d+)"), self._get_task),
            (("DELETE", r"/api/tasks/(\d+)"), self._delete_task),
            (("POST", r"/api/tasks/(\d+)/data"), self._post_data),
            (("GET", r"/api/tasks/(\d+)/data"), self._get_frame),
            (("GET", r"/api/tasks/(\d+)/data/meta"), self._get_meta),
            (("PATCH", r"/api/tasks/(\d+)/data/meta"), self._patch_meta),
            (("GET", r"/api/tasks/(\d+)/status"), lambda p, kw, t: Response(200, {"state": "Finished"})),
            (("GET", r"/api/requests/(\w+)"), self._get_request),
            (("GET", r"/api/jobs"), self._list_jobs),
            (("GET", r"/api/jobs/(\d+)"), self._get_job),
            (("GET", r"/api/(tasks|jobs)/(\d+)/annotations"), self._get_annotations),
            (("PUT", r"/api/(tasks|jobs)/(\d+)/annotations"), self._put_annotations),
            (("PATCH", r"/api/(tasks|jobs)/(\d+)/annotations"), self._patch_annotations),
            (("DELETE", r"/api/(tasks|jobs)/(\d+)/annotations"), self._delete_annotations),
        ]

    def _new(self, kind: str = "object") -> int:
        self._ids[kind] += 1
        return self._ids[kind]

    def _page(self, rows: list[dict[str, Any]], params: dict[str, Any], path: str) -> Response:
        size, page = int(params.get("page_size", 10)), int(params.get("page", 1))
        chunk = rows[(page - 1) * size : page * size]
        more = page * size < len(rows)
        nxt = f"https://{HOST}{path}?page={page + 1}&page_size={size}" if more else None
        return Response(
            200, {"count": len(rows), "next": nxt, "previous": None, "results": copy.deepcopy(chunk)}
        )

    # -- projects and labels ----------------------------------------------------------------------
    def add_project(self, name: str, labels: list[str | dict[str, Any]]) -> dict[str, Any]:
        """Create a project with labels (a name, or a label spec dict with ``name``, ``type``, ...).

        Args:
            name: the project name.
            labels: the labels.

        Returns:
            The project as the API shows it.
        """
        pid = self._new()
        self.projects[pid] = {"id": pid, "name": name}
        for spec in labels:
            self._add_label(pid, spec if isinstance(spec, dict) else {"name": spec})
        return copy.deepcopy(self.projects[pid])

    def _add_label(self, pid: int, spec: dict[str, Any], parent: int | None = None) -> dict[str, Any]:
        allowed = {"id", "name", "color", "type", "attributes", "sublabels", "svg", "has_parent"}
        _unknown(spec, allowed, "label")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise BadRequest("label: name is required")
        kind = spec.get("type", "any")
        if kind not in LABEL_TYPES:
            raise BadRequest(f"label: unknown type {kind!r}")
        if any(
            x["name"] == spec["name"] and x.get("parent_id") == parent and x["project_id"] == pid
            for x in self.labels.values()
        ):
            raise BadRequest(f"label: the name '{spec['name']}' is used twice")
        lid = self._new()
        label: dict[str, Any] = {
            "id": lid,
            "name": spec["name"],
            "color": spec.get("color", "#aabbcc"),
            "type": kind,
            "attributes": [],
            "sublabels": [],
            "project_id": pid,
            "task_id": None,
            "parent_id": parent,
        }
        for attr in spec.get("attributes", []):
            label["attributes"].append(
                {"id": self._new("attr"), **{k: v for k, v in attr.items() if k != "id"}}
            )
        self.labels[lid] = label
        for sub in spec.get("sublabels", []):
            label["sublabels"].append(self._add_label(pid, sub, lid))
        if kind == "skeleton" and not spec.get("sublabels"):
            raise BadRequest("label: a skeleton needs sublabels")
        return label

    def label_id(self, project_id: int, name: str, parent: str | None = None) -> int:
        """Id of a label (or of a sublabel of the skeleton called ``parent``).

        Args:
            project_id: the project.
            name: the label name.
            parent: the skeleton the sublabel belongs to.

        Returns:
            The label id.
        """
        pid = next(
            (x["id"] for x in self.labels.values() if x["project_id"] == project_id and x["name"] == parent),
            None,
        )
        return next(
            x["id"]
            for x in self.labels.values()
            if x["project_id"] == project_id
            and x["name"] == name
            and x.get("parent_id") == (pid if parent else None)
        )

    def add_attribute(self, label_id: int, name: str) -> int:
        """Give a label an attribute spec (a text attribute).

        Args:
            label_id: the label.
            name: the attribute name.

        Returns:
            The attribute spec id (``spec_id`` of the values).
        """
        spec = {
            "id": self._new("attr"),
            "name": name,
            "mutable": False,
            "input_type": "text",
            "default_value": "",
            "values": [],
        }
        self.labels[label_id]["attributes"].append(spec)
        return int(spec["id"])

    def rename_label(self, label_id: int, name: str) -> None:
        """Rename a label (as a person does in the CVAT interface).

        Args:
            label_id: the label.
            name: the new name.
        """
        self.labels[label_id]["name"] = name

    def label_names(self, project_id: int) -> list[str]:
        """Names of the top-level labels of a project, in id order.

        Args:
            project_id: the project.

        Returns:
            The names.
        """
        found = [x for x in self.labels.values() if x["project_id"] == project_id and not x.get("parent_id")]
        return [x["name"] for x in sorted(found, key=lambda x: x["id"])]

    def _list_projects(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        rows = [p for p in self.projects.values() if "name" not in params or params["name"] in p["name"]]
        return self._page(sorted(rows, key=lambda p: p["id"]), params, "/api/projects")

    def _post_project(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        body = kw.get("json") or {}
        _unknown(body, {"name", "labels", "bug_tracker", "target_storage", "source_storage"}, "project")
        if not body.get("name"):
            raise BadRequest("project: name is required")
        return Response(201, self.add_project(body["name"], body.get("labels", [])))

    def _get_project(self, params: dict[str, Any], kw: dict[str, Any], pid: str) -> Response:
        project = self.projects.get(int(pid))
        return Response(200, copy.deepcopy(project)) if project else Response(404, {"detail": "Not found."})

    def _patch_project(self, params: dict[str, Any], kw: dict[str, Any], pid: str) -> Response:
        project = self.projects.get(int(pid))
        if project is None:
            return Response(404, {"detail": "Not found."})
        body = kw.get("json") or {}
        _unknown(body, {"name", "labels", "bug_tracker"}, "project")
        for spec in body.get("labels", []):
            if "id" in spec:  # an update of an existing label
                label = self.labels.get(spec["id"])
                if label is None or label["project_id"] != project["id"]:
                    raise BadRequest(f"label: id {spec['id']} is not a label of the project")
                label["name"] = spec.get("name", label["name"])
                for attr in spec.get("attributes", []):  # attributes without an id are added
                    if "id" not in attr:
                        label["attributes"].append({"id": self._new("attr"), **attr})
            else:
                self._add_label(project["id"], spec)
        if "name" in body:
            project["name"] = body["name"]
        return Response(200, copy.deepcopy(project))

    def _list_labels(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        if "project_id" not in params:
            raise BadRequest("labels: one of project_id, task_id, job_id is required")
        pid = int(params["project_id"])
        rows = sorted((x for x in self.labels.values() if x["project_id"] == pid), key=lambda x: x["id"])
        return self._page(
            [{k: v for k, v in x.items() if k != "sublabels"} for x in rows], params, "/api/labels"
        )

    # -- tasks -------------------------------------------------------------------------------------
    def add_task(
        self,
        project_id: int,
        name: str,
        frames: list[tuple[str, int, int]],
        segment_size: int = 0,
        video: dict[str, int] | None = None,
    ) -> int:
        """Create a task that already has its data (a person created it in CVAT).

        Args:
            project_id: the project.
            name: the task name.
            frames: ``(file name, width, height)`` per frame (image tasks); for a video task the
                single entry is the video and the frame count comes from ``video["size"]``.
            segment_size: frames per job (0: one job).
            video: ``{"size", "start", "step"}`` to make a video task (CVAT lists one entry in the
                meta while the task has ``size`` frames).

        Returns:
            The task id.
        """
        task = _Task(self._new(), name, project_id, segment_size, 0)
        self.tasks[task.id] = task
        if video:
            fname, width, height = frames[0]
            step = int(video.get("step", 1))
            start = int(video.get("start", 0))
            task.frames = [{"name": fname, "width": width, "height": height, "related_files": 0}]
            task.size = int(video["size"])
            task.video, task.start_frame = True, start
            task.stop_frame = start + (task.size - 1) * step
            task.frame_filter = f"step={step}" if step > 1 else ""
        else:
            task.frames = [{"name": n, "width": w, "height": h, "related_files": 0} for n, w, h in frames]
            task.size = len(frames)
            task.stop_frame = max(task.size - 1, 0)
        for n in range(task.size):
            w, h = (frames[0][1], frames[0][2]) if video else (frames[n][1], frames[n][2])
            task.media[n] = self.png(w, h)
        self._finish_data(task)
        return task.id

    @staticmethod
    def png(width: int, height: int) -> bytes:
        """A small PNG of the given size.

        Args:
            width: pixels.
            height: pixels.

        Returns:
            The file bytes.
        """
        out = io.BytesIO()
        Image.new("RGB", (width, height), (90, 90, 90)).save(out, format="PNG")
        return out.getvalue()

    def _finish_data(self, task: _Task) -> None:
        task.has_data = True
        self.anns[task.id] = {"version": 0, "tags": [], "shapes": [], "tracks": []}
        step = task.segment_size or task.size or 1
        for start in range(0, task.size, step):
            self.add_job(task.id, start, min(start + step, task.size) - 1)

    def add_job(self, task_id: int, start: int, stop: int, kind: str = "annotation") -> int:
        """Add a job over task frames ``start``..``stop`` (inclusive).

        Args:
            task_id: the task.
            start: first TASK frame.
            stop: last TASK frame.
            kind: ``annotation`` or ``ground_truth``.

        Returns:
            The job id.
        """
        task = self.tasks[task_id]
        jid = self._new()
        self.jobs[jid] = {
            "id": jid,
            "task_id": task_id,
            "project_id": task.project_id,
            "start_frame": start,
            "stop_frame": stop,
            "frame_count": stop - start + 1,
            "state": "new",
            "stage": "annotation",
            "type": kind,
            "assignee": None,
            "mode": "interpolation" if task.video else "annotation",
        }
        task.jobs.append(jid)
        return jid

    def _task_json(self, task: _Task) -> dict[str, Any]:
        return {
            "id": task.id,
            "name": task.name,
            "project_id": task.project_id,
            "size": task.size,
            "segment_size": task.segment_size,
            "overlap": task.overlap,
            "mode": "interpolation" if task.video else ("annotation" if task.has_data else ""),
            "status": "annotation",
            "jobs": {"count": len(task.jobs)},
        }

    def _list_tasks(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        rows = [
            t
            for t in self.tasks.values()
            if "project_id" not in params or t.project_id == int(params["project_id"])
        ]
        return self._page(
            [self._task_json(t) for t in sorted(rows, key=lambda t: t.id)], params, "/api/tasks"
        )

    def _post_task(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        body = kw.get("json") or {}
        _unknown(
            body, {"name", "project_id", "segment_size", "overlap", "subset", "bug_tracker", "labels"}, "task"
        )
        if not body.get("name"):
            raise BadRequest("task: name is required")
        if body.get("labels") and body.get("project_id"):
            raise BadRequest("task: labels cannot be given together with a project")
        if body.get("project_id") not in self.projects:
            raise BadRequest(f"task: project {body.get('project_id')!r} does not exist")
        segment, overlap = int(body.get("segment_size", 0)), int(body.get("overlap", 0))
        if segment < 0 or (segment and not 0 <= overlap < segment):
            raise BadRequest("task: overlap must be smaller than segment_size")
        task = _Task(self._new(), body["name"], int(body["project_id"]), segment, overlap)
        self.tasks[task.id] = task
        return Response(201, self._task_json(task))

    def _get_task(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.get(int(tid))
        return Response(200, self._task_json(task)) if task else Response(404, {"detail": "Not found."})

    def _delete_task(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.pop(int(tid), None)
        if task is None:
            return Response(404, {"detail": "Not found."})
        for jid in task.jobs:
            self.jobs.pop(jid, None)
        self.anns.pop(task.id, None)
        return Response(204)

    def _post_data(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.get(int(tid))
        if task is None:
            return Response(404, {"detail": "Not found."})
        if task.has_data:
            raise BadRequest("data: the task already has data")
        data = kw.get("data") or {}
        _unknown(
            data, {"image_quality", "sorting_method", "start_frame", "stop_frame", "frame_filter"}, "data"
        )
        if "image_quality" not in data:
            raise BadRequest("data: image_quality is required")
        files = kw.get("files") or []
        if not files:
            raise BadRequest("data: no files")
        for number, (field_name, (fname, handle, _mime)) in enumerate(files):
            if field_name != f"client_files[{number}]":
                raise BadRequest(f"data: unexpected field {field_name}")
            raw = handle.read()
            try:
                width, height = Image.open(io.BytesIO(raw)).size
            except Exception:  # noqa: BLE001 - not an image
                raise BadRequest(f"data: {fname} is not an image") from None
            task.frames.append({"name": fname, "width": width, "height": height, "related_files": 0})
            task.media[number] = raw
        task.size = len(task.frames)
        task.stop_frame = task.size - 1
        self._finish_data(task)
        self.polls[f"rq{task.id}"] = 0
        return Response(202, {"rq_id": f"rq{task.id}"})

    def _get_request(self, params: dict[str, Any], kw: dict[str, Any], rq: str) -> Response:
        if rq not in self.polls:
            return Response(404, {"detail": "Not found."})
        self.polls[rq] += 1
        return Response(200, {"status": "finished" if self.polls[rq] >= 2 else "started"})

    def _get_frame(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.get(int(tid))
        if task is None:
            return Response(404, {"detail": "Not found."})
        if params.get("type") != "frame" or int(params.get("number", -1)) not in task.media:
            raise BadRequest("data: type=frame and a frame number of the task are required")
        return Response(200, None, raw=task.media[int(params["number"])])

    def _meta(self, task: _Task) -> dict[str, Any]:
        return {
            "chunk_size": 72,
            "size": task.size,
            "image_quality": 95,
            "start_frame": task.start_frame,
            "stop_frame": task.stop_frame,
            "frame_filter": task.frame_filter,
            "frames": copy.deepcopy(task.frames),
            "deleted_frames": list(task.deleted),
        }

    def _get_meta(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.get(int(tid))
        return Response(200, self._meta(task)) if task else Response(404, {"detail": "Not found."})

    def _patch_meta(self, params: dict[str, Any], kw: dict[str, Any], tid: str) -> Response:
        task = self.tasks.get(int(tid))
        if task is None:
            return Response(404, {"detail": "Not found."})
        body = kw.get("json") or {}
        _unknown(body, {"deleted_frames"}, "meta")
        frames = body.get("deleted_frames")
        if not isinstance(frames, list) or any(isinstance(f, bool) or not isinstance(f, int) for f in frames):
            raise BadRequest("meta: deleted_frames is a list of integers")
        if any(not 0 <= f < task.size for f in frames):
            raise BadRequest(f"meta: deleted_frames must be task frames (0..{task.size - 1})")
        task.deleted = sorted(set(frames))
        return Response(200, self._meta(task))

    def delete_frames(self, task_id: int, frames: list[int]) -> None:
        """Mark frames of a task deleted (a person does it in CVAT).

        Args:
            task_id: the task.
            frames: task frames.
        """
        task = self.tasks[task_id]
        task.deleted = sorted(set(task.deleted) | set(frames))

    def jobs_of(self, task_id: int) -> list[dict[str, Any]]:
        """The jobs of a task, in id order.

        Args:
            task_id: the task.

        Returns:
            A copy of the job objects.
        """
        return [copy.deepcopy(self.jobs[j]) for j in self.tasks[task_id].jobs]

    def _list_jobs(self, params: dict[str, Any], kw: dict[str, Any]) -> Response:
        rows = [
            j for j in self.jobs.values() if "task_id" not in params or j["task_id"] == int(params["task_id"])
        ]
        return self._page(sorted(rows, key=lambda j: j["id"]), params, "/api/jobs")

    def _get_job(self, params: dict[str, Any], kw: dict[str, Any], jid: str) -> Response:
        job = self.jobs.get(int(jid))
        return Response(200, copy.deepcopy(job)) if job else Response(404, {"detail": "Not found."})

    # -- annotations ---------------------------------------------------------------------------------
    def _scope(self, what: str, number: str) -> tuple[_Task, range]:
        """The task and the task frames an annotations endpoint covers (the task, or one job)."""
        if what == "tasks":
            task = self.tasks.get(int(number))
            if task is None:
                raise LookupError
            return task, range(0, task.size)
        job = self.jobs.get(int(number))
        if job is None:
            raise LookupError
        return self.tasks[job["task_id"]], range(job["start_frame"], job["stop_frame"] + 1)

    def _validator(self, task: _Task, frames: range, ids_ok: bool) -> Validator:
        labels = {i: x for i, x in self.labels.items() if x["project_id"] == task.project_id}
        return Validator(labels, task.size, frames, ids_ok)

    def _view(self, task: _Task, frames: range) -> dict[str, Any]:
        """The annotations of the task, or the part of them inside ``frames`` (a job)."""
        ann = copy.deepcopy(self.anns[task.id])
        if frames.start != 0 or frames.stop != task.size:
            ann["shapes"] = [s for s in ann["shapes"] if s["frame"] in frames]
            ann["tags"] = [t for t in ann["tags"] if t["frame"] in frames]
            tracks = []
            for tr in ann["tracks"]:
                kept = [s for s in tr["shapes"] if s["frame"] in frames]
                if kept:
                    tracks.append({**tr, "frame": kept[0]["frame"], "shapes": kept})
            ann["tracks"] = tracks
        return ann

    def _check_body(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise BadRequest("annotations: the body is an object")
        _unknown(body, ANNOTATION_KEYS, "annotations")
        for key in ("tags", "shapes", "tracks"):
            if not isinstance(body.get(key, []), list):
                raise BadRequest(f"annotations: {key} must be a list")
        return body

    def _normalise(self, check: Validator, body: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        return {
            "tags": [check.tag(x, f"tags[{n}]") for n, x in enumerate(body.get("tags", []))],
            "shapes": [check.shape(x, f"shapes[{n}]") for n, x in enumerate(body.get("shapes", []))],
            "tracks": [check.track(x, f"tracks[{n}]") for n, x in enumerate(body.get("tracks", []))],
        }

    def _assign(self, item: dict[str, Any], kind: str) -> dict[str, Any]:
        """Give an object (and the keyframes and parts it holds) fresh ids, as a stored object has."""
        item["id"] = self._new(kind)
        for sh in item.get("shapes", []) if kind == "track" else []:
            sh["id"] = self._new("shape")
        for part in item.get("elements", []):
            self._assign(part, "shape")
        return item

    def _get_annotations(
        self, params: dict[str, Any], kw: dict[str, Any], what: str, number: str
    ) -> Response:
        try:
            task, frames = self._scope(what, number)
        except LookupError:
            return Response(404, {"detail": "Not found."})
        return Response(200, self._view(task, frames))

    def _put_annotations(
        self, params: dict[str, Any], kw: dict[str, Any], what: str, number: str
    ) -> Response:
        try:
            task, frames = self._scope(what, number)
        except LookupError:
            return Response(404, {"detail": "Not found."})
        body = self._check_body(kw.get("json"))
        new = self._normalise(self._validator(task, frames, False), body)
        ann = self.anns[task.id]
        for key in ("tags", "shapes", "tracks"):
            outside = [x for x in ann[key] if not self._inside(x, key, frames)]  # a job keeps the others
            ann[key] = outside + [self._assign(x, key[:-1]) for x in new[key]]
        ann["version"] += 1
        return Response(200, self._view(task, frames))

    @staticmethod
    def _inside(item: dict[str, Any], key: str, frames: range) -> bool:
        if key == "tracks":
            return any(s["frame"] in frames for s in item["shapes"])
        return item["frame"] in frames

    def _patch_annotations(
        self, params: dict[str, Any], kw: dict[str, Any], what: str, number: str
    ) -> Response:
        try:
            task, frames = self._scope(what, number)
        except LookupError:
            return Response(404, {"detail": "Not found."})
        action = params.get("action")
        if action not in ("create", "update", "delete"):
            raise BadRequest("action must be create, update or delete")
        body = self._check_body(kw.get("json"))
        ann = self.anns[task.id]
        new = self._normalise(self._validator(task, frames, action != "create"), body)
        for key in ("tags", "shapes", "tracks"):
            self._patch_kind(ann, key, action, new[key], frames)
        ann["version"] += 1
        return Response(200, {**new, "version": ann["version"]})

    def _patch_kind(
        self, ann: dict[str, Any], key: str, action: str, items: list[dict[str, Any]], frames: range
    ) -> None:
        stored = ann[key]
        by_id = {x["id"]: n for n, x in enumerate(stored)}
        if action == "create":
            stored.extend(self._assign(x, key[:-1]) for x in items)
            return
        for item in items:
            if item.get("id") not in by_id:
                raise BadRequest(f"{key}: id {item.get('id')!r} does not exist")
        if action == "delete":
            gone = {x["id"] for x in items}
            ann[key] = [x for x in stored if x["id"] not in gone]
            return
        for item in items:
            current = by_id[item["id"]]
            if not self._inside(stored[current], key, frames):
                raise BadRequest(f"{key}: id {item['id']} is not in this job")
            stored[current] = item

    def _delete_annotations(
        self, params: dict[str, Any], kw: dict[str, Any], what: str, number: str
    ) -> Response:
        try:
            task, frames = self._scope(what, number)
        except LookupError:
            return Response(404, {"detail": "Not found."})
        ann = self.anns[task.id]
        for key in ("tags", "shapes", "tracks"):
            ann[key] = [x for x in ann[key] if not self._inside(x, key, frames)]
        ann["version"] += 1
        return Response(204)

    # -- direct state changes (another person working in CVAT) --------------------------------------
    def annotations(self, task_id: int) -> dict[str, Any]:
        """The annotations of a task as ``GET`` shows them.

        Args:
            task_id: the task.

        Returns:
            A deep copy: ``{version, tags, shapes, tracks}``.
        """
        return copy.deepcopy(self.anns[task_id])

    def put(self, task_id: int, body: dict[str, Any]) -> None:
        """Replace all the annotations of a task through the strict validation (ids are assigned).

        Args:
            task_id: the task.
            body: ``{tags, shapes, tracks}`` as a ``PUT`` takes them.

        Raises:
            BadRequest: the body is refused (a test mistake).
        """
        task = self.tasks[task_id]
        new = self._normalise(self._validator(task, range(0, task.size), False), self._check_body(body))
        ann = self.anns[task_id]
        for key in ("tags", "shapes", "tracks"):
            ann[key] = [self._assign(x, key[:-1]) for x in new[key]]
        ann["version"] += 1

    def add_shape(self, task_id: int, shape: dict[str, Any]) -> int:
        """Draw a new shape (``PATCH ?action=create`` semantics).

        Args:
            task_id: the task.
            shape: the shape (``type``, ``frame``, ``label_id``, ``points``, ...).

        Returns:
            The new shape id.
        """
        task = self.tasks[task_id]
        item = self._validator(task, range(0, task.size), False).shape(shape, "shape")
        ann = self.anns[task_id]
        ann["shapes"].append(self._assign(item, "shape"))
        ann["version"] += 1
        return int(item["id"])

    def add_tag(self, task_id: int, frame: int, label_id: int) -> int:
        """Add a tag.

        Args:
            task_id: the task.
            frame: the task frame.
            label_id: the label.

        Returns:
            The new tag id.
        """
        task = self.tasks[task_id]
        item = self._validator(task, range(0, task.size), False).tag(
            {"frame": frame, "label_id": label_id}, "tag"
        )
        ann = self.anns[task_id]
        ann["tags"].append(self._assign(item, "tag"))
        ann["version"] += 1
        return int(item["id"])

    def shape(self, task_id: int, shape_id: int) -> dict[str, Any]:
        """A stored shape (live object: tests may read it).

        Args:
            task_id: the task.
            shape_id: the shape id.

        Returns:
            The stored shape.
        """
        return next(s for s in self.anns[task_id]["shapes"] if s["id"] == shape_id)

    def edit_shape(self, task_id: int, shape_id: int, **changes: Any) -> None:
        """Change fields of a stored shape, validated like an update.

        Args:
            task_id: the task.
            shape_id: the shape id.
            **changes: the new field values (``points=[...]``, ``label_id=...``, ``rotation=30`` ...).
        """
        task = self.tasks[task_id]
        ann = self.anns[task_id]
        current = self.shape(task_id, shape_id)
        merged = {**current, **changes}
        item = self._validator(task, range(0, task.size), True).shape(merged, "shape")
        ann["shapes"][ann["shapes"].index(current)] = item
        ann["version"] += 1

    def delete_shape(self, task_id: int, shape_id: int) -> None:
        """Remove a shape.

        Args:
            task_id: the task.
            shape_id: the shape id.
        """
        ann = self.anns[task_id]
        ann["shapes"] = [s for s in ann["shapes"] if s["id"] != shape_id]
        ann["version"] += 1

    def edit_track(self, task_id: int, track_id: int, frame: int, **changes: Any) -> None:
        """Change (or add) the keyframe of a track on ``frame``, as the CVAT editor does.

        Args:
            task_id: the task.
            track_id: the track id.
            frame: the task frame of the keyframe.
            **changes: the keyframe fields to set (``points``, ``outside``, ``occluded``, ``rotation`` ...).
        """
        task = self.tasks[task_id]
        ann = self.anns[task_id]
        track = next(t for t in ann["tracks"] if t["id"] == track_id)
        keys = {s["frame"]: s for s in track["shapes"]}
        base = keys.get(frame) or {
            "type": track["shapes"][0]["type"],
            "frame": frame,
            "points": track["shapes"][0]["points"],
        }
        keys[frame] = {**base, **changes, "frame": frame}
        raw = {**track, "shapes": [keys[f] for f in sorted(keys)], "frame": min(keys)}
        item = self._validator(task, range(0, task.size), True).track(raw, "track")
        ann["tracks"][ann["tracks"].index(track)] = item
        ann["version"] += 1
