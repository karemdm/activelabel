"""Label Studio JSON export (list of tasks).

Supported results: ``rectanglelabels`` (rotated boxes become their enclosing box), ``polygonlabels``,
``keypointlabels`` (all points of an image form one pose object of class ``object``, ordered by the
keypoint names in order of first use) and ``choices`` (image classification). Values are percentages.
The most recent non-cancelled annotation of a task is used; tasks without one are ignored.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from activelabel.annotations.context import ParseContext, bbox_of_points
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label

UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
POSE_CLASS = "object"


def image_candidates(value: str) -> list[str]:
    """Names to try for a ``data.image`` value, best first.

    Handles ``/data/upload/1/1a2b3c4d-name.jpg`` (upload prefix removed),
    ``/data/local-files/?d=dir/name.jpg``
    (the ``d`` path) and URLs with a query string.
    """
    value = value.strip()
    if "?d=" in value:
        path = unquote(value.split("?d=", 1)[1].split("&", 1)[0])
    else:
        path = unquote(value.split("?", 1)[0].split("#", 1)[0])
    path = path.replace("\\", "/")
    base = path.rsplit("/", 1)[-1]
    stripped = UPLOAD_PREFIX.sub("", base)
    out = [path] if "?d=" in value and "/" in path else []
    out += [stripped, base]
    return list(dict.fromkeys(out))


def _task_image(task: dict[str, Any]) -> str:
    data = task.get("data")
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("image"), str):
        return str(data["image"])
    for value in data.values():
        if isinstance(value, str) and value.lower().split("?")[0].endswith(IMAGE_EXT):
            return value
    return ""


def _rect_box(value: dict[str, Any], width: float, height: float) -> tuple[float, float, float, float]:
    """Relative-percent rectangle to ``(x, y, w, h)`` in pixels, enclosing it when rotated."""
    x, y, w, h = (
        float(value[k]) * s / 100
        for k, s in (("x", width), ("y", height), ("width", width), ("height", height))
    )
    angle = math.radians(float(value.get("rotation") or 0))
    if not angle:
        return x, y, w, h
    cos, sin = math.cos(angle), math.sin(angle)
    corners = [
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))
    ]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def parse_labelstudio(ctx: ParseContext, files: list[Path]) -> None:
    """Read Label Studio JSON exports into ``ctx``."""
    pending: list[tuple[ImageRecord, dict[str, tuple[float, float]]]] = []
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, list) or not all(isinstance(t, dict) for t in data):
                raise ValueError("not a Label Studio export")
            for task in data:
                _task(ctx, task, pending)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)
    for rec, pts in pending:
        names = ctx.keypoint_names
        points = [[*pts[n], 2.0] if n in pts else [0.0, 0.0, 0.0] for n in names]
        ctx.add_label(
            rec, Label(POSE_CLASS, bbox=bbox_of_points(p for p in points if p[2]), keypoints=points)
        )


def _task(
    ctx: ParseContext, task: dict[str, Any], pending: list[tuple[ImageRecord, dict[str, tuple[float, float]]]]
) -> None:
    annotations = [
        a for a in task.get("annotations") or [] if isinstance(a, dict) and not a.get("was_cancelled")
    ]
    if not annotations:
        return
    value = _task_image(task)
    rec = ctx.match(*image_candidates(value), example=value) if value else None
    if value == "":
        ctx.problem("invalid_annotation", f"task {task.get('id')}: no image")
    if rec is None or not ctx.claim(rec, value):
        return
    pts: dict[str, tuple[float, float]] = {}
    for res in annotations[-1].get("result") or []:
        kind, val = res.get("type"), res.get("value") or {}
        width = float(res.get("original_width") or rec.width)
        height = float(res.get("original_height") or rec.height)
        where = rec.name
        if kind == "rectanglelabels" and val.get("rectanglelabels"):
            x, y, w, h = _rect_box(val, width, height)
            box = ctx.box(x, y, w, h, width, height, where)
            if box:
                ctx.add_label(rec, Label(str(val["rectanglelabels"][0]), bbox=box))
        elif kind == "polygonlabels" and val.get("polygonlabels"):
            raw = [(float(px) * width / 100, float(py) * height / 100) for px, py in val.get("points") or []]
            ring = ctx.ring(raw, width, height, where)
            if ring:
                ctx.add_label(
                    rec, Label(str(val["polygonlabels"][0]), bbox=bbox_of_points(ring), polygons=[ring])
                )
        elif kind == "keypointlabels" and val.get("keypointlabels"):
            name = str(val["keypointlabels"][0])
            if name not in ctx.keypoint_names:
                ctx.keypoint_names.append(name)
            pts[name] = (min(1.0, max(0.0, float(val["x"]) / 100)), min(1.0, max(0.0, float(val["y"]) / 100)))
        elif kind == "choices" and val.get("choices"):
            for choice in val["choices"]:
                ctx.add_label(rec, Label(str(choice)))
        elif kind in ("brushlabels", "ellipselabels", "polylinelabels"):
            ctx.problem("unsupported_shape", f"{where}: {kind}")
    if pts:
        ctx.add_class(POSE_CLASS)
        pending.append((rec, pts))
