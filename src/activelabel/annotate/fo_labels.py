"""FiftyOne labels -> internal labels (the inverse of ``fiftyone_bridge._ground_truth``).

FiftyOne is not imported here: labels are recognised by their class name, so the module is
cheap to import and easy to test with stand-in objects.
"""

from __future__ import annotations

import math
from typing import Any

from activelabel.labels import Label
from activelabel.tasks import Task

DECIMALS = 6


def _r(value: float) -> float:
    return round(float(value), DECIMALS)


def _is_nan(value: Any) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _box_of(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [_r(min(xs)), _r(min(ys)), _r(max(xs) - min(xs)), _r(max(ys) - min(ys))]


def _classification(value: Any) -> list[Label]:
    if type(value).__name__ == "Classifications":
        items = list(value.classifications or [])
        value = items[0] if items else None
    if value is None or not getattr(value, "label", None):
        return []
    return [Label(str(value.label))]


def _detections(value: Any, notes: dict[str, int]) -> list[Label]:
    out: list[Label] = []
    for det in value.detections or []:
        if not det.label or det.bounding_box is None:
            continue
        box = [_r(v) for v in det.bounding_box]
        if getattr(det, "mask", None) is not None:
            notes["mask_only"] = notes.get("mask_only", 0) + 1  # kept as a box: masks are not converted
        out.append(Label(str(det.label), bbox=box))
    return out


def _polylines(value: Any) -> list[Label]:
    out: list[Label] = []
    for poly in value.polylines or []:
        if not poly.label:
            continue
        rings = [[[_r(x), _r(y)] for x, y in ring] for ring in poly.points or [] if len(ring) >= 3]
        if not rings:
            continue
        flat = [p for ring in rings for p in ring]
        out.append(Label(str(poly.label), bbox=_box_of(flat), polygons=rings))
    return out


def _keypoints(value: Any) -> list[Label]:
    out: list[Label] = []
    for kp in value.keypoints or []:
        if not kp.label:
            continue
        points: list[list[float]] = []
        for x, y in kp.points or []:
            if _is_nan(x) or _is_nan(y):
                points.append([0.0, 0.0, 0.0])
            else:
                points.append([_r(x), _r(y), 2.0])
        visible = [p[:2] for p in points if p[2] > 0]
        out.append(Label(str(kp.label), bbox=_box_of(visible), keypoints=points))
    return out


def labels_and_notes(value: Any, task: Task) -> tuple[list[Label], dict[str, int]]:
    """Like :func:`labels_from_fo`, plus notes (``mask_only``: masks kept as boxes)."""
    notes: dict[str, int] = {}
    if value is None:
        return [], notes
    kind = type(value).__name__
    if kind in ("Classification", "Classifications"):
        labels = _classification(value)
    elif kind == "Detections":
        labels = _detections(value, notes)
    elif kind == "Polylines":
        labels = _polylines(value)
    elif kind == "Keypoints":
        labels = _keypoints(value)
    else:
        raise ValueError(f"Unsupported FiftyOne label type: {kind}")
    if task is Task.CLASSIFICATION:
        labels = labels[:1]
    return labels, notes


def labels_from_fo(value: Any, task: Task) -> list[Label]:
    """Internal labels of one FiftyOne label field value.

    Args:
        value: ``Classification`` (``Classifications``: the first), ``Detections``, ``Polylines``,
            ``Keypoints`` or None.
        task: task of the project (classification keeps one label).

    Returns:
        The labels, coordinates rounded to 6 decimals. Detections with a mask keep their box only.

    Raises:
        ValueError: unsupported label type.
    """
    return labels_and_notes(value, task)[0]
