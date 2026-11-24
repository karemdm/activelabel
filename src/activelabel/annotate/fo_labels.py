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
    """Round a coordinate to :data:`DECIMALS` decimals.

    Args:
        value: the number to round.

    Returns:
        The rounded float.
    """
    return round(float(value), DECIMALS)


def _is_nan(value: Any) -> bool:
    """Whether a value is missing: None, NaN or not a number.

    Args:
        value: the value to test.

    Returns:
        True when it is missing.
    """
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _box_of(points: list[list[float]]) -> list[float] | None:
    """Bounding box of points.

    Args:
        points: ``[x, y, ...]`` points.

    Returns:
        ``[x, y, w, h]`` rounded, or None when there are no points.
    """
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [_r(min(xs)), _r(min(ys)), _r(max(xs) - min(xs)), _r(max(ys) - min(ys))]


def _track(value: Any) -> int | None:
    """FiftyOne object-tracking ``index`` -> track id (None when it has none).

    Args:
        value: a FiftyOne label that may have an ``index``.

    Returns:
        The track id, or None.
    """
    index = getattr(value, "index", None)
    return None if index is None or _is_nan(index) else int(index)


def _classification(value: Any) -> list[Label]:
    """Labels of a ``Classification`` or ``Classifications`` (the first one).

    Args:
        value: the FiftyOne label.

    Returns:
        One label without geometry, or an empty list when there is no class.
    """
    if type(value).__name__ == "Classifications":
        items = list(value.classifications or [])
        value = items[0] if items else None
    if value is None or not getattr(value, "label", None):
        return []
    return [Label(str(value.label))]


def _detections(value: Any, notes: dict[str, int]) -> list[Label]:
    """Labels of a ``Detections`` value: those with a class and a box.

    Args:
        value: the ``Detections`` label.
        notes: counters; ``mask_only`` is increased for each detection that has a mask.

    Returns:
        One label with a box per detection.
    """
    out: list[Label] = []
    for det in value.detections or []:
        if not det.label or det.bounding_box is None:
            continue
        box = [_r(v) for v in det.bounding_box]
        if getattr(det, "mask", None) is not None:
            notes["mask_only"] = notes.get("mask_only", 0) + 1  # kept as a box: masks are not converted
        out.append(Label(str(det.label), bbox=box, track=_track(det)))
    return out


def _polylines(value: Any) -> list[Label]:
    """Labels of a ``Polylines`` value: those with a class and a ring of at least 3 points.

    Args:
        value: the ``Polylines`` label.

    Returns:
        One label with polygons and their bounding box per polyline.
    """
    out: list[Label] = []
    for poly in value.polylines or []:
        if not poly.label:
            continue
        rings = [[[_r(x), _r(y)] for x, y in ring] for ring in poly.points or [] if len(ring) >= 3]
        if not rings:
            continue
        flat = [p for ring in rings for p in ring]
        out.append(Label(str(poly.label), bbox=_box_of(flat), polygons=rings, track=_track(poly)))
    return out


def _inside(inner: list[float], outer: list[float]) -> bool:
    """Whether box ``inner`` ``[x, y, w, h]`` lies within box ``outer`` (small rounding allowed).

    Args:
        inner: the box to test ``[x, y, w, h]``.
        outer: the box that should contain it ``[x, y, w, h]``.

    Returns:
        True when ``inner`` is inside ``outer``.
    """
    eps = 1e-4
    return (
        inner[0] >= outer[0] - eps
        and inner[1] >= outer[1] - eps
        and inner[0] + inner[2] <= outer[0] + outer[2] + eps
        and inner[1] + inner[3] <= outer[1] + outer[3] + eps
    )


def _keypoints(value: Any) -> list[Label]:
    """Labels of a ``Keypoints`` value, restoring visibility and the original box kept by the bridge.

    Args:
        value: the ``Keypoints`` label.

    Returns:
        One label per keypoint set with its points ``[x, y, visibility]``.
    """
    out: list[Label] = []
    for kp in value.keypoints or []:
        if not kp.label:
            continue
        points: list[list[float]] = []
        sent = list(getattr(kp, "visibility", None) or [])  # written by fiftyone_bridge
        for i, (x, y) in enumerate(kp.points or []):
            if _is_nan(x) or _is_nan(y):
                points.append([0.0, 0.0, 0.0])
            else:  # an occluded point (1) stays occluded; a new or restored point is visible (2)
                points.append([_r(x), _r(y), 1.0 if i < len(sent) and sent[i] == 1 else 2.0])
        visible = [p[:2] for p in points if p[2] > 0]
        box = _box_of(visible)
        kept = getattr(kp, "box", None)  # the original box (a pose box is often larger than its points)
        if box and kept and len(kept) == 4 and _inside(box, [float(v) for v in kept]):
            box = [_r(float(v)) for v in kept]
        out.append(Label(str(kp.label), bbox=box, keypoints=points, track=_track(kp)))
    return out


def labels_and_notes(value: Any, task: Task) -> tuple[list[Label], dict[str, int]]:
    """Like :func:`labels_from_fo`, plus notes (``mask_only``: masks kept as boxes).

    Args:
        value: the FiftyOne label field value, or None.
        task: task of the project (classification keeps one label).

    Returns:
        The labels and the notes.

    Raises:
        ValueError: unsupported label type.
    """
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
