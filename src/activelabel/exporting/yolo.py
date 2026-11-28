"""YOLO (Ultralytics) text of an export: label lines and ``data.yaml``.

The inverse of :mod:`activelabel.annotations.yolo`, which reads the same shapes: ``cls cx cy w h`` for a box,
``cls x1 y1 x2 y2 ...`` for a polygon ring and ``cls cx cy w h x y v ...`` for a pose.
"""

from __future__ import annotations

from typing import Any

import yaml

from activelabel.annotations.context import bbox_of_points
from activelabel.labels import Label, clamp01
from activelabel.tasks import Task


def _number(value: float) -> str:
    """A coordinate as written in a label file.

    Args:
        value: a relative coordinate.

    Returns:
        The value kept inside 0..1 with six decimals.
    """
    return f"{clamp01(value):.6f}"


def _box_values(label: Label) -> list[float] | None:
    """The centre and size of an object's box.

    Args:
        label: an object label; a polygon or points give the box when it has none of its own.

    Returns:
        ``[cx, cy, w, h]`` (relative), or None when the label has no box.
    """
    box = label.bbox
    if box is None and label.polygons:
        box = bbox_of_points(p for ring in label.polygons for p in ring)
    if box is None:
        return None
    x, y, w, h = box
    return [x + w / 2, y + h / 2, w, h]


def label_lines(task: Task, labels: list[Label], classes: list[str]) -> list[str]:
    """The lines of the label file of one image.

    Args:
        task: ``detection``, ``segmentation`` or ``keypoints``.
        labels: the objects of the image.
        classes: class names; a label's index in this list is its class number.

    Returns:
        One line per box or pose, one per polygon ring of a segmentation object; objects without the shape of
        the task are left out. Empty for a background image.
    """
    index = {name: i for i, name in enumerate(classes)}
    lines: list[str] = []
    for label in labels:
        cls = index[label.label]
        if task is Task.SEGMENTATION:
            for ring in label.polygons or []:
                if len(ring) >= 3:
                    lines.append(f"{cls} " + " ".join(f"{_number(p[0])} {_number(p[1])}" for p in ring))
            continue
        box = _box_values(label)
        if box is None:
            continue
        line = f"{cls} " + " ".join(_number(v) for v in box)
        if task is Task.KEYPOINTS:
            if not label.keypoints:
                continue
            line += "".join(
                f" {_number(p[0])} {_number(p[1])} {int(p[2]) if len(p) > 2 else 2}" for p in label.keypoints
            )
        lines.append(line)
    return lines


def data_yaml(parts: dict[str, str | list[str]], classes: list[str], keypoints: int = 0) -> str:
    """The text of a ``data.yaml``.

    Args:
        parts: the image folders of each key (``train``, ``val``, ``test``), relative to the file; a list for
            several folders.
        classes: class names, in class number order.
        keypoints: points per object (pose); 0 for the other tasks.

    Returns:
        The YAML text, with the dataset folder ``.`` (the folder of the file).
    """
    data: dict[str, Any] = {"path": ".", **parts, "names": dict(enumerate(classes))}
    if keypoints:
        data["kpt_shape"] = [keypoints, 3]
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
