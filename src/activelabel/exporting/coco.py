"""COCO JSON of an export: boxes, polygons and keypoints.

The inverse of :mod:`activelabel.annotations.coco`: pixel coordinates, categories numbered from 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from activelabel.annotations.context import bbox_of_points
from activelabel.labels import Label
from activelabel.tasks import Task


def _area(ring: Sequence[Sequence[float]]) -> float:
    """Area of a polygon (shoelace formula).

    Args:
        ring: the ``(x, y)`` points.

    Returns:
        The area in the unit of the points.
    """
    total = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, [*ring[1:], ring[0]], strict=True))
    return abs(total) / 2


def _annotation(
    label: Label, category: int, width: int, height: int, number: int, image: int
) -> dict[str, Any] | None:
    """The COCO annotation of one object.

    Args:
        label: the object, with relative geometry.
        category: the category id of its class.
        width: width of the image in pixels.
        height: height of the image in pixels.
        number: the id of the annotation.
        image: the id of the image.

    Returns:
        The annotation, or None when the label has no box or polygon.
    """
    box = label.bbox
    if box is None and label.polygons:
        box = bbox_of_points(p for ring in label.polygons for p in ring)
    if box is None:
        return None
    x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    out: dict[str, Any] = {
        "id": number,
        "image_id": image,
        "category_id": category,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }
    if label.polygons:
        rings = [[(p[0] * width, p[1] * height) for p in ring] for ring in label.polygons if len(ring) >= 3]
        out["segmentation"] = [[v for point in ring for v in point] for ring in rings]
        out["area"] = sum(_area(ring) for ring in rings)
    if label.keypoints:
        flat = [
            v for p in label.keypoints for v in (p[0] * width, p[1] * height, int(p[2]) if len(p) > 2 else 2)
        ]
        out["keypoints"] = flat
        out["num_keypoints"] = sum(1 for p in label.keypoints if (p[2] if len(p) > 2 else 2) > 0)
    return out


def coco_document(
    task: Task,
    images: list[tuple[str, int, int, list[Label]]],
    classes: list[str],
    keypoints: int = 0,
) -> dict[str, Any]:
    """The COCO file of one group of images.

    Args:
        task: ``detection``, ``segmentation`` or ``keypoints``.
        images: ``(file name, width, height, labels)`` of each image; the names are relative to the folder of
            the group.
        classes: class names; category ``i + 1`` is ``classes[i]``.
        keypoints: points per object (pose); 0 for the other tasks.

    Returns:
        The document: ``images``, ``annotations`` and ``categories`` (with the keypoint names ``kp1``... when
        the task is ``keypoints``).
    """
    category = {name: i + 1 for i, name in enumerate(classes)}
    names = [f"kp{k + 1}" for k in range(keypoints)]
    cats = [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "",
            **({"keypoints": names, "skeleton": []} if keypoints else {}),
        }
        for i, name in enumerate(classes)
    ]
    entries: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for position, (name, width, height, labels) in enumerate(images, 1):
        entries.append({"id": position, "file_name": name, "width": width, "height": height})
        for label in labels:
            made = _annotation(label, category[label.label], width, height, len(annotations) + 1, position)
            if made is not None and (task is not Task.KEYPOINTS or "keypoints" in made):
                annotations.append(made)
    return {"images": entries, "annotations": annotations, "categories": cats}
