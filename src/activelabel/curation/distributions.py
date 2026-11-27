"""Distributions of the annotations of a project: class balance, objects per image, box sizes (#623).

Pure numbers from ``annotations.json`` and the manifest (no FiftyOne): the Curation step and the preview of
the split show them as bars, and the FiftyOne App has the same charts in its Histograms panel.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from typing import Any

import numpy as np

from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.labels import AnnotationSet
from activelabel.tasks import Task

#: Buckets of the number of objects in an image: ``(key, first, last)``, ``last`` None = no limit.
COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-10", 4, 10),
    ("11+", 11, None),
)

#: Buckets of the area of a box relative to its image: ``(key, upper limit)``; a box is in the first bucket
#: whose limit is above its area, the last one has no limit. ``tiny`` is under 1% of the image, ``small``
#: under 5%, ``medium`` under 25% and ``large`` the rest.
AREA_BUCKETS: tuple[tuple[str, float | None], ...] = (
    ("tiny", 0.01),
    ("small", 0.05),
    ("medium", 0.25),
    ("large", None),
)

#: Tasks whose objects have a box (the box sizes are shown for them).
BOX_TASKS = (Task.DETECTION, Task.SEGMENTATION, Task.KEYPOINTS)


def _count_histogram(counts: list[int]) -> list[dict[str, Any]]:
    """Images per bucket of ``COUNT_BUCKETS``.

    Args:
        counts: number of objects of every image.

    Returns:
        ``[{"key", "min", "max", "images"}]``, one entry per bucket (``max`` None = no limit).
    """
    values = np.asarray(counts, dtype=int)
    out = []
    for key, low, high in COUNT_BUCKETS:
        inside = values >= low if high is None else (values >= low) & (values <= high)
        out.append({"key": key, "min": low, "max": high, "images": int(inside.sum())})
    return out


def _area_histogram(areas: list[float]) -> list[dict[str, Any]]:
    """Boxes per bucket of ``AREA_BUCKETS``.

    Args:
        areas: relative area (0 to 1) of every box.

    Returns:
        ``[{"key", "below", "objects"}]``, one entry per bucket (``below`` None = no limit).
    """
    values = np.asarray(areas, dtype=float)
    out = []
    low = -1.0
    for key, high in AREA_BUCKETS:
        inside = values >= low if high is None else (values >= low) & (values < high)
        out.append({"key": key, "below": high, "objects": int(inside.sum())})
        low = high if high is not None else low
    return out


def _by_objects(row: dict[str, Any]) -> tuple[int, str]:
    """Sort key of a class row: most objects first, then by name.

    Args:
        row: ``{"name", "objects", "images"}``.

    Returns:
        The key.
    """
    return -int(row["objects"]), str(row["name"])


def distributions(
    annotations: AnnotationSet | None,
    manifest: Manifest,
    classes: list[str],
    task: Task,
    image_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Class balance, objects per image and box sizes of the active images.

    Excluded images are left out. Classification counts images per class only (one class per image) and
    regression has no classes, so neither has the object histograms.

    Args:
        annotations: the ground truth of the project, or None when there is none yet.
        manifest: images of the project.
        classes: classes of the project (listed even when they have no object).
        task: task of the project.
        image_ids: keep only these images (a part of a dataset version); None keeps every active image.

    Returns:
        ``{"task", "images", "annotated", "without_annotations", "classes": [{"name", "objects", "images"}]
        (most objects first), "objects_per_image": [...] or None, "box_area": [...] or None, "boxes"}``.
    """
    keep = None if image_ids is None else set(image_ids)
    active = [r.id for r in manifest.all() if r.status == ACTIVE and (keep is None or r.id in keep)]
    labels = annotations.images if annotations else {}
    regression = task is Task.REGRESSION
    objects: Counter[str] = Counter()
    images: Counter[str] = Counter()
    per_image: list[int] = []
    areas: list[float] = []
    without = 0
    for image_id in active:
        found = [] if regression else list(labels.get(image_id) or [])
        if task is Task.CLASSIFICATION:
            found = found[:1]
        without += not found
        per_image.append(len(found))
        objects.update(lb.label for lb in found)
        images.update({lb.label for lb in found})
        areas += [
            max(0.0, lb.bbox[2]) * max(0.0, lb.bbox[3]) for lb in found if lb.bbox and len(lb.bbox) >= 4
        ]
    names = [] if regression else list(classes) + [c for c in objects if c not in classes]
    rows = [{"name": c, "objects": objects.get(c, 0), "images": images.get(c, 0)} for c in names]
    rows.sort(key=_by_objects)
    counted = task not in (Task.CLASSIFICATION, Task.REGRESSION)
    return {
        "task": task.value,
        "images": len(active),
        "annotated": len(active) - without if not regression else 0,
        "without_annotations": without if not regression else 0,
        "classes": rows,
        "objects_per_image": _count_histogram(per_image) if counted else None,
        "box_area": _area_histogram(areas) if task in BOX_TASKS else None,
        "boxes": len(areas),
    }
