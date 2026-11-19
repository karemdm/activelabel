"""Convert the ground truth of a project to another task (when the change is sensible)."""

from __future__ import annotations

from activelabel.annotations.validation import to_detection
from activelabel.labels import AnnotationSet, Label
from activelabel.tasks import Task

OBJECT_TASKS = (Task.DETECTION, Task.SEGMENTATION, Task.KEYPOINTS)
CONVERSIONS: frozenset[tuple[Task, Task]] = frozenset(
    {(Task.SEGMENTATION, Task.DETECTION), (Task.KEYPOINTS, Task.DETECTION)}
    | {(src, Task.CLASSIFICATION) for src in OBJECT_TASKS}
)


def conversion_possible(from_task: Task, to_task: Task) -> bool:
    """Whether ground truth of ``from_task`` can become ``to_task`` without inventing data."""
    return (from_task, to_task) in CONVERSIONS


def convert_annotations(annotations: AnnotationSet, to_task: Task) -> AnnotationSet | None:
    """Ground truth converted to another task.

    Segmentation or keypoints to detection keeps the box of each object (polygons and keypoints
    are dropped). Detection, segmentation or keypoints to classification gives each image the
    distinct classes of its objects (an image without objects gets an empty list).

    Args:
        annotations: the current ground truth (not modified).
        to_task: the new task.

    Returns:
        A new set, or None when no sensible conversion exists.
    """
    if not conversion_possible(annotations.task, to_task):
        return None
    images: dict[str, list[Label]] = {}
    for image_id, labels in annotations.images.items():
        if to_task is Task.CLASSIFICATION:
            images[image_id] = [Label(name) for name in dict.fromkeys(lb.label for lb in labels)]
        else:
            boxes = [to_detection(lb) for lb in labels]
            images[image_id] = [b for b in boxes if b]
    source = {**annotations.source, "converted_from": annotations.task.value}
    return AnnotationSet(to_task, list(annotations.classes), images, source)
