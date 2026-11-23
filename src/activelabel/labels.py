"""Internal annotation format, the same for every task and every imported format.

Coordinates are relative to the image size (0..1), top-left origin, like FiftyOne.
A project keeps its ground truth in ``<project>/annotations.json``::

    {"task": "detection", "classes": ["cat", "dog"], "source": {...},
     "images": {"<image id>": [{"label": "cat", "bbox": [x, y, w, h]}, ...]}}
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task

Point = list[float]  # [x, y]
Keypoint = list[float]  # [x, y, visibility]; visibility 0 = not labeled, 1 = hidden, 2 = visible


@dataclass
class Label:
    """One annotation: a class for the image, or an object.

    Attributes:
        label: class name.
        bbox: ``[x, y, width, height]`` (relative) for objects; None for image classes.
        polygons: one or more rings of ``[x, y]`` points (relative), for segmentation.
        keypoints: ``[x, y, visibility]`` per point (relative), for pose.
        score: confidence, for predictions; None for ground truth.
        track: identity of the object across frames (the CVAT track id; FiftyOne ``index``); None for
            objects that are not tracked.
        model: the saved model that made a pre-annotation (``m1``…); None for labels made by people.
    """

    label: str
    bbox: list[float] | None = None
    polygons: list[list[Point]] | None = None
    keypoints: list[Keypoint] | None = None
    score: float | None = None
    track: int | None = None
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form without empty fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Label:
        """Inverse of :meth:`as_dict`."""
        return cls(
            label=str(data["label"]),
            bbox=data.get("bbox"),
            polygons=data.get("polygons"),
            keypoints=data.get("keypoints"),
            score=data.get("score"),
            track=data.get("track"),
            model=data.get("model"),
        )


def task_of(labels: list[Label]) -> Task | None:
    """Task these labels fit best (None when there are no labels)."""
    if not labels:
        return None
    if any(lb.keypoints for lb in labels):
        return Task.KEYPOINTS
    if any(lb.polygons for lb in labels):
        return Task.SEGMENTATION
    if any(lb.bbox for lb in labels):
        return Task.DETECTION
    return Task.CLASSIFICATION


@dataclass
class AnnotationSet:
    """Ground truth of a project: labels per image id."""

    task: Task
    classes: list[str]
    images: dict[str, list[Label]]
    source: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "task": self.task.value,
            "classes": self.classes,
            "source": self.source,
            "images": {k: [lb.as_dict() for lb in v] for k, v in self.images.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationSet:
        """Inverse of :meth:`as_dict`."""
        images = {k: [Label.from_dict(x) for x in v] for k, v in data.get("images", {}).items()}
        return cls(Task(data["task"]), list(data.get("classes", [])), images, dict(data.get("source", {})))


def annotations_path(project_dir: Path) -> Path:
    """File with the ground truth of a project."""
    return project_dir / "annotations.json"


def load_annotations(project_dir: Path) -> AnnotationSet | None:
    """Ground truth of a project, or None when there is none."""
    data = read_json(annotations_path(project_dir))
    return AnnotationSet.from_dict(data) if data else None


_ANN_LOCKS: dict[Path, threading.RLock] = {}
_ANN_GUARD = threading.Lock()


def annotations_lock(project_dir: Path) -> threading.RLock:
    """Lock to hold around a read-modify-write of a project's ``annotations.json``.

    Every writer (review apply, sync, pre-annotation, accept, image deletion, imports) takes it, so a
    change made by one is never lost by another that read the file before it.
    """
    with _ANN_GUARD:
        return _ANN_LOCKS.setdefault(Path(project_dir).resolve(), threading.RLock())


def save_annotations(project_dir: Path, annotations: AnnotationSet) -> None:
    """Save the ground truth of a project (stamps ``source.saved``)."""
    annotations.source = {**annotations.source, "saved": now()}
    write_json(annotations_path(project_dir), annotations.as_dict())
