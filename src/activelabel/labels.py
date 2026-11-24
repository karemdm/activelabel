"""Internal annotation format, the same for every task and every imported format.

Coordinates are relative to the image size (0..1), top-left origin, like FiftyOne.
A project keeps its ground truth in ``<project>/annotations.json``::

    {"task": "detection", "classes": ["cat", "dog"], "source": {...},
     "images": {"<image id>": [{"label": "cat", "bbox": [x, y, w, h]}, ...]}}
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task

Point = list[float]  # [x, y]
Keypoint = list[float]  # [x, y, visibility]; visibility 0 = not labeled, 1 = hidden, 2 = visible
MISSING_CORNER = (0.0, 0.0)  # where tools that cannot leave a point empty (CVAT) put a point not labeled


def clamp01(value: float) -> float:
    """``value`` limited to the range 0..1 (relative coordinates)."""
    return min(1.0, max(0.0, value))


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


NO_GEOMETRY = "no_geometry"
POLYGON_TOO_FEW_POINTS = "polygon_too_few_points"
BOX_INVALID = "box_invalid"
KEYPOINTS_COUNT_MISMATCH = "keypoints_count_mismatch"
MULTIPLE_CLASSES = "multiple_classes"
UNKNOWN_CLASS = "unknown_class"
LABEL_PROBLEMS = (
    NO_GEOMETRY,
    POLYGON_TOO_FEW_POINTS,
    BOX_INVALID,
    KEYPOINTS_COUNT_MISMATCH,
    MULTIPLE_CLASSES,
    UNKNOWN_CLASS,
)
MAX_PROBLEM_EXAMPLES = 5


def has_geometry(task: Task, label: Label) -> bool:
    """Whether a label has the shape its task needs (classification needs none)."""
    if task is Task.CLASSIFICATION:
        return True
    if task is Task.DETECTION:
        return bool(label.bbox)
    if task is Task.SEGMENTATION:
        return bool(label.polygons)
    return bool(label.keypoints)


def geometry_problem(task: Task, label: Label, keypoints: int | None = None) -> str | None:
    """The first thing wrong with the shape of one label for a task, or None when it is usable.

    Args:
        task: the task the label must fit.
        label: the label.
        keypoints: how many points a pose label must have; None leaves the count unchecked.

    Returns:
        ``no_geometry`` (the shape the task needs is missing), ``box_invalid`` (width or height not
        positive, or the box lies entirely outside the image), ``polygon_too_few_points`` (a ring with
        fewer than 3 points) or ``keypoints_count_mismatch``; None when the label is fine.
    """
    if not has_geometry(task, label):
        return NO_GEOMETRY
    if task is Task.DETECTION and label.bbox:
        x, y, w, h = label.bbox[:4]
        if w <= 0 or h <= 0 or x >= 1 or y >= 1 or x + w <= 0 or y + h <= 0:
            return BOX_INVALID
    elif task is Task.SEGMENTATION and any(len(ring) < 3 for ring in label.polygons or []):
        return POLYGON_TOO_FEW_POINTS
    elif task is Task.KEYPOINTS and keypoints is not None and len(label.keypoints or []) != keypoints:
        return KEYPOINTS_COUNT_MISMATCH
    return None


def common_keypoint_count(labels_by_image: dict[str, list[Label]]) -> int | None:
    """The most common number of points among the pose labels (None when there are none)."""
    sizes = Counter(len(lb.keypoints) for items in labels_by_image.values() for lb in items if lb.keypoints)
    return sizes.most_common(1)[0][0] if sizes else None


@dataclass
class LabelCheck:
    """What :func:`validate_labels` found.

    Attributes:
        counts: for every category of ``LABEL_PROBLEMS``: how many labels (``multiple_classes``: how many
            images) have the problem. Every category is present, zero when none.
        examples: for every category with a problem, up to 5 image ids that have it.
        labels: labels examined.
    """

    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LABEL_PROBLEMS, 0))
    examples: dict[str, list[str]] = field(default_factory=dict)
    labels: int = 0

    def add(self, problem: str, image_id: str) -> None:
        """Count one occurrence of ``problem`` on an image."""
        self.counts[problem] += 1
        found = self.examples.setdefault(problem, [])
        if image_id not in found and len(found) < MAX_PROBLEM_EXAMPLES:
            found.append(image_id)

    @property
    def ok(self) -> bool:
        """True when nothing was found."""
        return not any(self.counts.values())

    def problems(self) -> dict[str, int]:
        """Only the categories that have something, with their counts."""
        return {k: v for k, v in self.counts.items() if v}


def validate_labels(
    task: Task,
    labels_by_image: dict[str, list[Label]],
    classes: list[str] | None = None,
    keypoints: int | None = None,
) -> LabelCheck:
    """Check the content of labels against a task (pure: nothing is changed or written).

    Used wherever labels enter or leave a project (import, sync, data split, training, pre-annotation) so
    they all agree on what is usable. A label counts under ONE geometry problem (see
    :func:`geometry_problem`); the class checks are separate.

    Args:
        task: the task the labels must fit.
        labels_by_image: labels per image id.
        classes: the project's classes; labels naming another class (ignoring case) count as
            ``unknown_class``. None or empty skips that check.
        keypoints: the number of points a pose label must have. None: the most common number among the
            pose labels is taken as the expected one, so labels that differ from it are reported.

    Returns:
        The counts and examples per category. For classification ``multiple_classes`` counts the images
        that have more than one distinct class (those keep only the first on import).
    """
    check = LabelCheck()
    known = {c.lower() for c in classes or []}
    expected = keypoints
    if task is Task.KEYPOINTS and expected is None:
        sizes = Counter(
            len(lb.keypoints) for items in labels_by_image.values() for lb in items if lb.keypoints
        )
        expected = sizes.most_common(1)[0][0] if sizes else None
    for image_id, items in labels_by_image.items():
        for label in items:
            check.labels += 1
            problem = geometry_problem(task, label, expected)
            if problem:
                check.add(problem, image_id)
            if known and label.label.lower() not in known:
                check.add(UNKNOWN_CLASS, image_id)
        if task is Task.CLASSIFICATION and len({lb.label.lower() for lb in items}) > 1:
            check.add(MULTIPLE_CLASSES, image_id)
    return check


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
