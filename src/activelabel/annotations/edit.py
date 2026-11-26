"""Edit the labels of one image inside ActiveLabel (``GET/PUT /projects/{id}/images/{image_id}/labels``).

The person reads the labels of an image with a ``revision`` and sends the whole new list back with that
revision. The save is refused when the labels changed meanwhile (another edit, a synchronisation, an
import, a pre-annotation), when a class is not in the project (unless the person agrees to add it) and when
a label does not pass the same checks as an import: the shape the project type needs, geometry inside the
image and the number of points of a pose.

A saved edit is a local change: the CVAT link of the project marks the image as edited here
(:func:`activelabel.sync.links.mark_edited`), so the next comparison sends it instead of calling it a
conflict, even when the change is smaller than the comparison's tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from activelabel.errors import CodedError
from activelabel.labels import (
    BOX_INVALID,
    KEYPOINTS_COUNT_MISMATCH,
    MULTIPLE_CLASSES,
    NO_GEOMETRY,
    AnnotationSet,
    Label,
    annotations_lock,
    geometry_problem,
    load_annotations,
    save_annotations,
)
from activelabel.projects import Project, ProjectStore
from activelabel.sync import links
from activelabel.tasks import Task

OUTSIDE_IMAGE = "outside_image"
SHAPE_INVALID = "shape_invalid"
GEOMETRY_NOT_FOR_TASK = "geometry_not_for_task"
EDIT_PROBLEMS = (
    NO_GEOMETRY,
    BOX_INVALID,
    "polygon_too_few_points",
    KEYPOINTS_COUNT_MISMATCH,
    MULTIPLE_CLASSES,
    OUTSIDE_IMAGE,
    SHAPE_INVALID,
    GEOMETRY_NOT_FOR_TASK,
)
EDGE = 1e-6  # rounding allowed past the border of the image (relative units)
MAX_LISTED = 20  # problems listed in the error; the count covers all of them
ALLOWED = {  # the shapes a label of each project type may carry
    Task.CLASSIFICATION: set(),
    Task.DETECTION: {"bbox"},
    Task.SEGMENTATION: {"bbox", "polygons"},
    Task.KEYPOINTS: {"bbox", "keypoints"},
}


@dataclass
class Saved:
    """What :func:`save_image_labels` stored.

    Attributes:
        labels: the labels of the image now (class names in the project's spelling).
        revision: their new revision.
        classes_added: the classes added to the project by this save.
    """

    labels: list[Label]
    revision: str
    classes_added: list[str]


def revision(labels: list[Label]) -> str:
    """Revision of the labels of one image: changes whenever any field of any label changes.

    Args:
        labels: the labels of the image, in their stored order.

    Returns:
        The first 16 hex digits of the SHA-256 of their JSON form.
    """
    body = json.dumps([lb.as_dict() for lb in labels], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def image_labels(project: Project, project_dir: Any, image_id: str) -> dict[str, Any]:
    """The labels of one image with the project's classes and their revision.

    Args:
        project: the project (its classes).
        project_dir: the project folder.
        image_id: id of the image (an image without labels gives an empty list).

    Returns:
        ``{labels, classes, revision}``.
    """
    annotations = load_annotations(project_dir)
    labels = annotations.images.get(image_id, []) if annotations else []
    return {
        "labels": [lb.as_dict() for lb in labels],
        "classes": list(project.classes),
        "revision": revision(labels),
    }


def _numbers(value: Any, size: int | None = None) -> bool:
    """Whether ``value`` is a list of finite numbers (of length ``size`` when given).

    Args:
        value: the value to test.
        size: the length required, or None for any length.

    Returns:
        True when it is such a list.
    """
    if not isinstance(value, list) or (size is not None and len(value) != size):
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)


def _shape_ok(label: Label) -> bool:
    """Whether the geometry of a label has the right structure (lengths, finite numbers, visibility).

    Args:
        label: the label.

    Returns:
        True when the box has 4 numbers, every polygon point 2 and every keypoint 3 (``[x, y, visibility]``,
        visibility 0, 1 or 2).
    """
    if label.bbox is not None and not _numbers(label.bbox, 4):
        return False
    rings = label.polygons
    if rings is not None and not (
        isinstance(rings, list) and all(isinstance(r, list) and all(_numbers(p, 2) for p in r) for r in rings)
    ):
        return False
    points = label.keypoints
    if points is not None and not (isinstance(points, list) and all(_numbers(p, 3) for p in points)):
        return False
    return all(p[2] in (0, 1, 2) for p in points or [])


def _inside(value: float) -> bool:
    """Whether a relative coordinate lies on the image (with :data:`EDGE` of rounding).

    Args:
        value: the coordinate.

    Returns:
        True when it is between 0 and 1.
    """
    return -EDGE <= value <= 1 + EDGE


def _outside(label: Label) -> bool:
    """Whether some geometry of a label goes past the border of the image.

    Args:
        label: a label whose shape passed :func:`_shape_ok`.

    Returns:
        True when the box, a polygon point or a labelled keypoint lies outside the image.
    """
    if label.bbox:
        x, y, w, h = label.bbox
        if not all(_inside(v) for v in (x, y, x + w, y + h)):
            return True
    points = [p for ring in label.polygons or [] for p in ring]
    points += [p for p in label.keypoints or [] if p[2] > 0]
    return any(not (_inside(p[0]) and _inside(p[1])) for p in points)


def _expected_points(images: dict[str, list[Label]], image_id: str, labels: list[Label]) -> dict[str, int]:
    """How many points the poses of each class must have: the most common count of the other images.

    Args:
        images: the project's labels by image id.
        image_id: the image being edited (its stored labels do not count).
        labels: the new labels of the image (they decide for a class no other image has).

    Returns:
        The number of points by class name (lower case).
    """
    others: dict[str, Counter[int]] = {}
    for key, items in images.items():
        for lb in items if key != image_id else []:
            if lb.keypoints:
                others.setdefault(lb.label.lower(), Counter())[len(lb.keypoints)] += 1
    mine: dict[str, Counter[int]] = {}
    for lb in labels:
        if isinstance(lb.keypoints, list) and lb.label.lower() not in others:
            mine.setdefault(lb.label.lower(), Counter())[len(lb.keypoints)] += 1
    return {name: found.most_common(1)[0][0] for name, found in {**mine, **others}.items()}


def label_problem(task: Task, label: Label, points: int | None) -> str | None:
    """The first thing that keeps one label from being saved, or None when it can be saved.

    Args:
        task: the project type.
        label: the label.
        points: how many points a pose of this class must have, or None when not known.

    Returns:
        One of :data:`EDIT_PROBLEMS` but ``multiple_classes``, or None.
    """
    if not _shape_ok(label):
        return SHAPE_INVALID
    present = {name for name in ("bbox", "polygons", "keypoints") if getattr(label, name) is not None}
    if present - ALLOWED[task]:
        return GEOMETRY_NOT_FOR_TASK
    problem = geometry_problem(task, label, points)
    if problem:
        return problem
    return OUTSIDE_IMAGE if _outside(label) else None


def check_labels(task: Task, labels: list[Label], points: dict[str, int]) -> list[dict[str, Any]]:
    """Every problem of a new list of labels for one image (pure: nothing is written).

    Args:
        task: the project type.
        labels: the new labels of the image.
        points: the number of points a pose of each class (lower case) must have.

    Returns:
        ``[{index, problem}]`` in label order (``index`` counts from 0); for classification more than one
        label is one ``multiple_classes`` problem at index 1.
    """
    found: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if not label.label.strip():
            found.append({"index": index, "problem": SHAPE_INVALID})
            continue
        problem = label_problem(task, label, points.get(label.label.lower()))
        if problem:
            found.append({"index": index, "problem": problem})
    if task is Task.CLASSIFICATION and len(labels) > 1:
        found.append({"index": 1, "problem": MULTIPLE_CLASSES})
    return found


def _new_classes(project: Project, labels: list[Label]) -> list[str]:
    """The classes of ``labels`` the project does not have (ignoring case), in order of first use.

    Args:
        project: the project.
        labels: the new labels.

    Returns:
        The new class names as the labels spell them.
    """
    known = {c.lower() for c in project.classes}
    out: list[str] = []
    for lb in labels:
        name = lb.label.strip()
        if name.lower() not in known:
            known.add(name.lower())
            out.append(name)
    return out


def save_image_labels(
    store: ProjectStore,
    project_id: int,
    image_id: str,
    labels: list[Label],
    expected: str,
    add_classes: bool = False,
) -> Saved:
    """Replace the labels of one image, when they did not change since the person read them.

    The check of the revision and the write happen under the annotations lock. No safety copy is made (one
    image changes; the copies are kept for bulk changes). The image is then marked as edited here in the
    project's links (see :func:`activelabel.sync.links.mark_edited`).

    Args:
        store: the project store.
        project_id: id of the project.
        image_id: id of the image (the caller checks that it is in the project).
        labels: the new labels of the image, in order (an empty list: the image has no labels).
        expected: the revision the person read (:func:`revision`).
        add_classes: add the classes the project does not have instead of refusing them.

    Returns:
        The saved labels, their revision and the classes added.

    Raises:
        CodedError: ``labels_changed`` (the labels of the image changed meanwhile; ``params.revision`` is
            the current one), ``labels_invalid`` (a label does not pass the checks: ``count``, the first
            ``index`` counted from 1 and its ``problem``, and up to 20 ``problems``), ``classes_new`` (a class
            is not in the project and ``add_classes`` is False: ``classes``, ``count``), or a code of
            :func:`activelabel.projects.check_classes` for a class list the project cannot hold.
    """
    project_dir = store.dir(project_id)
    with annotations_lock(project_dir):
        project = store.get(project_id)
        annotations = load_annotations(project_dir)
        if annotations is None:
            annotations = AnnotationSet(project.task, list(project.classes), {}, {"format": "activelabel"})
        current = annotations.images.get(image_id, [])
        if revision(current) != expected:
            raise CodedError(
                "labels_changed",
                "The labels of this image changed since you opened it: reload them and edit again.",
                revision=revision(current),
            )
        problems = check_labels(project.task, labels, _expected_points(annotations.images, image_id, labels))
        if problems:
            first = problems[0]
            raise CodedError(
                "labels_invalid",
                f"{len(problems)} labels cannot be saved (the first: label {first['index'] + 1}, "
                f"{first['problem']}).",
                count=len(problems),
                index=first["index"] + 1,
                problem=first["problem"],
                problems=problems[:MAX_LISTED],
            )
        added = _new_classes(project, labels)
        if added and not add_classes:
            raise CodedError(
                "classes_new",
                f"These classes are not in the project: {', '.join(added)}. Add them, or pick another class.",
                classes=added,
                count=len(added),
            )
        if added:  # the store refuses a class list it cannot hold (CodedError) before writing anything
            project = store.update(project_id, classes=[*project.classes, *added])
        canon = {c.lower(): c for c in project.classes}
        saved = [
            Label(**{**lb.__dict__, "label": canon.get(lb.label.strip().lower(), lb.label.strip())})
            for lb in labels
        ]
        annotations.images[image_id] = saved
        annotations.task = project.task
        annotations.classes = list(project.classes)
        save_annotations(project_dir, annotations)
        links.mark_edited(project_dir, image_id, saved)
    return Saved(saved, revision(saved), added)
