"""Compare the labels brought back from a tool with the project ground truth, and apply them."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.annotate.base import Pulled
from activelabel.backups import make_backup
from activelabel.images.manifest import Manifest
from activelabel.labels import (
    AnnotationSet,
    Label,
    annotations_lock,
    has_geometry,
    load_annotations,
    save_annotations,
)
from activelabel.projects import Project, ProjectStore
from activelabel.sync.links import label_hash
from activelabel.tasks import Task

SAME_CLASS_IOU = 0.5  # same class and this much overlap: the same object
OTHER_CLASS_IOU = 0.7  # another class at the same place: the class was changed
TOLERANCE = 0.005  # geometry moved less than this (relative units) counts as unchanged
MAX_EXAMPLES = 20
MAX_PROBLEM_EXAMPLES = 5


# -- geometry -------------------------------------------------------------------
def _box(label: Label) -> list[float] | None:
    """Bounding box of a label, taken from its box, polygons or visible keypoints.

    Args:
        label: the label.

    Returns:
        ``[x, y, w, h]``, or None when the label has no geometry.
    """
    if label.bbox:
        return label.bbox
    points: list[list[float]] = []
    if label.polygons:
        points = [p for ring in label.polygons for p in ring]
    elif label.keypoints:
        points = [p for p in label.keypoints if p[2] > 0]
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two boxes.

    Args:
        a: first box ``[x, y, w, h]``.
        b: second box ``[x, y, w, h]``.

    Returns:
        A value from 0 to 1 (two empty boxes at the same place count as 1).
    """
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 1.0 if a == b else 0.0
    return inter / union


def _similarity(a: Label, b: Label) -> float:
    """How alike two labels are: keypoint distance for poses, box overlap for the rest.

    Args:
        a: first label.
        b: second label.

    Returns:
        A value from 0 (unrelated) to 1 (same place).
    """
    if a.keypoints and b.keypoints:
        if len(a.keypoints) != len(b.keypoints):
            return 0.0
        dists = [
            math.dist(p[:2], q[:2])
            for p, q in zip(a.keypoints, b.keypoints, strict=True)
            if p[2] > 0 and q[2] > 0
        ]
        if not dists:
            return 1.0 if not any(p[2] > 0 for p in a.keypoints + b.keypoints) else 0.0
        return max(0.0, 1.0 - (sum(dists) / len(dists)) / 0.2)
    box_a, box_b = _box(a), _box(b)
    return _iou(box_a, box_b) if box_a and box_b else 0.0


def _moved(a: Label, b: Label) -> bool:
    """Whether the geometry of two matched labels differs by more than :data:`TOLERANCE`.

    Args:
        a: the old label.
        b: the new label.

    Returns:
        True when keypoints, polygons or box moved (or keypoint visibility changed).
    """
    if a.keypoints and b.keypoints:
        for p, q in zip(a.keypoints, b.keypoints, strict=True):
            if (p[2] > 0) != (q[2] > 0):
                return True
            if p[2] > 0 and math.dist(p[:2], q[:2]) > TOLERANCE:
                return True
        return False
    if a.polygons and b.polygons:
        if [len(r) for r in a.polygons] != [len(r) for r in b.polygons]:
            return True
        return any(
            abs(p[i] - q[i]) > TOLERANCE
            for ra, rb in zip(a.polygons, b.polygons, strict=True)
            for p, q in zip(ra, rb, strict=True)
            for i in (0, 1)
        )
    if a.bbox and b.bbox:
        return any(abs(x - y) > TOLERANCE for x, y in zip(a.bbox, b.bbox, strict=True))
    return False


# -- one image ------------------------------------------------------------------
@dataclass
class ImageChanges:
    """Changes of one image."""

    added: list[Label] = field(default_factory=list)
    removed: list[Label] = field(default_factory=list)
    edited: list[Label] = field(default_factory=list)  # the new version of each edited label

    def __bool__(self) -> bool:
        """Whether anything changed.

        Returns:
            True when there is an added, removed or edited label.
        """
        return bool(self.added or self.removed or self.edited)


def compare_image(task: Task, old: list[Label], new: list[Label]) -> ImageChanges:
    """Match old and new labels of one image and sort them into added, removed and edited.

    Args:
        task: task of the project (classification compares only the first label).
        old: the labels in the project.
        new: the labels brought back from the tool.

    Returns:
        The added, removed and edited labels (edited ones in their new version).
    """
    changes = ImageChanges()
    if task is Task.CLASSIFICATION:
        before, after = old[:1], new[:1]
        if before and after:
            if before[0].label != after[0].label:
                changes.edited.append(after[0])
        elif after:
            changes.added.append(after[0])
        elif before:
            changes.removed.append(before[0])
        return changes
    free_old, free_new = list(range(len(old))), list(range(len(new)))
    for min_iou, same_class in ((SAME_CLASS_IOU, True), (OTHER_CLASS_IOU, False)):
        pairs = sorted(
            (
                (_similarity(old[i], new[j]), i, j)
                for i in free_old
                for j in free_new
                if (old[i].label == new[j].label) == same_class
            ),
            reverse=True,
        )
        for sim, i, j in pairs:
            if sim >= min_iou and i in free_old and j in free_new:
                free_old.remove(i)
                free_new.remove(j)
                if old[i].label != new[j].label or _moved(old[i], new[j]):
                    changes.edited.append(new[j])
    changes.removed = [old[i] for i in free_old]
    changes.added = [new[j] for j in free_new]
    return changes


# -- cleaning -------------------------------------------------------------------
def clean_labels(project: Project, labels: list[Label]) -> tuple[list[Label], dict[str, int]]:
    """Labels that fit the project: class names match the class list ignoring case, geometry present.

    Args:
        project: the project (task and class list).
        labels: labels brought back from the tool.

    Returns:
        The usable labels and counts of what was dropped: ``no_geometry``, ``multiple_classes``.
    """
    known = {c.lower(): c for c in project.classes}
    dropped = {"no_geometry": 0, "multiple_classes": 0}
    out: list[Label] = []
    for label in labels:
        if not has_geometry(project.task, label):
            dropped["no_geometry"] += 1
            continue
        out.append(Label(**{**label.__dict__, "label": known.get(label.label.lower(), label.label)}))
    if project.task is Task.CLASSIFICATION and len(out) > 1:
        dropped["multiple_classes"] = len(out) - 1
        out = out[:1]
    return out, dropped


# -- what the tool changed ------------------------------------------------------
def settle(
    pulled: dict[str, list[Label]], current: dict[str, list[Label]]
) -> tuple[dict[str, list[Label]], list[str]]:
    """The labels to diff and apply: the tool's version only for the images edited in the tool.

    An image the tool did not change keeps the project's current labels, so a project edit made after the
    images were sent is not undone. An image changed in the tool whose project labels also changed since
    they were written into the tool is a conflict: the tool's version still wins (the person reviews the
    summary before applying), and the image is listed.

    Args:
        pulled: what the tool's pull returned (a :class:`~activelabel.annotate.base.Pulled`, or a plain
            dict when the tool cannot tell which images were edited: then all of them are taken).
        current: the project's labels now, by image id.

    Returns:
        The labels of each pulled image and the ids of the images edited on both sides.
    """
    if not isinstance(pulled, Pulled):
        return dict(pulled), []
    labels = {i: (items if i in pulled.edited else list(current.get(i, []))) for i, items in pulled.items()}
    conflicts = [
        i
        for i in pulled
        if i in pulled.edited and i in pulled.written and pulled.written[i] != label_hash(current.get(i, []))
    ]
    return labels, conflicts


# -- diff -----------------------------------------------------------------------
_MESSAGES = {
    "unknown_class": "Labels with classes that are not in the project (added to the project on apply).",
    "no_geometry": "Labels without the shape this task needs (they are skipped on apply).",
    "multiple_classes": "Images with more than one class (classification keeps the first).",
    "missing_images": "Images the tool did not return (they are left unchanged).",
    "edited_on_both_sides": (
        "Images edited in the tool whose project labels also changed since they were sent: applying "
        "replaces the project's version with the tool's."
    ),
}


@dataclass
class _Tally:
    """What :func:`_compare_all` counts while it goes over the images."""

    names: dict[str, str]
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    problems: dict[str, dict[str, Any]] = field(default_factory=dict)

    def problem(self, code: str, image_id: str, count: int = 1) -> None:
        """Count a problem and keep the name of the image as an example (a few per code).

        Args:
            code: the problem code (a key of ``_MESSAGES``).
            image_id: id of the image where it happened.
            count: how many times to count it.
        """
        entry = self.problems.setdefault(code, {"count": 0, "examples": []})
        entry["count"] += count
        name = self.names.get(image_id, image_id)
        if name not in entry["examples"] and len(entry["examples"]) < MAX_PROBLEM_EXAMPLES:
            entry["examples"].append(name)

    def bump(self, label: Label, kind: str) -> None:
        """Count a label as added, removed or edited under its class.

        Args:
            label: the label.
            kind: ``added``, ``removed`` or ``edited``.
        """
        self.per_class.setdefault(label.label, {"added": 0, "removed": 0, "edited": 0})[kind] += 1


def _compare_all(
    project_dir: Path,
    image_ids: list[str],
    new_labels: dict[str, list[Label]],
    project: Project,
    conflicts: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, list[Label]]]:
    """Clean the new labels and compare them with the ground truth, image by image.

    Args:
        project_dir: the project folder.
        image_ids: the images of the session.
        new_labels: labels brought back from the tool, by image id.
        project: the project (task and class list).
        conflicts: images edited both in the tool and in the project (see :func:`settle`).

    Returns:
        The summary (see :func:`diff`) and the cleaned labels of each image present in ``new_labels``.
    """
    annotations = load_annotations(project_dir)
    old_all = annotations.images if annotations else {}
    names = {r.id: r.name for r in Manifest(project_dir).all()}
    known = {c.lower() for c in project.classes}
    tally = _Tally(names)
    totals = {"added": 0, "removed": 0, "edited": 0}
    unknown: dict[str, None] = {}
    examples: list[dict[str, Any]] = []
    cleaned: dict[str, list[Label]] = {}
    changed = 0
    for image_id in conflicts or []:
        tally.problem("edited_on_both_sides", image_id)

    for image_id in image_ids:
        if image_id not in new_labels:
            tally.problem("missing_images", image_id)
            continue
        labels, dropped = clean_labels(project, new_labels[image_id])
        cleaned[image_id] = labels
        for code, count in dropped.items():
            if count:
                tally.problem(code, image_id, count)
        for label in labels:
            if label.label.lower() not in known:
                unknown.setdefault(label.label)
                tally.problem("unknown_class", image_id)
        changes = compare_image(project.task, old_all.get(image_id, []), labels)
        if changes:
            changed += 1
            for kind in ("added", "removed", "edited"):
                items = getattr(changes, kind)
                totals[kind] += len(items)
                for label in items:
                    tally.bump(label, kind)
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "image_id": image_id,
                        "name": names.get(image_id, image_id),
                        "added": len(changes.added),
                        "removed": len(changes.removed),
                        "edited": len(changes.edited),
                    }
                )
    result = {
        "images_changed": changed,
        **totals,
        "per_class": tally.per_class,
        "unknown_classes": list(unknown),
        "problems": [
            {"level": "warning", "code": code, "message": _MESSAGES[code], **entry}
            for code, entry in tally.problems.items()
        ],
        "examples": examples,
    }
    return result, cleaned


def diff(
    project_dir: Path,
    image_ids: list[str],
    new_labels: dict[str, list[Label]],
    project: Project,
    conflicts: list[str] | None = None,
) -> dict[str, Any]:
    """Summary of the changes between the project ground truth and ``new_labels`` for these images.

    Args:
        project_dir: the project folder.
        image_ids: the images of the session.
        new_labels: labels brought back from the tool, by image id.
        project: the project (task and class list).
        conflicts: images edited both in the tool and in the project (``edited_on_both_sides``).

    Returns:
        ``{images_changed, added, removed, edited, per_class, unknown_classes, problems, examples}``
        (see docs/ANNOTATION.md).
    """
    return _compare_all(project_dir, image_ids, new_labels, project, conflicts)[0]


# -- apply ----------------------------------------------------------------------
def apply(
    store: ProjectStore,
    project_id: int,
    manifest: Manifest,
    image_ids: list[str],
    new_labels: dict[str, list[Label]],
    reason: str = "apply",
) -> dict[str, Any]:
    """Make ``new_labels`` the ground truth of these images (after a backup of ``annotations.json``).

    Unknown classes are added to the project. Labels without the geometry the task needs are skipped;
    classification keeps one class per image.

    Args:
        store: the project store (to read the project and add classes).
        project_id: id of the project.
        manifest: the images of the project; ids that are not in it are ignored.
        image_ids: the images of the session.
        new_labels: labels brought back from the tool, by image id.
        reason: why the backup is made (see :data:`activelabel.backups.REASONS`): ``apply`` for a FiftyOne
            session, ``sync`` for a CVAT synchronisation.

    Returns:
        Counts: the diff counts plus ``images_updated``, ``new_classes``, ``skipped`` and ``backup``.
    """
    project = store.get(project_id)
    project_dir = store.dir(project_id)
    existing = {r.id for r in manifest.all()}
    ids = [i for i in image_ids if i in existing]
    with annotations_lock(project_dir):
        return _apply_locked(store, project_id, project, project_dir, ids, new_labels, reason)


def _apply_locked(
    store: ProjectStore,
    project_id: int,
    project: Project,
    project_dir: Path,
    ids: list[str],
    new_labels: dict[str, list[Label]],
    reason: str,
) -> dict[str, Any]:
    """Do the work of :func:`apply` while the annotations file is locked.

    Args:
        store: the project store (to add classes).
        project_id: id of the project.
        project: the project as read before the lock.
        project_dir: the project folder.
        ids: the images to update (already restricted to those of the manifest).
        new_labels: labels brought back from the tool, by image id.
        reason: why the backup is made, as in :func:`apply`.

    Returns:
        The same counts as :func:`apply`.
    """
    summary, cleaned = _compare_all(project_dir, ids, new_labels, project)
    annotations = load_annotations(project_dir)
    backup = ""
    if annotations is None:
        annotations = AnnotationSet(project.task, list(project.classes), {}, {"format": "activelabel"})
    else:
        backup = make_backup(project_dir, reason)
    for image_id, labels in cleaned.items():
        if labels or image_id in annotations.images:
            annotations.images[image_id] = labels
    new_classes = list(summary["unknown_classes"])
    if new_classes:
        project = store.update(project_id, classes=[*project.classes, *new_classes])
    annotations.classes = list(project.classes)
    save_annotations(project_dir, annotations)
    skipped = sum(p["count"] for p in summary["problems"] if p["code"] in ("no_geometry", "multiple_classes"))
    return {
        "images_changed": summary["images_changed"],
        "added": summary["added"],
        "removed": summary["removed"],
        "edited": summary["edited"],
        "images_updated": len(cleaned),
        "new_classes": new_classes,
        "skipped": skipped,
        "backup": backup,
    }
