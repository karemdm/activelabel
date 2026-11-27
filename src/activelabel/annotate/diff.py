"""Compare the labels brought back from a tool with the project ground truth, and apply them."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.annotate.base import Pulled
from activelabel.backups import make_backup
from activelabel.images.manifest import Manifest
from activelabel.images.tags import image_tags as clean_tags
from activelabel.images.tags import load_tags, set_tags
from activelabel.labels import (
    REPEATED_TARGET,
    AnnotationSet,
    Label,
    annotations_lock,
    common_keypoint_count,
    geometry_problem,
    load_annotations,
    same_value,
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
SKIPPED_ON_APPLY = (  # problems of labels that apply leaves out (counted as ``skipped``)
    "no_geometry",
    "box_invalid",
    "polygon_too_few_points",
    "keypoints_count_mismatch",
    "multiple_classes",
)


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


def _compare_values(old: list[Label], new: list[Label]) -> ImageChanges:
    """Changes of the values of one image of a regression project: labels are matched by target.

    Args:
        old: the values in the project (one label per target).
        new: the values brought back from the tool.

    Returns:
        A value of a target that only one side has is added or removed; a value that differs is edited.
    """
    changes = ImageChanges()
    before = {lb.label: lb for lb in old}
    after = {lb.label: lb for lb in new}
    for target, label in after.items():
        if target not in before:
            changes.added.append(label)
        elif not same_value(before[target].value, label.value):
            changes.edited.append(label)
    changes.removed = [lb for target, lb in before.items() if target not in after]
    return changes


def compare_image(
    task: Task, old: list[Label], new: list[Label], origins: list[str | None] | None = None
) -> ImageChanges:
    """Match old and new labels of one image and sort them into added, removed and edited.

    Labels whose origin is known (the tool kept the identity of the label it was given) are paired with
    that project label first, however far they moved; the others are paired by overlap.

    Args:
        task: task of the project (classification compares only the first label; regression matches
            the values by target, see :func:`_compare_values`).
        old: the labels in the project.
        new: the labels brought back from the tool.
        origins: for each label of ``new``, the ``label_hash`` of the project label it was written from, or
            None (drawn in the tool, or not known); None for no origins at all.

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
    if task is Task.REGRESSION:
        return _compare_values(old, new)
    free_old, free_new = list(range(len(old))), list(range(len(new)))
    hashes = [label_hash([lb]) for lb in old]
    for j, origin in enumerate(origins or []):
        i = next((i for i in free_old if origin and hashes[i] == origin), None)
        if i is None or j >= len(new):
            continue
        free_old.remove(i)
        free_new.remove(j)
        if old[i].label != new[j].label or _moved(old[i], new[j]):
            changes.edited.append(new[j])
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
def clean_labels(
    project: Project, labels: list[Label], keypoints: dict[str, int] | None = None
) -> tuple[list[Label], dict[str, int]]:
    """Labels that fit the project (see :func:`clean_indexed`), without their positions.

    Args:
        project: the project (task and class list).
        labels: labels brought back from the tool.
        keypoints: how many points a pose label of each class (lower case) must have.

    Returns:
        The usable labels and counts of what was dropped.
    """
    kept, dropped = clean_indexed(project, labels, keypoints)
    return [label for _, label in kept], dropped


def clean_indexed(
    project: Project, labels: list[Label], keypoints: dict[str, int] | None = None
) -> tuple[list[tuple[int, Label]], dict[str, int]]:
    """Labels that fit the project: class names match the class list ignoring case, usable geometry.

    Args:
        project: the project (task and class list).
        labels: labels brought back from the tool.
        keypoints: how many points a pose label of each class (lower case) must have; a class that is
            not in it, or None, is not checked (a CVAT skeleton of another class may have other points).

    Returns:
        The usable labels, each with its position in ``labels``, and counts of what was dropped:
        ``multiple_classes``, ``repeated_target`` (regression: a second value of one target) and the codes of
        :func:`activelabel.labels.geometry_problem` (``no_geometry``, ``box_invalid``,
        ``polygon_too_few_points``, ``keypoints_count_mismatch``).
    """
    known = {c.lower(): c for c in project.classes}
    dropped = {"no_geometry": 0, "multiple_classes": 0}
    out: list[tuple[int, Label]] = []
    for index, label in enumerate(labels):
        expected = (keypoints or {}).get(label.label.lower())
        problem = geometry_problem(project.task, label, expected)
        if problem is not None:
            dropped[problem] = dropped.get(problem, 0) + 1
            continue
        out.append((index, Label(**{**label.__dict__, "label": known.get(label.label.lower(), label.label)})))
    if project.task is Task.CLASSIFICATION and len(out) > 1:
        dropped["multiple_classes"] = len(out) - 1
        out = out[:1]
    if project.task is Task.REGRESSION:  # one value per target: the first is kept
        first: list[tuple[int, Label]] = []
        for item in out:
            if not any(f.label == item[1].label for _, f in first):
                first.append(item)
        if len(first) < len(out):
            dropped[REPEATED_TARGET] = len(out) - len(first)
        out = first
    return out, dropped


def keypoint_counts(old: dict[str, list[Label]], new: dict[str, list[Label]]) -> dict[str, int]:
    """How many points the poses of each class have: the most common count in the project, else in the tool.

    Args:
        old: the project's labels per image id.
        new: the labels brought back from the tool per image id.

    Returns:
        The expected number of points per class name (lower case); classes without poses are absent.
    """
    counts: dict[str, int] = {}
    for source in (new, old):  # the project's count wins over the tool's
        classes = {lb.label.lower() for items in source.values() for lb in items if lb.keypoints}
        for name in classes:
            of_class = {k: [lb for lb in v if lb.label.lower() == name] for k, v in source.items()}
            found = common_keypoint_count(of_class)
            if found is not None:
                counts[name] = found
    return counts


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


def tag_changes(
    old_labels: list[Label], new_labels: list[Label], old_tags: list[str], new_tags: list[str] | None
) -> tuple[int, int]:
    """How many tags one image gains and loses: its own tags and the tags of its labels.

    Args:
        old_labels: the labels in the project.
        new_labels: the labels brought back.
        old_tags: the image's tags in the project.
        new_tags: the image's tags in the tool, or None when the tool has no image tags.

    Returns:
        ``(added, removed)``.
    """
    before = Counter(t for lb in old_labels for t in lb.tags or [])
    after = Counter(t for lb in new_labels for t in lb.tags or [])
    added, removed = sum((after - before).values()), sum((before - after).values())
    if new_tags is not None:
        mine, theirs = set(old_tags), set(clean_tags(new_tags))
        added, removed = added + len(theirs - mine), removed + len(mine - theirs)
    return added, removed


# -- diff -----------------------------------------------------------------------
PROBLEM_MESSAGES = {
    "unknown_class": "Labels with classes that are not in the project (added to the project on apply).",
    "no_geometry": "Labels without the shape this task needs (they are skipped on apply).",
    "multiple_classes": "Images with more than one class (classification keeps the first).",
    "missing_images": "Images the tool did not return (they are left unchanged).",
    "box_invalid": "Boxes without area or outside the image (they are skipped on apply).",
    "polygon_too_few_points": "Polygons with fewer than 3 points (they are skipped on apply).",
    "keypoints_count_mismatch": (
        "Poses with another number of points than the project's (they are skipped on apply)."
    ),
    "open_polyline": "Open lines of fewer than 3 points are not polygons (they are skipped on apply).",
    "mask_only": "Detections with a mask: only their box is kept.",
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
            code: the problem code (a key of ``PROBLEM_MESSAGES``).
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
    tool_problems: dict[str, dict[str, int]] | None = None,
    image_tags: dict[str, list[str]] | None = None,
    origins: dict[str, list[str | None]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[Label]]]:
    """Clean the new labels and compare them with the ground truth, image by image.

    Args:
        project_dir: the project folder.
        image_ids: the images of the session.
        new_labels: labels brought back from the tool, by image id.
        project: the project (task and class list).
        conflicts: images edited both in the tool and in the project (see :func:`settle`).
        tool_problems: ``{image_id: {code: count}}`` for shapes the tool held that could not be read as
            they are (``Pulled.problems``).
        image_tags: the image tags in the tool (``Pulled.tags``), or None when it has none.
        origins: for the labels of each image, the project label each was written from (``Pulled.origins``,
            see :func:`compare_image`), or None.

    Returns:
        The summary (see :func:`diff`) and the cleaned labels of each image present in ``new_labels``.
    """
    annotations = load_annotations(project_dir)
    old_all = annotations.images if annotations else {}
    names = {r.id: r.name for r in Manifest(project_dir).all()}
    known = {c.lower() for c in project.classes}
    tally = _Tally(names)
    totals = {"added": 0, "removed": 0, "edited": 0}
    tag_totals = {"added": 0, "removed": 0}
    stored_tags = load_tags(project_dir)
    unknown: dict[str, None] = {}
    examples: list[dict[str, Any]] = []
    cleaned: dict[str, list[Label]] = {}
    changed = 0
    for image_id in conflicts or []:
        tally.problem("edited_on_both_sides", image_id)
    for image_id, notes in (tool_problems or {}).items():
        for code, count in notes.items():
            tally.problem(code, image_id, count)
    keypoints = keypoint_counts(old_all, new_labels) if project.task is Task.KEYPOINTS else None

    for image_id in image_ids:
        if image_id not in new_labels:
            tally.problem("missing_images", image_id)
            continue
        kept, dropped = clean_indexed(project, new_labels[image_id], keypoints)
        labels = [label for _, label in kept]
        known_origins = (origins or {}).get(image_id) or []
        image_origins = [known_origins[i] if i < len(known_origins) else None for i, _ in kept]
        cleaned[image_id] = labels
        for code, count in dropped.items():
            if count:
                tally.problem(code, image_id, count)
        for label in labels:
            if label.label.lower() not in known:
                unknown.setdefault(label.label)
                tally.problem("unknown_class", image_id)
        changes = compare_image(project.task, old_all.get(image_id, []), labels, image_origins)
        theirs = image_tags.get(image_id) if image_tags is not None else None
        tags_added, tags_removed = tag_changes(
            old_all.get(image_id, []), labels, stored_tags.get(image_id, []), theirs
        )
        tag_totals["added"] += tags_added
        tag_totals["removed"] += tags_removed
        if changes or tags_added or tags_removed:
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
                        "tags_added": tags_added,
                        "tags_removed": tags_removed,
                    }
                )
    result = {
        "images_changed": changed,
        **totals,
        "per_class": tally.per_class,
        "tags": tag_totals,
        "unknown_classes": list(unknown),
        "problems": [
            {"level": "warning", "code": code, "message": PROBLEM_MESSAGES[code], **entry}
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
    tool_problems: dict[str, dict[str, int]] | None = None,
    image_tags: dict[str, list[str]] | None = None,
    origins: dict[str, list[str | None]] | None = None,
) -> dict[str, Any]:
    """Summary of the changes between the project ground truth and ``new_labels`` for these images.

    Args:
        project_dir: the project folder.
        image_ids: the images of the session.
        new_labels: labels brought back from the tool, by image id.
        project: the project (task and class list).
        conflicts: images edited both in the tool and in the project (``edited_on_both_sides``).
        tool_problems: ``{image_id: {code: count}}`` for shapes the tool could not give as they are.
        image_tags: the image tags in the tool by image id (``Pulled.tags``), or None.
        origins: the project label each label was written from, by image id (``Pulled.origins``), or None.

    Returns:
        ``{images_changed, added, removed, edited, per_class, tags, unknown_classes, problems, examples}``
        (see docs/ANNOTATION.md); ``tags`` is ``{added, removed}``: the tags of the images and of their labels
        gained and lost, also given per image in ``examples`` (``tags_added``, ``tags_removed``). An image
        whose only change is its tags counts in ``images_changed``.
    """
    return _compare_all(
        project_dir, image_ids, new_labels, project, conflicts, tool_problems, image_tags, origins
    )[0]


# -- apply ----------------------------------------------------------------------
def apply(
    store: ProjectStore,
    project_id: int,
    manifest: Manifest,
    image_ids: list[str],
    new_labels: dict[str, list[Label]],
    reason: str = "apply",
    last_session: dict[str, Any] | None = None,
    image_tags: dict[str, list[str]] | None = None,
    origins: dict[str, list[str | None]] | None = None,
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
        last_session: the edit session applied (``{tool, session, at}``), recorded in the annotations'
            ``source`` as ``last_session`` next to the original import information; None records nothing.
        image_tags: the image tags brought back from the tool, by image id: they become the tags of those
            images in the project (``tags.json``); None leaves the image tags as they are.
        origins: the project label each brought-back label was written from (see :func:`compare_image`), so
            the counts pair them as the review did; None pairs by overlap only.

    Returns:
        Counts: the diff counts plus ``tags`` (``{added, removed}``), ``images_updated``, ``new_classes``,
        ``skipped`` and ``backup``.
    """
    project = store.get(project_id)
    project_dir = store.dir(project_id)
    existing = {r.id for r in manifest.all()}
    ids = [i for i in image_ids if i in existing]
    with annotations_lock(project_dir):
        counts = _apply_locked(
            store,
            project_id,
            project,
            project_dir,
            ids,
            new_labels,
            reason,
            last_session,
            image_tags,
            origins,
        )
        if image_tags:
            set_tags(project_dir, {i: image_tags[i] for i in ids if i in image_tags})
        return counts


def _apply_locked(
    store: ProjectStore,
    project_id: int,
    project: Project,
    project_dir: Path,
    ids: list[str],
    new_labels: dict[str, list[Label]],
    reason: str,
    last_session: dict[str, Any] | None = None,
    image_tags: dict[str, list[str]] | None = None,
    origins: dict[str, list[str | None]] | None = None,
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
        last_session: the applied session, as in :func:`apply`.
        image_tags: the image tags brought back, as in :func:`apply` (only counted here).
        origins: the origins of the labels brought back, as in :func:`apply`.

    Returns:
        The same counts as :func:`apply`.
    """
    summary, cleaned = _compare_all(
        project_dir, ids, new_labels, project, image_tags=image_tags, origins=origins
    )
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
    if last_session:
        annotations.source = {**annotations.source, "last_session": last_session}
    save_annotations(project_dir, annotations)
    skipped = sum(p["count"] for p in summary["problems"] if p["code"] in SKIPPED_ON_APPLY)
    return {
        "images_changed": summary["images_changed"],
        "added": summary["added"],
        "removed": summary["removed"],
        "edited": summary["edited"],
        "tags": summary["tags"],
        "images_updated": len(cleaned),
        "new_classes": new_classes,
        "skipped": skipped,
        "backup": backup,
    }
