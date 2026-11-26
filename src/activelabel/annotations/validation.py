"""Check a parse result against a project and say what the user can do."""

from __future__ import annotations

from collections import Counter

from activelabel.annotations.context import bbox_of_points
from activelabel.annotations.models import MAX_EXAMPLES, PROBLEMS, ParseResult, Problem, ValidationReport
from activelabel.labels import KEYPOINTS_COUNT_MISMATCH, NO_GEOMETRY, Label, task_of, validate_labels
from activelabel.projects import Project
from activelabel.tasks import Task

IMPORT = "import"
CHANGE_TASK = "change_task_and_import"
CONVERT = "convert_and_import"
DISCARD = "discard"
CONVERTIBLE_TO_DETECTION = (Task.SEGMENTATION, Task.KEYPOINTS)


def validate(result: ParseResult, project: Project) -> ValidationReport:
    """Build the report the user sees before importing.

    Options: ``import`` (task matches), ``change_task_and_import`` (set the project task to the one the
    labels fit), ``convert_and_import`` (segmentation or keypoints to a detection project, using the box
    of each object), ``discard``. Detection to classification is never offered. When nothing can be
    imported only ``discard`` is offered.

    Labels without the shape the task needs (a box in a segmentation file, a pose with another number of
    points than most) are reported as ``no_geometry`` / ``keypoints_count_mismatch`` warnings, with the number
    and example image ids; :func:`activelabel.annotations.importer.apply` leaves them out.

    Args:
        result: what the parser read.
        project: the project the files are meant for.

    Returns:
        The report with the inferred task, counts, problems and the options the user has.
    """
    labels = result.all_labels()
    inferred = task_of(labels)
    problems = [Problem(p.level, p.code, p.message, p.count, list(p.examples)) for p in result.problems]
    if result.images_matched == 0:
        # one message: "no_images_matched" carries the same count and examples as "unmatched_entries"
        problems = [p for p in problems if p.code != "unmatched_entries"]
        level, message = PROBLEMS["no_images_matched"]
        problems.insert(
            0, Problem(level, "no_images_matched", message, result.unmatched_count, result.unmatched[:5])
        )
    elif not labels:
        level, message = PROBLEMS["no_labels"]
        problems.insert(0, Problem(level, "no_labels", message, 0, []))
    if inferred is Task.CLASSIFICATION:
        many = [k for k, items in result.labels.items() if len({lb.label.lower() for lb in items}) > 1]
        if many:
            problems.append(
                Problem(
                    "warning",
                    "multiple_classes",
                    f"{len(many)} images have more than one class; only the first one is kept",
                    len(many),
                    many[:MAX_EXAMPLES],
                )
            )
    matches = inferred is not None and inferred == project.task
    converts = project.task is Task.DETECTION and inferred in CONVERTIBLE_TO_DETECTION
    if inferred is not None and (
        matches or not converts
    ):  # labels the task cannot use are left out on import
        found = validate_labels(inferred, result.labels)
        for code in (NO_GEOMETRY, KEYPOINTS_COUNT_MISMATCH):
            if found.counts[code]:
                level, message = PROBLEMS[code]
                problems.append(Problem(level, code, message, found.counts[code], found.examples[code]))
    can_import = bool(labels) and not any(p.level == "error" for p in problems)
    options = [DISCARD]
    if can_import and inferred is not None:
        if matches:
            options.insert(0, IMPORT)
        else:
            options.insert(0, CHANGE_TASK)
            if project.task is Task.DETECTION and inferred in CONVERTIBLE_TO_DETECTION:
                options.insert(0, CONVERT)
    known = {c.lower() for c in project.classes}
    with_labels = sum(1 for v in result.labels.values() if v)
    return ValidationReport(
        format=result.format,
        inferred_task=inferred.value if inferred else None,
        project_task=project.task.value,
        task_matches=matches,
        classes_found=list(result.classes),
        classes_new=[c for c in result.classes if known and c.lower() not in known],
        counts={
            "annotation_files": len(result.files),
            "images_matched": result.images_matched,
            "images_with_labels": with_labels,
            "images_without_labels": max(0, result.images_total - with_labels),
            "labels_total": len(labels),
            "labels_per_class": dict(Counter(lb.label for lb in labels)),
        },
        problems=problems,
        can_import=can_import,
        options=options,
        unmatched={"count": result.unmatched_count, "examples": result.unmatched[:5]},
    )


def to_detection(label: Label) -> Label | None:
    """Box-only version of a label (its box, or the box of its polygons or visible keypoints).

    Args:
        label: any label.

    Returns:
        A label with only the box (and class and track), or None when the label has no box to keep.
    """
    bbox = label.bbox
    if bbox is None and label.polygons:
        bbox = bbox_of_points(p for ring in label.polygons for p in ring)
    if bbox is None and label.keypoints:
        bbox = bbox_of_points(k for k in label.keypoints if k[2] > 0)
    return Label(label.label, bbox=bbox, track=label.track) if bbox else None
