"""Apply staged annotation files to a project's ground truth."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from activelabel.annotations.parsing import detect_format, parse
from activelabel.annotations.staging import discard_staged, get_target, load_staged
from activelabel.annotations.target import build_target
from activelabel.annotations.validation import CHANGE_TASK, CONVERT, DISCARD, IMPORT, to_detection, validate
from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, Label, annotations_lock, load_annotations, save_annotations
from activelabel.projects import ProjectStore
from activelabel.storage import now
from activelabel.tasks import Task

ACTIONS = (IMPORT, CHANGE_TASK, CONVERT, DISCARD)
MODES = ("replace", "merge")


def apply(
    store: ProjectStore, project_id: int, manifest: Manifest, token: str, action: str, mode: str = "replace"
) -> dict[str, Any]:
    """Run staged annotation files through parse and validate again, then act.

    Args:
        store: projects.
        project_id: target project.
        manifest: images of the project (active ones are matched).
        token: staging token from :func:`activelabel.annotations.staging.stage`.
        action: import, change_task_and_import, convert_and_import or discard.
        mode: ``replace`` swaps the ground truth; ``merge`` adds labels only for images that have
            none yet and keeps the existing ones.

    Returns:
        Summary: action, mode, format, task, images/labels imported, kept (merge), added classes.

    Raises:
        ValueError: unknown action or mode, action not allowed for these files, nothing to import,
            or merge into labels of another task.
    """
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    project_dir = store.dir(project_id)
    project = store.get(project_id)
    if action == DISCARD:
        discard_staged(project_dir, token)
        return {"action": DISCARD, "project_id": project_id}
    files = load_staged(project_dir, token)
    fmt = detect_format(files)
    if fmt is None:
        raise ValueError("The annotation format could not be recognised.")
    saved_target = get_target(project_dir, token)
    target = None
    if saved_target:
        target = build_target(
            project_dir, manifest, saved_target.get("task"), str(saved_target.get("class_name") or "")
        )
    result = parse(files, fmt, target.images() if target else manifest.active(), target)
    report = validate(result, project)
    if action not in report.options:
        raise ValueError(f"Action '{action}' is not possible for these files (options: {report.options}).")
    if report.inferred_task is None:
        raise ValueError("The type of these annotations could not be recognised.")
    task = Task(report.inferred_task) if action == CHANGE_TASK else project.task

    labels = {k: [replace(lb) for lb in v] for k, v in result.labels.items()}
    converted = 0
    if action == CONVERT:
        for key, items in labels.items():
            boxes = [to_detection(lb) for lb in items]
            converted += sum(1 for b in boxes if b)
            labels[key] = [b for b in boxes if b]
    reduced = 0
    if task is Task.CLASSIFICATION:  # one class per image: keep the first
        for key, items in labels.items():
            if len({lb.label.lower() for lb in items}) > 1:
                reduced += 1
            labels[key] = items[:1]
    canon = {c.lower(): c for c in project.classes}
    for items in labels.values():
        for lb in items:
            lb.label = canon.get(lb.label.lower(), lb.label)
    new_classes = list(project.classes)
    seen = {c.lower() for c in new_classes}
    for items in labels.values():
        for lb in items:
            if lb.label.lower() not in seen:
                new_classes.append(lb.label)
                seen.add(lb.label.lower())
    for name in result.classes:  # classes declared in the files but not used
        if name.lower() not in seen:
            new_classes.append(canon.get(name.lower(), name))
            seen.add(name.lower())

    # read, merge and write under one lock: a review, sync or pre-annotation saved meanwhile is kept
    with annotations_lock(project_dir):
        existing = load_annotations(project_dir)
        kept = added = n_labels = 0
        final: dict[str, list[Label]]
        if mode == "merge" and existing is not None:
            if existing.task != task and any(existing.images.values()):
                raise ValueError("Existing labels are of another task; use replace instead of merge.")
            final = dict(existing.images)
            for key, items in labels.items():
                if final.get(key):
                    kept += 1
                elif items:
                    final[key] = items
                    added += 1
                    n_labels += len(items)
                else:
                    final.setdefault(key, [])
        elif target is not None and existing is not None:
            # the files are for one video task: only the images of that task are replaced
            mine = {r.id for r in target.frames.values()}
            if existing.task != task and any(v for k, v in existing.images.items() if k not in mine):
                raise ValueError(
                    "Other images have labels of another task; change the task of the project first."
                )
            final = {k: v for k, v in existing.images.items() if k not in mine}
            final.update(labels)
            added = sum(1 for v in labels.values() if v)
            n_labels = sum(len(v) for v in labels.values())
        else:
            final = labels
            added = sum(1 for v in labels.values() if v)
            n_labels = sum(len(v) for v in labels.values())
        source = {"format": fmt, "files": result.files, "imported": now(), "mode": mode}
        save_annotations(project_dir, AnnotationSet(task, new_classes, final, source))
        changes: dict[str, Any] = {}
        if task != project.task:
            changes["task"] = task
        if new_classes != project.classes:
            changes["classes"] = new_classes
        updated = store.update(project_id, **changes) if changes else project
    discard_staged(project_dir, token)
    return {
        "action": action,
        "mode": mode,
        "project_id": project_id,
        "format": fmt,
        "task": task.value,
        "images_imported": added,
        "labels_imported": n_labels,
        "images_kept": kept,
        "images_added": added,
        "converted": converted,
        "classes_reduced": reduced,
        "classes_added": [c for c in updated.classes if c not in project.classes],
        "problems": [p.as_dict() for p in report.problems],
    }
