"""Shared steps of the CVAT and Label Studio imports.

Unique project name, task inference (majority vote, polygon -> box conversion, kinds that do not
fit are dropped and counted), creation of the project and its rollback when the import fails or is
cancelled, and the final report. Secrets never reach this module.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.annotate.base import ToolError
from activelabel.annotations.models import Problem
from activelabel.annotations.validation import to_detection
from activelabel.images.ingest import IngestReport
from activelabel.jobs import Progress
from activelabel.labels import AnnotationSet, Label, annotations_lock, save_annotations
from activelabel.projects import MAX_DESCRIPTION, MAX_NAME, Project, ProjectStore, clean_classes
from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task

TOOL_NAMES = {"cvat": "CVAT", "labelstudio": "Label Studio"}
ORIGIN_FILE = "origin.json"
# Order used to break ties between kinds with the same number of annotations (richest first).
KIND_ORDER = (Task.KEYPOINTS, Task.SEGMENTATION, Task.DETECTION, Task.CLASSIFICATION)
MAX_ITEMS = 200  # most remote projects/tasks listed


class RemoteNotFound(ToolError):
    """The remote project or task does not exist (HTTP 404 in the API)."""


def kind_of(label: Label) -> Task:
    """What a label is: keypoints, polygons, a box or a class of the whole image."""
    if label.keypoints:
        return Task.KEYPOINTS
    if label.polygons:
        return Task.SEGMENTATION
    if label.bbox:
        return Task.DETECTION
    return Task.CLASSIFICATION


@dataclass
class TaskPlan:
    """Result of :func:`plan_task`: the task, the labels that fit it and what was changed."""

    task: Task
    labels: dict[str, list[Label]]
    problems: list[Problem] = field(default_factory=list)


def _problem(code: str, message: str, counts: Counter[str]) -> Problem:
    return Problem("warning", code, message, sum(counts.values()), [f"{k}: {n}" for k, n in counts.items()])


def plan_task(labels: dict[str, list[Label]], hint: Task = Task.DETECTION) -> TaskPlan:
    """Choose the task of an imported project from its annotations.

    The most frequent kind wins (ties: keypoints, segmentation, detection, classification). Polygons
    and keypoints become boxes in a detection project; every other kind that does not fit is left
    out. Classification keeps the first class of each image. Without annotations the task is ``hint``.

    Args:
        labels: labels per image id (an empty list is an image annotated without objects).
        hint: task used when there are no annotations at all.

    Returns:
        The task, the converted labels and the problems (counts of converted / left out annotations).
    """
    counts: Counter[Task] = Counter(kind_of(lb) for items in labels.values() for lb in items)
    if not counts:
        return TaskPlan(hint, {k: list(v) for k, v in labels.items()})
    task = max(KIND_ORDER, key=lambda t: (counts[t], -KIND_ORDER.index(t)))
    out: dict[str, list[Label]] = {}
    converted: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    multiple = 0
    for key, items in labels.items():
        kept: list[Label] = []
        for lb in items:
            kind = kind_of(lb)
            if kind is task:
                kept.append(lb)
            elif task is Task.DETECTION and kind in (Task.SEGMENTATION, Task.KEYPOINTS):
                box = to_detection(lb)
                if box is not None:
                    kept.append(box)
                    converted[kind.value] += 1
                else:
                    dropped[kind.value] += 1
            else:
                dropped[kind.value] += 1
        if task is Task.CLASSIFICATION and len({lb.label.lower() for lb in kept}) > 1:
            multiple += 1
        if items and not kept:
            continue  # everything it had was left out: the image stays unannotated
        out[key] = kept[:1] if task is Task.CLASSIFICATION else kept
    problems: list[Problem] = []
    if converted:
        problems.append(
            _problem("converted_to_box", "Polygons or keypoints were converted to boxes.", converted)
        )
    if dropped:
        problems.append(
            _problem(
                "dropped_other_kind", f"Annotations that do not fit {task.value} were left out.", dropped
            )
        )
    if multiple:
        message = "Some images have more than one class; only the first one is kept."
        problems.append(Problem("warning", "multiple_classes", message, multiple))
    return TaskPlan(task, out, problems)


def unique_name(store: ProjectStore, name: str) -> str:
    """``name`` or ``name (2)``, ``name (3)``... so that no project has it (ignoring case)."""
    base = " ".join(name.split())[:MAX_NAME].strip() or "Imported project"
    taken = {p.name.lower() for p in store.list()}
    candidate, n = base, 1
    while candidate.lower() in taken:
        n += 1
        suffix = f" ({n})"
        candidate = base[: MAX_NAME - len(suffix)].rstrip() + suffix
    return candidate


def file_id(path: Path) -> str:
    """Content id of an image file, the same as the one the project stores it under."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class FixedMessage(Progress):
    """Progress handle that keeps one message (the ingest would show each file name)."""

    def __init__(self, inner: Progress, message: str) -> None:
        self._inner = inner
        self._message = message

    def update(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """Forward the counts with the fixed message (also checks for cancellation)."""
        self._inner.update(done, total, self._message)

    def check(self) -> None:
        """Raise Cancelled when the job was cancelled."""
        self._inner.check()


@dataclass
class Origin:
    """Where a project came from (``<project>/origin.json``)."""

    tool: str
    url: str
    remote_id: str
    kind: str
    name: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, with the import time."""
        return {**self.__dict__, "imported_at": now()}


def read_origin(project_dir: Path) -> dict[str, Any] | None:
    """Origin of an imported project (None for projects created here)."""
    data = read_json(project_dir / ORIGIN_FILE)
    return data if isinstance(data, dict) else None


@contextmanager
def new_project(
    store: ProjectStore, origin: Origin, name: str, description: str, hint: Task
) -> Iterator[Project]:
    """Create the project; delete it again when the block fails or the job is cancelled."""
    text = (description or "").strip()  # the origin badge says where it came from
    project = store.create(unique_name(store, name or origin.name), text[:MAX_DESCRIPTION], hint)
    try:
        yield project
    except BaseException:
        shutil.rmtree(store.dir(project.id), ignore_errors=True)
        raise


def ingest_problems(report: IngestReport) -> list[Problem]:
    """Rejected and repeated images as problems."""
    out: list[Problem] = []
    if report.rejected:
        out.append(
            Problem(
                "warning",
                "images_rejected",
                "Some files are not readable images and were skipped.",
                len(report.rejected),
                [f"{r['name']} ({r['reason']})" for r in report.rejected[:5]],
            )
        )
    if report.duplicates:
        out.append(
            Problem(
                "info",
                "images_duplicate",
                "Identical images were added once.",
                len(report.duplicates),
                report.duplicates[:5],
            )
        )
    return out


def finish(
    store: ProjectStore,
    project: Project,
    labels: dict[str, list[Label]],
    declared: list[str],
    origin: Origin,
    report: IngestReport,
    problems: list[Problem],
    hint: Task,
) -> dict[str, Any]:
    """Save task, classes, annotations and origin of the new project; return the job result.

    Args:
        store: projects.
        project: the created project.
        labels: labels per image id of the project (only ids that are images of the project).
        declared: class names the remote project declares (kept even when unused).
        origin: where the project came from.
        report: what the image import did.
        problems: problems found so far (the task plan and the ingest add theirs).
        hint: task when there are no annotations.

    Returns:
        The job result described in docs/IMPORT.md.
    """
    plan = plan_task(labels, hint)
    names = list(declared) if plan.task is not Task.KEYPOINTS else []
    for items in plan.labels.values():
        names += [lb.label for lb in items]
    classes = clean_classes(names)
    canon = {c.lower(): c for c in classes}
    for items in plan.labels.values():
        for lb in items:
            lb.label = canon.get(lb.label.lower(), lb.label)
    project_dir = store.dir(project.id)
    if plan.labels:
        source = {"format": origin.tool, "imported": now(), "remote_id": origin.remote_id}
        with annotations_lock(project_dir):
            save_annotations(project_dir, AnnotationSet(plan.task, classes, plan.labels, source))
    updated = store.update(project.id, task=plan.task, classes=classes)
    write_json(project_dir / ORIGIN_FILE, origin.as_dict())
    all_problems = [*problems, *plan.problems, *ingest_problems(report)]
    return {
        "project_id": project.id,
        "name": updated.name,
        "task": plan.task.value,
        "images_added": report.added,
        "images_rejected": len(report.rejected),
        "labels": sum(len(v) for v in plan.labels.values()),
        "classes": classes,
        "problems": [p.as_dict() for p in all_problems],
    }
