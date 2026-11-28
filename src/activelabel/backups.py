"""Safety copies of a project's ``annotations.json``: make them, list them and restore one.

A copy is made before anything that replaces labels in bulk (a class removed, a task change that discards the
labels, a FiftyOne or CVAT apply, a class renamed in CVAT, an import that replaces, a restore). It is saved
next to the original as ``annotations.<UTC time>Z[.<reason>].bak.json``; the reason is one of :data:`REASONS`.
Copies made before this naming existed have no ``Z``; their time is the file's modification time. Only the
newest :data:`KEEP_BACKUPS` copies of a project are kept.
"""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.labels import AnnotationSet, annotations_lock, annotations_path, save_annotations
from activelabel.projects import ProjectStore
from activelabel.storage import DamagedFile, read_json

REASONS = ("classes", "task", "apply", "sync", "rename", "replace", "restore")
KEEP_BACKUPS = 20  # copies kept per project; older ones are deleted when a new one is made
STAMP = "%Y%m%dT%H%M%S"
NAME = re.compile(
    r"annotations\.(?P<stamp>\d{8}T\d{6})(?P<utc>Z?)(?:-(?P<n>\d+))?(?:\.(?P<reason>[a-z]+))?\.bak\.json"
)


def make_backup(project_dir: Path, reason: str = "") -> str:
    """Copy ``annotations.json`` to a new safety copy and delete the copies beyond :data:`KEEP_BACKUPS`.

    Call it while holding :func:`activelabel.labels.annotations_lock`, before the file is changed.

    Args:
        project_dir: the project folder.
        reason: why the copy is made (one of :data:`REASONS`), kept in the name; empty when unknown.

    Returns:
        The file name of the copy, or an empty string when the project has no ``annotations.json``.

    Raises:
        ValueError: ``reason`` is not one of :data:`REASONS`.
    """
    if reason and reason not in REASONS:
        raise ValueError(f"Unknown backup reason: {reason}")
    source = annotations_path(project_dir)
    if not source.exists():
        return ""
    stamp = datetime.now(UTC).strftime(STAMP) + "Z"
    suffix = f".{reason}" if reason else ""
    # two copies in the same second must not overwrite each other, and the later one sorts as newer
    same = (NAME.fullmatch(p.name) for p in project_dir.glob(f"annotations.{stamp}*.bak.json"))
    n = 1 + max((int(m["n"] or 1) for m in same if m is not None), default=0)
    name = f"annotations.{stamp}{suffix}.bak.json" if n == 1 else f"annotations.{stamp}-{n}{suffix}.bak.json"
    shutil.copyfile(source, project_dir / name)
    prune_backups(project_dir)
    return name


def prune_backups(project_dir: Path, keep: int = KEEP_BACKUPS) -> list[str]:
    """Delete the oldest safety copies of a project so that at most ``keep`` remain.

    Args:
        project_dir: the project folder.
        keep: how many of the newest copies to keep.

    Returns:
        The names of the copies deleted, oldest last.
    """
    removed = []
    for path, _match, _created_at in _copies(project_dir)[keep:]:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def _copies(project_dir: Path) -> list[tuple[Path, re.Match[str], str]]:
    """The safety copy files of a project, newest first (without reading them).

    Args:
        project_dir: the project folder.

    Returns:
        ``(path, match of its name, creation time in UTC)`` per copy.
    """
    found = []
    for path in project_dir.glob("annotations.*.bak.json"):
        match = NAME.fullmatch(path.name)
        if match is not None:
            found.append((path, match, _created(path, match["stamp"], bool(match["utc"]))))
    return sorted(found, key=lambda item: (item[2], int(item[1]["n"] or 1)), reverse=True)


def _created(path: Path, stamp: str, utc: bool) -> str:
    """Time a copy was made, ISO 8601 in UTC.

    Args:
        path: the copy.
        stamp: the time written in its name.
        utc: whether the name says that the time is UTC.

    Returns:
        The time from the name; the file's modification time for a copy made before names carried UTC.
    """
    if utc:
        moment = datetime.strptime(stamp, STAMP).replace(tzinfo=UTC)
    else:
        moment = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return moment.isoformat(timespec="seconds")


def _counts(path: Path) -> tuple[int, int] | None:
    """Images with labels and labels of a copy, or None when it cannot be read as annotations.

    Args:
        path: the copy.

    Returns:
        ``(images, labels)``, or None.
    """
    try:
        data = read_json(path)
        found = AnnotationSet.from_dict(data) if data else None
    except (DamagedFile, KeyError, ValueError, TypeError, AttributeError):
        return None
    if found is None:
        return None
    return sum(1 for v in found.images.values() if v), sum(len(v) for v in found.images.values())


def list_backups(project_dir: Path) -> list[dict[str, Any]]:
    """The safety copies of a project, newest first.

    Args:
        project_dir: the project folder.

    Returns:
        One dict per copy: ``name``, ``created`` (UTC), ``images`` and ``labels`` (None when the copy
        cannot be read), ``reason`` (empty when unknown) and ``valid`` (it can be restored).
    """
    found = []
    for path, match, created in _copies(project_dir):
        counts = _counts(path)
        found.append(
            {
                "name": path.name,
                "created": created,
                "images": counts[0] if counts else None,
                "labels": counts[1] if counts else None,
                "reason": match["reason"] or "",
                "valid": counts is not None,
            }
        )
    return found


def backup_path(project_dir: Path, name: str) -> Path:
    """The file of a safety copy, after checking that ``name`` is really the name of one.

    Args:
        project_dir: the project folder.
        name: the name sent by the caller; only the names :func:`make_backup` writes are accepted, so
            nothing outside the project folder can be reached.

    Returns:
        The path of the copy.

    Raises:
        CodedError: the name is not that of a safety copy (``backup_name_invalid``).
        FileNotFoundError: no such copy.
    """
    if NAME.fullmatch(name) is None:
        raise CodedError("backup_name_invalid", "That is not the name of a safety copy.")
    path = project_dir / name
    if not path.is_file():
        raise FileNotFoundError("That safety copy no longer exists.")
    return path


def restore_backup(store: ProjectStore, project_id: int, name: str) -> dict[str, Any]:
    """Make a safety copy the ground truth again (after a copy of the current file).

    The project takes the task of the copy, and gets the classes of the copy it does not have yet (so no
    label is left with a class the project does not know).

    Args:
        store: the projects.
        project_id: the project.
        name: file name of the copy (see :func:`list_backups`).

    Returns:
        ``restored`` (the name), ``backup`` (the copy of the file that was replaced, empty when there was
        none), ``images`` and ``labels`` of the restored ground truth.

    Raises:
        CodedError: invalid name, or a copy that cannot be read (``backup_unreadable``).
        FileNotFoundError: no such copy.
    """
    project_dir = store.dir(project_id)
    path = backup_path(project_dir, name)
    with annotations_lock(project_dir):
        counts = _counts(path)
        if counts is None:
            raise CodedError(
                "backup_unreadable", "That safety copy cannot be read, so it cannot be restored."
            )
        restored = AnnotationSet.from_dict(read_json(path))
        project = store.get(project_id)
        known = {c.lower() for c in project.classes}
        classes = [*project.classes]
        for label in [*restored.classes, *(lb.label for v in restored.images.values() for lb in v)]:
            if label.lower() not in known:
                known.add(label.lower())
                classes.append(label)
        saved = make_backup(project_dir, "restore")
        if restored.task != project.task or classes != project.classes:
            project = store.update(project_id, task=restored.task, classes=classes)
        restored.classes = list(project.classes)
        save_annotations(project_dir, restored)
    return {"restored": name, "backup": saved, "images": counts[0], "labels": counts[1]}
