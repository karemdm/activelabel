"""Annotation sessions of a project, kept in ``<project>/annotate.json``."""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.labels import load_annotations
from activelabel.preannotate import unreviewed_images
from activelabel.storage import now, read_json, write_json
from activelabel.tasks_store import TaskNotFound, TaskStore, jobs

SCOPES = ("all", "unlabeled", "selected", "issue", "preannotated", "task")
OPEN, PULLED, APPLIED, DISCARDED = "open", "pulled", "applied", "discarded"
_lock = threading.RLock()


class SessionNotFound(KeyError):
    """No session with this id."""


def _file(project_dir: Path) -> Path:
    """Path of ``annotate.json`` in the project folder.

    Args:
        project_dir: the project folder.

    Returns:
        The path of the sessions file.
    """
    return project_dir / "annotate.json"


def session_dir(project_dir: Path, session_id: str) -> Path:
    """Folder for the files of a session (pulled labels).

    Args:
        project_dir: the project folder.
        session_id: id of the session.

    Returns:
        The folder path (not created here).
    """
    return project_dir / "annotate" / session_id


def new_id() -> str:
    """A short unique session id.

    Returns:
        Eight hexadecimal characters.
    """
    return uuid.uuid4().hex[:8]


def _task_images(manifest: Manifest, task: int | None, job: int | None) -> set[str]:
    """Ids of the images of a task, or of one of its jobs.

    Args:
        manifest: the images of the project.
        task: id of the task.
        job: number of the job (1-based), or None for the whole task.

    Returns:
        The image ids.

    Raises:
        CodedError: no task chosen, or the task or the job does not exist.
    """
    if task is None:
        raise CodedError("task_required", "Choose a task.")
    try:
        found = TaskStore(manifest.project_dir).get(task)
    except TaskNotFound:
        raise CodedError("task_not_found", "This task does not exist.", task=task) from None
    if job is None:
        return set(found.images)
    for item in jobs(found):
        if item.n == job:
            return set(item.image_ids)
    raise CodedError("job_not_found", "This job does not exist.", task=task, job=job)


def select_images(
    manifest: Manifest,
    scope: str,
    ids: list[str] | None = None,
    issue: str = "",
    task: int | None = None,
    job: int | None = None,
) -> list[str]:
    """Ids of the images a session covers.

    Args:
        manifest: the images of the project (excluded ones are never chosen).
        scope: one of :data:`SCOPES`.
        ids: for scope ``selected``, the image ids to keep.
        issue: for scope ``issue``, the quality issue the images must have.
        task: for scope ``task``, the id of the task.
        job: for scope ``task``, the job (1-based) to keep; None for the whole task.

    Returns:
        The ids of the chosen images, in manifest order.

    Raises:
        CodedError: unknown scope, unknown task or job, or nothing selected (message is readable).
    """
    active = manifest.active()
    if scope == "all":
        chosen = [r.id for r in active]
    elif scope == "unlabeled":
        annotations = load_annotations(manifest.project_dir)
        done = set(annotations.images) if annotations else set()
        chosen = [r.id for r in active if r.id not in done]
    elif scope == "selected":
        wanted = set(ids or [])
        chosen = [r.id for r in active if r.id in wanted]
    elif scope == "preannotated":  # labelled by a model, not reviewed yet
        pending = set(unreviewed_images(manifest.project_dir))
        chosen = [r.id for r in active if r.id in pending]
    elif scope == "task":
        wanted = _task_images(manifest, task, job)
        chosen = [r.id for r in active if r.id in wanted]
    elif scope == "issue":
        if not issue:
            raise CodedError("issue_required", "Choose an issue.")
        chosen = [r.id for r in active if issue in r.issues]
    else:
        raise CodedError(
            "scope_unknown",
            f"Unknown scope '{scope}' (use: {', '.join(SCOPES)}).",
            scope=scope,
        )
    if not chosen:
        raise CodedError(
            "no_images_match", "No images match this choice. Pick other images or another scope."
        )
    return chosen


def _load(project_dir: Path) -> list[dict[str, Any]]:
    """Read the sessions file.

    Args:
        project_dir: the project folder.

    Returns:
        The sessions as stored (an empty list when there is no file).
    """
    data = read_json(_file(project_dir), {"sessions": []}) or {}
    return list(data.get("sessions", []))


def _save(project_dir: Path, sessions: list[dict[str, Any]]) -> None:
    """Write the sessions file.

    Args:
        project_dir: the project folder.
        sessions: all the sessions to keep.
    """
    write_json(_file(project_dir), {"sessions": sessions})


def create(
    project_dir: Path,
    tool: str,
    scope: str,
    image_ids: list[str],
    links: list[dict[str, str]],
    ref: dict[str, Any],
    session_id: str | None = None,
    task: int | None = None,
    job: int | None = None,
) -> dict[str, Any]:
    """Record a new open session.

    Args:
        project_dir: the project folder.
        tool: name of the annotation tool.
        scope: how the images were chosen (see :data:`SCOPES`).
        image_ids: ids of the images of the session.
        links: links to open in the tool (``{"label", "url"}``).
        ref: the tool's own reference of the session.
        session_id: id to use; a new one is made when None.
        task: for scope ``task``, the id of the task the images belong to.
        job: for scope ``task``, the job of that task (None: the whole task).

    Returns:
        The new session.

    Raises:
        ValueError: no images.
    """
    if not image_ids:
        raise ValueError("No images to annotate. Pick at least one image.")
    session = {
        "id": session_id or new_id(),
        "tool": tool,
        "scope": scope,
        "images": len(image_ids),
        "image_ids": list(image_ids),
        "created": now(),
        "status": OPEN,
        "links": links,
        "ref": ref,
    }
    if task is not None:
        session["task"] = task
        if job is not None:
            session["job"] = job
    with _lock:
        _save(project_dir, [*_load(project_dir), session])
    return session


def list_sessions(project_dir: Path) -> list[dict[str, Any]]:
    """Sessions, newest first.

    Args:
        project_dir: the project folder.

    Returns:
        The sessions ordered by creation time, newest first.
    """
    return sorted(_load(project_dir), key=lambda s: s["created"], reverse=True)


def get(project_dir: Path, session_id: str) -> dict[str, Any]:
    """One session.

    Args:
        project_dir: the project folder.
        session_id: id of the session.

    Returns:
        The session.

    Raises:
        SessionNotFound: unknown id.
    """
    for session in _load(project_dir):
        if session["id"] == session_id:
            return session
    raise SessionNotFound(session_id)


def update(project_dir: Path, session_id: str, **fields: Any) -> dict[str, Any]:
    """Change fields of a session (status, diff, ...).

    Args:
        project_dir: the project folder.
        session_id: id of the session.
        **fields: the fields to set; they replace the stored values.

    Returns:
        The session after the change.

    Raises:
        SessionNotFound: unknown id.
    """
    with _lock:
        sessions = _load(project_dir)
        for session in sessions:
            if session["id"] == session_id:
                session.update(fields)
                _save(project_dir, sessions)
                return session
    raise SessionNotFound(session_id)


def close(project_dir: Path, session_id: str) -> None:
    """Forget a session and its files.

    Args:
        project_dir: the project folder.
        session_id: id of the session.

    Raises:
        SessionNotFound: unknown id.
    """
    with _lock:
        sessions = _load(project_dir)
        kept = [s for s in sessions if s["id"] != session_id]
        if len(kept) == len(sessions):
            raise SessionNotFound(session_id)
        _save(project_dir, kept)
    shutil.rmtree(session_dir(project_dir, session_id), ignore_errors=True)


def protected_images(project_dir: Path) -> set[str]:
    """Ids of the images of FiftyOne sessions still being edited (not applied or discarded).

    Their ``ground_truth`` in the FiftyOne dataset may hold edits not yet brought back, so refreshing
    the dataset keeps the ones edited in FiftyOne (``fiftyone_bridge.is_edited``) and rewrites the rest.

    Args:
        project_dir: the project folder.

    Returns:
        The image ids.
    """
    return {
        i
        for s in _load(project_dir)
        if s.get("tool") == "fiftyone" and s.get("status") in (OPEN, PULLED)
        for i in s.get("image_ids", [])
    }


def public(session: dict[str, Any]) -> dict[str, Any]:
    """Session as the API returns it (no image ids, label hashes or private tool reference).

    Args:
        session: a stored session.

    Returns:
        A copy of the session without ``image_ids``, ``ref`` and ``base``.
    """
    return {k: v for k, v in session.items() if k not in ("image_ids", "ref", "base")}
