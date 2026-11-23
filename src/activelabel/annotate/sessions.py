"""Annotation sessions of a project, kept in ``<project>/annotate.json``."""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from activelabel.images.manifest import Manifest
from activelabel.labels import load_annotations
from activelabel.storage import now, read_json, write_json

SCOPES = ("all", "unlabeled", "selected", "issue", "preannotated")
OPEN, PULLED, APPLIED, DISCARDED = "open", "pulled", "applied", "discarded"
_lock = threading.RLock()


class SessionNotFound(KeyError):
    """No session with this id."""


def _file(project_dir: Path) -> Path:
    return project_dir / "annotate.json"


def session_dir(project_dir: Path, session_id: str) -> Path:
    """Folder for the files of a session (pulled labels)."""
    return project_dir / "annotate" / session_id


def new_id() -> str:
    """A short unique session id."""
    return uuid.uuid4().hex[:8]


def select_images(manifest: Manifest, scope: str, ids: list[str] | None = None, issue: str = "") -> list[str]:
    """Ids of the images a session covers.

    Raises:
        ValueError: unknown scope or nothing selected (message is readable).
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
        from activelabel.preannotate import unreviewed_images

        pending = set(unreviewed_images(manifest.project_dir))
        chosen = [r.id for r in active if r.id in pending]
    elif scope == "issue":
        if not issue:
            raise ValueError("Choose an issue.")
        chosen = [r.id for r in active if issue in r.issues]
    else:
        raise ValueError(f"Unknown scope '{scope}' (use: {', '.join(SCOPES)}).")
    if not chosen:
        raise ValueError("No images match this choice. Pick other images or another scope.")
    return chosen


def _load(project_dir: Path) -> list[dict[str, Any]]:
    data = read_json(_file(project_dir), {"sessions": []}) or {}
    return list(data.get("sessions", []))


def _save(project_dir: Path, sessions: list[dict[str, Any]]) -> None:
    write_json(_file(project_dir), {"sessions": sessions})


def create(
    project_dir: Path,
    tool: str,
    scope: str,
    image_ids: list[str],
    links: list[dict[str, str]],
    ref: dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Record a new open session.

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
    with _lock:
        _save(project_dir, [*_load(project_dir), session])
    return session


def list_sessions(project_dir: Path) -> list[dict[str, Any]]:
    """Sessions, newest first."""
    return sorted(_load(project_dir), key=lambda s: s["created"], reverse=True)


def get(project_dir: Path, session_id: str) -> dict[str, Any]:
    """One session.

    Raises:
        SessionNotFound: unknown id.
    """
    for session in _load(project_dir):
        if session["id"] == session_id:
            return session
    raise SessionNotFound(session_id)


def update(project_dir: Path, session_id: str, **fields: Any) -> dict[str, Any]:
    """Change fields of a session (status, diff, ...).

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


def public(session: dict[str, Any]) -> dict[str, Any]:
    """Session as the API returns it (without the image ids and the tool's private reference)."""
    return {k: v for k, v in session.items() if k not in ("image_ids", "ref")}
