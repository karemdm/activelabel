"""HTTP API of the safety copies of a project's annotations (see docs/DESIGN.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from activelabel import backups
from activelabel.errors import CodedError
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


@router.get("/projects/{project_id}/annotations/backups")
def list_backups(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The safety copies of the annotations, newest first: name, created (UTC), images, labels, reason.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{items}``, newest first.
    """
    state.project(project_id)
    return {"items": backups.list_backups(state.store.dir(project_id))}


@router.post("/projects/{project_id}/annotations/backups/{name}/restore")
def restore_backup(project_id: int, name: str, state: AppState = STATE) -> dict[str, Any]:
    """Make a safety copy the annotations again (the current file is copied first).

    Refused with 409 while an activity, training, export or pre-annotation of the project runs; the
    project takes the task of the copy and gets the classes of the copy it does not have.

    Args:
        project_id: Id of the project.
        name: File name of the safety copy.
        state: The shared state of the application.

    Returns:
        The result of the restore (counts, and the copy made of the current file).

    Raises:
        ApiError: 404 ``backup_not_found``; 400 for a coded refusal; 409 while the project is busy.
    """
    state.project(project_id)
    state.busy(project_id, training=True)
    try:
        return backups.restore_backup(state.store, project_id, name)
    except FileNotFoundError as exc:
        raise ApiError(404, "backup_not_found", str(exc)) from exc
    except CodedError as exc:
        raise bad_request(exc) from exc
