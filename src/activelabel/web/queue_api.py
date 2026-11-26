"""HTTP API of the global work queue (see docs/DESIGN.md, "Work queue")."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from activelabel.projects import ProjectNotFound
from activelabel.storage import DamagedFile
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
JOB_GONE = "This activity no longer exists (it already finished or the server restarted)."


def _snapshot(state: AppState, project_id: int | None = None) -> dict[str, Any]:
    """The queue with the name of the project of each item.

    Args:
        state: The shared state of the application.
        project_id: Only the jobs of this project; None for every project.

    Returns:
        The scheduler's snapshot; every item and interrupted entry also has ``project_name`` (empty for
        project 0, which is where the activities of no project belong, or for a deleted project).
    """
    snapshot = state.scheduler.snapshot(project_id)
    names: dict[int, str] = {}
    for line in (*snapshot["items"], *snapshot["interrupted"]):
        owner = int(line.get("project_id") or 0)
        if owner not in names:
            try:
                names[owner] = state.store.get(owner).name if owner else ""
            except (ProjectNotFound, DamagedFile):
                names[owner] = ""
        line["project_name"] = names[owner]
    return snapshot


@router.get("/queue")
def get_queue(project: int | None = Query(None), state: AppState = STATE) -> dict[str, Any]:
    """The work queue: limits, lanes, running and waiting jobs, and the jobs a restart interrupted.

    Args:
        project: Only the jobs of this project (the lanes still count every project); all when omitted.
        state: The shared state of the application.

    Returns:
        ``{limits, lanes, items, interrupted}``, each item with ``project_name``.
    """
    return _snapshot(state, project)


@router.post("/queue/{job_id}/cancel")
def cancel_queued(job_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a waiting or running job of any project, whoever owns it (runner or job manager).

    Args:
        job_id: Id of the job.
        state: The shared state of the application.

    Returns:
        The queue after the request (as :func:`get_queue` for every project).
    """
    if not state.runner.cancel_job(job_id):
        try:
            state.jobs.cancel(job_id)
        except KeyError:
            raise ApiError(404, "activity_not_found", JOB_GONE) from None
    return _snapshot(state)
