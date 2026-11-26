"""HTTP API of the uniqueness of the images of the curation (scores over the similarity map's embeddings)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from activelabel.curation import uniqueness_job
from activelabel.errors import CodedError
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


def _shown(state: AppState, project_id: int) -> dict[str, Any]:
    """The state of the uniqueness with ``outdated`` (the scored images differ from the active ones).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The saved state (``{}`` when it was never computed); the ``scores`` stay in it.
    """
    state.project(project_id)  # 404 for an unknown project
    data = uniqueness_job.state(state.runner, state.store.dir(project_id))
    if not data:
        return {}
    active = {r.id for r in state.manifest(project_id).active()}
    scored = data.get("scores")
    outdated = False
    if data.get("status") == "done" and isinstance(scored, dict):  # images without embeddings are `missing`
        outdated = bool(set(scored) - active) or len(active - set(scored)) != data.get("missing", 0)
    return {**data, "outdated": outdated}


@router.get("/projects/{project_id}/curation/uniqueness")
def uniqueness_state(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The uniqueness of a project: state of the computation and, once done, the score of each image.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state (``{}`` when it was never computed) with ``outdated``.
    """
    return _shown(state, project_id)


@router.post("/projects/{project_id}/curation/uniqueness")
def start_uniqueness(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Queue the scoring (CPU lane of the global queue; it needs the similarity map's embeddings).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state, queued.

    Raises:
        ApiError: 409 ``uniqueness_running`` when one is queued or running, 409 ``similarity_map_needed``
            without the embeddings of the similarity map.
    """
    project = state.project(project_id)
    project_dir = state.store.dir(project_id)
    if uniqueness_job.state(state.runner, project_dir).get("status") in uniqueness_job.ACTIVE:
        raise ApiError(409, "uniqueness_running", "The uniqueness is already being computed.")
    try:
        return uniqueness_job.start(state.runner, project, project_dir)
    except CodedError as exc:
        raise ApiError(409, exc.code, str(exc)) from exc


@router.post("/projects/{project_id}/curation/uniqueness/cancel")
def cancel_uniqueness(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running scoring (the previous scores stay).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state.
    """
    state.project(project_id)
    return uniqueness_job.cancel(state.runner, project_id, state.store.dir(project_id))
