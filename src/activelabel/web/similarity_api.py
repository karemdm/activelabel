"""HTTP API of the similarity map of the curation (embeddings, 2-D map and groups)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.curation import similarity_job
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class StartBody(BaseModel):
    """Body of POST /projects/{id}/curation/similarity."""

    device: str = "auto"


def _active_ids(state: AppState, project_id: int) -> set[str]:
    """Ids of the active images of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The ids.
    """
    return {r.id for r in state.manifest(project_id).active()}


@router.get("/projects/{project_id}/curation/similarity")
def similarity_state(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The similarity map of a project: state of the computation and, once done, the map and its groups.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state (``{}`` when the map was never computed) with ``outdated``.
    """
    active = _active_ids(state, project_id)
    return similarity_job.state(state.runner, state.store.dir(project_id), active)


@router.post("/projects/{project_id}/curation/similarity")
def start_similarity(
    project_id: int, body: StartBody | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Queue the computation of the map (behind any training run: one heavy job at a time).

    Args:
        project_id: Id of the project.
        body: Where the network runs.
        state: The shared state of the application.

    Returns:
        The saved state, queued.
    """
    project = state.project(project_id)
    manifest = state.manifest_with_images(project_id)
    project_dir = state.store.dir(project_id)
    current = similarity_job.state(state.runner, project_dir, {r.id for r in manifest.active()})
    if current.get("status") in similarity_job.ACTIVE:
        raise ApiError(409, "similarity_running", "The similarity map is already being computed.")
    return similarity_job.start(state.runner, project, project_dir, body.device if body else "auto")


@router.post("/projects/{project_id}/curation/similarity/cancel")
def cancel_similarity(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running computation (the previous map stays).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state.
    """
    state.project(project_id)
    return similarity_job.cancel(state.runner, project_id, state.store.dir(project_id))
