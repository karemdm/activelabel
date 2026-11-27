"""HTTP API of the similarity map of the curation (embeddings, 2-D map and groups)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.curation import clustering_job, similarity_job
from activelabel.errors import CodedError
from activelabel.training import devices
from activelabel.training.environment import TrainingEnvironment
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class StartBody(BaseModel):
    """Body of POST /projects/{id}/curation/similarity."""

    device: str = "auto"


class ClusteringBody(BaseModel):
    """Body of POST /projects/{id}/curation/similarity/clustering."""

    method: str = "auto"
    groups: int = 0
    min_size: int = 5


def _active_ids(state: AppState, project_id: int) -> set[str]:
    """Ids of the active images of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The ids.
    """
    return {r.id for r in state.manifest(project_id).active()}


@router.get("/projects/{project_id}/curation/similarity/requirements")
def similarity_requirements(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """What the map needs before it is asked for (#664): packages to install, devices and images.

    The interface reads it before starting a map: it confirms the first install of PyTorch with the same text
    as the training dialog, offers the devices and refuses a project with too few images.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{packages, missing, install_needed, devices, images, min_images}``: ``missing`` are the packages
        the training environment lacks (all of them before it exists); ``devices`` the device choices of
        this computer; ``images`` the active images.
    """
    images = len(_active_ids(state, project_id))
    env = TrainingEnvironment(state.settings.workspace)
    missing = env.missing(similarity_job.PACKAGES) if env.exists() else list(similarity_job.PACKAGES)
    return {
        "packages": list(similarity_job.PACKAGES),
        "missing": missing,
        "install_needed": bool(missing),
        "devices": devices.available(env),
        "images": images,
        "min_images": similarity_job.MIN_IMAGES,
    }


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
    """Queue the computation of the map (global queue, in the lane of its device).

    Args:
        project_id: Id of the project.
        body: Where the network runs.
        state: The shared state of the application.

    Returns:
        The saved state, queued.

    Raises:
        ApiError: 409 ``similarity_running`` when the map is queued or running, 409 ``clustering_running``
            while a regrouping is, 400 ``similarity_too_few_images`` below ``MIN_IMAGES`` active images and
            400 for a device this computer does not have.
    """
    project = state.project(project_id)
    manifest = state.manifest_with_images(project_id)
    project_dir = state.store.dir(project_id)
    if clustering_job.state(state.runner, project_dir).get("status") in clustering_job.ACTIVE:
        raise ApiError(409, "clustering_running", "The map is being regrouped.")
    active = {r.id for r in manifest.active()}
    current = similarity_job.state(state.runner, project_dir, active)
    if current.get("status") in similarity_job.ACTIVE:
        raise ApiError(409, "similarity_running", "The similarity map is already being computed.")
    if len(active) < similarity_job.MIN_IMAGES:
        raise ApiError(
            400,
            "similarity_too_few_images",
            f"A map needs at least {similarity_job.MIN_IMAGES} images.",
            min=similarity_job.MIN_IMAGES,
            images=len(active),
        )
    try:
        device = devices.check_here(body.device if body else "auto", state.settings.workspace)
    except devices.DeviceError as exc:
        raise bad_request(exc) from None
    return similarity_job.start(state.runner, project, project_dir, device)


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


@router.get("/projects/{project_id}/curation/similarity/clustering")
def clustering_state(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The regrouping of the map: method asked for and state of the job.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The ``clustering`` entry (``{}`` when the map was never regrouped).
    """
    state.project(project_id)
    return clustering_job.state(state.runner, state.store.dir(project_id))


@router.post("/projects/{project_id}/curation/similarity/clustering")
def start_clustering(project_id: int, body: ClusteringBody, state: AppState = STATE) -> dict[str, Any]:
    """Queue the regrouping of the map (CPU lane of the global queue; the points stay, the groups change).

    Args:
        project_id: Id of the project.
        body: The method and its number (``groups`` for ``kmeans``, ``min_size`` for ``hdbscan``).
        state: The shared state of the application.

    Returns:
        The ``clustering`` entry, queued.

    Raises:
        ApiError: 409 ``clustering_running`` while a regrouping is queued or running, 409
            ``similarity_running`` while the map is, 409 ``similarity_map_needed`` without a map, 400
            ``cluster_method_unknown``, ``cluster_groups_range`` or ``cluster_min_size_small`` for a method or
            number out of range.
    """
    project = state.project(project_id)
    project_dir = state.store.dir(project_id)
    if clustering_job.state(state.runner, project_dir).get("status") in clustering_job.ACTIVE:
        raise ApiError(409, "clustering_running", "The map is already being regrouped.")
    active = _active_ids(state, project_id)
    if similarity_job.state(state.runner, project_dir, active).get("status") in similarity_job.ACTIVE:
        raise ApiError(409, "similarity_running", "The similarity map is being computed.")
    try:
        return clustering_job.start(
            state.runner, project, project_dir, body.method, body.groups, body.min_size
        )
    except CodedError as exc:
        if exc.code == "similarity_map_needed":
            raise ApiError(409, exc.code, str(exc)) from exc
        raise bad_request(exc) from exc


@router.post("/projects/{project_id}/curation/similarity/clustering/cancel")
def cancel_clustering(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running regrouping (the groups stay as they were).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The ``clustering`` entry.
    """
    state.project(project_id)
    return clustering_job.cancel(state.runner, project_id, state.store.dir(project_id))
