"""HTTP API of the label check: possible label errors and hard images from a saved model's predictions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.curation import mistakenness_job
from activelabel.errors import CodedError
from activelabel.training import devices
from activelabel.training.models import ModelNotFound
from activelabel.training.runner import RunnerError
from activelabel.web.errors import ApiError, bad_request, need_fiftyone
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class StartBody(BaseModel):
    """Body of POST /projects/{id}/curation/mistakenness."""

    model: int
    device: str = "auto"


@router.get("/projects/{project_id}/curation/mistakenness")
def mistakenness_state(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The label check of a project: state of the job and, once done, the numbers of each image.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state (``{}`` when it was never run); the ``scores`` stay in it.
    """
    state.project(project_id)  # 404 for an unknown project
    return mistakenness_job.state_of(state.runner, state.store.dir(project_id))


@router.post("/projects/{project_id}/curation/mistakenness")
def start_mistakenness(project_id: int, body: StartBody, state: AppState = STATE) -> dict[str, Any]:
    """Queue the label check with a saved model (the lane of the chosen device in the global queue).

    Args:
        project_id: Id of the project.
        body: The number of the saved model and the device (``auto``, ``cpu``, ``cuda:N`` or ``mps``).
        state: The shared state of the application.

    Returns:
        The saved state, queued.

    Raises:
        ApiError: 404 ``model_not_found``; 400 for a device that is not here or a model that cannot predict
            on this project; 409 ``mistakenness_running`` when one is queued or running, 409
            ``no_labelled_images`` without labelled images; 501 ``fiftyone_not_installed`` without FiftyOne
            (the comparison is FiftyOne Brain's).
    """
    project = state.project(project_id)
    need_fiftyone()
    project_dir = state.store.dir(project_id)
    try:
        device = devices.check_here(body.device, state.settings.workspace)
    except devices.DeviceError as error:
        raise bad_request(error) from None
    if mistakenness_job.state_of(state.runner, project_dir).get("status") in mistakenness_job.ACTIVE:
        raise ApiError(409, "mistakenness_running", "The labels are already being checked.")
    try:
        return mistakenness_job.start(state.runner, project, project_dir, body.model, device)
    except ModelNotFound:
        raise ApiError(404, "model_not_found", "This model does not exist.") from None
    except RunnerError as error:
        raise ApiError(400, error.code or "mistakenness_model_unusable", str(error), **error.params) from None
    except CodedError as error:
        raise ApiError(409, error.code, str(error), **error.params) from None


@router.post("/projects/{project_id}/curation/mistakenness/cancel")
def cancel_mistakenness(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running label check (the previous results stay).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The saved state.
    """
    state.project(project_id)
    return mistakenness_job.cancel(state.runner, project_id, state.store.dir(project_id))
