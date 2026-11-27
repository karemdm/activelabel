"""HTTP API of the model evaluation: a saved model's metrics on one part of a dataset version (#611)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.curation import evaluation_job
from activelabel.datasets import DatasetNotFound
from activelabel.errors import CodedError
from activelabel.training import devices
from activelabel.training.models import ModelNotFound, ModelStore
from activelabel.training.runner import RunnerError
from activelabel.web.errors import ApiError, bad_request, need_fiftyone
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
#: Codes of an evaluation that cannot start because of what was asked (HTTP 400); the others are 409.
ASKED = {
    "evaluation_task_unsupported",
    "evaluation_needs_shapely",
    "part_unknown",
    "fold_required",
    "fold_not_found",
    "version_has_no_folds",
}


class StartBody(BaseModel):
    """Body of POST /projects/{id}/models/{n}/evaluation."""

    device: str = "auto"
    version: int | None = None
    part: str | None = None
    fold: int | None = None


def _model_dir(state: AppState, project_id: int, model: int) -> Any:
    """The folder of the project, after checking the model exists.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        model: Number of the saved model.

    Returns:
        The folder of the project.

    Raises:
        ApiError: 404 ``model_not_found``.
    """
    state.project(project_id)  # 404 for an unknown project
    project_dir = state.store.dir(project_id)
    try:
        ModelStore(project_dir).get(model)
    except ModelNotFound:
        raise ApiError(404, "model_not_found", "This model does not exist.") from None
    return project_dir


@router.get("/projects/{project_id}/models/{model}/evaluation")
def evaluation_state(project_id: int, model: int, state: AppState = STATE) -> dict[str, Any]:
    """The evaluation of a model: state of the job and, once done, its result.

    Args:
        project_id: Id of the project.
        model: Number of the saved model.
        state: The shared state of the application.

    Returns:
        The saved entry (job fields and ``result``; only ``active`` when it was never evaluated); ``active``
        is the number of the model being evaluated now, or None.

    Raises:
        ApiError: 404 ``model_not_found``.
    """
    project_dir = _model_dir(state, project_id, model)
    return evaluation_job.state_of(state.runner, project_dir, model)


@router.post("/projects/{project_id}/models/{model}/evaluation")
def start_evaluation(
    project_id: int, model: int, body: StartBody | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Queue the evaluation of a model (the lane of the chosen device in the global queue).

    Args:
        project_id: Id of the project.
        model: Number of the saved model.
        body: The device (``auto``, ``cpu``, ``cuda:N`` or ``mps``) and, to change what is evaluated, the
            dataset version, its part (``train``, ``val`` or ``test``) and fold. Without them: the version
            the model was trained on, its ``test`` part (``val`` when that is empty).
        state: The shared state of the application.

    Returns:
        The model's saved entry, queued.

    Raises:
        ApiError: 404 ``model_not_found`` or ``dataset_not_found``; 400 for a device that is not here, a model
            that cannot predict on this project, a keypoints or regression project
            (``evaluation_task_unsupported``) or a part or fold that does not fit; 409
            ``evaluation_running`` when a model is being evaluated, ``no_images_in_part`` and
            ``no_dataset_version``; 501 ``fiftyone_not_installed`` without FiftyOne (the evaluation is
            FiftyOne's).
    """
    project = state.project(project_id)
    need_fiftyone()
    project_dir = _model_dir(state, project_id, model)
    body = body or StartBody()
    try:
        device = devices.check_here(body.device, state.settings.workspace)
    except devices.DeviceError as error:
        raise bad_request(error) from None
    try:
        return evaluation_job.start(
            state.runner, project, project_dir, model, device, body.version, body.part, body.fold
        )
    except ModelNotFound:
        raise ApiError(404, "model_not_found", "This model does not exist.") from None
    except DatasetNotFound:
        raise ApiError(404, "dataset_not_found", "This dataset version does not exist.") from None
    except RunnerError as error:
        raise ApiError(400, error.code or "evaluation_model_unusable", str(error), **error.params) from None
    except CodedError as error:
        raise ApiError(400 if error.code in ASKED else 409, error.code, str(error), **error.params) from None


@router.post("/projects/{project_id}/models/{model}/evaluation/cancel")
def cancel_evaluation(project_id: int, model: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running evaluation (the previous result stays).

    Args:
        project_id: Id of the project.
        model: Number of the saved model.
        state: The shared state of the application.

    Returns:
        The saved entry.

    Raises:
        ApiError: 404 ``model_not_found``.
    """
    project_dir = _model_dir(state, project_id, model)
    evaluation_job.cancel(state.runner, project_id, project_dir, model)
    return evaluation_job.state_of(state.runner, project_dir, model)
