"""HTTP API of the FiftyOne runs of a project's analyses (#629)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from activelabel import fiftyone_bridge, optional
from activelabel.curation import fiftyone_runs
from activelabel.web.errors import ApiError, need_fiftyone
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


@router.get("/projects/{project_id}/curation/runs")
def list_runs(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The runs recorded on the project's FiftyOne dataset (brain, evaluation, annotation and our own).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{"runs": [{type, key, timestamp, config, ours}]}``; empty before the project used FiftyOne and
        without FiftyOne (the light install).
    """
    state.project(project_id)
    if not optional.FIFTYONE:  # the light install: no dataset, no runs
        return {"runs": []}
    dataset = fiftyone_bridge.existing_dataset(state.store.dir(project_id), project_id)
    return {"runs": fiftyone_runs.listing(dataset) if dataset is not None else []}


@router.delete("/projects/{project_id}/curation/runs/{key}")
def delete_run(project_id: int, key: str, state: AppState = STATE) -> dict[str, Any]:
    """Delete the FiftyOne run of one of ActiveLabel's analyses (its results in ActiveLabel stay).

    Args:
        project_id: Id of the project.
        key: The run key (one of ``fiftyone_runs.RUN_KEYS``).
        state: The shared state of the application.

    Returns:
        The runs after the deletion (as :func:`list_runs`).

    Raises:
        ApiError: 404 ``run_not_found`` for an unknown key or a run that is not one of ActiveLabel's; 501
            ``fiftyone_not_installed`` without FiftyOne.
    """
    state.project(project_id)
    need_fiftyone()
    dataset = fiftyone_bridge.existing_dataset(state.store.dir(project_id), project_id)
    if dataset is None or not fiftyone_runs.delete(dataset, key):
        raise ApiError(
            404, "run_not_found", "This run does not exist or is not one of ActiveLabel's.", key=key
        )
    return {"runs": fiftyone_runs.listing(dataset)}
