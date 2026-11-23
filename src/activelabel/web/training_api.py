"""HTTP API of the model catalogue and training runs (see docs/TRAINING.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.projects import Project
from activelabel.training.catalog import COMMON_PARAMS, METRICS, PROJECT_TYPES, models_for
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.runs import RunBusy, RunNotFound, RunStore
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class DeleteRun(BaseModel):
    """Body of DELETE /projects/{id}/runs/{n}."""

    confirm: bool = False


def start_run(state: AppState, project_id: int, number: int) -> None:
    """Hand a queued run to the runner (it starts when no other run is running)."""
    state.runner.enqueue(project_id, number)


def _store(state: AppState, project_id: int) -> RunStore:
    state.project(project_id)
    return RunStore(state.store.dir(project_id))


def _summary(store: RunStore, run: dict[str, Any]) -> dict[str, Any]:
    config = store.config(run["number"])
    return {
        **run,
        "model": config.model,
        "dataset": config.dataset,
        "fold": config.fold,
        "start": config.start,
        "mode": config.mode,
    }


def _checked(state: AppState, project: Project, config: RunConfig) -> RunConfig:
    """Validate against the catalogue and the dataset version; fill in its fingerprint."""
    try:
        version = DatasetStore(state.store.dir(project.id)).get(config.dataset)
    except DatasetNotFound:
        raise HTTPException(404, f"Dataset version {config.dataset} does not exist.") from None
    if version["type"] != project.task.value:
        raise HTTPException(400, "This dataset version was made for another project type.")
    try:
        checked = validate(config, project.task.value, len(version["folds"] or []))
    except ConfigError as error:
        raise HTTPException(400, str(error)) from None
    checked.dataset_fingerprint = version["fingerprint"]
    return checked


def _create(state: AppState, project: Project, config: RunConfig) -> dict[str, Any]:
    store = _store(state, project.id)
    run = store.create(_checked(state, project, config))
    start_run(state, project.id, run["number"])
    return _summary(store, store.get(run["number"]))


@router.get("/training/catalog")
def catalog(type: str | None = None) -> dict[str, Any]:  # noqa: A002 - the query parameter is named type
    """Models (with their hyperparameter specs) a project type offers; all without ``type``."""
    if type is not None and type not in PROJECT_TYPES:
        raise HTTPException(400, f"The type must be one of: {', '.join(PROJECT_TYPES)}.")
    return {
        "models": [m.as_dict() for m in models_for(type)],
        "common": [p.as_dict() for p in COMMON_PARAMS],
        "metrics": {kind: {"choices": list(v[0]), "default": v[1]} for kind, v in METRICS.items()},
    }


@router.get("/projects/{project_id}/runs")
def list_runs(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every run of the project, newest first."""
    store = _store(state, project_id)
    return {"runs": [_summary(store, r) for r in store.list()]}


@router.post("/projects/{project_id}/runs")
def create_run(project_id: int, body: dict[str, Any], state: AppState = STATE) -> dict[str, Any]:
    """Validate a configuration and store the run as queued."""
    project = state.project(project_id)
    try:
        config = RunConfig.from_dict(body)
    except ConfigError as error:
        raise HTTPException(400, str(error)) from None
    return _create(state, project, config)


@router.get("/projects/{project_id}/runs/{number}")
def get_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Configuration, status, metrics per epoch and results of a run."""
    store = _store(state, project_id)
    try:
        run = store.get(number)
        return {
            **_summary(store, run),
            "config": store.config(number).to_dict(),
            "metrics": store.metrics(number),
            "results": store.results(number),
        }
    except RunNotFound:
        raise HTTPException(404, "This run does not exist.") from None


@router.delete("/projects/{project_id}/runs/{number}")
def delete_run(
    project_id: int, number: int, body: DeleteRun | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a run (needs ``{"confirm": true}``); a running run cannot be deleted."""
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise HTTPException(400, "Confirm the deletion of this run.")
    try:
        store.delete(number)
    except RunNotFound:
        raise HTTPException(404, "This run does not exist.") from None
    except RunBusy:
        raise HTTPException(409, "This run is running: cancel it before deleting.") from None
    return {"deleted": True}


@router.post("/projects/{project_id}/runs/{number}/reproduce")
def reproduce_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """A new run with the same configuration."""
    project = state.project(project_id)
    try:
        config = _store(state, project_id).config(number)
    except RunNotFound:
        raise HTTPException(404, "This run does not exist.") from None
    return _create(state, project, config)


@router.post("/projects/{project_id}/runs/{number}/cancel")
def cancel_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running run (a finished run is left as it is)."""
    store = _store(state, project_id)
    try:
        state.runner.cancel(project_id, number)
        return _summary(store, store.get(number))
    except RunNotFound:
        raise HTTPException(404, "This run does not exist.") from None
