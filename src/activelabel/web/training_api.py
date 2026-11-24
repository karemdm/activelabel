"""HTTP API of the model catalogue and training runs (see docs/TRAINING.md)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.projects import Project
from activelabel.training.catalog import COMMON_PARAMS, METRICS, PROJECT_TYPES, get_model, models_for
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.models import ModelStore
from activelabel.training.runs import QUEUED, RunBusy, RunNotFound, RunStore
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
    """A run with the main fields of its configuration (``config_error`` when the file is unreadable)."""
    try:
        config = store.config(run["number"])
    except (OSError, ValueError) as exc:  # a missing or damaged config.yaml must not hide every run
        return {**run, "config_error": str(exc) or type(exc).__name__}
    return {
        **run,
        "model": config.model,
        "dataset": config.dataset,
        "fold": config.fold,
        "start": config.start,
        "mode": config.mode,
        "metric": config.metric,
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
    model = get_model(checked.model)
    if model is not None and model.family == "ultralytics":
        raise HTTPException(400, "Training YOLO models is not available yet.")
    if version.get("incomplete"):
        raise HTTPException(
            400, "Some images of this dataset version were deleted from the project: make a new version."
        )
    if not version.get("folds") and not (version.get("sizes") or {}).get("train"):
        raise HTTPException(400, "The training part of this dataset version has no images: make another one.")
    if not version.get("folds") and not (version.get("sizes") or {}).get("val"):
        raise HTTPException(
            400, "This dataset version has no validation images: make a version with validation."
        )
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


@router.get("/training/environment")
def environment(state: AppState = STATE) -> dict[str, Any]:
    """Packages installed in the training environment and the GPU found (nothing is installed here)."""
    from activelabel.training.environment import TrainingEnvironment

    env = TrainingEnvironment(state.settings.workspace)
    gpu = env.gpu()
    exists = env.exists()
    installed: dict[str, str] = env.installed() if exists else {}
    return {
        "exists": exists,
        "packages": installed,
        "gpu": None if gpu is None else gpu.as_dict(),
    }


@router.get("/projects/{project_id}/runs/{number}/log")
def run_log(project_id: int, number: int, lines: int = 200, state: AppState = STATE) -> dict[str, Any]:
    """The last ``lines`` lines of a run's log (1..2000)."""
    store = _store(state, project_id)
    try:
        folder = store._require(number)  # noqa: SLF001 - folder of an existing run
    except RunNotFound:
        raise HTTPException(404, "This training run does not exist.") from None
    count = max(1, min(int(lines), 2000))
    return {"lines": _tail(folder / "log.txt", count)}


def _tail(path: Path, count: int, chunk: int = 64 * 1024) -> list[str]:
    """The last ``count`` lines of a text file, reading only its end."""
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = position = handle.tell()
        data = b""
        while position > 0 and data.count(b"\n") <= count:
            position = max(0, position - chunk)
            handle.seek(position)
            data = handle.read(end - position)
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-count:] if position == 0 else lines[1:][-count:]  # the first piece may be cut


@router.get("/projects/{project_id}/runs")
def list_runs(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every run of the project, newest first."""
    store = _store(state, project_id)
    saved = {m["run"]: m["number"] for m in ModelStore(state.store.dir(project_id)).list()}
    return {"runs": [{**_summary(store, r), "model_number": saved.get(r["number"])} for r in store.list()]}


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
        metrics = store.metrics(number)
        current = run.get("current")
        if not metrics and current:  # a grid or all-folds run shows the child that is training
            try:
                metrics = store.metrics(number, current)
            except RunNotFound:
                metrics = []
        summary = _summary(store, run)
        return {
            **summary,
            "config": None if "config_error" in summary else store.config(number).to_dict(),
            "metrics": metrics,
            "results": store.results(number),
        }
    except RunNotFound:
        raise HTTPException(404, "This run does not exist.") from None


@router.get("/projects/{project_id}/runs/{number}/children/{child}")
def get_child(project_id: int, number: int, child: str, state: AppState = STATE) -> dict[str, Any]:
    """One training of a grid or all-folds run: its status, configuration, metrics and results."""
    store = _store(state, project_id)
    try:
        entry = next((c for c in store.get(number).get("children") or [] if c.get("id") == child), None)
        if entry is None:
            raise RunNotFound(child)
        return {
            **entry,
            "config": store.child_config(number, child).to_dict(),
            "metrics": store.metrics(number, child),
            "results": store.results(number, child),
        }
    except RunNotFound:
        raise HTTPException(404, "This training does not exist.") from None


@router.delete("/projects/{project_id}/runs/{number}")
def delete_run(
    project_id: int, number: int, body: DeleteRun | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a run (needs ``{"confirm": true}``); a running run cannot be deleted."""
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise HTTPException(400, "Confirm the deletion of this run.")
    try:
        if store.get(number)["status"] == QUEUED:  # also leave the queue of the runner
            state.runner.cancel(project_id, number)
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
    except (OSError, ValueError):
        raise HTTPException(409, "The configuration of this run is missing or damaged.") from None
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
