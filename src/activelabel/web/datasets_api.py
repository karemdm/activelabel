"""HTTP API of dataset versions (see docs/DATASETS.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.datasets import DatasetNotFound, DatasetStore, Plan, parse_options, plan_split
from activelabel.tasks_store import TaskStore
from activelabel.training.runs import QUEUED, RUNNING, RunStore
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class DatasetBody(BaseModel):
    """Body of the preview and create endpoints (the options of docs/DATASETS.md, plus a note)."""

    tasks: list[int] | None = None
    split: dict[str, Any] | None = None
    kfold: int | None = None
    background: bool = False
    seed: int = 42
    videos: str = "auto"
    gap_seconds: float = 0.5
    note: str = ""
    allow_empty: bool = False


class DeleteDataset(BaseModel):
    """Body of DELETE /projects/{id}/datasets/{n}."""

    confirm: bool = False
    confirm_runs: bool = False


def _plan(state: AppState, project_id: int, body: DatasetBody) -> Plan:
    project = state.project(project_id)
    project_dir = state.store.dir(project_id)
    data = body.model_dump(exclude={"note", "allow_empty"})
    try:
        options = parse_options(data, [t.id for t in TaskStore(project_dir).load()])
        return plan_split(project_dir, project, options)
    except ValueError as error:
        raise bad_request(error) from None


def _numbers(runs: list[int]) -> str:
    return ", ".join(f"r{n}" for n in runs)


def _runs_using(state: AppState, project_id: int, version: int) -> tuple[list[int], list[int]]:
    """Numbers of the unfinished and of the finished training runs that use a dataset version."""
    runs = RunStore(state.store.dir(project_id))
    active: list[int] = []
    finished: list[int] = []
    for run in reversed(runs.list()):
        try:
            if runs.config(run["number"]).dataset != version:
                continue
        except (OSError, ValueError):
            continue  # a run whose configuration is unreadable cannot name its version
        (active if run["status"] in (QUEUED, RUNNING) else finished).append(run["number"])
    return active, finished


def _store(state: AppState, project_id: int) -> DatasetStore:
    state.project(project_id)
    return DatasetStore(state.store.dir(project_id))


@router.get("/projects/{project_id}/datasets")
def list_datasets(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every dataset version of the project, newest first."""
    return {"versions": _store(state, project_id).versions()}


@router.post("/projects/{project_id}/datasets/preview")
def preview_dataset(project_id: int, body: DatasetBody, state: AppState = STATE) -> dict[str, Any]:
    """What a version would contain (sizes, stats, warnings); nothing is saved."""
    return _plan(state, project_id, body).preview()


@router.post("/projects/{project_id}/datasets")
def create_dataset(project_id: int, body: DatasetBody, state: AppState = STATE) -> dict[str, Any]:
    """Save a new dataset version and return its summary.

    A hold-out split whose training or validation part (asked to have images) ended up empty is
    refused with 409 unless the body has ``allow_empty: true``: such a version cannot train a model.
    """
    plan = _plan(state, project_id, body)
    blocked = [p for p in plan.empty_parts() if p in ("train", "val")]
    if blocked and not body.allow_empty:
        names = " and ".join("training" if p == "train" else "validation" for p in blocked)
        raise ApiError(
            409,
            "dataset_part_empty",
            f"The {names} part would have no images (too few images or groups of images for these "
            "proportions): a model cannot be trained on this version.",
            parts=blocked,
        )
    try:
        return _store(state, project_id).create(plan, body.note)
    except ValueError as error:
        raise bad_request(error) from None


@router.get("/projects/{project_id}/datasets/{number}")
def get_dataset(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """One version without its copy of the labels."""
    try:
        return _store(state, project_id).get(number)
    except DatasetNotFound:
        raise ApiError(404, "dataset_not_found", "This dataset version does not exist.") from None


@router.delete("/projects/{project_id}/datasets/{number}")
def delete_dataset(
    project_id: int, number: int, body: DeleteDataset | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a version (needs ``{"confirm": true}``); its number is not reused.

    Refused with 409 while a queued or running training run uses the version. Finished runs that use
    it cannot be repeated afterwards (saved models keep working): they must be acknowledged with
    ``{"confirm_runs": true}``, otherwise the answer is 409 naming them.
    """
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise ApiError(400, "confirm_required", "Confirm the deletion of this dataset version.")
    active, finished = _runs_using(state, project_id, number)
    if active:
        raise ApiError(
            409,
            "dataset_in_use",
            f"Training run(s) {_numbers(active)} use this version and have not finished: cancel them first.",
            runs=active,
        )
    if finished and not body.confirm_runs:
        raise ApiError(
            409,
            "dataset_used_by_runs",
            f"Training run(s) {_numbers(finished)} used this version and could not be repeated after "
            "it is deleted (saved models keep working): confirm to delete it anyway.",
            runs=finished,
        )
    try:
        store.delete(number)
    except DatasetNotFound:
        raise ApiError(404, "dataset_not_found", "This dataset version does not exist.") from None
    return {"deleted": True}
