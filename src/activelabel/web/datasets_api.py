"""HTTP API of dataset versions (see docs/DATASETS.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.datasets import DatasetNotFound, DatasetStore, Plan, parse_options, plan_split
from activelabel.tasks_store import TaskStore
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


class DeleteDataset(BaseModel):
    """Body of DELETE /projects/{id}/datasets/{n}."""

    confirm: bool = False


def _plan(state: AppState, project_id: int, body: DatasetBody) -> Plan:
    project = state.project(project_id)
    project_dir = state.store.dir(project_id)
    data = body.model_dump(exclude={"note"})
    try:
        options = parse_options(data, [t.id for t in TaskStore(project_dir).load()])
        return plan_split(project_dir, project, options)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None


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
    """Save a new dataset version and return its summary."""
    plan = _plan(state, project_id, body)
    try:
        return _store(state, project_id).create(plan, body.note)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None


@router.get("/projects/{project_id}/datasets/{number}")
def get_dataset(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """One version without its copy of the labels."""
    try:
        return _store(state, project_id).get(number)
    except DatasetNotFound:
        raise HTTPException(404, "This dataset version does not exist.") from None


@router.delete("/projects/{project_id}/datasets/{number}")
def delete_dataset(
    project_id: int, number: int, body: DeleteDataset | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a version (needs ``{"confirm": true}``); its number is not reused."""
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise HTTPException(400, "Confirm the deletion of this dataset version.")
    try:
        store.delete(number)
    except DatasetNotFound:
        raise HTTPException(404, "This dataset version does not exist.") from None
    return {"deleted": True}
