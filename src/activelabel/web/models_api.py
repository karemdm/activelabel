"""HTTP API of the saved models of a project (step 6, see docs/TRAINING.md)."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from activelabel.preannotate import QUEUED, RUNNING, PreannotationStore
from activelabel.training.models import EXPORT_FILES, WEIGHTS, ModelError, ModelNotFound, ModelStore
from activelabel.training.public_models import get_public, public_for, suggest_mapping
from activelabel.training.runs import RunNotFound
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
DOWNLOADS = {"pytorch": WEIGHTS, **EXPORT_FILES}


class SaveModel(BaseModel):
    """Body of POST /projects/{id}/models."""

    run: int
    note: str = ""


class PublicModelBody(BaseModel):
    """Body of POST /projects/{id}/models/public."""

    public: str
    mapping: dict[str, str]
    note: str = ""


class ChangeModel(BaseModel):
    """Body of PATCH /projects/{id}/models/{n}: only the fields given change."""

    note: str | None = None
    current: bool | None = None


class DeleteModel(BaseModel):
    """Body of DELETE /projects/{id}/models/{n}."""

    confirm: bool = False


class ExportModel(BaseModel):
    """Body of POST /projects/{id}/models/{n}/export."""

    format: str = "onnx"


def _store(state: AppState, project_id: int) -> ModelStore:
    state.project(project_id)
    return ModelStore(state.store.dir(project_id))


def _missing() -> HTTPException:
    return HTTPException(404, "This model does not exist.")


@router.get("/projects/{project_id}/models")
def list_models(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every saved model of the project (newest first) with its results, and the current one."""
    store = _store(state, project_id)
    models = [{**card, "results": store.results(card["number"])} for card in store.list()]
    return {"models": models, "current": store.current()}


@router.get("/projects/{project_id}/models/public-catalog")
def public_catalog(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Public pretrained models for this project's type, each with a suggested mapping to its classes."""
    project = state.project(project_id)
    classes = list(project.classes)
    return {
        "models": [
            {**m.as_dict(), "suggested": suggest_mapping(m, classes)} for m in public_for(project.task.value)
        ],
        "project_classes": classes,
    }


@router.post("/projects/{project_id}/models/public")
def add_public_model(project_id: int, body: PublicModelBody, state: AppState = STATE) -> dict[str, Any]:
    """Add a public pretrained model to the project, with its category → class mapping."""
    project = state.project(project_id)
    model = get_public(body.public)
    if model is None:
        raise HTTPException(404, "This public model does not exist.")
    if model not in public_for(project.task.value):
        raise HTTPException(400, "This public model does not make the labels this project uses.")
    try:
        return ModelStore(state.store.dir(project_id)).add_public(
            body.public, body.mapping, list(project.classes), body.note
        )
    except ModelError as error:
        raise HTTPException(400, str(error)) from None


@router.post("/projects/{project_id}/models")
def save_model(project_id: int, body: SaveModel, state: AppState = STATE) -> dict[str, Any]:
    """Save the best weights of a finished training run as the next model."""
    store = _store(state, project_id)
    try:
        return store.save_run(body.run, body.note.strip()[:500])
    except RunNotFound:
        raise HTTPException(404, "This training run does not exist.") from None
    except ModelError as error:
        raise HTTPException(400, str(error)) from None


@router.get("/projects/{project_id}/models/{number}")
def get_model_card(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """The card of a model with its results."""
    store = _store(state, project_id)
    try:
        return {**store.get(number), "results": store.results(number)}
    except ModelNotFound:
        raise _missing() from None


@router.patch("/projects/{project_id}/models/{number}")
def change_model(project_id: int, number: int, body: ChangeModel, state: AppState = STATE) -> dict[str, Any]:
    """Change the note, or mark the model as current (``current: false`` leaves no current model)."""
    store = _store(state, project_id)
    try:
        if body.note is not None:
            store.update(number, note=body.note.strip()[:500])
        if body.current is not None:
            if body.current:
                store.set_current(number)
            elif store.current() == number:
                store.set_current(None)
        return store.get(number)
    except ModelNotFound:
        raise _missing() from None


@router.delete("/projects/{project_id}/models/{number}")
def delete_model(
    project_id: int, number: int, body: DeleteModel | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a model and its files (needs ``{"confirm": true}``)."""
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise HTTPException(400, "Confirm the deletion of this model.")
    using = [
        r
        for r in PreannotationStore(state.store.dir(project_id)).list()
        if r.get("model") == number and r.get("status") in (QUEUED, RUNNING)
    ]
    if using:
        raise HTTPException(409, "A pre-annotation with this model is queued or running: wait for it to end.")
    try:
        store.delete(number)
    except ModelNotFound:
        raise _missing() from None
    except ModelError as error:
        raise HTTPException(409, str(error)) from None
    except OSError:  # a file still in use (Windows): nothing was removed from the list
        raise HTTPException(409, "A file of this model is in use: try again in a moment.") from None
    return {"deleted": True}


@router.post("/projects/{project_id}/models/{number}/export")
def export_model(project_id: int, number: int, body: ExportModel, state: AppState = STATE) -> dict[str, Any]:
    """Queue an export of the model (ONNX); it runs when no training or other export is running."""
    _store(state, project_id)
    try:
        return state.runner.enqueue_export(project_id, number, body.format)
    except ModelNotFound:
        raise _missing() from None
    except ModelError as error:
        raise HTTPException(400, str(error)) from None


@router.get("/projects/{project_id}/models/{number}/download/{kind}")
def download_model(project_id: int, number: int, kind: str, state: AppState = STATE) -> FileResponse:
    """The weights (``pytorch``) or an export (``onnx``) of a model, as a file download."""
    store = _store(state, project_id)
    name = DOWNLOADS.get(kind)
    if name is None:
        raise HTTPException(404, "Unknown download.")
    try:
        card = store.get(number)
    except ModelNotFound:
        raise _missing() from None
    if card.get("source") == "public":
        raise HTTPException(404, "A public model has no weights here: they are downloaded when it is used.")
    try:
        path = store.file(number, name)
    except ModelNotFound:
        raise HTTPException(404, "This file does not exist (export the model first).") from None
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", state.project(project_id).name).strip("-") or "project"
    filename = f"{project}-m{number}-{card['model']}{path.suffix}"
    return FileResponse(path, filename=filename, media_type="application/octet-stream")
