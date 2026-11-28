"""HTTP API of the saved models of a project (step 6, see docs/TRAINING.md)."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from activelabel.curation import evaluation_job
from activelabel.jobs import QUEUED, RUNNING
from activelabel.preannotate import PreannotationStore
from activelabel.textcheck import message_of
from activelabel.training.models import EXPORT_FILES, WEIGHTS, ModelError, ModelNotFound, ModelStore
from activelabel.training.public_models import get_public, public_for, suggest_mapping
from activelabel.training.runs import RunNotFound
from activelabel.web.errors import ApiError, bad_request
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
    """The model store of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The model store of the project.
    """
    state.project(project_id)
    return ModelStore(state.store.dir(project_id))


def _missing() -> ApiError:
    """The HTTP 404 error for a model that does not exist.

    Returns:
        The 404 error for a model that does not exist.
    """
    return ApiError(404, "model_not_found", "This model does not exist.")


@router.get("/projects/{project_id}/models")
def list_models(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every saved model of the project (newest first) with its results, and the current one.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{models, current}``.
    """
    store = _store(state, project_id)
    models = [{**card, "results": store.results(card["number"])} for card in store.list()]
    return {"models": models, "current": store.current()}


@router.get("/projects/{project_id}/models/public-catalog")
def public_catalog(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Public pretrained models for this project's type, each with a suggested mapping to its classes.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{models, project_classes}``: the public models for the project type, each with a suggested mapping.
    """
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
    """Add a public pretrained model to the project, with its category → class mapping.

    Args:
        project_id: Id of the project.
        body: The public model, the mapping to the project classes and a note.
        state: The shared state of the application.

    Returns:
        The new model.
    """
    project = state.project(project_id)
    model = get_public(body.public)
    if model is None:
        raise ApiError(404, "public_model_not_found", "This public model does not exist.")
    if model not in public_for(project.task.value):
        raise ApiError(
            400, "public_model_wrong_type", "This public model does not make the labels this project uses."
        )
    try:
        return ModelStore(state.store.dir(project_id)).add_public(
            body.public, body.mapping, list(project.classes), body.note
        )
    except ModelError as error:
        raise bad_request(error) from None


@router.post("/projects/{project_id}/models")
def save_model(project_id: int, body: SaveModel, state: AppState = STATE) -> dict[str, Any]:
    """Save the best weights of a finished training run as the next model.

    Args:
        project_id: Id of the project.
        body: The run to save the weights of and a note.
        state: The shared state of the application.

    Returns:
        The new model.
    """
    store = _store(state, project_id)
    try:
        return store.save_run(body.run, body.note.strip()[:500])
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
    except ModelError as error:
        raise bad_request(error) from None


@router.get("/projects/{project_id}/models/{number}")
def get_model_card(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """The card of a model with its results.

    Args:
        project_id: Id of the project.
        number: Number of the model.
        state: The shared state of the application.

    Returns:
        The model with its results.
    """
    store = _store(state, project_id)
    try:
        return {**store.get(number), "results": store.results(number)}
    except ModelNotFound:
        raise _missing() from None


@router.patch("/projects/{project_id}/models/{number}")
def change_model(project_id: int, number: int, body: ChangeModel, state: AppState = STATE) -> dict[str, Any]:
    """Change the note, or mark the model as current (``current: false`` leaves no current model).

    Args:
        project_id: Id of the project.
        number: Number of the model.
        body: The new note and/or whether the model is the current one.
        state: The shared state of the application.

    Returns:
        The model.
    """
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
    """Delete a model and its files (needs ``{"confirm": true}``).

    Args:
        project_id: Id of the project.
        number: Number of the model.
        body: The confirmation.
        state: The shared state of the application.

    Returns:
        ``{deleted: true}``. Its evaluation goes too: the entry, and in FiftyOne its fields, run and view.
    """
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise ApiError(400, "confirm_required", "Confirm the deletion of this model.")
    using = [
        r
        for r in PreannotationStore(state.store.dir(project_id)).list()
        if r.get("model") == number and r.get("status") in (QUEUED, RUNNING)
    ]
    if using or evaluation_job.active_model(state.store.dir(project_id)) == number:
        raise ApiError(
            409,
            "model_in_use",
            "A pre-annotation or an evaluation with this model is queued or running: wait for it to end.",
        )
    try:
        store.delete(number)
    except ModelNotFound:
        raise _missing() from None
    except ModelError as error:
        raise ApiError(409, error.code, message_of(error), **error.params) from None
    except OSError:  # a file still in use (Windows): nothing was removed from the list
        raise ApiError(
            409, "model_file_in_use", "A file of this model is in use: try again in a moment."
        ) from None
    project_dir = state.store.dir(project_id)
    evaluation_job.forget(project_dir, project_id, number)  # its entry, fields, run and view (#611)
    return {"deleted": True}


@router.post("/projects/{project_id}/models/{number}/export")
def export_model(project_id: int, number: int, body: ExportModel, state: AppState = STATE) -> dict[str, Any]:
    """Queue an export of the model (ONNX); it runs when no training or other export is running.

    Args:
        project_id: Id of the project.
        number: Number of the model.
        body: The export format.
        state: The shared state of the application.

    Returns:
        The queued export.
    """
    _store(state, project_id)
    try:
        return state.runner.enqueue_export(project_id, number, body.format)
    except ModelNotFound:
        raise _missing() from None
    except ModelError as error:
        raise bad_request(error) from None


@router.get("/projects/{project_id}/models/{number}/download/{kind}")
def download_model(project_id: int, number: int, kind: str, state: AppState = STATE) -> FileResponse:
    """The weights (``pytorch``) or an export (``onnx``) of a model, as a file download.

    Args:
        project_id: Id of the project.
        number: Number of the model.
        kind: ``pytorch`` (weights) or ``onnx`` (export).
        state: The shared state of the application.

    Returns:
        The file to download.
    """
    store = _store(state, project_id)
    name = DOWNLOADS.get(kind)
    if name is None:
        raise ApiError(404, "download_unknown", "Unknown download.", kind=kind)
    try:
        card = store.get(number)
    except ModelNotFound:
        raise _missing() from None
    if card.get("source") == "public":
        raise ApiError(
            404,
            "public_model_no_weights",
            "A public model has no weights here: they are downloaded when it is used.",
        )
    try:
        path = store.file(number, name)
    except ModelNotFound:
        raise ApiError(
            404, "model_file_missing", "This file does not exist (export the model first)."
        ) from None
    project = re.sub(r"[^A-Za-z0-9._-]+", "-", state.project(project_id).name).strip("-") or "project"
    filename = f"{project}-m{number}-{card['model']}{path.suffix}"
    return FileResponse(path, filename=filename, media_type="application/octet-stream")
