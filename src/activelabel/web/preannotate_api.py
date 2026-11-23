"""HTTP API of pre-annotation: labels from a saved model, to be reviewed (see docs/PREANNOTATE.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel import preannotate
from activelabel.training.catalog import get_model
from activelabel.training.models import ModelNotFound, ModelStore
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class StartBody(BaseModel):
    """Body of POST /projects/{id}/preannotations."""

    model: int
    task: int
    scope: str = "unlabeled"
    threshold: float = 0.5


class AcceptBody(BaseModel):
    """Body of POST /projects/{id}/preannotations/accept: one task, or every task when omitted."""

    task: int | None = None


def _models(state: AppState, project_id: int) -> list[dict[str, Any]]:
    project_task = state.project(project_id).task
    out = []
    for card in ModelStore(state.store.dir(project_id)).list():
        spec = get_model(card["model"])
        kind = card.get("task_kind") or (spec.task_kind if spec else "")
        out.append(
            {
                "number": card["number"],
                "model": card["model"],
                "label": card.get("label") or (spec.label if spec else card["model"]),
                "public": card.get("source") == "public",
                "task_kind": kind,
                "current": card.get("current", False),
                "metric": card.get("metric"),
                "value": card.get("value"),
                "supported": kind in preannotate.KINDS
                and preannotate.fits(kind, project_task)
                and not (spec is not None and spec.family == "ultralytics"),
            }
        )
    return out


@router.get("/projects/{project_id}/preannotations")
def overview(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Models that can pre-annotate, tasks with their counts, and the runs (newest first)."""
    state.project(project_id)
    project_dir = state.store.dir(project_id)
    return {
        "models": _models(state, project_id),
        "tasks": preannotate.task_counts(project_dir),
        "runs": preannotate.PreannotationStore(project_dir).list(),
    }


@router.post("/projects/{project_id}/preannotations")
def start(project_id: int, body: StartBody, state: AppState = STATE) -> dict[str, Any]:
    """Queue a pre-annotation of one task's images with a saved model."""
    state.project(project_id)
    project_dir = state.store.dir(project_id)
    try:
        card = ModelStore(project_dir).get(body.model)
    except ModelNotFound:
        raise HTTPException(404, "This model does not exist.") from None
    info = next((m for m in _models(state, project_id) if m["number"] == body.model), None)
    if not info or not info["supported"]:
        kind = card.get("task_kind") or "these"
        if kind in preannotate.KINDS and not preannotate.fits(kind, state.project(project_id).task):
            raise HTTPException(400, f"This model makes {kind} labels, which this project does not use.")
        raise HTTPException(400, f"Pre-annotation with {kind} models comes later.")
    if not 0.0 < body.threshold < 1.0:
        raise HTTPException(400, "The minimum confidence must be between 0 and 1.")
    try:
        images = preannotate.targets(project_dir, body.task, body.scope)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    if not images:
        raise HTTPException(400, "No image of this task matches this choice.")
    runs = preannotate.PreannotationStore(project_dir)
    if any(
        r["status"] in ("queued", "running") and r["task"] == body.task and r["scope"] == body.scope
        for r in runs.list()
    ):
        raise HTTPException(409, "These images are already being pre-annotated.")
    task_name = next((t["name"] for t in preannotate.task_counts(project_dir) if t["id"] == body.task), "")
    run = runs.create(
        model=body.model,
        model_label=info["label"],
        task=body.task,
        task_name=task_name,
        scope=body.scope,
        threshold=round(body.threshold, 3),
        total=len(images),
    )
    state.runner.enqueue_preannotation(project_id, run["number"])
    return run


@router.post("/projects/{project_id}/preannotations/{number}/cancel")
def cancel(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running pre-annotation (nothing is written)."""
    state.project(project_id)
    store = preannotate.PreannotationStore(state.store.dir(project_id))
    try:
        state.runner.cancel_preannotation(project_id, number)
        return store.get(number)
    except preannotate.RunNotFound:
        raise HTTPException(404, "This pre-annotation does not exist.") from None


@router.post("/projects/{project_id}/preannotations/accept")
def accept(project_id: int, body: AcceptBody, state: AppState = STATE) -> dict[str, Any]:
    """Keep the pre-annotations of a task (or of every task) as they are, marked as reviewed."""
    state.project(project_id)
    project_dir = state.store.dir(project_id)
    try:
        images = preannotate.unreviewed_images(project_dir, body.task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    return {"accepted": preannotate.accept(project_dir, images)}
