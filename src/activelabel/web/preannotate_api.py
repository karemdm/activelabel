"""HTTP API of pre-annotation: labels from a saved model, to be reviewed (see docs/PREANNOTATE.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel import preannotate, preannotate_options
from activelabel.errors import CodedError
from activelabel.tasks import refuse_regression
from activelabel.training import devices
from activelabel.training.catalog import get_model
from activelabel.training.models import ModelNotFound, ModelStore
from activelabel.training.public_models import get_public
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class StartBody(BaseModel):
    """Body of POST /projects/{id}/preannotations."""

    model: int
    task: int
    scope: str = "unlabeled"
    threshold: float | None = None  # the minimum confidence of older clients: ``options.confidence`` wins
    device: str = "auto"
    options: dict[str, Any] | None = None  # inference parameters (see preannotate_options); missing: defaults
    classes: list[str] | None = None  # classes to write; None: every class of the model the project has


class AcceptBody(BaseModel):
    """Body of POST /projects/{id}/preannotations/accept: one task, or every task when omitted."""

    task: int | None = None


def _models(state: AppState, project_id: int) -> list[dict[str, Any]]:
    """The models of a project that can pre-annotate.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The models of the project that can pre-annotate, each with its details.
    """
    project = state.project(project_id)
    project_task = project.task
    out = []
    for card in ModelStore(state.store.dir(project_id)).list():
        public = card.get("source") == "public"
        spec = None if public else get_model(card["model"])
        # a model this version no longer knows (catalogue changed): listed, but it cannot run
        missing = get_public(str(card.get("public") or "")) is None if public else spec is None
        kind = card.get("task_kind") or (spec.task_kind if spec else "")
        out.append(
            {
                "number": card["number"],
                "model": card["model"],
                "label": card.get("label") or (spec.label if spec else card["model"]),
                "public": public,
                "missing": missing,
                "task_kind": kind,
                "current": card.get("current", False),
                "metric": card.get("metric"),
                "classes": preannotate_options.model_classes(card, list(project.classes)),
                "options": preannotate_options.schema(kind),
                "value": card.get("value"),
                "supported": not missing
                and kind in preannotate.KINDS
                and preannotate.fits(kind, project_task)
                and not (spec is not None and spec.family == "ultralytics"),
            }
        )
    return out


@router.get("/projects/{project_id}/preannotations")
def overview(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Models that can pre-annotate, tasks with their counts, and the runs (newest first).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{models, tasks, runs}``; each model has its ``classes`` (``name``, ``in_project``) and the
        ``options`` its kind takes (see :func:`activelabel.preannotate_options.schema`).
    """
    state.project(project_id)
    project_dir = state.store.dir(project_id)
    return {
        "models": _models(state, project_id),
        "tasks": preannotate.task_counts(project_dir),
        "runs": preannotate.PreannotationStore(project_dir).list(),
    }


@router.post("/projects/{project_id}/preannotations")
def start(project_id: int, body: StartBody, state: AppState = STATE) -> dict[str, Any]:
    """Queue a pre-annotation of one task's images with a saved model.

    Args:
        project_id: Id of the project.
        body: The task, the model, the device (``auto``, ``cpu``, ``cuda:N`` or ``mps``), the ``options``
            (inference parameters; ``threshold`` is the older name of ``options.confidence``) and the
            ``classes`` to write.
        state: The shared state of the application.

    Returns:
        The queued pre-annotation run.
    """
    try:
        refuse_regression(state.project(project_id).task)
    except CodedError as error:
        raise bad_request(error) from None
    project_dir = state.store.dir(project_id)
    try:
        card = ModelStore(project_dir).get(body.model)
    except ModelNotFound:
        raise ApiError(404, "model_not_found", "This model does not exist.") from None
    info = next((m for m in _models(state, project_id) if m["number"] == body.model), None)
    if info and info["missing"]:
        raise ApiError(
            400,
            "model_not_in_catalogue",
            "This model is not in the catalogue of this version of ActiveLabel.",
        )
    if not info or not info["supported"]:
        kind = card.get("task_kind") or "these"
        if kind in preannotate.KINDS and not preannotate.fits(kind, state.project(project_id).task):
            raise ApiError(
                400,
                "model_kind_mismatch",
                f"This model makes {kind} labels, which this project does not use.",
                kind=kind,
            )
        raise ApiError(
            400,
            "preannotation_kind_unsupported",
            f"Pre-annotation does not support this model ({kind}): YOLO models cannot be used for it.",
            kind=kind,
        )
    try:
        device = devices.check_here(body.device, state.settings.workspace)
    except devices.DeviceError as error:
        raise bad_request(error) from None
    given = dict(body.options or {})
    if body.threshold is not None and "confidence" not in given:
        threshold = round(body.threshold, 3)
        if not 0.0 < threshold < 1.0:  # also refuses NaN
            raise ApiError(400, "threshold_range", "The minimum confidence must be between 0.001 and 0.999.")
        given["confidence"] = threshold
    try:
        options = preannotate_options.resolve(str(info.get("task_kind") or ""), given)
        classes = preannotate_options.resolve_classes(info.get("classes") or [], body.classes)
    except CodedError as error:
        raise bad_request(error) from None
    try:
        images = preannotate.targets(project_dir, body.task, body.scope)
    except ValueError as error:
        raise bad_request(error) from None
    if not images:
        raise ApiError(400, "no_images_for_choice", "No image of this task matches this choice.")
    runs = preannotate.PreannotationStore(project_dir)
    if any(
        r["status"] in ("queued", "running") and r["task"] == body.task and r["scope"] == body.scope
        for r in runs.list()
    ):
        raise ApiError(409, "preannotation_running", "These images are already being pre-annotated.")
    task_name = next((t["name"] for t in preannotate.task_counts(project_dir) if t["id"] == body.task), "")
    run = runs.create(
        model=body.model,
        model_label=info["label"],
        task=body.task,
        task_name=task_name,
        scope=body.scope,
        threshold=options["confidence"],
        options=options,
        classes=classes,
        total=len(images),
        device=device,
    )
    state.runner.enqueue_preannotation(project_id, run["number"])
    return run


@router.post("/projects/{project_id}/preannotations/{number}/cancel")
def cancel(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running pre-annotation (nothing is written).

    Args:
        project_id: Id of the project.
        number: Number of the pre-annotation run.
        state: The shared state of the application.

    Returns:
        The pre-annotation run.
    """
    state.project(project_id)
    store = preannotate.PreannotationStore(state.store.dir(project_id))
    try:
        state.runner.cancel_preannotation(project_id, number)
        return store.get(number)
    except preannotate.RunNotFound:
        raise ApiError(404, "preannotation_not_found", "This pre-annotation does not exist.") from None


@router.post("/projects/{project_id}/preannotations/accept")
def accept(project_id: int, body: AcceptBody, state: AppState = STATE) -> dict[str, Any]:
    """Keep the pre-annotations of a task (or of every task) as they are, marked as reviewed.

    Args:
        project_id: Id of the project.
        body: The task to accept (every task when omitted).
        state: The shared state of the application.

    Returns:
        ``{accepted}``: how many images were marked as reviewed.
    """
    state.project(project_id)
    project_dir = state.store.dir(project_id)
    try:
        images = preannotate.unreviewed_images(project_dir, body.task)
    except CodedError as error:
        raise ApiError(404, error.code, str(error), **error.params) from None
    return {"accepted": preannotate.accept(project_dir, images)}
