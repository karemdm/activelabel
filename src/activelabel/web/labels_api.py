"""HTTP API of the labels of one image, edited inside ActiveLabel (see docs/ANNOTATION.md)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from activelabel import fiftyone_bridge
from activelabel.annotations.edit import image_labels, save_image_labels
from activelabel.errors import CodedError
from activelabel.fiftyone_names import recorded_name
from activelabel.labels import Label
from activelabel.tasks import Task
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
CONFLICTS = ("labels_changed", "classes_new")  # answered with HTTP 409; the other refusals are 400
log = logging.getLogger(__name__)


class LabelIn(BaseModel):
    """One label of the body of PUT ``/labels``: the class and the shape the project type needs.

    ``track``, ``score`` and ``model`` are sent back as GET gave them, so a tracked object keeps its
    identity and an untouched pre-annotation stays one.
    """

    label: str
    bbox: list[float] | None = None
    polygons: list[list[list[float]]] | None = None
    keypoints: list[list[float]] | None = None
    track: int | None = None
    score: float | None = None
    model: str | None = None


class LabelsBody(BaseModel):
    """Body of PUT ``/projects/{id}/images/{image_id}/labels``."""

    labels: list[LabelIn]
    revision: str
    add_classes: bool = False


def _check_image(state: AppState, project_id: int, image_id: str) -> None:
    """HTTP 404 when the project or the image does not exist.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        image_id: Id of the image.
    """
    if image_id not in state.manifest(project_id).load():
        raise ApiError(404, "image_not_found", "This image is not in the project.")


def refresh_fiftyone(
    project_dir: Path, project_id: int, image_id: str, task: Task, labels: list[Label]
) -> None:
    """Show the edited labels in the project's FiftyOne dataset, when the project has one (best effort).

    Runs after the answer was sent; a project never opened in FiftyOne is skipped without loading it.

    Args:
        project_dir: Folder of the project.
        project_id: Id of the project.
        image_id: The image edited.
        task: Type of the project.
        labels: The labels of the image now.
    """
    if recorded_name(project_dir) is None:
        return
    try:
        dataset = fiftyone_bridge.existing_dataset(project_dir, project_id)
        if dataset is not None:
            fiftyone_bridge.show_project_labels(dataset, image_id, task, labels)
    except Exception:  # noqa: BLE001 - the edit is saved; the dataset is refreshed again when it is opened
        log.warning("Could not show the edited labels of %s in FiftyOne", image_id, exc_info=True)


@router.get("/projects/{project_id}/images/{image_id}/labels")
def get_labels(project_id: int, image_id: str, state: AppState = STATE) -> dict[str, Any]:
    """The labels of one image, the project's classes and the revision to send back with an edit.

    Args:
        project_id: Id of the project.
        image_id: Id of the image.
        state: The shared state of the application.

    Returns:
        ``{labels, classes, revision}``.
    """
    project = state.project(project_id)
    _check_image(state, project_id, image_id)
    return image_labels(project, state.store.dir(project_id), image_id)


@router.put("/projects/{project_id}/images/{image_id}/labels")
def put_labels(
    project_id: int, image_id: str, body: LabelsBody, background: BackgroundTasks, state: AppState = STATE
) -> dict[str, Any]:
    """Replace the labels of one image (all of them: the list sent is the new list).

    Args:
        project_id: Id of the project.
        image_id: Id of the image.
        body: The new labels, the revision they were read at and whether new classes may be added.
        background: Where the refresh of the FiftyOne dataset is queued (after the answer).
        state: The shared state of the application.

    Returns:
        ``{labels, revision, classes}``: the labels saved, their new revision and the project's classes.
    """
    project = state.project(project_id)
    _check_image(state, project_id, image_id)
    state.busy(project_id)
    labels = [Label.from_dict(item.model_dump()) for item in body.labels]
    try:
        saved = save_image_labels(state.store, project_id, image_id, labels, body.revision, body.add_classes)
    except CodedError as exc:
        if exc.code in CONFLICTS:
            raise ApiError(409, exc.code, str(exc), **exc.params) from None
        raise bad_request(exc) from None
    project_dir = state.store.dir(project_id)
    background.add_task(refresh_fiftyone, project_dir, project_id, image_id, project.task, saved.labels)
    return {
        "labels": [lb.as_dict() for lb in saved.labels],
        "revision": saved.revision,
        "classes": list(state.project(project_id).classes),
    }
