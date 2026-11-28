"""HTTP API of the values of a regression project: one image's numbers are edited here (#619)."""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.labels import AnnotationSet, Label, annotations_lock, load_annotations, save_annotations
from activelabel.tasks import Task
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class ImageValues(BaseModel):
    """The values to set on one image.

    Attributes:
        values: ``{target: number}``; ``null`` clears the value of that target. Targets not named stay
            as they are.
    """

    values: dict[str, float | None]


@router.put("/projects/{project_id}/images/{image_id}/values")
def set_values(project_id: int, image_id: str, body: ImageValues, state: AppState = STATE) -> dict[str, Any]:
    """Set (or clear) the values of some targets of one image of a regression project.

    Args:
        project_id: Id of the project.
        image_id: Id of the image.
        body: The values to set.
        state: The shared state of the application.

    Returns:
        ``{image_id, values}``: all the values of the image after the change, ``{target: number}``.

    Raises:
        ApiError: ``values_not_regression`` (the project has no values), ``image_not_found``,
            ``target_unknown`` (not a target of the project) or ``value_invalid`` (not a finite number).
    """
    project = state.project(project_id)
    if project.task is not Task.REGRESSION:
        raise ApiError(400, "values_not_regression", "Only a regression project has values per image.")
    try:
        state.manifest(project_id).get(image_id)
    except KeyError:
        raise ApiError(404, "image_not_found", "This image is not in the project.") from None
    for target, value in body.values.items():
        if target not in project.classes:
            raise ApiError(
                400, "target_unknown", f"'{target}' is not a target of this project.", target=target
            )
        if value is not None and not math.isfinite(value):
            raise ApiError(400, "value_invalid", "A value must be a finite number.", target=target)
    project_dir = state.store.dir(project_id)
    with annotations_lock(project_dir):
        found = load_annotations(project_dir) or AnnotationSet(Task.REGRESSION, list(project.classes), {}, {})
        kept = [lb for lb in found.images.get(image_id, []) if lb.label not in body.values]
        kept += [Label(t, value=v) for t, v in body.values.items() if v is not None]
        order = {name: i for i, name in enumerate(project.classes)}
        kept.sort(key=lambda lb: order.get(lb.label, len(order)))
        found.images[image_id] = kept
        found.classes = list(project.classes)
        save_annotations(project_dir, found)
    return {"image_id": image_id, "values": {lb.label: lb.value for lb in kept}}
