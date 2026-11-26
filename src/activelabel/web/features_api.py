"""What is installed and how much space it takes, with the removal of the training environment (#218).

docs/DEVELOPMENT.md, *Light install*: the core always; FiftyOne as the ``fiftyone`` extra; the packages of
training and pre-annotation in the separate training environment, installed on first use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel import optional
from activelabel.curation.similarity_job import KIND as SIMILARITY_KIND
from activelabel.training.environment import TrainingEnvironment, remove_tree
from activelabel.training.runner import RUNNER_KINDS
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
#: Kinds of job that use the training environment (it cannot be removed while one waits or runs).
ENVIRONMENT_KINDS = (*RUNNER_KINDS, SIMILARITY_KIND)


def folder_size(path: Path) -> int:
    """Bytes of the files under a folder (links are not followed; unreadable files are skipped).

    Args:
        path: the folder.

    Returns:
        The total size; 0 when the folder does not exist.
    """
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                continue
    return total


def _report(state: AppState) -> dict[str, Any]:
    """The optional parts and the training environment.

    Args:
        state: The shared state of the application.

    Returns:
        ``{fiftyone, training_environment}`` (see :func:`features`).
    """
    env = TrainingEnvironment(state.settings.workspace)
    exists = env.exists()
    return {
        "fiftyone": {
            "installed": optional.FIFTYONE,
            "version": optional.version("fiftyone"),
            "install": optional.FIFTYONE_INSTALL,
            "provides": list(optional.FIFTYONE_PROVIDES),
        },
        "training_environment": {
            "exists": exists,
            "path": str(env.path),
            "bytes": folder_size(env.path) if exists else 0,
            "packages": env.installed() if exists else {},
        },
    }


@router.get("/features")
def features(state: AppState = STATE) -> dict[str, Any]:
    """What is installed: FiftyOne (an extra of the Python package) and the training environment.

    Args:
        state: The shared state of the application.

    Returns:
        ``{fiftyone: {installed, version, install, provides}, training_environment: {exists, path, bytes,
        packages}}``: ``install`` is the command that adds FiftyOne; ``provides`` the features it brings.
    """
    return _report(state)


class RemoveEnvironment(BaseModel):
    """Body of DELETE /training/environment."""

    confirm: bool = False


@router.delete("/training/environment")
def remove_environment(body: RemoveEnvironment, state: AppState = STATE) -> dict[str, Any]:
    """Remove the training environment (its packages are installed again by the next run that needs them).

    Args:
        body: ``{"confirm": true}``.
        state: The shared state of the application.

    Returns:
        What is installed afterwards (as :func:`features`).

    Raises:
        ApiError: 400 ``confirm_required`` without the confirmation; 409 ``busy_training`` while a training
            run, pre-annotation, export or similarity map waits or runs.
    """
    if not body.confirm:
        raise ApiError(400, "confirm_required", "Confirm the removal of the training environment.")
    items = state.scheduler.snapshot()["items"]
    if any(item["kind"] in ENVIRONMENT_KINDS for item in items):
        raise ApiError(
            409,
            "busy_training",
            "Wait for the training, export or pre-annotation of this project to finish, or cancel it.",
        )
    remove_tree(TrainingEnvironment(state.settings.workspace).path)
    return _report(state)
