"""HTTP API of the model catalogue and training runs (see docs/TRAINING.md)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.jobs import DONE
from activelabel.projects import Project
from activelabel.storage import now
from activelabel.training import augmentations, devices
from activelabel.training.catalog import (
    COMMON_PARAMS,
    DEFAULT_ARCH,
    METRICS,
    PROJECT_TYPES,
    device_size,
    get_model,
    models_for,
)
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.environment import TrainingEnvironment
from activelabel.training.licences import (
    LICENCES,
    LicenceNotAccepted,
    LicenceStore,
    licence_of_model,
)
from activelabel.training.models import WEIGHTS, ModelStore
from activelabel.training.quality import quality_alerts
from activelabel.training.runner import EXTRA_PACKAGES, RunnerError, check_resume
from activelabel.training.runs import QUEUED, RunBusy, RunNotFound, RunStore
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

LOG = logging.getLogger(__name__)
CONFIG_MESSAGES = {
    "config_missing": "The configuration file of this run is missing.",
    "config_damaged": "The configuration file of this run is damaged.",
}
#: Fields of a stored warning of a run.
WARNING_KEYS = ("code", "params", "message")
#: the project types the catalogues answer for
ASKED_TYPES = PROJECT_TYPES

router = APIRouter(prefix="/api")
STATE = Depends(get_state)


class DeleteRun(BaseModel):
    """Body of DELETE /projects/{id}/runs/{n}."""

    confirm: bool = False


def start_run(state: AppState, project_id: int, number: int) -> None:
    """Hand a queued run to the runner (it starts when no other run is running).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        number: Number of the run.
    """
    state.runner.enqueue(project_id, number)


def _store(state: AppState, project_id: int) -> RunStore:
    """The run store of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The run store of the project.
    """
    state.project(project_id)
    return RunStore(state.store.dir(project_id))


def _warnings(run: dict[str, Any]) -> list[dict[str, Any]]:
    """The warnings of a run as ``{code, params, message}`` (runs made before the codes kept plain text).

    Args:
        run: The run record.

    Returns:
        The warnings; an old text warning has ``code`` "" and empty ``params``.
    """
    out = []
    for warning in run.get("warnings") or []:
        if isinstance(warning, dict):
            out.append(
                {
                    "code": "",
                    "params": {},
                    "message": "",
                    **{k: warning[k] for k in warning if k in WARNING_KEYS},
                }
            )
        else:
            out.append({"code": "", "params": {}, "message": str(warning)})
    return out


def _summary(store: RunStore, run: dict[str, Any]) -> dict[str, Any]:
    """A run with the main fields of its configuration.

    When ``config.yaml`` cannot be read the run has ``config_error`` (``config_missing`` or
    ``config_damaged``) and a plain ``config_message``; the detail (which may name a path) goes to the log.

    Args:
        store: The run store.
        run: The run record.

    Returns:
        The run with the main fields of its configuration; ``warnings`` as ``{code, params, message}``
        and ``error_code`` / ``error_params`` (empty when the failure has no code), for the translations;
        ``alerts`` as ``{code, params}`` (a finished run made before the alerts gets them from its results,
        without the validation size check).
    """
    alerts = run.get("alerts")
    if alerts is None and run.get("status") == DONE:
        alerts = quality_alerts(store.results(run["number"]), store.metrics(run["number"]))
    run = {
        **run,
        "alerts": alerts or [],
        "warnings": _warnings(run),
        "error_code": run.get("error_code") or "",
        "error_params": run.get("error_params") or {},
    }
    try:
        config = store.config(run["number"])
    except (OSError, ValueError) as exc:  # a missing or damaged config.yaml must not hide every run
        LOG.warning("Run %s: unreadable config.yaml: %s", run["number"], exc)
        code = "config_missing" if isinstance(exc, FileNotFoundError) else "config_damaged"
        return {**run, "config_error": code, "config_message": CONFIG_MESSAGES[code]}
    return {
        **run,
        "model": config.model,
        "dataset": config.dataset,
        "fold": config.fold,
        "start": config.start,
        "mode": config.mode,
        "metric": config.metric,
        "augmentations": config.augmentations,
    }


def _checked(state: AppState, project: Project, config: RunConfig) -> RunConfig:
    """Validate against the catalogue and the dataset version; fill in its fingerprint.

    Args:
        state: The shared state of the application.
        project: The project.
        config: The configuration to check.

    Returns:
        The configuration, validated and with the dataset fingerprint.

    Raises:
        ApiError: the dataset or the configuration is not right.
    """
    try:
        version = DatasetStore(state.store.dir(project.id)).get(config.dataset)
    except DatasetNotFound:
        raise ApiError(
            404, "dataset_not_found", "This dataset version does not exist.", version=config.dataset
        ) from None
    if version["type"] != project.task.value:
        raise ApiError(400, "dataset_wrong_type", "This dataset version was made for another project type.")
    try:
        checked = validate(config, project.task.value, len(version["folds"] or []))
    except ConfigError as error:
        raise bad_request(error) from None
    model = get_model(checked.model)
    licence = licence_of_model(model) if model is not None else None
    if licence is not None:
        try:
            LicenceStore(state.settings.workspace).require([licence])
        except LicenceNotAccepted as error:
            raise bad_request(error) from None
    if version.get("incomplete"):
        raise ApiError(
            400,
            "dataset_incomplete",
            "Some images of this dataset version were deleted from the project: make a new version.",
        )
    gone = [c for c in version["classes"] if c not in project.classes]
    if gone:  # the version is frozen: the way out is a new version, not a model with classes nobody can use
        raise ApiError(
            400,
            "dataset_classes_removed",
            f"This dataset version has classes the project no longer has ({', '.join(gone[:5])}): "
            "make a new version.",
            classes=gone,
        )
    if not version.get("folds") and not (version.get("sizes") or {}).get("train"):
        raise ApiError(
            400,
            "dataset_train_empty",
            "The training part of this dataset version has no images: make another one.",
        )
    if not version.get("folds") and not (version.get("sizes") or {}).get("val"):
        raise ApiError(
            400,
            "dataset_no_validation",
            "This dataset version has no validation images: make a version with validation.",
        )
    checked.dataset_fingerprint = version["fingerprint"]
    return checked


def _device(state: AppState, choice: Any) -> str:
    """The device a run asks for, checked against this computer.

    Args:
        state: The shared state of the application.
        choice: The requested device (``None`` means ``auto``).

    Returns:
        The device choice to store.
    """
    try:
        return devices.check_here(choice, state.settings.workspace)
    except devices.DeviceError as error:
        raise bad_request(error) from None


def _create(state: AppState, project: Project, config: RunConfig, device: str = "auto") -> dict[str, Any]:
    """Store a validated configuration as a queued run and hand it to the runner.

    Args:
        state: The shared state of the application.
        project: The project.
        config: The validated configuration.
        device: The checked device choice of the run.

    Returns:
        The new run, queued.
    """
    store = _store(state, project.id)
    run = store.create(_checked(state, project, config), device)
    start_run(state, project.id, run["number"])
    return _summary(store, store.get(run["number"]))


@router.get("/training/catalog")
def catalog(type: str | None = None) -> dict[str, Any]:  # noqa: A002 - the query parameter is named type
    """Models (with their hyperparameter specs) a project type offers; all without ``type``.

    Args:
        type: Project type to list the models of (all when omitted).

    Returns:
        ``{models, common, metrics, default_arch}``: each model has its ``arch`` and ``variant`` (the dialog's
        architecture and size fields), ``experimental``, ``params_m`` and ``file_mb``; ``default_arch`` is the
        architecture a family proposes first (``{"ultralytics": "YOLO26"}``).
    """
    if type is not None and type not in ASKED_TYPES:
        raise ApiError(
            400,
            "type_invalid",
            f"The type must be one of: {', '.join(ASKED_TYPES)}.",
            choices=list(ASKED_TYPES),
        )
    return {
        "models": [m.as_dict() for m in models_for(type)],
        "common": [p.as_dict() for p in COMMON_PARAMS],
        "metrics": {kind: {"choices": list(v[0]), "default": v[1]} for kind, v in METRICS.items()},
        "default_arch": dict(DEFAULT_ARCH),
    }


@router.get("/training/augmentations")
def augmentation_catalog(type: str | None = None) -> dict[str, Any]:  # noqa: A002 - the query parameter is named type
    """Augmentation techniques and preset lists per kind of task (of the type's models; all without a type).

    Args:
        type: project type, or ``None`` for every kind of task.

    Returns:
        ``{"levels": [...], "kinds": {kind: {"techniques": [...], "presets": {level: [entries]}}}}``.

    Raises:
        ApiError: unknown project type.
    """
    if type is not None and type not in ASKED_TYPES:
        raise ApiError(
            400,
            "type_invalid",
            f"The type must be one of: {', '.join(ASKED_TYPES)}.",
            choices=list(ASKED_TYPES),
        )
    kinds = sorted({m.task_kind for m in models_for(type)})
    return {"levels": list(augmentations.LEVELS), "kinds": {k: augmentations.catalogue(k) for k in kinds}}


@router.get("/training/environment")
def environment(model: str | None = None, state: AppState = STATE) -> dict[str, Any]:
    """Packages installed in the training environment and the GPU found (nothing is installed here).

    With ``model`` (a catalogue id) the answer also has ``missing`` (packages that model needs and the
    environment lacks) and ``install_needed``: true when the first run with it will install packages.

    Args:
        model: Catalogue id of a model, to also list what it is missing.
        state: The shared state of the application.

    Returns:
        ``{exists, packages, gpu, devices, default_size}`` (plus ``missing`` and ``install_needed`` with
        ``model``); ``default_size`` is the model size the dialog proposes first here (``n`` without a GPU,
        ``s`` on a GPU under 8 GB, ``m`` from 8 GB);
        ``devices`` are the device choices of this computer (``auto``, ``cpu``, ``cuda:N``, ``mps``).
    """
    env = TrainingEnvironment(state.settings.workspace)
    gpu = env.gpu()
    exists = env.exists()
    installed: dict[str, str] = env.installed() if exists else {}
    answer: dict[str, Any] = {
        "exists": exists,
        "packages": installed,
        "gpu": None if gpu is None else gpu.as_dict(),
        "devices": devices.available(env),
        "default_size": device_size(None if gpu is None else gpu.memory_mb, gpu is not None),
    }
    if model is not None:
        spec = get_model(model)
        if spec is None:
            raise ApiError(404, "model_not_found", "This model does not exist.")
        needed = [*spec.packages, *EXTRA_PACKAGES]
        answer["missing"] = env.missing(needed) if exists else needed
        answer["install_needed"] = bool(answer["missing"])
        licence = licence_of_model(spec)
        answer["licence"] = licence.id if licence else None
        answer["licence_accepted"] = licence is None or LicenceStore(state.settings.workspace).is_accepted(
            licence.id
        )
    return answer


class AcceptLicence(BaseModel):
    """Body of POST /training/licences/{id}/accept."""

    accept: bool = False


@router.get("/training/licences")
def licences(state: AppState = STATE) -> dict[str, Any]:
    """The licences that must be accepted before their package is installed, and whether each was.

    Args:
        state: The shared state of the application.

    Returns:
        ``{"licences": [{id, name, package, url, accepted, accepted_at}]}``.
    """
    return {"licences": LicenceStore(state.settings.workspace).status()}


@router.post("/training/licences/{licence_id}/accept")
def accept_licence(licence_id: str, body: AcceptLicence, state: AppState = STATE) -> dict[str, Any]:
    """Record that the person accepts a licence (the date is kept in the workspace settings).

    Args:
        licence_id: Id of the licence (``ultralytics``).
        body: ``accept`` must be true.
        state: The shared state of the application.

    Returns:
        The licence with its state, as in :func:`licences`.

    Raises:
        ApiError: unknown licence (404), or ``accept`` is not true (400, ``licence_not_confirmed``).
    """
    if licence_id not in LICENCES:
        raise ApiError(404, "licence_not_found", "This licence does not exist.")
    if not body.accept:
        raise ApiError(400, "licence_not_confirmed", "Send accept: true to accept the licence.")
    store = LicenceStore(state.settings.workspace)
    store.accept(licence_id)
    return next(entry for entry in store.status() if entry["id"] == licence_id)


@router.get("/projects/{project_id}/runs/{number}/log")
def run_log(project_id: int, number: int, lines: int = 200, state: AppState = STATE) -> dict[str, Any]:
    """The last ``lines`` lines of a run's log (1..2000).

    Args:
        project_id: Id of the project.
        number: Number of the run.
        lines: How many lines (1 to 2000).
        state: The shared state of the application.

    Returns:
        ``{lines}``: the end of the log.
    """
    store = _store(state, project_id)
    try:
        folder = store._require(number)  # noqa: SLF001 - folder of an existing run
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
    count = max(1, min(int(lines), 2000))
    return {"lines": _tail(folder / "log.txt", count)}


def _tail(path: Path, count: int, chunk: int = 64 * 1024) -> list[str]:
    """The last ``count`` lines of a text file, reading only its end.

    Args:
        path: The text file.
        count: How many lines.
        chunk: Bytes read per step from the end.

    Returns:
        The last lines (empty when the file does not exist).
    """
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
    """Every run of the project, newest first, with its model number and whether it still has its weights.

    The weights move to the model when a run is saved, and go with it when the model is deleted.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{runs}``, newest first; each run has ``model_number`` (``None`` if not saved) and ``has_weights``.
    """
    store = _store(state, project_id)
    saved = {m["run"]: m["number"] for m in ModelStore(state.store.dir(project_id)).list()}
    return {
        "runs": [
            {
                **_summary(store, r),
                "model_number": saved.get(r["number"]),
                "has_weights": (store.path(r["number"]) / WEIGHTS).exists(),
            }
            for r in store.list()
        ]
    }


@router.post("/projects/{project_id}/runs")
def create_run(project_id: int, body: dict[str, Any], state: AppState = STATE) -> dict[str, Any]:
    """Validate a configuration and store the run as queued.

    Args:
        project_id: Id of the project.
        body: The run configuration, plus ``device`` (``auto`` when absent: not part of the configuration).
        state: The shared state of the application.

    Returns:
        The new run, queued.
    """
    project = state.project(project_id)
    options = dict(body)
    choice = options.pop("device", None)
    try:
        config = RunConfig.from_dict(options)
    except ConfigError as error:
        raise bad_request(error) from None
    return _create(state, project, config, _device(state, choice))


@router.get("/projects/{project_id}/runs/{number}")
def get_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Configuration, status, metrics per epoch and results of a run.

    Args:
        project_id: Id of the project.
        number: Number of the run.
        state: The shared state of the application.

    Returns:
        The run with its configuration, metrics and results.
    """
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
        raise ApiError(404, "run_not_found", "This run does not exist.") from None


@router.get("/projects/{project_id}/runs/{number}/children/{child}")
def get_child(project_id: int, number: int, child: str, state: AppState = STATE) -> dict[str, Any]:
    """One training of a grid or all-folds run: its status, configuration, metrics and results.

    Args:
        project_id: Id of the project.
        number: Number of the run.
        child: Id of the child training.
        state: The shared state of the application.

    Returns:
        The child with its configuration, metrics and results.
    """
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
        raise ApiError(404, "child_not_found", "This training does not exist.") from None


@router.delete("/projects/{project_id}/runs/{number}")
def delete_run(
    project_id: int, number: int, body: DeleteRun | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Delete a run (needs ``{"confirm": true}``); a running run cannot be deleted.

    Args:
        project_id: Id of the project.
        number: Number of the run.
        body: The confirmation.
        state: The shared state of the application.

    Returns:
        ``{deleted: true}``.
    """
    store = _store(state, project_id)
    if not (body and body.confirm):
        raise ApiError(400, "confirm_required", "Confirm the deletion of this run.")
    try:
        if store.get(number)["status"] == QUEUED:  # also leave the queue of the runner
            state.runner.cancel(project_id, number)
        store.delete(number)
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
    except RunBusy:
        raise ApiError(409, "run_running", "This run is running: cancel it before deleting.") from None
    return {"deleted": True}


@router.post("/projects/{project_id}/runs/{number}/reproduce")
def reproduce_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """A new run with the same configuration (and the same device choice).

    Args:
        project_id: Id of the project.
        number: Number of the run.
        state: The shared state of the application.

    Returns:
        The new run, queued.
    """
    project = state.project(project_id)
    try:
        store = _store(state, project_id)
        config = store.config(number)
        choice = store.get(number).get("device_choice")
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
    except (OSError, ValueError):
        raise ApiError(
            409, "run_config_damaged", "The configuration of this run is missing or damaged."
        ) from None
    return _create(state, project, config, _device(state, choice))


@router.post("/projects/{project_id}/runs/{number}/resume")
def resume_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Queue a failed or cancelled run again: it goes on from its last checkpoint (same number, same folder).

    Args:
        project_id: Id of the project.
        number: Number of the run.
        state: The shared state of the application.

    Returns:
        The run, queued, with ``resumes`` (``{at, from_epoch}`` per resume).
    """
    store = _store(state, project_id)
    try:
        check_resume(state.store.dir(project_id), store, number)
        run = store.get(number)
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
    except (OSError, ValueError):
        raise ApiError(
            409, "run_config_damaged", "The configuration of this run is missing or damaged."
        ) from None
    except RunnerError as error:
        status = 400 if error.code == "resume_composite" else 409
        raise ApiError(status, error.code or "run_not_resumable", str(error)) from None
    resumes = [*(run.get("resumes") or []), {"at": now(), "from_epoch": run.get("checkpoint_epoch")}]
    store.update(
        number,
        status=QUEUED,
        resume=True,
        resumes=resumes,
        error="",
        error_code="",
        error_params={},
        finished=None,
        phase="",
    )
    start_run(state, project_id, number)
    return _summary(store, store.get(number))


@router.post("/projects/{project_id}/runs/{number}/cancel")
def cancel_run(project_id: int, number: int, state: AppState = STATE) -> dict[str, Any]:
    """Cancel a queued or running run (a finished run is left as it is).

    Args:
        project_id: Id of the project.
        number: Number of the run.
        state: The shared state of the application.

    Returns:
        The run.
    """
    store = _store(state, project_id)
    try:
        state.runner.cancel(project_id, number)
        return _summary(store, store.get(number))
    except RunNotFound:
        raise ApiError(404, "run_not_found", "This run does not exist.") from None
