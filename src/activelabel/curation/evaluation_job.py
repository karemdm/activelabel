"""The job that evaluates a saved model on one part of a dataset version, in FiftyOne (global queue; #611).

It lives next to the label check (:mod:`activelabel.curation.mistakenness_job`), whose way of working it
follows: it predicts with the model in the training environment, writes the predictions into the project's
FiftyOne dataset and lets FiftyOne compare them with the ground truth (:mod:`evaluation_fiftyone`). Phases:
``environment``, ``predicting`` (the worker ``predict.py`` runs the model on the images of the part that have
labels; cancelling kills it) and ``fiftyone``.

The state is one file, ``evaluations.json``: ``{"models": {"<N>": entry}}``, an entry per model with the
job's fields (``status``, ``job``, ``device``, ``phase``, ``done``, ``total``, ``version``, ``part``,
``fold``, ``error``...) and, once one finished, its ``result``. A new run that is cancelled or fails leaves
the previous ``result``. Only one evaluation runs at a time in a project.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

from activelabel import fiftyone_bridge, preannotate, preannotate_options
from activelabel.constants import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from activelabel.curation import evaluation_fiftyone, fiftyone_runs
from activelabel.curation.state_store import StateStore
from activelabel.datasets import DatasetStore
from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Progress, readable_error
from activelabel.labels import Label, load_annotations
from activelabel.projects import Project
from activelabel.storage import now
from activelabel.tasks import Task
from activelabel.training.models import ModelStore
from activelabel.training.runner import TrainingRunner

LOG = logging.getLogger(__name__)
KIND = "evaluation"
STATE_FILE = "evaluations.json"
WORKER = "predict"  # name of the worker process (what cancelling kills)
#: Lowest confidence of a prediction kept: low, so the metrics see the doubtful predictions too.
CONFIDENCE = 0.05
PROGRESS_EVERY = 1.0  # seconds between writes of the progress to the state file
ACTIVE = (QUEUED, RUNNING)


def store(project_dir: Path) -> StateStore:
    """The evaluations file of a project.

    Args:
        project_dir: Folder of the project.

    Returns:
        The store of ``evaluations.json``.
    """
    return StateStore(project_dir, STATE_FILE)


def entry_of(project_dir: Path, model: int) -> dict[str, Any]:
    """The saved entry of one model.

    Args:
        project_dir: Folder of the project.
        model: Number of the saved model.

    Returns:
        Its job fields and ``result``; ``{}`` when the model was never evaluated.
    """
    entry = store(project_dir).read().get("models", {}).get(str(model), {})
    return entry if isinstance(entry, dict) else {}


def _patch(project_dir: Path, model: int, **fields: Any) -> dict[str, Any]:
    """Change some fields of one model's entry and save the file.

    Args:
        project_dir: Folder of the project.
        model: Number of the saved model.
        **fields: The fields to set.

    Returns:
        The entry after the change.
    """
    state = store(project_dir)
    with state.lock:
        models = dict(state.read().get("models", {}))
        entry = {**models.get(str(model), {}), **fields}
        models[str(model)] = entry
        state.update(models=models)
    return entry


def active_model(project_dir: Path) -> int | None:
    """The model being evaluated now (queued or running), if any.

    Args:
        project_dir: Folder of the project.

    Returns:
        The model number, or None.
    """
    for number, entry in store(project_dir).read().get("models", {}).items():
        if isinstance(entry, dict) and entry.get("status") in ACTIVE:
            return int(number)
    return None


def target(
    project_dir: Path, card: dict[str, Any], version: int | None, part: str | None, fold: int | None
) -> tuple[int, str, int | None, list[str]]:
    """What to evaluate on: the dataset version, its part (and fold) and the image ids.

    Without a version it is the one the model was trained on (the newest one when that was deleted, or for
    a public model); without a part it is ``test`` when that has images, else ``val`` (of fold 0 in a K-fold
    version).

    Args:
        project_dir: Folder of the project.
        card: The card of the saved model.
        version: The version asked for, or None.
        part: ``train``, ``val`` or ``test``, or None.
        fold: The fold of a K-fold version, or None.

    Returns:
        ``(version, part, fold, image ids)``.

    Raises:
        DatasetNotFound: the version asked for does not exist.
        CodedError: ``no_dataset_version`` when the project has none; a part or fold that does not fit.
    """
    datasets = DatasetStore(project_dir)
    if version is None:
        trained = card.get("dataset")
        versions: list[int] = [v["version"] for v in datasets.versions()]
        if not versions:
            raise CodedError(
                "no_dataset_version", "Create a data split first (step 4): it holds the test part."
            )
        number = trained if trained in versions else versions[0]
    else:
        number = version
    folds = datasets.read(number).get("folds")
    if part is None:
        part = "test" if datasets.image_ids(number, "test") else "val"
    if folds is not None and part != "test" and fold is None:
        fold = 0
    return number, part, fold, datasets.image_ids(number, part, fold)


def evaluable(project_dir: Path, task: Task, ids: list[str]) -> list[dict[str, str]]:
    """The images among ``ids`` that can be evaluated: active, reviewed by a person and with a file.

    A classification image needs its class; for objects an image reviewed with no objects counts (a model
    that finds objects there is wrong).

    Args:
        project_dir: Folder of the project.
        task: Task of the project.
        ids: Image ids of the part.

    Returns:
        ``[{"id", "path"}]`` with absolute paths, in the order of ``ids``.
    """
    annotations = load_annotations(project_dir)
    images: dict[str, list[Label]] = annotations.images if annotations else {}
    if task is Task.CLASSIFICATION:
        reviewed = preannotate.annotated_ids(images, ids)
    else:
        reviewed = {i for i in ids if preannotate.image_state(images, i) == "human"}
    manifest = Manifest(project_dir)
    active = {record.id: record for record in manifest.active()}
    items = []
    for image_id in ids:
        record = active.get(image_id)
        path = manifest.file_path(record) if record is not None else None
        if image_id in reviewed and path is not None and path.exists():
            items.append({"id": image_id, "path": str(path.resolve())})
    return items


class _Events:
    """Takes the worker's events: the labels predicted per image, unreadable images and the progress."""

    def __init__(self, project_dir: Path, model: int, progress: Progress) -> None:
        """Start collecting the events of one evaluation.

        Args:
            project_dir: Folder of the project.
            model: Number of the saved model.
            progress: Progress handle of the job.
        """
        self.predictions: dict[str, list[Label]] = {}
        self.unreadable: list[str] = []
        self._project_dir = project_dir
        self._model = model
        self._progress = progress
        self._last = 0.0

    def __call__(self, event: dict[str, Any]) -> None:
        """Take one event of the worker (``prediction``, ``skipped`` or ``progress``; others are ignored).

        Args:
            event: A JSON event of the worker.
        """
        kind = event.get("event")
        if kind == "prediction":
            self.predictions[str(event["id"])] = preannotate.mask_labels(event.get("labels") or [])
        elif kind == "skipped":
            self.unreadable.append(str(event.get("id")))
        elif kind == "progress" and time.monotonic() - self._last >= PROGRESS_EVERY:
            self._last = time.monotonic()
            _patch(self._project_dir, self._model, done=event.get("step", 0), total=event.get("steps", 0))
            self._progress.update(done=event.get("step"), total=event.get("steps"))


def start(
    runner: TrainingRunner,
    project: Project,
    project_dir: Path,
    model: int,
    device: str = "auto",
    version: int | None = None,
    part: str | None = None,
    fold: int | None = None,
) -> dict[str, Any]:
    """Queue the evaluation of a model (one at a time in a project).

    Args:
        runner: The training runner (its queue holds the heavy jobs of every project).
        project: The project.
        project_dir: Folder of the project.
        model: Number of the saved model.
        device: Where the model runs (``auto``, ``cpu``, ``cuda:N`` or ``mps``): the job's lane.
        version: The dataset version to evaluate on (default: the one the model was trained on).
        part: ``train``, ``val`` or ``test`` (default: ``test`` when it has images, else ``val``).
        fold: The fold of a K-fold version (default 0 for ``val``).

    Returns:
        The model's entry in ``evaluations.json``, queued.

    Raises:
        CodedError: ``evaluation_task_unsupported`` (keypoints, regression), ``evaluation_running``,
            ``no_dataset_version``, ``no_images_in_part``, a part or fold that does not fit the version.
        DatasetNotFound: the version does not exist.
        ModelNotFound: unknown model.
        RunnerError: a model that cannot predict on this project (see ``TrainingRunner.predictor``).
    """
    evaluation_fiftyone.refuse(project.task)
    card = ModelStore(project_dir).get(model)
    runner.predictor(project.id, model)  # refused now rather than in the queue
    version, part, fold, ids = target(project_dir, card, version, part, fold)
    images = len(evaluable(project_dir, project.task, ids))
    if not images:
        raise CodedError(
            "no_images_in_part",
            "No image of this part has labels a person reviewed.",
            version=version,
            part=part,
        )
    state = store(project_dir)
    with state.lock:  # the job starts only after its id is saved (it reads the state first)
        busy = active_model(project_dir)
        if busy is not None:
            raise CodedError("evaluation_running", "Another model is being evaluated.", model=busy)
        _patch(
            project_dir,
            model,
            status=QUEUED,
            device=device,
            version=version,
            part=part,
            fold=fold,
            images=images,
            created=now(),
            started="",
            finished="",
            error="",
            phase="",
            done=0,
            total=0,
            message="",
        )
        job_id = runner.enqueue_tool(
            project.id,
            KIND,
            partial(run, runner, project, project_dir, model),
            {"model": model},
            device=device,
            on_cancel=partial(cancel, runner, project.id, project_dir, model),
        )
        return _patch(project_dir, model, job=job_id)


def state_of(runner: TrainingRunner, project_dir: Path, model: int) -> dict[str, Any]:
    """The entry as the interface shows it (one left queued or running by a previous start is interrupted).

    Args:
        runner: The training runner.
        project_dir: Folder of the project.
        model: Number of the saved model.

    Returns:
        The saved entry (``{}`` before the first evaluation) with ``active``: the model being evaluated
        now (any), or None.
    """
    entry = entry_of(project_dir, model)
    if entry.get("status") in ACTIVE and runner.tool_job(str(entry.get("job") or "")) is None:
        entry = _patch(project_dir, model, status=FAILED, error="interrupted", finished=now(), phase="")
    return {**entry, "active": active_model(project_dir)} if entry else {"active": active_model(project_dir)}


def cancel(runner: TrainingRunner, project_id: int, project_dir: Path, model: int) -> dict[str, Any]:
    """Cancel a queued or running evaluation (its worker is killed; the previous result stays).

    Args:
        runner: The training runner.
        project_id: Id of the project.
        project_dir: Folder of the project.
        model: Number of the saved model.

    Returns:
        The saved entry.
    """
    state = store(project_dir)
    with state.lock:
        entry = entry_of(project_dir, model)
        if entry.get("status") not in ACTIVE:
            return entry
        runner.cancel_tool(project_id, str(entry.get("job") or ""), WORKER)
        if entry.get("status") == QUEUED:
            return _patch(project_dir, model, status=CANCELLED, finished=now(), phase="")
    return entry


def run(runner: TrainingRunner, project: Project, project_dir: Path, model: int, progress: Progress) -> None:
    """Body of the job: predict on the images of the part, then evaluate them in FiftyOne.

    Args:
        runner: The training runner (environment, worker processes).
        project: The project.
        project_dir: Folder of the project.
        model: Number of the saved model.
        progress: Progress handle of the job (checked for cancellation).
    """
    state = store(project_dir)
    with state.lock:
        entry = entry_of(project_dir, model)
        if entry.get("status") != QUEUED:  # cancelled while waiting
            return
        _patch(project_dir, model, status=RUNNING, started=now(), phase="environment", done=0, total=0)
    try:
        predictor = runner.predictor(project.id, model)
        version, part, fold = int(entry["version"]), str(entry["part"]), entry.get("fold")
        ids = DatasetStore(project_dir).image_ids(version, part, fold)
        items = evaluable(project_dir, project.task, ids)
        if not items:
            raise CodedError("no_images_in_part", "No image of this part has labels a person reviewed.")
        events = _Events(project_dir, model, progress)
        options = {**preannotate_options.resolve(predictor.kind, None), "confidence": CONFIDENCE}
        runner.predict(
            project.id,
            predictor,
            items,
            options,
            str(entry.get("device") or "auto"),
            progress,
            partial(_phase_then, project_dir, model, events),
            project_dir / "evaluation.log",
            partial(_tell, project_dir, model),
        )
        progress.check()
        _patch(project_dir, model, phase="fiftyone", done=len(items))
        dataset = fiftyone_bridge.sync_dataset(project, Manifest(project_dir), load_annotations(project_dir))
        result = evaluation_fiftyone.evaluate(dataset, project.task, model, events.predictions)
        result.update(version=version, part=part, fold=fold, evaluated=now(), unreadable=events.unreadable)
        _record(dataset, model, entry, result)
        _patch(project_dir, model, status=DONE, finished=now(), phase="", result=result)
    except Cancelled:
        _patch(project_dir, model, status=CANCELLED, finished=now(), phase="")
    except Exception as exc:  # noqa: BLE001 - shown to the person
        LOG.exception("The evaluation of model %s of project %s failed", model, project.id)
        _patch(project_dir, model, status=FAILED, error=readable_error(exc), finished=now(), phase="")


def _record(dataset: Any, model: int, entry: dict[str, Any], result: dict[str, Any]) -> None:
    """Record the evaluation as a run of the dataset; a failure there is only logged.

    Args:
        dataset: The project's FiftyOne dataset.
        model: Number of the saved model.
        entry: The model's entry (``device``).
        result: The result of the evaluation (``version``, ``part``, ``images``, ``fields``).
    """
    try:
        fiftyone_runs.record(
            dataset,
            "evaluation",
            suffix=f"_m{model}",
            model=model,
            device=entry.get("device"),
            version=result["version"],
            part=result["part"],
            eval_key=evaluation_fiftyone.key_of(model),
            fields=result["fields"],
            images=result["images"],
        )
    except Exception:  # noqa: BLE001 - the results are kept without the run
        LOG.exception("Could not record the evaluation as a FiftyOne run")


def _phase_then(project_dir: Path, model: int, events: _Events, event: dict[str, Any]) -> None:
    """Pass a worker event on, marking the ``predicting`` phase at the first one.

    Args:
        project_dir: Folder of the project.
        model: Number of the saved model.
        events: The collector of the events.
        event: A JSON event of the worker.
    """
    if (
        not events.predictions
        and not events.unreadable
        and entry_of(project_dir, model).get("phase") != "predicting"
    ):
        _patch(project_dir, model, phase="predicting", message="")
    events(event)


def _tell(project_dir: Path, model: int, message: str) -> None:
    """Show what the environment set-up is doing.

    Args:
        project_dir: Folder of the project.
        model: Number of the saved model.
        message: The step being done (a code the interface translates, such as ``train_env_install:torch``).
    """
    _patch(project_dir, model, message=message)
