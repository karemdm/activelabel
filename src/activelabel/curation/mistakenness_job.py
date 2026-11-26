"""The job that looks for possible label errors with a saved model (global queue, device lane; #605, #612).

Phases: ``environment`` (the model's packages in the training environment), ``predicting`` (the worker
``predict.py`` runs the model on every active image that has labels; cancelling kills it) and
``fiftyone`` (the predictions written next to the ground truth and FiftyOne Brain's
``compute_mistakenness``, plus ``compute_hardness`` for a classification model, which sends its logits; see
:mod:`activelabel.curation.mistakenness_fiftyone`). The state and the per-image results are kept in
``mistakenness.json``: ``scores`` maps an image id to ``{"mistakenness", "missing", "spurious",
"hardness"}``. The labels themselves are never changed.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

from activelabel import fiftyone_bridge, preannotate, preannotate_options
from activelabel.constants import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from activelabel.curation import fiftyone_runs, mistakenness_fiftyone
from activelabel.curation.state_store import StateStore
from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Progress, readable_error
from activelabel.labels import Label, load_annotations
from activelabel.projects import Project
from activelabel.storage import now
from activelabel.training.runner import TrainingRunner

LOG = logging.getLogger(__name__)
KIND = "mistakenness"
STATE_FILE = "mistakenness.json"
WORKER = "predict"  # name of the worker process (what cancelling kills)
#: Lowest confidence of a prediction kept: low, so the comparison also sees the doubtful ones.
CONFIDENCE = 0.05
PROGRESS_EVERY = 1.0  # seconds between writes of the progress to the state file
ACTIVE = (QUEUED, RUNNING)


def store(project_dir: Path) -> StateStore:
    """The label-check file of a project.

    Args:
        project_dir: Folder of the project.

    Returns:
        The store of ``mistakenness.json``.
    """
    return StateStore(project_dir, STATE_FILE)


def labelled_images(project_dir: Path) -> list[dict[str, str]]:
    """The active images with labels whose file exists, as the worker reads them.

    Args:
        project_dir: Folder of the project.

    Returns:
        ``[{"id", "path"}]`` with absolute paths.
    """
    annotations = load_annotations(project_dir)
    labelled = {i for i, labels in (annotations.images if annotations else {}).items() if labels}
    manifest = Manifest(project_dir)
    items = []
    for record in manifest.active():
        path = manifest.file_path(record)
        if record.id in labelled and path is not None and path.exists():
            items.append({"id": record.id, "path": str(path.resolve())})
    return items


class _Events:
    """Takes the worker's events: the labels predicted per image, unreadable images and the progress."""

    def __init__(self, state: StateStore, progress: Progress) -> None:
        """Start collecting the events of one check.

        Args:
            state: The label-check file of the project.
            progress: Progress handle of the job.
        """
        self.predictions: dict[str, list[Label]] = {}
        self.logits: dict[str, list[float]] = {}
        self.unreadable: list[str] = []
        self._state = state
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
            if isinstance(event.get("logits"), list):
                self.logits[str(event["id"])] = [float(x) for x in event["logits"]]
        elif kind == "skipped":
            self.unreadable.append(str(event.get("id")))
        elif kind == "progress" and time.monotonic() - self._last >= PROGRESS_EVERY:
            self._last = time.monotonic()
            self._state.update(done=event.get("step", 0), total=event.get("steps", 0))
            self._progress.update(done=event.get("step"), total=event.get("steps"))


def start(
    runner: TrainingRunner, project: Project, project_dir: Path, model: int, device: str = "auto"
) -> dict[str, Any]:
    """Queue the check (the caller checked that none is queued or running).

    Args:
        runner: The training runner (its queue holds the heavy jobs of every project).
        project: The project.
        project_dir: Folder of the project.
        model: Number of the saved model whose predictions are compared with the labels.
        device: Where the model runs (``auto``, ``cpu``, ``cuda:N`` or ``mps``): the job's lane.

    Returns:
        The state saved in ``mistakenness.json``.

    Raises:
        CodedError: ``no_labelled_images`` when no active image has labels.
        ModelNotFound: unknown model.
        RunnerError: a model that cannot predict on this project (see ``TrainingRunner.predictor``).
    """
    runner.predictor(project.id, model)  # refused now rather than in the queue
    if not labelled_images(project_dir):
        raise CodedError("no_labelled_images", "Label some images first: the check compares labels.")
    state = store(project_dir)
    with state.lock:  # the job starts only after its id is saved (it reads the state first)
        state.update(
            status=QUEUED,
            model=model,
            device=device,
            created=now(),
            started="",
            finished="",
            error="",
            fiftyone_error="",
            phase="",
        )
        job_id = runner.enqueue_tool(
            project.id,
            KIND,
            partial(run, runner, project, project_dir),
            {"model": model},
            device=device,
            on_cancel=partial(cancel, runner, project.id, project_dir),
        )
        return state.update(job=job_id)


def state_of(runner: TrainingRunner, project_dir: Path) -> dict[str, Any]:
    """The state as the interface shows it (one left queued or running by a previous start is interrupted).

    Args:
        runner: The training runner.
        project_dir: Folder of the project.

    Returns:
        The saved state (``{}`` before the first check).
    """
    state = store(project_dir)
    data = state.read()
    if data.get("status") in ACTIVE and runner.tool_job(str(data.get("job") or "")) is None:
        data = state.update(status=FAILED, error="interrupted", finished=now(), phase="")
    return data


def cancel(runner: TrainingRunner, project_id: int, project_dir: Path) -> dict[str, Any]:
    """Cancel a queued or running check (its worker is killed; the previous results stay).

    Args:
        runner: The training runner.
        project_id: Id of the project.
        project_dir: Folder of the project.

    Returns:
        The saved state.
    """
    state = store(project_dir)
    with state.lock:
        data = state.read()
        if data.get("status") not in ACTIVE:
            return data
        runner.cancel_tool(project_id, str(data.get("job") or ""), WORKER)
        if data.get("status") == QUEUED:
            return state.update(status=CANCELLED, finished=now(), phase="")
    return data


def run(runner: TrainingRunner, project: Project, project_dir: Path, progress: Progress) -> None:
    """Body of the job: predict on the labelled images, then compare with the labels in FiftyOne.

    Args:
        runner: The training runner (environment, worker processes).
        project: The project.
        project_dir: Folder of the project.
        progress: Progress handle of the job (checked for cancellation).
    """
    state = store(project_dir)
    with state.lock:
        data = state.read()
        if data.get("status") != QUEUED:  # cancelled while waiting
            return
        state.update(status=RUNNING, started=now(), phase="environment", done=0, total=0, message="")
    try:
        predictor = runner.predictor(project.id, int(data["model"]))
        items = labelled_images(project_dir)
        if not items:
            raise CodedError("no_labelled_images", "Label some images first: the check compares labels.")
        events = _Events(state, progress)
        options = {**preannotate_options.resolve(predictor.kind, None), "confidence": CONFIDENCE}
        runner.predict(
            project.id,
            predictor,
            items,
            options,
            str(data.get("device") or "auto"),
            progress,
            partial(_phase_then, state, events),
            project_dir / "mistakenness.log",
            partial(_tell, state),
            logits=predictor.kind == "classification",
        )
        progress.check()
        state.update(phase="fiftyone", done=len(items))
        manifest = Manifest(project_dir)
        dataset = fiftyone_bridge.sync_dataset(project, manifest, load_annotations(project_dir))
        classes = [c if c in project.classes else None for c in predictor.card.get("classes") or []]
        result = mistakenness_fiftyone.score(
            dataset, project.task, events.predictions, events.logits or None, classes
        )
        _record(dataset, data, bool(events.logits), len(result))
        state.update(
            status=DONE,
            finished=now(),
            phase="",
            scores=result,
            unreadable=events.unreadable,
            checked=len(result),
        )
    except Cancelled:
        state.update(status=CANCELLED, finished=now(), phase="")
    except Exception as exc:  # noqa: BLE001 - shown to the person
        LOG.exception("The label check of project %s failed", project.id)
        state.update(status=FAILED, error=readable_error(exc), finished=now(), phase="")


def _record(dataset: Any, data: dict[str, Any], logits: bool, images: int) -> None:
    """Record the check as a run of the dataset (#629); a failure there is only logged.

    Args:
        dataset: The project's FiftyOne dataset.
        data: The state of the check (``model``, ``device``).
        logits: Whether the model sent its logits (then ``hardness`` was computed too).
        images: How many images were checked.
    """
    fields = [mistakenness_fiftyone.PRED_FIELD, mistakenness_fiftyone.MISTAKENNESS]
    fields += (
        [mistakenness_fiftyone.HARDNESS]
        if logits
        else [mistakenness_fiftyone.MISSING, mistakenness_fiftyone.SPURIOUS]
    )
    try:
        fiftyone_runs.record(
            dataset,
            "mistakenness",
            model=data.get("model"),
            device=data.get("device"),
            fields=fields,
            images=images,
        )
    except Exception:  # noqa: BLE001 - the results are kept without the run
        LOG.exception("Could not record the label check as a FiftyOne run")


def _phase_then(state: StateStore, events: _Events, event: dict[str, Any]) -> None:
    """Pass a worker event on, marking the ``predicting`` phase at the first one.

    Args:
        state: The label-check file of the project.
        events: The collector of the events.
        event: A JSON event of the worker.
    """
    if not events.predictions and not events.unreadable and state.read().get("phase") != "predicting":
        state.update(phase="predicting", message="")
    events(event)


def _tell(state: StateStore, message: str) -> None:
    """Show what the environment set-up is doing.

    Args:
        state: The label-check file of the project.
        message: The step being done (a code the interface translates, such as ``train_env_install:torch``).
    """
    state.update(message=message)
