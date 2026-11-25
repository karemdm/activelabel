"""Runs training in a separate process, one run at a time for the whole app (docs/TRAINING.md).

A run goes through the phases ``environment`` (packages), ``preparing`` (the temporary folder and
``spec.json``) and ``training`` (the worker process). The worker is a standalone script that talks
by JSON lines on stdout (see ``workers/``); the runner turns them into ``run.json``,
``metrics.jsonl``, ``results.json`` and ``log.txt``. Every download goes to a temporary folder that
is removed when the run ends, whatever the outcome.

A grid search or a run over every fold of a K-fold version trains several *children* one after
another (``r<N>/c1``, ``c2``...), in the same slot of the queue and the same temporary folder
(pretrained weights are downloaded once). The run ranks the grid combinations by the mean of the
selection metric over the folds; only the weights of the best child are kept, in the run's folder.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, Any

from activelabel import __version__, preannotate
from activelabel.datasets import DatasetNotFound, DatasetStore, common_count
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, JobManager, Progress
from activelabel.labels import Label, geometry_problem
from activelabel.projects import ProjectStore
from activelabel.storage import now, write_json
from activelabel.tasks import Task
from activelabel.training import augmentations
from activelabel.training.catalog import ModelSpec, get_model
from activelabel.training.config import RunConfig
from activelabel.training.environment import TrainingEnvironment, kill_tree, remove_tree
from activelabel.training.models import EXPORT_FILES, FORMATS, ModelError, ModelNotFound, ModelStore
from activelabel.training.public_models import get_public
from activelabel.training.runs import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    RunNotFound,
    RunStore,
)

LOG = logging.getLogger(__name__)
WORKERS_DIR = Path(__file__).parent / "workers"
#: Worker script (a file of ``workers/``) of each kind of task.
WORKER_FILES = {
    "classification": "classification.py",
    "detection": "detection.py",
    "instance": "detection.py",
    "keypoints": "detection.py",
    "semantic": "segmentation.py",
}
OBJECT_KINDS = ("detection", "instance", "keypoints", "semantic")
EXTRA_PACKAGES = ("pillow", "numpy")  # the workers need nothing else
EXPORT_SCRIPT = WORKERS_DIR / "export_onnx.py"
PUBLIC_PACKAGES = ("torch", "torchvision")  # what a public torchvision model needs
PREDICT_SCRIPT = WORKERS_DIR / "predict.py"
EXPORT_PACKAGES = {"onnx": ("onnx",)}  # installed only when a model is exported
#: Kinds of task that can be exported to ONNX (the graph is checked with onnx.checker). Detection models
#: (torchvision) trace into graphs that break on images without candidate boxes: they come later.
ONNX_KINDS = ("classification", "semantic")
#: Hyperparameters the runner reads itself and hands to the worker in another form: the ``augmentation``
#: level becomes the list of techniques ``spec["augmentations"]``.
RESOLVED_PARAMS = ("augmentation",)
PACKAGES_RECORDED = ("torch", "torchvision", "timm", "pillow", "numpy")
COMES_LATER = "comes later"
#: Status fields of a child that are also shown on the run itself (progress of the current child).
TOP_FIELDS = ("phase", "epoch", "step", "steps", "message")
PROGRESS_EVERY = 1.0  # seconds between run.json writes for step progress


class RunnerError(RuntimeError):
    """A readable reason why a run cannot start or failed."""


def worker_script(task_kind: str) -> Path:
    """Script that trains this kind of task.

    Args:
        task_kind: kind of task of the model (a key of ``WORKER_FILES``).

    Returns:
        Path of the worker script in ``workers/``.

    Raises:
        RunnerError: the kind of task has no worker yet.
    """
    name = WORKER_FILES.get(task_kind)
    script = WORKERS_DIR / name if name else None
    if script is None or not script.exists():
        raise RunnerError(f"Training {task_kind} models {COMES_LATER}.")
    return script


def download_env(tmp: Path) -> dict[str, str]:
    """Environment variables that send every download and cache of the worker to ``tmp``.

    Args:
        tmp: temporary folder of the run.

    Returns:
        Variable names and values (cache folders inside ``tmp``, unbuffered UTF-8 output).
    """
    cache = {
        "TORCH_HOME": tmp / "torch",
        "HF_HOME": tmp / "hf",
        "XDG_CACHE_HOME": tmp / "cache",
        "PIP_CACHE_DIR": tmp / "pip",
        "TMPDIR": tmp,
        "TEMP": tmp,
        "TMP": tmp,
    }
    out = {k: str(v) for k, v in cache.items()}
    out["PYTHONUNBUFFERED"] = "1"
    out["PYTHONIOENCODING"] = "utf-8"
    return out


def selection_key(metric: str) -> str:
    """Name of the validation metric in an ``epoch`` event.

    Args:
        metric: name of the selection metric (for example ``accuracy``).

    Returns:
        The key of that metric in the event (``val_<metric>``).
    """
    return f"val_{metric}"


KIND_TASK = {
    "detection": Task.DETECTION,
    "instance": Task.SEGMENTATION,
    "semantic": Task.SEGMENTATION,
    "keypoints": Task.KEYPOINTS,
}


def _objects(
    own: list[dict[str, Any]], classes: list[str], kind: str, keypoints: int | None
) -> list[dict[str, Any]]:
    """Objects of an image for the spec: the class index and the one shape this kind of model trains on.

    A box for detection, polygons (3 or more points each) for instance segmentation (a box stands in
    for the polygon only in semantic segmentation), points for pose (all with ``keypoints`` points).
    Labels of another class or without that shape are left out.

    Args:
        own: the label dicts of the image.
        classes: class names of the dataset version; the position is the class index.
        kind: kind of task (``detection``, ``instance``, ``semantic`` or ``keypoints``).
        keypoints: points every object must have (pose), or ``None``.

    Returns:
        One dict per usable label: ``label`` (class index) plus its shape keys.
    """
    out = []
    task = KIND_TASK[kind]
    for label in own:
        if label.get("label") not in classes:
            continue
        item = Label.from_dict(label)
        if kind == "semantic" and not item.polygons and item.bbox:
            shapes = {"bbox": item.bbox}
        elif geometry_problem(task, item, keypoints) is not None:
            continue
        else:
            names = {"detection": ("bbox",), "instance": ("polygons", "bbox"), "semantic": ("polygons",)}
            shapes = {k: label[k] for k in names.get(kind, ("keypoints", "bbox")) if label.get(k)}
        out.append({"label": classes.index(label["label"]), **shapes})
    return out


def build_spec(
    project_dir: Path, config: RunConfig, model: ModelSpec, run_dir: Path, version: dict[str, Any]
) -> dict[str, Any]:
    """The ``spec.json`` of the worker: absolute image paths and class indexes.

    Items are ``{"path", "label"}`` for classification and ``{"path", "objects"}`` for the other
    kinds (``objects``: ``label`` class index plus the label's ``bbox``, ``polygons`` and ``keypoints``);
    an image without objects is kept as a background image. ``augmentations`` is the resolved list of
    techniques the worker applies to the training images (a preset level gives its list).

    Args:
        project_dir: folder of the project.
        config: configuration of the single training (not a grid or an all-folds run).
        model: catalogue entry of the model.
        run_dir: folder where the worker writes its outputs.
        version: the dataset version (``classes``, ``parts``, ``folds``, ``labels``, ``type``).

    Returns:
        The spec dict written to ``spec.json``.

    Raises:
        RunnerError: a grid or all-folds config, images no longer in the project, labels without the
            shape the model trains on, an empty train or validation part, no keypoints, or keypoint flip
            pairs that name a point the labels do not have.
    """
    if config.fold == "all" or config.mode != "single":
        raise RunnerError("A grid or all-folds run is trained one child at a time.")
    folds = version.get("folds")
    parts: dict[str, list[str]] = {k: list(v) for k, v in (version.get("parts") or {}).items()}
    if folds:
        fold = config.fold if isinstance(config.fold, int) else 0
        parts["val"] = list(folds[fold])
        parts["train"] = [i for k, f in enumerate(folds) if k != fold for i in f]
    classes: list[str] = list(version["classes"])
    manifest = Manifest(project_dir)
    records = manifest.load()
    labels = version.get("labels") or {}
    out_parts: dict[str, list[dict[str, Any]]] = {}
    expected_points = (
        common_count(len(lb.get("keypoints") or []) for own in labels.values() for lb in own)
        if model.task_kind == "keypoints"
        else None
    )
    unusable: list[str] = []
    missing = 0
    for name in ("train", "val", "test"):
        items: list[dict[str, Any]] = []
        for image_id in parts.get(name, []):
            record = records.get(image_id)
            path = manifest.file_path(record) if record else None
            if path is None or not path.exists():
                missing += 1
                continue
            own = labels.get(image_id) or []
            if model.task_kind in OBJECT_KINDS:
                objects = _objects(own, classes, model.task_kind, expected_points)
                if own and not objects:
                    unusable.append(image_id)
                items.append({"path": str(path.resolve()), "objects": objects})
                continue
            if not own or own[0].get("label") not in classes:
                continue
            items.append({"path": str(path.resolve()), "label": classes.index(own[0]["label"])})
        out_parts[name] = items
    if missing:
        raise RunnerError(f"{missing} image(s) of this dataset version are no longer in the project.")
    if unusable:
        raise RunnerError(
            f"{len(unusable)} image(s) of this dataset version have labels without the shape a "
            f"{model.task_kind} model trains on (for example {', '.join(unusable[:3])}): make a new "
            "version, which leaves such labels out."
        )
    if not out_parts["train"]:
        raise RunnerError("The training part has no labelled images.")
    if not out_parts["val"]:
        raise RunnerError("The validation part has no labelled images: make a version with validation.")
    extra: dict[str, Any] = {}
    if model.task_kind in OBJECT_KINDS:
        for name in ("train", "val"):
            if not any(i["objects"] for i in out_parts[name]):
                raise RunnerError(f"The {name} part has no labelled objects.")
        if model.task_kind == "keypoints":
            extra["num_keypoints"] = expected_points or 0
            if not extra["num_keypoints"]:
                raise RunnerError("The labels have no keypoints.")
    params = config.effective_params(model)
    techniques = augmentations.resolve(params["augmentation"], config.augmentations, model.task_kind)
    highest, points = augmentations.max_flip_index(techniques), extra.get("num_keypoints", 0)
    if highest >= points:
        raise RunnerError(
            f"The keypoint flip pairs name point {highest}, but the labels have {points} points "
            "(numbered from 0)."
        )
    return {
        **extra,
        "augmentations": techniques,
        "task": model.task_kind,
        "model": model.id,
        "family": model.family,
        "start": config.start,
        "params": params,
        "seed": params["seed"],
        "metric": config.metric or model.default_metric(),
        "classes": classes,
        "type": version["type"],
        "fold": config.fold if isinstance(config.fold, int) else None,
        "parts": out_parts,
        "output_dir": str(run_dir.resolve()),
    }


@dataclass
class _Target:
    """Where one training writes: its folder (metrics, results, weights), the log and its status."""

    dir: Path
    log: Path
    update: Callable[..., Any]


def _pump_lines(stream: IO[str], lines: queue.Queue[str | None]) -> None:
    """Move the lines of a process's output to a queue (``None`` marks the end); runs in a thread.

    Args:
        stream: output stream of the process.
        lines: queue that receives every line, then ``None`` at the end of the stream.
    """
    for line in stream:
        lines.put(line)
    lines.put(None)


def _send_message(update: Callable[..., Any], message: str) -> None:
    """Show a set-up message (environment preparation) through ``update(message=...)``.

    Args:
        update: function that stores the status fields.
        message: text to show.
    """
    update(message=message)


def _report_step(store: RunStore, number: int, progress: Progress, message: str) -> None:
    """Show a set-up message (environment preparation) in the job and in the run.

    Args:
        store: runs of the project.
        number: number of the run.
        progress: progress handle of the job.
        message: text to show.
    """
    progress.update(message=message)
    store.update(number, message=message)


def _update_child(store: RunStore, number: int, index: int, **fields: Any) -> None:
    """Write the fields a child training reports: the top-level ones also go to the run itself.

    Args:
        store: runs of the project.
        number: number of the run.
        index: position of the child in the run's ``children``.
        **fields: status fields of the child; those in ``TOP_FIELDS`` are copied to the run, and
            ``message`` is kept only there.
    """
    top = {k: v for k, v in fields.items() if k in TOP_FIELDS}
    own = {k: v for k, v in fields.items() if k != "message"}
    store.update_child(number, index, top=top, **own)


class _PredictionEvents:
    """Collects the events of a pre-annotation process: labels per image, unreadable images, progress."""

    def __init__(
        self, runs: preannotate.PreannotationStore, number: int, progress: Progress, total: int
    ) -> None:
        """Start collecting events of one pre-annotation run.

        Args:
            runs: store of the pre-annotation runs.
            number: number of the pre-annotation run.
            progress: progress handle of the job.
            total: number of images sent to the process (used until it reports its own total).
        """
        self.predictions: dict[str, list[Label]] = {}
        self.unreadable: list[str] = []
        self._runs = runs
        self._number = number
        self._progress = progress
        self._total = total
        self._last = 0.0

    def __call__(self, event: dict[str, Any]) -> None:
        """Take one event of the process (``prediction``, ``skipped`` or ``progress``).

        Args:
            event: a JSON event of the process; other kinds are ignored.
        """
        kind = event.get("event")
        if kind == "prediction":
            self.predictions[str(event["id"])] = preannotate.mask_labels(event.get("labels") or [])
        elif kind == "skipped":
            self.unreadable.append(str(event.get("id")))
        elif kind == "progress" and time.monotonic() - self._last >= PROGRESS_EVERY:
            self._last = time.monotonic()
            self._runs.update(self._number, done=event.get("step", 0), total=event.get("steps", self._total))
            self._progress.update(done=event.get("step"), total=event.get("steps"))


def _describe(entry: dict[str, Any]) -> str:
    """Short text for a child in the log: its grid values and fold (``training`` when it has neither).

    Args:
        entry: the child's entry of ``children`` (``values``, ``fold``).

    Returns:
        For example ``lr=0.01, fold 2``.
    """
    parts = [f"{k}={v}" for k, v in (entry.get("values") or {}).items()]
    if entry.get("fold") is not None:
        parts.append(f"fold {int(entry['fold']) + 1}")
    return ", ".join(parts) or "training"


def rank_children(
    entries: list[dict[str, Any]], results: dict[str, dict[str, Any] | None], metric: str, n_folds: int
) -> dict[str, Any]:
    """Rank the grid combinations by the mean validation ``metric`` over their folds (higher is better).

    Args:
        entries: the ``children`` of ``run.json`` (``id``, ``combo``, ``fold``, ``values``).
        results: ``results.json`` of each finished child by id (``None`` for a failed one).
        metric: selection metric (a key of ``results["val"]``).
        n_folds: folds per combination (1 without K-fold).

    Returns:
        ``{"ranking", "best_combo", "best_child", "best_value", "best_fold_value", "folds"}``
        (``best_value`` is the mean over the folds, ``best_fold_value`` the score of ``best_child``).
        A ranking row has
        ``rank``, ``combo``, ``values``, ``children``, ``scores`` (per finished child), ``mean``,
        ``std`` and ``complete``; complete combinations come first. ``best_child`` is the best fold
        of the best combination (``None`` when no child finished).
    """
    combos: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        combos.setdefault(int(entry["combo"]), []).append(entry)
    ranking: list[dict[str, Any]] = []
    for combo, members in combos.items():
        scores: dict[str, float] = {}
        for entry in members:
            value = ((results.get(entry["id"]) or {}).get("val") or {}).get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[entry["id"]] = float(value)
        values = list(scores.values())
        ranking.append(
            {
                "combo": combo,
                "values": members[0].get("values") or {},
                "children": [e["id"] for e in members],
                "scores": scores,
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if values else None,
                "complete": len(values) == len(members),
            }
        )
    ranking.sort(
        key=lambda r: (r["mean"] is not None, r["complete"], r["mean"] or 0.0, -r["combo"]), reverse=True
    )
    for position, row in enumerate(ranking, start=1):
        row["rank"] = position
    top = ranking[0] if ranking and ranking[0]["mean"] is not None else None
    best_child = max(top["scores"], key=lambda c: top["scores"][c]) if top else None
    return {
        "ranking": ranking,
        "best_combo": top["combo"] if top else None,
        "best_child": best_child,
        "best_value": top["mean"] if top else None,
        "best_fold_value": top["scores"][best_child] if top and best_child else None,
        "folds": n_folds,
    }


_remove_tree = remove_tree


class TrainingRunner:
    """Queue of runs of every project of the app; one runs at a time."""

    def __init__(
        self,
        workspace: Path,
        project_dir: Callable[[int], Path],
        project_ids: Callable[[], list[int]] | None = None,
        project: Callable[[int], Any] | None = None,
        environment: TrainingEnvironment | None = None,
        scripts: Callable[[str], Path] = worker_script,
        poll: float = 0.2,
    ) -> None:
        """Create the runner; nothing runs until a run is queued.

        Args:
            workspace: folder of the app's data (temporary folders go to ``training/tmp`` inside it).
            project_dir: function that gives the folder of a project by id.
            project_ids: function that lists the project ids (used by ``recover``); none by default.
            project: function that gives a project by id; by default it is read from its folder.
            environment: training environment; by default one inside ``workspace``.
            scripts: function that gives the worker script of a kind of task.
            poll: seconds to wait for a line of output before checking for cancellation.
        """
        self.workspace = Path(workspace)
        self._project_dir = project_dir
        self._project_ids = project_ids or (lambda: [])
        self._project_lookup = project
        self.environment = environment or TrainingEnvironment(self.workspace)
        self._scripts = scripts
        self._poll = poll
        self.tmp_root = self.workspace / "training" / "tmp"
        self._jobs = JobManager(workers=1)
        self._lock = threading.Lock()
        self._waiting: list[tuple[int, int]] = []
        self._job_ids: dict[tuple[int, int], str] = {}
        self._procs: dict[tuple[int, int], subprocess.Popen[str]] = {}
        self._tmp: dict[tuple[int, int], Path] = {}
        self._export_procs: dict[tuple[int, int, str], subprocess.Popen[str]] = {}
        self._pre_jobs: dict[tuple[int, int], str] = {}

    def _project(self, project_id: int) -> Any:
        """The project (type and classes), read from its folder when no lookup was given.

        Args:
            project_id: id of the project.

        Returns:
            The project.
        """
        if self._project_lookup is not None:
            return self._project_lookup(project_id)
        return ProjectStore(self._project_dir(project_id).parent).get(project_id)

    # -- public ------------------------------------------------------------

    def store(self, project_id: int) -> RunStore:
        """Runs of a project.

        Args:
            project_id: id of the project.

        Returns:
            The run store of the project.
        """
        return RunStore(self._project_dir(project_id))

    def enqueue(self, project_id: int, number: int) -> None:
        """Queue a stored run; it starts when no other run is running.

        Args:
            project_id: id of the project.
            number: number of the stored run.
        """
        key = (project_id, number)
        with self._lock:
            self._waiting.append(key)
            job = self._jobs.submit(
                project_id, "training", lambda p: self._execute(key, p), {"number": number}
            )
            self._job_ids[key] = job.id
        self._write_positions()

    def cancel(self, project_id: int, number: int) -> dict[str, Any]:
        """Cancel a queued or running run (a running one has its process tree killed).

        Args:
            project_id: id of the project.
            number: number of the run.

        Returns:
            The run as it is after the call (a running one is waited for up to 10 seconds; one that
            was neither queued nor running is returned unchanged).

        Raises:
            RunNotFound: unknown run.
        """
        key = (project_id, number)
        store = self.store(project_id)
        run = store.get(number)
        if run["status"] not in (QUEUED, RUNNING):
            return run
        with self._lock:
            job_id = self._job_ids.get(key)
            was_waiting = key in self._waiting
            if was_waiting:
                self._waiting.remove(key)
                self._job_ids.pop(key, None)  # the job is cancelled below and will never run
            proc = self._procs.get(key)
        if job_id:
            self._jobs.cancel(job_id)
        if proc is not None:
            kill_tree(proc)
        if run["status"] == QUEUED and (was_waiting or not job_id):
            run = store.update(number, status=CANCELLED, finished=now(), queue_position=None, phase="")
            self._write_positions()
        elif run["status"] == RUNNING:
            deadline = time.monotonic() + 10
            while run["status"] == RUNNING and time.monotonic() < deadline:
                time.sleep(0.05)
                run = store.get(number)
        return run

    def enqueue_export(self, project_id: int, number: int, fmt: str) -> dict[str, Any]:
        """Queue the export of a saved model (in the same queue as the training runs).

        An export already queued or running is left as it is.

        Args:
            project_id: id of the project.
            number: number of the saved model.
            fmt: export format (one of ``FORMATS``).

        Returns:
            The model card.

        Raises:
            ModelNotFound: unknown model.
            ModelError: unknown format, or a public model (it has no weights to export).
        """
        if fmt not in FORMATS:
            raise ModelError(
                "format_invalid", f"The format must be one of: {', '.join(FORMATS)}.", choices=list(FORMATS)
            )
        store = ModelStore(self._project_dir(project_id))
        card = store.get(number)
        if card.get("source") == "public":
            raise ModelError(
                "public_model_no_export",
                "A public model has no weights here to export (they are downloaded when used).",
            )
        if ((card.get("exports") or {}).get(fmt) or {}).get("status") in (QUEUED, RUNNING):
            return card
        card = store.set_export(number, fmt, status=QUEUED, error="", message="", requested=now())
        self._jobs.submit(
            project_id,
            "export",
            lambda p: self._export(project_id, number, fmt, p),
            {"number": number, "format": fmt},
        )
        return card

    def enqueue_preannotation(self, project_id: int, number: int) -> None:
        """Queue a stored pre-annotation run (in the same queue as the training runs).

        Args:
            project_id: id of the project.
            number: number of the pre-annotation run.
        """
        job = self._jobs.submit(
            project_id, "preannotate", lambda p: self._preannotate(project_id, number, p), {"number": number}
        )
        with self._lock:
            self._pre_jobs[(project_id, number)] = job.id

    def cancel_preannotation(self, project_id: int, number: int) -> None:
        """Cancel a queued or running pre-annotation (nothing is written for a cancelled run).

        Args:
            project_id: id of the project.
            number: number of the pre-annotation run.

        Raises:
            preannotate.RunNotFound: unknown run.
        """
        runs = preannotate.PreannotationStore(self._project_dir(project_id))
        run = runs.get(number)
        if run["status"] not in (QUEUED, RUNNING):
            return
        with self._lock:
            job_id = self._pre_jobs.get((project_id, number))
            proc = self._export_procs.get((project_id, number, "predict"))
        if job_id:
            self._jobs.cancel(job_id)
        if proc is not None:
            kill_tree(proc)
        if run["status"] == QUEUED:
            runs.update(number, status=CANCELLED, finished=now())
            with self._lock:
                self._pre_jobs.pop((project_id, number), None)
        else:
            deadline = time.monotonic() + 10
            while runs.get(number)["status"] == RUNNING and time.monotonic() < deadline:
                time.sleep(0.05)

    def has_active(self, project_id: int) -> bool:
        """Whether a training run, ONNX export or pre-annotation of the project is queued or running.

        Args:
            project_id: id of the project.

        Returns:
            ``True`` when the project has an active job.
        """
        return bool(self._jobs.active(project_id))

    def recover(self) -> None:
        """At startup: runs left ``running`` were interrupted (failed); queued ones go back in line.

        Exports and pre-annotations left queued or running were interrupted too (they can be asked again),
        and temporary folders left by a process that was killed are removed. A failing step is logged
        and skipped: it never stops the app from starting.
        """
        try:
            if self.tmp_root.exists():
                for leftover in list(self.tmp_root.iterdir()):
                    _remove_tree(leftover)
        except OSError:
            LOG.exception("Could not clean the temporary training folder")
        try:
            project_ids = list(self._project_ids())
        except Exception:  # noqa: BLE001 - startup must go on
            LOG.exception("Could not list the projects to recover")
            return
        for project_id in project_ids:
            self._recover_tools(project_id)
        queued: list[tuple[int, int]] = []
        for project_id in project_ids:
            try:
                store = self.store(project_id)
                for number in store.numbers():  # oldest first
                    try:
                        queued.extend(self._recover_run(store, project_id, store.get(number)))
                    except Exception:  # noqa: BLE001 - one broken run must not stop the others
                        LOG.exception("Could not recover run %s of project %s", number, project_id)
            except Exception:  # noqa: BLE001
                LOG.exception("Could not recover the runs of project %s", project_id)
        for project_id, number in queued:
            try:
                self.enqueue(project_id, number)
            except Exception:  # noqa: BLE001
                LOG.exception("Could not queue run %s of project %s again", number, project_id)

    def _recover_run(self, store: RunStore, project_id: int, run: dict[str, Any]) -> list[tuple[int, int]]:
        """Fail a run left running (and its unfinished children); report a queued one to be queued again.

        Args:
            store: runs of the project.
            project_id: id of the project.
            run: the run as stored.

        Returns:
            ``[(project_id, number)]`` when the run was queued, otherwise an empty list.
        """
        number = run["number"]
        if run["status"] == RUNNING:
            fields: dict[str, Any] = {}
            if run.get("children"):  # a composite run: its unfinished children were interrupted too
                fields["children"] = [
                    {**c, "status": FAILED, "error": "interrupted", "finished": now()}
                    if c.get("status") in (QUEUED, RUNNING)
                    else c
                    for c in run["children"]
                ]
            store.update(
                number,
                status=FAILED,
                error="interrupted",
                finished=now(),
                phase="",
                current=None,
                queue_position=None,
                **fields,
            )
        elif run["status"] == QUEUED:
            return [(project_id, number)]
        return []

    def _recover_tools(self, project_id: int) -> None:
        """Pre-annotations and exports left queued or running are marked as interrupted.

        Args:
            project_id: id of the project.
        """
        try:
            pre = preannotate.PreannotationStore(self._project_dir(project_id))
            for run in pre.list():
                if run["status"] in (QUEUED, RUNNING):
                    pre.update(run["number"], status=FAILED, error="interrupted", finished=now())
        except Exception:  # noqa: BLE001
            LOG.exception("Could not recover the pre-annotations of project %s", project_id)
        try:
            models = ModelStore(self._project_dir(project_id))
            for card in models.list():
                for fmt, export in (card.get("exports") or {}).items():
                    if (export or {}).get("status") in (QUEUED, RUNNING):
                        models.set_export(
                            card["number"], fmt, status=FAILED, error="interrupted", finished=now()
                        )
        except Exception:  # noqa: BLE001
            LOG.exception("Could not recover the exports of project %s", project_id)

    def shutdown(self) -> None:
        """Stop the running process (if any) and the queue."""
        self._jobs.shutdown()
        with self._lock:
            procs = [*self._procs.values(), *self._export_procs.values()]
        for proc in procs:
            kill_tree(proc)

    # -- export of a saved model ---------------------------------------------

    def _export(self, project_id: int, number: int, fmt: str, progress: Progress) -> None:
        """Job of an export: set up the environment, run the export process and record the outcome.

        The result (done, cancelled or failed with a readable error) is written on the model card; the
        temporary folder and the half-written file are always removed.

        Args:
            project_id: id of the project.
            number: number of the saved model.
            fmt: export format.
            progress: progress handle of the job (checked for cancellation).
        """
        store = ModelStore(self._project_dir(project_id))
        try:
            store.set_export(number, fmt, status=RUNNING, started=now(), message="")
        except KeyError:
            return  # the model was deleted while waiting
        tmp: Path | None = None
        try:
            card = store.get(number)
            folder = store.path(number)
            part = (folder / EXPORT_FILES[fmt]).with_suffix(".part")
            config = RunConfig.from_yaml((folder / "config.yaml").read_text(encoding="utf-8"))
            model = get_model(card["model"])
            if model is None:
                raise RunnerError(f"Unknown model '{card['model']}'.")
            if model.family == "ultralytics" or model.task_kind not in ONNX_KINDS:
                raise RunnerError(
                    f"ONNX export of {model.task_kind} models {COMES_LATER}: download the PyTorch weights."
                )
            worker = self._scripts(model.task_kind).stem
            (folder / EXPORT_FILES[fmt]).unlink(missing_ok=True)  # an old export is not kept as the new one
            part.unlink(missing_ok=True)

            self.environment.ensure(
                [*model.packages, *EXTRA_PACKAGES, *EXPORT_PACKAGES[fmt]],
                progress.check,
                partial(_send_message, partial(store.set_export, number, fmt)),
            )
            progress.check()
            store.set_export(number, fmt, message="")
            self.tmp_root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f"m{number}-{fmt}-", dir=self.tmp_root))
            spec = {
                "checkpoint": str((folder / "best.pt").resolve()),
                "output": str((folder / EXPORT_FILES[fmt]).resolve()),
                "worker": worker,
                "task": model.task_kind,
                "build": {
                    "model": model.id,
                    "family": model.family,
                    "classes": list(card.get("classes") or []),
                    "params": config.effective_params(model),
                    "start": config.start,
                },
            }
            spec_file = tmp / "spec.json"
            spec_file.write_text(json.dumps(spec), encoding="utf-8")
            script = self._scripts(model.task_kind).parent / EXPORT_SCRIPT.name
            cmd = [str(self.environment.python()), str(script), str(spec_file)]
            info = self._run_export(project_id, number, fmt, cmd, tmp, folder / "export.log", progress)
            store.set_export(number, fmt, status=DONE, finished=now(), message="", **info)
        except Cancelled:
            self._drop_partial_export(store, number, fmt)
            store.set_export(number, fmt, status=CANCELLED, finished=now(), message="")
        except ModelNotFound:
            self._export_failed(store, number, fmt, "This model was deleted.")
        except FileNotFoundError:
            message = "A file of this model is missing (best.pt): save the training run as a model again."
            self._export_failed(store, number, fmt, message)
        except Exception as exc:  # noqa: BLE001 - shown to the person
            self._export_failed(store, number, fmt, str(exc) or type(exc).__name__)
        finally:
            self._drop_partial_export(store, number, fmt)
            if tmp is not None:
                _remove_tree(tmp)

    @staticmethod
    def _drop_partial_export(store: ModelStore, number: int, fmt: str) -> None:
        """Remove the half-written export file (before a status says the export is over).

        Args:
            store: models of the project.
            number: number of the saved model.
            fmt: export format.
        """
        try:
            (store.path(number) / EXPORT_FILES[fmt]).with_suffix(".part").unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _export_failed(store: ModelStore, number: int, fmt: str, message: str) -> None:
        """Remove the partial file and mark the export as failed (nothing is recorded if the model is gone).

        Args:
            store: models of the project.
            number: number of the saved model.
            fmt: export format.
            message: error to show.
        """
        TrainingRunner._drop_partial_export(store, number, fmt)
        try:
            store.set_export(number, fmt, status=FAILED, error=message, finished=now())
        except ModelNotFound:
            pass  # the model folder is gone: nothing to record

    def _preannotate(self, project_id: int, number: int, progress: Progress) -> None:
        """Job of a pre-annotation: check the model and project, predict, then write the labels.

        Nothing is written when the project's type or classes changed meanwhile. The outcome (done with
        counts, cancelled or failed with a readable error) is written on the pre-annotation run.

        Args:
            project_id: id of the project.
            number: number of the pre-annotation run.
            progress: progress handle of the job (checked for cancellation).
        """
        project_dir = self._project_dir(project_id)
        runs = preannotate.PreannotationStore(project_dir)
        try:
            run = runs.get(number)
        except preannotate.RunNotFound:
            run = None
        if run is None or run["status"] != QUEUED:  # deleted, or cancelled while waiting
            with self._lock:
                self._pre_jobs.pop((project_id, number), None)
            return
        runs.update(number, status=RUNNING, started=now(), phase="environment")
        tmp: Path | None = None
        try:
            models = ModelStore(project_dir)
            card = models.get(int(run["model"]))
            folder = models.path(int(run["model"]))
            public = get_public(str(card.get("public") or "")) if card.get("source") == "public" else None
            if card.get("source") == "public" and public is None:
                raise RunnerError("This public model is no longer in the catalogue.")
            model = None if public else get_model(card["model"])
            if public is None and model is None:
                raise RunnerError(f"Unknown model '{card['model']}'.")
            kind = public.task_kind if public else model.task_kind  # type: ignore[union-attr]
            if model is not None and (model.family == "ultralytics" or kind not in preannotate.KINDS):
                raise RunnerError(f"Pre-annotation with {kind} models {COMES_LATER}.")
            project = self._project(project_id)
            if not preannotate.fits(kind, project.task):
                raise RunnerError(
                    f"This model makes {kind} labels, but the project is now {project.task.value}."
                )
            model_classes = list(card.get("classes") or [])
            removed = [c for c in model_classes if c not in project.classes]
            if model_classes and len(removed) == len(model_classes):
                raise RunnerError(
                    "None of the classes of this model is in the project (any more): "
                    f"{', '.join(removed[:5])}."
                )
            expected_points, have_points = (
                card.get("num_keypoints"),
                preannotate.project_keypoints(project_dir),
            )
            if kind == "keypoints" and expected_points and have_points and expected_points != have_points:
                raise RunnerError(
                    f"This model finds {expected_points} keypoints per object, but the labels of the "
                    f"project have {have_points}: use a model trained on this skeleton."
                )
            image_ids = preannotate.targets(project_dir, int(run["task"]), str(run["scope"]))
            if not image_ids:
                raise RunnerError("No image of this task needs pre-annotation now.")
            manifest = Manifest(project_dir)
            records = manifest.load()
            items = []
            for image_id in image_ids:
                path = manifest.file_path(records[image_id])
                if path is not None and path.exists():
                    items.append({"id": image_id, "path": str(path.resolve())})
            if not items:
                raise RunnerError("None of the images of this task could be read.")

            packages = PUBLIC_PACKAGES if public else model.packages  # type: ignore[union-attr]
            update = partial(_send_message, partial(runs.update, number))
            self.environment.ensure([*packages, *EXTRA_PACKAGES], progress.check, update)
            progress.check()
            runs.update(number, phase="predicting", message="", total=len(items), done=0)
            self.tmp_root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f"p{number}-", dir=self.tmp_root))
            spec: dict[str, Any] = {
                "task": kind,
                "classes": list(card.get("classes") or []),
                "threshold": float(run["threshold"]),
                "model": f"m{card['number']}",
                "items": items,
            }
            if public:
                spec["public"] = {
                    "builder": public.builder,
                    "categories": list(public.categories),
                    "mapping": dict(card.get("mapping") or {}),
                    "task": public.task_kind,
                }
            else:
                assert model is not None
                config = RunConfig.from_yaml((folder / "config.yaml").read_text(encoding="utf-8"))
                spec["checkpoint"] = str((folder / "best.pt").resolve())
                spec["worker"] = self._scripts(model.task_kind).stem
                spec["build"] = {
                    "model": model.id,
                    "family": model.family,
                    "classes": list(card.get("classes") or []),
                    "params": config.effective_params(model),
                    "start": config.start,
                }
            spec_file = tmp / "spec.json"
            spec_file.write_text(json.dumps(spec), encoding="utf-8")
            events = _PredictionEvents(runs, number, progress, len(items))
            script = self._scripts(kind).parent / PREDICT_SCRIPT.name
            cmd = [str(self.environment.python()), str(script), str(spec_file)]
            log_file = project_dir / "preannotations" / f"p{number}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._run_export(
                project_id, number, "predict", cmd, tmp, log_file, progress, events, "pre-annotation"
            )
            now_project = self._project(project_id)  # it may have been edited while the model ran
            if now_project.task is not project.task or list(now_project.classes) != list(project.classes):
                raise RunnerError(
                    "The project's type or classes changed while the pre-annotation was running: "
                    "nothing was written. Run it again."
                )
            counts: dict[str, Any] = dict(
                preannotate.apply_predictions(project_dir, now_project, events.predictions, str(run["scope"]))
            )
            counts["unreadable"] = len(events.unreadable)
            counts["removed_classes"] = removed  # classes of the model the project does not have
            runs.update(number, status=DONE, finished=now(), phase="", done=len(items), counts=counts)
        except Cancelled:
            runs.update(number, status=CANCELLED, finished=now(), phase="")
        except ModelNotFound:
            runs.update(number, status=FAILED, error="This model was deleted.", finished=now(), phase="")
        except FileNotFoundError:
            message = "A file of this model is missing (best.pt): save the training run as a model again."
            runs.update(number, status=FAILED, error=message, finished=now(), phase="")
        except Exception as exc:  # noqa: BLE001 - shown to the person
            runs.update(number, status=FAILED, error=str(exc) or type(exc).__name__, finished=now(), phase="")
        finally:
            with self._lock:
                self._pre_jobs.pop((project_id, number), None)
            if tmp is not None:
                _remove_tree(tmp)

    def _run_export(
        self,
        project_id: int,
        number: int,
        fmt: str,
        cmd: list[str],
        tmp: Path,
        log_file: Path,
        progress: Progress,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        what: str = "export",
    ) -> dict[str, Any]:
        """Run a tool process (export, pre-annotation); returns the ``done`` event's results.

        ``on_event`` receives every JSON event; ``prediction`` lines are not written to the log.

        Args:
            project_id: id of the project.
            number: number of the model (export) or of the pre-annotation run.
            fmt: export format, or ``"predict"`` for a pre-annotation (part of the process key).
            cmd: command line of the process.
            tmp: temporary folder (working directory and download cache).
            log_file: file that receives the process output.
            progress: progress handle of the job (checked for cancellation).
            on_event: function that receives every JSON event of the process.
            what: name of the process in the error message.

        Returns:
            The ``results`` of the ``done`` event.

        Raises:
            RunnerError: the process failed, reported an error or ended without results.
            Cancelled: the job was cancelled (the process tree is killed).
        """
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        key = (project_id, number, fmt)
        state: dict[str, Any] = {"results": None, "error": "", "tail": []}
        with log_file.open("a", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tmp,
                env={**os.environ, **download_env(tmp)},
                creationflags=creation,
            )
            with self._lock:
                self._export_procs[key] = proc
            lines: queue.Queue[str | None] = queue.Queue()

            assert proc.stdout is not None
            reader = threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True)
            reader.start()
            try:
                while True:
                    progress.check()
                    try:
                        line = lines.get(timeout=self._poll)
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    text = line.rstrip("\r\n")
                    try:
                        event = json.loads(text)
                    except ValueError:
                        event = None
                    if not (isinstance(event, dict) and event.get("event") == "prediction"):
                        log.write(text + "\n")
                        log.flush()
                    if not isinstance(event, dict):
                        if text.strip():
                            state["tail"] = [*state["tail"], text.strip()][-3:]  # for the error message
                        continue
                    if on_event is not None:
                        on_event(event)
                    if event.get("event") == "done":
                        state["results"] = event.get("results") or {}
                    elif event.get("event") == "error":
                        state["error"] = str(event.get("message", "error"))
                code = proc.wait()
                progress.check()
            except BaseException:
                kill_tree(proc)
                raise
            finally:
                reader.join(timeout=2)
                with self._lock:
                    self._export_procs.pop(key, None)
        if code != 0 or state["error"] or state["results"] is None:
            detail = f" Last output: {' | '.join(state['tail'])}" if state["tail"] else ""
            raise RunnerError(state["error"] or f"The {what} process stopped with code {code}.{detail}")
        return dict(state["results"])

    # -- one run -----------------------------------------------------------

    def _write_positions(self) -> None:
        """Write the queue position (from 1) on every run still waiting."""
        with self._lock:
            waiting = list(self._waiting)
        for position, (project_id, number) in enumerate(waiting, start=1):
            try:
                self.store(project_id).update(number, queue_position=position)
            except (RunNotFound, KeyError):
                continue

    def _execute(self, key: tuple[int, int], progress: Progress) -> None:
        """Job of a training run: mark it running, call ``_run`` and record done, cancelled or failed.

        A run deleted or cancelled while waiting is skipped. The process and temporary folder are
        always released.

        Args:
            key: ``(project_id, run number)``.
            progress: progress handle of the job.
        """
        project_id, number = key
        store = self.store(project_id)
        with self._lock:
            if key in self._waiting:
                self._waiting.remove(key)
        self._write_positions()
        try:
            waiting = store.get(number)["status"] == QUEUED
        except RunNotFound:
            waiting = False
        if not waiting:  # deleted, or cancelled while waiting
            with self._lock:
                self._job_ids.pop(key, None)
            return
        store.update(number, status=RUNNING, started=now(), queue_position=None, phase="environment")
        try:
            self._run(key, store, progress)
        except Cancelled:
            store.update(number, status=CANCELLED, finished=now(), phase="")
        except Exception as exc:  # noqa: BLE001 - shown to the person
            store.update(
                number, status=FAILED, error=str(exc) or type(exc).__name__, finished=now(), phase=""
            )
        else:
            store.update(number, status=DONE, finished=now(), phase="")
        finally:
            with self._lock:
                self._procs.pop(key, None)
                self._job_ids.pop(key, None)
                tmp = self._tmp.pop(key, None)
            if tmp is not None:
                _remove_tree(tmp)

    def _run(self, key: tuple[int, int], store: RunStore, progress: Progress) -> None:
        """Prepare the environment and the temporary folder, then train (one training or the children).

        Args:
            key: ``(project_id, run number)``.
            store: runs of the project.
            progress: progress handle of the job.

        Raises:
            RunnerError: unknown or not yet supported model, or the dataset version was deleted.
        """
        project_id, number = key
        run_dir = store.path(number)
        config = store.config(number)
        model = get_model(config.model)
        if model is None:
            raise RunnerError(f"Unknown model '{config.model}'.")
        if model.family == "ultralytics":
            raise RunnerError(f"Training YOLO models {COMES_LATER}.")
        script = self._scripts(model.task_kind)

        env = self.environment
        env.ensure(
            [*model.packages, *EXTRA_PACKAGES], progress.check, partial(_report_step, store, number, progress)
        )
        progress.check()

        store.update(number, phase="preparing", message="")
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"r{number}-", dir=self.tmp_root))
        with self._lock:
            self._tmp[key] = tmp
        project_dir = self._project_dir(project_id)
        try:
            version = DatasetStore(project_dir).read(config.dataset)
        except (DatasetNotFound, KeyError):
            raise RunnerError("The data split version of this run was deleted.") from None
        installed = {k.lower(): v for k, v in env.installed().items()}
        store.update(
            number,
            device=env.device_info(),
            packages={k: installed[k] for k in PACKAGES_RECORDED if k in installed},
            activelabel_version=__version__,
            epoch=0,
            classes=list(version.get("classes") or []),
        )
        python = str(env.python())
        if config.is_composite():
            self._run_children(key, store, progress, config, model, version, [python, str(script)], tmp)
            return
        spec = build_spec(project_dir, config, model, run_dir, version)
        spec_file = tmp / "spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        store.update(number, epochs=spec["params"]["epochs"], augmentations=spec["augmentations"])
        progress.check()
        store.update(number, phase="training")
        target = _Target(run_dir, run_dir / "log.txt", lambda **f: store.update(number, **f))
        self._train(key, target, progress, [python, str(script), str(spec_file)], tmp, spec)

    def _run_children(
        self,
        key: tuple[int, int],
        store: RunStore,
        progress: Progress,
        config: RunConfig,
        model: ModelSpec,
        version: dict[str, Any],
        cmd: list[str],
        tmp: Path,
    ) -> None:
        """Train every child of a grid or all-folds run, then rank them and keep the best weights.

        A failed child is recorded and the others still train; a cancellation marks the remaining
        children cancelled and propagates.

        Args:
            key: ``(project_id, run number)``.
            store: runs of the project.
            progress: progress handle of the job.
            config: configuration of the composite run.
            model: catalogue entry of the model.
            version: the dataset version.
            cmd: worker command without the spec file (the spec of each child is appended).
            tmp: temporary folder of the run.

        Raises:
            Cancelled: the job was cancelled.
            RunnerError: every child failed.
        """
        project_id, number = key
        run_dir = store.path(number)
        n_folds = len(version.get("folds") or []) if config.fold == "all" else 1
        children = config.children(len(version.get("folds") or []))
        names = sorted(config.grid)
        entries: list[dict[str, Any]] = []
        for i, child in enumerate(children):
            values = {n: (child.start if n == "start" else child.params.get(n)) for n in names}
            entries.append(
                {
                    "id": f"c{i + 1}",
                    "combo": i // n_folds + 1,
                    "fold": child.fold,
                    "values": values,
                    "status": QUEUED,
                }
            )
        store.update(number, children=entries, trainings=len(entries), phase="training")
        log_file = run_dir / "log.txt"
        for i, child in enumerate(children):
            try:
                progress.check()
            except Cancelled:  # cancelled between two children: the rest never start
                for j in range(i, len(entries)):
                    store.update_child(number, j, status=CANCELLED, finished=now())
                raise
            cid = entries[i]["id"]
            child_dir = run_dir / cid
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "config.yaml").write_text(child.to_yaml(), encoding="utf-8", newline="\n")
            with log_file.open("a", encoding="utf-8") as log:
                log.write(f"== {cid}: {_describe(entries[i])} ==\n")

            try:
                spec = build_spec(self._project_dir(project_id), child, model, child_dir, version)
                spec_dir = tmp / cid
                spec_dir.mkdir(parents=True, exist_ok=True)
                spec_file = spec_dir / "spec.json"
                spec_file.write_text(json.dumps(spec), encoding="utf-8")
                epochs = spec["params"]["epochs"]
                store.update_child(
                    number,
                    i,
                    top={
                        "current": cid,
                        "phase": "training",
                        "epoch": 0,
                        "epochs": epochs,
                        "step": None,
                        "steps": None,
                    },
                    status=RUNNING,
                    started=now(),
                    epochs=epochs,
                    augmentations=spec["augmentations"],
                )
                target = _Target(child_dir, log_file, partial(_update_child, store, number, i))
                self._train(key, target, progress, [*cmd, str(spec_file)], spec_dir, spec)
            except Cancelled:
                for j in range(i, len(entries)):
                    store.update_child(number, j, status=CANCELLED, finished=now())
                raise
            except Exception as exc:  # noqa: BLE001 - the other children still train
                message = str(exc) or type(exc).__name__
                store.update_child(number, i, status=FAILED, error=message, finished=now())
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(f"{cid} failed: {message}\n")
            else:
                store.update_child(number, i, status=DONE, finished=now())
        self._finish_children(store, number, config.metric or model.default_metric(), n_folds)

    def _finish_children(self, store: RunStore, number: int, metric: str, n_folds: int) -> None:
        """Rank the combinations; copy the best child's curves, results and weights to the run.

        Args:
            store: runs of the project.
            number: number of the run.
            metric: selection metric.
            n_folds: folds per combination.

        Raises:
            RunnerError: no child finished.
        """
        run_dir = store.path(number)
        entries = store.get(number).get("children") or []
        results = {
            e["id"]: (store.results(number, e["id"]) if e["status"] == DONE else None) for e in entries
        }
        summary = rank_children(entries, results, metric, n_folds)
        best_id = summary["best_child"]
        if best_id is None:
            errors = [e.get("error") for e in entries if e.get("error")]
            raise RunnerError("Every training of this run failed" + (f": {errors[0]}" if errors else "."))
        best_dir = run_dir / best_id
        for name in ("metrics.jsonl", "best.pt", "last.pt"):
            if (best_dir / name).exists():
                shutil.copy2(best_dir / name, run_dir / name)
        for entry in entries:  # only the best child keeps weights (in the run's folder)
            for weights in (run_dir / entry["id"]).glob("*.pt"):
                weights.unlink(missing_ok=True)
        best_results = results[best_id] or {}
        write_json(run_dir / "results.json", {**best_results, **summary, "metric": metric})
        store.update(
            number,
            current=None,
            best={
                "epoch": best_results.get("best_epoch"),
                "value": summary["best_value"],
                "best_fold_value": summary["best_fold_value"],
            },
            best_child=best_id,
        )

    def _train(
        self,
        key: tuple[int, int],
        target: _Target,
        progress: Progress,
        cmd: list[str],
        tmp: Path,
        spec: dict[str, Any],
    ) -> None:
        """Run one worker process, handle its events and write ``results.json``.

        Args:
            key: ``(project_id, run number)`` (the process is registered under it).
            target: where this training writes its files and status.
            progress: progress handle of the job.
            cmd: command line of the worker, including the spec file.
            tmp: temporary folder (working directory and download cache).
            spec: the spec given to the worker (``metric`` selects the validation key).

        Raises:
            RunnerError: the process failed, reported an error or ended without results.
        """
        run_dir = target.dir
        metric_key = selection_key(spec["metric"])
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = {**os.environ, **download_env(tmp)}
        lines: queue.Queue[str | None] = queue.Queue()
        state: dict[str, Any] = {"results": None, "error": "", "last": 0.0, "warnings": []}
        with target.log.open("a", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tmp,
                env=env,
                creationflags=creation,
            )
            with self._lock:
                self._procs[key] = proc

            assert proc.stdout is not None
            reader = threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True)
            reader.start()
            try:
                eof = False
                while not eof:
                    progress.check()
                    try:
                        line = lines.get(timeout=self._poll)
                    except queue.Empty:
                        continue
                    if line is None:
                        eof = True
                        continue
                    self._handle(line.rstrip("\r\n"), target, progress, log, metric_key, state)
                code = proc.wait()
                progress.check()  # a cancelled run also ends its pipe: do not report it as a failure
            except BaseException:
                kill_tree(proc)
                raise
            finally:
                reader.join(timeout=2)
        if code != 0 or state["error"]:
            raise RunnerError(state["error"] or f"The training process stopped with code {code}.")
        if state["results"] is not None:
            write_json(run_dir / "results.json", state["results"])
        if not (run_dir / "results.json").exists():
            raise RunnerError("The training process ended without results.")

    def _handle(
        self,
        text: str,
        target: _Target,
        progress: Progress,
        log: Any,
        metric_key: str,
        state: dict[str, Any],
    ) -> None:
        """Handle one output line of the worker: a JSON event, or plain text that goes to the log.

        Args:
            text: the line without its line break.
            target: where this training writes its files and status.
            progress: progress handle of the job.
            log: open log file.
            metric_key: key of the selection metric in ``epoch`` events.
            state: mutable state of the process (``results``, ``error``, ``last`` write time, ``warnings``).
        """
        if not text.strip():
            return
        try:
            event = json.loads(text)
            if not isinstance(event, dict):
                raise ValueError
        except ValueError:
            log.write(text + "\n")
            log.flush()
            return
        kind = event.get("event")
        if kind == "log":
            log.write(str(event.get("message", "")) + "\n")
            log.flush()
        elif kind == "warning":
            message = str(event.get("message", "warning"))
            log.write("warning: " + message + "\n")
            log.flush()
            if message not in state["warnings"]:
                state["warnings"].append(message)
                target.update(warnings=list(state["warnings"]))
        elif kind == "progress":
            done, total = event.get("step"), event.get("steps")
            progress.update(done=done, total=total, message=event.get("phase"))
            if time.monotonic() - state["last"] >= PROGRESS_EVERY:
                state["last"] = time.monotonic()
                target.update(
                    phase=event.get("phase") or "training",
                    epoch=event.get("epoch", 0),
                    step=done,
                    steps=total,
                )
        elif kind == "epoch":
            row = {k: v for k, v in event.items() if k != "event"}
            with (target.dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
            fields: dict[str, Any] = {"epoch": row.get("epoch", 0), "step": None, "steps": None}
            if row.get("best"):
                fields["best"] = {"epoch": row.get("epoch"), "value": row.get(metric_key)}
            target.update(**fields)
            progress.update(done=row.get("epoch", 0))
            log.write(text + "\n")
            log.flush()
        elif kind == "done":
            state["results"] = event.get("results")
        elif kind == "error":
            state["error"] = str(event.get("message", "error"))
            log.write(state["error"] + "\n")
            log.flush()
