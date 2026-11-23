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
from pathlib import Path
from typing import Any

from activelabel import __version__, preannotate
from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, JobManager, Progress
from activelabel.labels import Label
from activelabel.storage import now, write_json
from activelabel.training.catalog import ModelSpec, get_model
from activelabel.training.config import RunConfig
from activelabel.training.environment import TrainingEnvironment, kill_tree, remove_tree
from activelabel.training.models import EXPORT_FILES, FORMATS, ModelError, ModelNotFound, ModelStore
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
PREDICT_SCRIPT = WORKERS_DIR / "predict.py"
EXPORT_PACKAGES = {"onnx": ("onnx",)}  # installed only when a model is exported
#: Kinds of task whose ONNX graph is checked to give the same results as PyTorch. Detection models
#: (torchvision) trace into graphs that break on images without candidate boxes: they come later.
ONNX_KINDS = ("classification", "semantic")
PACKAGES_RECORDED = ("torch", "torchvision", "timm", "pillow", "numpy")
COMES_LATER = "comes later"
#: Status fields of a child that are also shown on the run itself (progress of the current child).
TOP_FIELDS = ("phase", "epoch", "step", "steps", "message")
PROGRESS_EVERY = 1.0  # seconds between run.json writes for step progress


class RunnerError(RuntimeError):
    """A readable reason why a run cannot start or failed."""


def worker_script(task_kind: str) -> Path:
    """Script that trains this kind of task.

    Raises:
        RunnerError: the kind of task has no worker yet.
    """
    name = WORKER_FILES.get(task_kind)
    script = WORKERS_DIR / name if name else None
    if script is None or not script.exists():
        raise RunnerError(f"Training {task_kind} models {COMES_LATER}.")
    return script


def download_env(tmp: Path) -> dict[str, str]:
    """Environment variables that send every download and cache of the worker to ``tmp``."""
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
    """Name of the validation metric in an ``epoch`` event."""
    return f"val_{metric}"


def _objects(own: list[dict[str, Any]], classes: list[str]) -> list[dict[str, Any]]:
    """Objects of an image for the spec: the class index and whichever shapes the label has."""
    out = []
    for label in own:
        if label.get("label") not in classes:
            continue
        shapes = {k: label[k] for k in ("bbox", "polygons", "keypoints") if label.get(k)}
        if shapes:
            out.append({"label": classes.index(label["label"]), **shapes})
    return out


def build_spec(
    project_dir: Path, config: RunConfig, model: ModelSpec, run_dir: Path, version: dict[str, Any]
) -> dict[str, Any]:
    """The ``spec.json`` of the classification worker: absolute image paths and class indexes.

    Raises:
        RunnerError: a fold ``"all"`` (later), a part without images, or missing image files.

    Items are ``{"path", "label"}`` for classification and ``{"path", "objects"}`` for the other
    kinds (``objects``: ``label`` class index plus the label's ``bbox``, ``polygons`` and ``keypoints``);
    an image without objects is kept as a background image.
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
                items.append({"path": str(path.resolve()), "objects": _objects(own, classes)})
                continue
            if not own or own[0].get("label") not in classes:
                continue
            items.append({"path": str(path.resolve()), "label": classes.index(own[0]["label"])})
        out_parts[name] = items
    if missing:
        raise RunnerError(f"{missing} image(s) of this dataset version are no longer in the project.")
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
            extra["num_keypoints"] = max(
                (
                    len(o["keypoints"])
                    for p in out_parts.values()
                    for i in p
                    for o in i["objects"]
                    if "keypoints" in o
                ),
                default=0,
            )
            if not extra["num_keypoints"]:
                raise RunnerError("The labels have no keypoints.")
    params = config.effective_params(model)
    return {
        **extra,
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


def _describe(entry: dict[str, Any]) -> str:
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
        """The project (type and classes), read from its folder when no lookup was given."""
        if self._project_lookup is not None:
            return self._project_lookup(project_id)
        from activelabel.projects import ProjectStore

        return ProjectStore(self._project_dir(project_id).parent).get(project_id)

    # -- public ------------------------------------------------------------

    def store(self, project_id: int) -> RunStore:
        """Runs of a project."""
        return RunStore(self._project_dir(project_id))

    def enqueue(self, project_id: int, number: int) -> None:
        """Queue a stored run; it starts when no other run is running."""
        key = (project_id, number)
        with self._lock:
            self._waiting.append(key)
            job = self._jobs.submit(project_id, "training", lambda p: self._execute(key, p))
            self._job_ids[key] = job.id
        self._write_positions()

    def cancel(self, project_id: int, number: int) -> dict[str, Any]:
        """Cancel a queued or running run (a running one has its process tree killed).

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

        Returns the model card. An export already queued or running is left as it is.

        Raises:
            ModelNotFound: unknown model.
            ModelError: unknown format.
        """
        if fmt not in FORMATS:
            raise ModelError(f"The format must be one of: {', '.join(FORMATS)}.")
        store = ModelStore(self._project_dir(project_id))
        card = store.get(number)
        if ((card.get("exports") or {}).get(fmt) or {}).get("status") in (QUEUED, RUNNING):
            return card
        card = store.set_export(number, fmt, status=QUEUED, error="", message="", requested=now())
        self._jobs.submit(project_id, "export", lambda p: self._export(project_id, number, fmt, p))
        return card

    def enqueue_preannotation(self, project_id: int, number: int) -> None:
        """Queue a stored pre-annotation run (in the same queue as the training runs)."""
        job = self._jobs.submit(project_id, "preannotate", lambda p: self._preannotate(project_id, number, p))
        with self._lock:
            self._pre_jobs[(project_id, number)] = job.id

    def cancel_preannotation(self, project_id: int, number: int) -> None:
        """Cancel a queued or running pre-annotation (nothing is written for a cancelled run).

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
        """Whether a training run, ONNX export or pre-annotation of the project is queued or running."""
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
        """Pre-annotations and exports left queued or running are marked as interrupted."""
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

            def step(message: str) -> None:
                store.set_export(number, fmt, message=message)

            self.environment.ensure(
                [*model.packages, *EXTRA_PACKAGES, *EXPORT_PACKAGES[fmt]], progress.check, step
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
            store.set_export(number, fmt, status=CANCELLED, finished=now(), message="")
        except ModelNotFound:
            self._export_failed(store, number, fmt, "This model was deleted.")
        except FileNotFoundError:
            message = "A file of this model is missing (best.pt): save the training run as a model again."
            self._export_failed(store, number, fmt, message)
        except Exception as exc:  # noqa: BLE001 - shown to the person
            self._export_failed(store, number, fmt, str(exc) or type(exc).__name__)
        finally:
            try:
                (store.path(number) / EXPORT_FILES[fmt]).with_suffix(".part").unlink(missing_ok=True)
            except OSError:
                pass
            if tmp is not None:
                _remove_tree(tmp)

    @staticmethod
    def _export_failed(store: ModelStore, number: int, fmt: str, message: str) -> None:
        try:
            store.set_export(number, fmt, status=FAILED, error=message, finished=now())
        except ModelNotFound:
            pass  # the model folder is gone: nothing to record

    def _preannotate(self, project_id: int, number: int, progress: Progress) -> None:
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
            config = RunConfig.from_yaml((folder / "config.yaml").read_text(encoding="utf-8"))
            model = get_model(card["model"])
            if model is None:
                raise RunnerError(f"Unknown model '{card['model']}'.")
            if model.family == "ultralytics" or model.task_kind not in preannotate.KINDS:
                raise RunnerError(f"Pre-annotation with {model.task_kind} models {COMES_LATER}.")
            project = self._project(project_id)
            if not preannotate.fits(model.task_kind, project.task):
                raise RunnerError(
                    f"This model makes {model.task_kind} labels, but the project is now {project.task.value}."
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

            def step(message: str) -> None:
                runs.update(number, message=message)

            self.environment.ensure([*model.packages, *EXTRA_PACKAGES], progress.check, step)
            progress.check()
            runs.update(number, phase="predicting", message="", total=len(items), done=0)
            self.tmp_root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f"p{number}-", dir=self.tmp_root))
            spec = {
                "checkpoint": str((folder / "best.pt").resolve()),
                "worker": self._scripts(model.task_kind).stem,
                "task": model.task_kind,
                "build": {
                    "model": model.id,
                    "family": model.family,
                    "classes": list(card.get("classes") or []),
                    "params": config.effective_params(model),
                    "start": config.start,
                },
                "classes": list(card.get("classes") or []),
                "threshold": float(run["threshold"]),
                "model": f"m{card['number']}",
                "items": items,
            }
            spec_file = tmp / "spec.json"
            spec_file.write_text(json.dumps(spec), encoding="utf-8")
            predictions: dict[str, list[Label]] = {}
            unreadable: list[str] = []
            last = {"t": 0.0}

            def on_event(event: dict[str, Any]) -> None:
                kind = event.get("event")
                if kind == "prediction":
                    predictions[str(event["id"])] = [Label.from_dict(x) for x in event.get("labels") or []]
                elif kind == "skipped":
                    unreadable.append(str(event.get("id")))
                elif kind == "progress" and time.monotonic() - last["t"] >= PROGRESS_EVERY:
                    last["t"] = time.monotonic()
                    runs.update(number, done=event.get("step", 0), total=event.get("steps", len(items)))
                    progress.update(done=event.get("step"), total=event.get("steps"))

            script = self._scripts(model.task_kind).parent / PREDICT_SCRIPT.name
            cmd = [str(self.environment.python()), str(script), str(spec_file)]
            log_file = project_dir / "preannotations" / f"p{number}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._run_export(
                project_id, number, "predict", cmd, tmp, log_file, progress, on_event, "pre-annotation"
            )
            counts = preannotate.apply_predictions(project_dir, project, predictions, str(run["scope"]))
            counts["unreadable"] = len(unreadable)
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

            def pump() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=pump, daemon=True)
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
        with self._lock:
            waiting = list(self._waiting)
        for position, (project_id, number) in enumerate(waiting, start=1):
            try:
                self.store(project_id).update(number, queue_position=position)
            except (RunNotFound, KeyError):
                continue

    def _execute(self, key: tuple[int, int], progress: Progress) -> None:
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
        project_id, number = key
        run_dir = store.path(number)
        config = store.config(number)
        model = get_model(config.model)
        if model is None:
            raise RunnerError(f"Unknown model '{config.model}'.")
        if model.family == "ultralytics":
            raise RunnerError(f"Training YOLO models {COMES_LATER}.")
        script = self._scripts(model.task_kind)

        def step(message: str) -> None:
            progress.update(message=message)
            store.update(number, message=message)

        env = self.environment
        env.ensure([*model.packages, *EXTRA_PACKAGES], progress.check, step)
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
        store.update(number, epochs=spec["params"]["epochs"])
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
        """Train every child of a grid or all-folds run, then rank them and keep the best weights."""
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
            progress.check()
            cid = entries[i]["id"]
            child_dir = run_dir / cid
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "config.yaml").write_text(child.to_yaml(), encoding="utf-8", newline="\n")
            with log_file.open("a", encoding="utf-8") as log:
                log.write(f"== {cid}: {_describe(entries[i])} ==\n")

            def update(index: int = i, **fields: Any) -> None:
                top = {k: v for k, v in fields.items() if k in TOP_FIELDS}
                own = {k: v for k, v in fields.items() if k != "message"}
                store.update_child(number, index, top=top, **own)

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
                )
                target = _Target(child_dir, log_file, update)
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
        """Rank the combinations; copy the best child's curves, results and weights to the run."""
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

            def pump() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=pump, daemon=True)
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
