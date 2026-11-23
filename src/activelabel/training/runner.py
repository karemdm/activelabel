"""Runs training in a separate process, one run at a time for the whole app (docs/TRAINING.md).

A run goes through the phases ``environment`` (packages), ``preparing`` (the temporary folder and
``spec.json``) and ``training`` (the worker process). The worker is a standalone script that talks
by JSON lines on stdout (see ``workers/``); the runner turns them into ``run.json``,
``metrics.jsonl``, ``results.json`` and ``log.txt``. Every download goes to a temporary folder that
is removed when the run ends, whatever the outcome.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from activelabel import __version__
from activelabel.datasets import DatasetStore
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, JobManager, Progress
from activelabel.storage import now, write_json
from activelabel.training.catalog import ModelSpec, get_model
from activelabel.training.config import RunConfig
from activelabel.training.environment import TrainingEnvironment, kill_tree
from activelabel.training.runs import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    RunNotFound,
    RunStore,
)

WORKERS_DIR = Path(__file__).parent / "workers"
WORKER_SCRIPTS = {"classification": WORKERS_DIR / "classification.py"}
EXTRA_PACKAGES = ("pillow", "numpy")  # the workers need nothing else
PACKAGES_RECORDED = ("torch", "torchvision", "timm", "pillow", "numpy")
COMES_LATER = "comes later"
PROGRESS_EVERY = 1.0  # seconds between run.json writes for step progress


class RunnerError(RuntimeError):
    """A readable reason why a run cannot start or failed."""


def worker_script(task_kind: str) -> Path:
    """Script that trains this kind of task.

    Raises:
        RunnerError: the kind of task has no worker yet.
    """
    script = WORKER_SCRIPTS.get(task_kind)
    if script is None:
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


def build_spec(
    project_dir: Path, config: RunConfig, model: ModelSpec, run_dir: Path, version: dict[str, Any]
) -> dict[str, Any]:
    """The ``spec.json`` of the classification worker: absolute image paths and class indexes.

    Raises:
        RunnerError: a fold ``"all"`` (later), a part without images, or missing image files.
    """
    if config.fold == "all":
        raise RunnerError(f"Training all folds {COMES_LATER}: choose one fold.")
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
        items = []
        for image_id in parts.get(name, []):
            record = records.get(image_id)
            path = manifest.file_path(record) if record else None
            if path is None or not path.exists():
                missing += 1
                continue
            own = labels.get(image_id) or []
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
    params = config.effective_params(model)
    return {
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


def _remove_tree(path: Path) -> None:
    for attempt in range(5):
        shutil.rmtree(path, ignore_errors=attempt < 4)
        if not path.exists():
            return
        time.sleep(0.2)


class TrainingRunner:
    """Queue of runs of every project of the app; one runs at a time."""

    def __init__(
        self,
        workspace: Path,
        project_dir: Callable[[int], Path],
        project_ids: Callable[[], list[int]] | None = None,
        environment: TrainingEnvironment | None = None,
        scripts: Callable[[str], Path] = worker_script,
        poll: float = 0.2,
    ) -> None:
        self.workspace = Path(workspace)
        self._project_dir = project_dir
        self._project_ids = project_ids or (lambda: [])
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

    def recover(self) -> None:
        """At startup: runs left ``running`` were interrupted (failed); queued ones go back in line."""
        queued: list[tuple[int, int]] = []
        for project_id in self._project_ids():
            store = self.store(project_id)
            for run in reversed(store.list()):  # oldest first
                if run["status"] == RUNNING:
                    store.update(run["number"], status=FAILED, error="interrupted", finished=now(), phase="")
                elif run["status"] == QUEUED:
                    queued.append((project_id, run["number"]))
        for project_id, number in queued:
            self.enqueue(project_id, number)

    def shutdown(self) -> None:
        """Stop the running process (if any) and the queue."""
        self._jobs.shutdown()
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            kill_tree(proc)

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
            if store.get(number)["status"] != QUEUED:
                return  # cancelled while waiting
        except RunNotFound:
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
        if config.mode != "single":
            raise RunnerError(f"Grid search {COMES_LATER}.")
        script = self._scripts(model.task_kind)
        if config.fold == "all":
            raise RunnerError(f"Training all folds {COMES_LATER}: choose one fold.")

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
        version = DatasetStore(self._project_dir(project_id)).read(config.dataset)
        spec = build_spec(self._project_dir(project_id), config, model, run_dir, version)
        spec_file = tmp / "spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        installed = {k.lower(): v for k, v in env.installed().items()}
        store.update(
            number,
            device=env.device_info(),
            packages={k: installed[k] for k in PACKAGES_RECORDED if k in installed},
            activelabel_version=__version__,
            epochs=spec["params"]["epochs"],
            epoch=0,
        )
        progress.check()
        store.update(number, phase="training")
        self._train(key, store, progress, [str(env.python()), str(script), str(spec_file)], tmp, spec)

    def _train(
        self,
        key: tuple[int, int],
        store: RunStore,
        progress: Progress,
        cmd: list[str],
        tmp: Path,
        spec: dict[str, Any],
    ) -> None:
        number = key[1]
        run_dir = store.path(number)
        metric_key = selection_key(spec["metric"])
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        env = {**os.environ, **download_env(tmp)}
        lines: queue.Queue[str | None] = queue.Queue()
        state: dict[str, Any] = {"results": None, "error": "", "last": 0.0}
        with (run_dir / "log.txt").open("a", encoding="utf-8", errors="replace") as log:
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
                    self._handle(line.rstrip("\r\n"), store, number, progress, log, metric_key, state)
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
        store: RunStore,
        number: int,
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
        elif kind == "progress":
            done, total = event.get("step"), event.get("steps")
            progress.update(done=done, total=total, message=event.get("phase"))
            if time.monotonic() - state["last"] >= PROGRESS_EVERY:
                state["last"] = time.monotonic()
                store.update(
                    number,
                    phase=event.get("phase") or "training",
                    epoch=event.get("epoch", 0),
                    step=done,
                    steps=total,
                )
        elif kind == "epoch":
            row = {k: v for k, v in event.items() if k != "event"}
            with (store.path(number) / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
            fields: dict[str, Any] = {"epoch": row.get("epoch", 0), "step": None, "steps": None}
            if row.get("best"):
                fields["best"] = {"epoch": row.get("epoch"), "value": row.get(metric_key)}
            store.update(number, **fields)
            progress.update(done=row.get("epoch", 0))
            log.write(text + "\n")
            log.flush()
        elif kind == "done":
            state["results"] = event.get("results")
        elif kind == "error":
            state["error"] = str(event.get("message", "error"))
            log.write(state["error"] + "\n")
            log.flush()
