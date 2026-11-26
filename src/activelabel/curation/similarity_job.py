"""The job that computes the similarity map of a project (in the global queue: one heavy job per device).

Phases: ``environment`` (torch and torchvision in the training environment, installed on first use),
``embedding`` (the worker ``embed.py`` reads every active image), ``map`` (2-D map and groups, in the app)
and ``fiftyone`` (the map and the groups written to the project's FiftyOne dataset). The state is kept in
``similarity.json`` (see :mod:`activelabel.curation.similarity`); a failure to write to FiftyOne does not
lose the map, it is only reported (``fiftyone_error``).
"""

from __future__ import annotations

import logging
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from activelabel import fiftyone_bridge, optional
from activelabel.constants import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from activelabel.curation import similarity, similarity_fiftyone
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Progress, readable_error
from activelabel.labels import load_annotations
from activelabel.projects import Project
from activelabel.storage import now, read_json
from activelabel.training.environment import remove_tree
from activelabel.training.runner import TrainingRunner

LOG = logging.getLogger(__name__)
WORKER = "embed"
KIND = "similarity"
PACKAGES = ("torch", "torchvision", "pillow", "numpy")
#: Fewest active images for a map (t-SNE needs 5; fewer images are easier to look at one by one).
MIN_IMAGES = 5
BACKBONE = "mobilenet_v3_small"
PROGRESS_EVERY = 1.0  # seconds between writes of the progress to similarity.json
ACTIVE = (QUEUED, RUNNING)


def images_to_embed(manifest: Manifest) -> list[dict[str, str]]:
    """The active images whose file exists, as the worker reads them.

    Args:
        manifest: Images of the project.

    Returns:
        ``[{"id", "path"}]`` with absolute paths.
    """
    items = []
    for record in manifest.active():
        path = manifest.file_path(record)
        if path is not None and path.exists():
            items.append({"id": record.id, "path": str(path.resolve())})
    return items


class _Events:
    """Takes the worker's events: progress (written now and then) and unreadable images."""

    def __init__(self, store: similarity.SimilarityStore, progress: Progress) -> None:
        """Start collecting the events of one computation.

        Args:
            store: The similarity files of the project.
            progress: Progress handle of the job.
        """
        self.unreadable: list[str] = []
        self._store = store
        self._progress = progress
        self._last = 0.0

    def __call__(self, event: dict[str, Any]) -> None:
        """Take one event of the worker (``skipped`` or ``progress``; others are ignored).

        Args:
            event: A JSON event of the worker.
        """
        kind = event.get("event")
        if kind == "skipped":
            self.unreadable.append(str(event.get("id")))
        elif kind == "progress" and time.monotonic() - self._last >= PROGRESS_EVERY:
            self._last = time.monotonic()
            self._store.update(done=event.get("step", 0), total=event.get("steps", 0))
            self._progress.update(done=event.get("step"), total=event.get("steps"))


def start(
    runner: TrainingRunner, project: Project, project_dir: Path, device: str = "auto"
) -> dict[str, Any]:
    """Queue the computation of the map (the caller checked that none is queued or running).

    Args:
        runner: The training runner (the job waits in the global queue, in the lane of its device).
        project: The project.
        project_dir: Folder of the project.
        device: Where the network runs (``auto``, ``cpu``, ``cuda:N`` or ``mps``).

    Returns:
        The state saved in ``similarity.json``.
    """
    store = similarity.SimilarityStore(project_dir)
    with store.lock:  # the job starts only after its id is saved (it reads the state first)
        store.update(
            status=QUEUED, created=now(), started="", finished="", error="", fiftyone_error="", phase=""
        )
        job_id = runner.enqueue_tool(
            project.id,
            KIND,
            partial(run, runner, project, project_dir, device),
            {},
            device=device,
            on_cancel=partial(cancel, runner, project.id, project_dir),
        )
        return store.update(job=job_id)


def state(runner: TrainingRunner, project_dir: Path, active_ids: set[str]) -> dict[str, Any]:
    """The state of the map as the interface shows it.

    A computation left queued or running by a previous start of the app (its job is unknown) is reported as
    interrupted.

    Args:
        runner: The training runner.
        project_dir: Folder of the project.
        active_ids: Ids of the active images now.

    Returns:
        The saved state (``{}`` without a map) with ``outdated`` (the images changed since the map was made).
    """
    store = similarity.SimilarityStore(project_dir)
    data = store.read()
    if data.get("status") in ACTIVE and runner.tool_job(str(data.get("job") or "")) is None:
        data = store.update(status=FAILED, error="interrupted", finished=now(), phase="")
    if not data:
        return {}
    return {**data, "outdated": similarity.is_outdated(data, active_ids)}


def cancel(runner: TrainingRunner, project_id: int, project_dir: Path) -> dict[str, Any]:
    """Cancel a queued or running computation (the previous map, if any, stays).

    Args:
        runner: The training runner.
        project_id: Id of the project.
        project_dir: Folder of the project.

    Returns:
        The saved state.
    """
    store = similarity.SimilarityStore(project_dir)
    with store.lock:
        data = store.read()
        if data.get("status") not in ACTIVE:
            return data
        runner.cancel_tool(project_id, str(data.get("job") or ""), WORKER)
        if data.get("status") == QUEUED:
            return store.update(status=CANCELLED, finished=now(), phase="")
    return data


def run(runner: TrainingRunner, project: Project, project_dir: Path, device: str, progress: Progress) -> None:
    """Body of the job: embeddings in the training environment, then the map, then FiftyOne.

    The outcome is written to ``similarity.json``: ``done`` with the map, ``cancelled``, or ``failed`` with a
    readable ``error``; a previous map is kept until a new one is ready.

    Args:
        runner: The training runner (environment, temporary folders, worker processes).
        project: The project.
        project_dir: Folder of the project.
        device: Where the network runs.
        progress: Progress handle of the job (checked for cancellation).
    """
    store = similarity.SimilarityStore(project_dir)
    with store.lock:
        if store.read().get("status") != QUEUED:  # cancelled while waiting
            return
        store.update(status=RUNNING, started=now(), phase="environment", done=0, total=0, message="")
    tmp: Path | None = None
    try:
        manifest = Manifest(project_dir)
        items = images_to_embed(manifest)
        listed = {i["id"] for i in items}
        missing = [
            r.id for r in manifest.active() if r.id not in listed
        ]  # no file: not read, like unreadable
        if not items:
            raise ValueError("The project has no image to map.")
        runner.environment.ensure(PACKAGES, progress.check, partial(_tell, store))
        progress.check()
        store.update(phase="embedding", message="", total=len(items), done=0)
        runner.tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="similarity-", dir=runner.tmp_root))
        events = _Events(store, progress)
        spec = {"items": items, "backbone": BACKBONE, "device": device, "output": str(tmp / "out")}
        results = runner.run_worker(
            project.id, WORKER, spec, tmp, project_dir / "similarity.log", progress, events
        )
        ids = [str(x) for x in read_json(tmp / "out" / "ids.json", [])]
        vectors = np.load(tmp / "out" / "vectors.npy")
        progress.check()
        store.update(phase="map", done=len(items))
        result = similarity.build_map(ids, vectors)
        store.save_vectors(ids, vectors, str(results.get("backbone") or BACKBONE))
        progress.check()
        store.update(phase="fiftyone")
        fiftyone_error = _publish(project, manifest, project_dir, result)
        store.update(
            **result,
            status=DONE,
            finished=now(),
            phase="",
            backbone=str(results.get("backbone") or BACKBONE),
            unreadable=events.unreadable + missing,  # else the new map would be "outdated" at once
            fiftyone_error=fiftyone_error,
            clustering={},  # a new map has its own groups (``auto``): a regrouping no longer applies
        )
    except Cancelled:
        store.update(status=CANCELLED, finished=now(), phase="")
    except Exception as exc:  # noqa: BLE001 - shown to the person
        LOG.exception("The similarity map of project %s failed", project.id)
        store.update(status=FAILED, error=readable_error(exc), finished=now(), phase="")
    finally:
        if tmp is not None:
            remove_tree(tmp)


def _tell(store: similarity.SimilarityStore, message: str) -> None:
    """Show what the environment set-up is doing.

    Args:
        store: The similarity files of the project.
        message: The step being done (a code the interface translates, such as ``train_env_install:torch``).
    """
    store.update(message=message)


def _publish(project: Project, manifest: Manifest, project_dir: Path, result: dict[str, Any]) -> str:
    """Write the map and the groups to the project's FiftyOne dataset (refreshed first).

    Args:
        project: The project.
        manifest: Images of the project.
        project_dir: Folder of the project.
        result: The map (see :func:`activelabel.curation.similarity.build_map`).

    Returns:
        ``""`` when it worked, else the readable error (the map itself is kept either way).
    """
    if not optional.FIFTYONE:  # the light install: the map is shown by ActiveLabel only
        return (
            f"FiftyOne is not installed (the map is not in its App). Add it with: {optional.FIFTYONE_INSTALL}"
        )
    try:
        dataset = fiftyone_bridge.sync_dataset(project, manifest, load_annotations(project_dir))
        similarity_fiftyone.publish(dataset, result)
    except Exception as exc:  # noqa: BLE001 - the map is still useful without FiftyOne
        LOG.exception("Could not write the similarity map of project %s to FiftyOne", project.id)
        return readable_error(exc)
    return ""
