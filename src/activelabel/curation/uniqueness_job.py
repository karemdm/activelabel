"""The job that scores the uniqueness of a project's images (global queue, CPU lane; #605, #610).

It needs the embeddings of the similarity map (compute the map first). Phases: ``scoring`` (in chunks:
cancelling stops between two of them) and ``fiftyone`` (the scores written to the field ``uniqueness`` of
the project's FiftyOne dataset; a failure there keeps the scores and is only reported as
``fiftyone_error``). The state is kept in ``uniqueness.json`` (:class:`~activelabel.curation.uniqueness.
UniquenessStore`).
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

from activelabel import fiftyone_bridge, optional
from activelabel.constants import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from activelabel.curation import fiftyone_runs, similarity, uniqueness
from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Progress, readable_error
from activelabel.labels import load_annotations
from activelabel.projects import Project
from activelabel.scheduler import CPU
from activelabel.storage import now
from activelabel.training.runner import TrainingRunner

LOG = logging.getLogger(__name__)
KIND = "uniqueness"
FIELD = "uniqueness"  # field of the FiftyOne samples
ACTIVE = (QUEUED, RUNNING)


def start(runner: TrainingRunner, project: Project, project_dir: Path) -> dict[str, Any]:
    """Queue the scoring (the caller checked that none is queued or running).

    Args:
        runner: The training runner (its queue holds the heavy jobs of every project).
        project: The project.
        project_dir: Folder of the project.

    Returns:
        The state saved in ``uniqueness.json``.

    Raises:
        CodedError: ``similarity_map_needed`` when the similarity map (its embeddings) was never computed.
    """
    if similarity.SimilarityStore(project_dir).load_vectors() is None:
        raise CodedError(
            "similarity_map_needed", "Compute the similarity map first: uniqueness uses its embeddings."
        )
    store = uniqueness.UniquenessStore(project_dir)
    with store.lock:  # the job starts only after its id is saved (it reads the state first)
        store.update(status=QUEUED, created=now(), started="", finished="", error="", fiftyone_error="")
        job_id = runner.enqueue_tool(
            project.id,
            KIND,
            partial(run, project, project_dir),
            {},
            device=CPU,
            on_cancel=partial(cancel, runner, project.id, project_dir),
        )
        return store.update(job=job_id)


def state(runner: TrainingRunner, project_dir: Path) -> dict[str, Any]:
    """The state as the interface shows it (one left queued or running by a previous start is interrupted).

    Args:
        runner: The training runner.
        project_dir: Folder of the project.

    Returns:
        The saved state (``{}`` before the first scoring).
    """
    store = uniqueness.UniquenessStore(project_dir)
    data = store.read()
    if data.get("status") in ACTIVE and runner.tool_job(str(data.get("job") or "")) is None:
        data = store.update(status=FAILED, error="interrupted", finished=now())
    return data


def cancel(runner: TrainingRunner, project_id: int, project_dir: Path) -> dict[str, Any]:
    """Cancel a queued or running scoring (the previous scores, if any, stay).

    Args:
        runner: The training runner.
        project_id: Id of the project.
        project_dir: Folder of the project.

    Returns:
        The saved state.
    """
    store = uniqueness.UniquenessStore(project_dir)
    with store.lock:
        data = store.read()
        if data.get("status") not in ACTIVE:
            return data
        runner.cancel_tool(project_id, str(data.get("job") or ""), KIND)
        if data.get("status") == QUEUED:
            return store.update(status=CANCELLED, finished=now())
    return data


def run(project: Project, project_dir: Path, progress: Progress) -> None:
    """Body of the job: score the active images that have embeddings, then write the scores to FiftyOne.

    Args:
        project: The project.
        project_dir: Folder of the project.
        progress: Progress handle of the job (checked between chunks).
    """
    store = uniqueness.UniquenessStore(project_dir)
    with store.lock:
        if store.read().get("status") != QUEUED:  # cancelled while waiting
            return
        store.update(status=RUNNING, started=now(), phase="scoring")
    try:
        saved = similarity.SimilarityStore(project_dir).load_vectors()
        if saved is None:
            raise CodedError("similarity_map_needed", "Compute the similarity map first.")
        ids, vectors, backbone = saved
        manifest = Manifest(project_dir)
        active = {r.id for r in manifest.active()}
        rows = [i for i, image_id in enumerate(ids) if image_id in active]
        values = uniqueness.scores(vectors[rows], partial(_report, progress))
        result = {ids[i]: round(float(v), 6) for i, v in zip(rows, values, strict=True)}
        progress.check()
        store.update(phase="fiftyone")
        fiftyone_error = _publish(project, manifest, project_dir, result)
        store.update(
            status=DONE,
            finished=now(),
            phase="",
            scores=result,
            backbone=backbone,
            missing=len(active - set(result)),
            fiftyone_error=fiftyone_error,
        )
    except Cancelled:
        store.update(status=CANCELLED, finished=now(), phase="")
    except Exception as exc:  # noqa: BLE001 - shown to the person
        LOG.exception("The uniqueness of project %s failed", project.id)
        store.update(status=FAILED, error=readable_error(exc), finished=now(), phase="")


def _report(progress: Progress, done: int, total: int) -> None:
    """Report the rows scored so far (and stop here when the job was cancelled).

    Args:
        progress: Progress handle of the job.
        done: Rows scored.
        total: Rows in all.
    """
    progress.update(done=done, total=total)


def _publish(project: Project, manifest: Manifest, project_dir: Path, result: dict[str, float]) -> str:
    """Write the scores to the field ``uniqueness`` of the project's FiftyOne dataset (refreshed first).

    Args:
        project: The project.
        manifest: Images of the project.
        project_dir: Folder of the project.
        result: Image id to score.

    Returns:
        ``""`` when it worked, else the readable error (the scores are kept either way).
    """
    if not optional.FIFTYONE:  # the light install: the scores are shown by ActiveLabel only
        return f"FiftyOne is not installed (the scores are not in its App): {optional.FIFTYONE_INSTALL}"
    try:
        dataset = fiftyone_bridge.sync_dataset(project, manifest, load_annotations(project_dir))
        dataset.set_values(FIELD, result, key_field="image_id")
        fiftyone_runs.record(dataset, "uniqueness", field=FIELD, images=len(result))
    except Exception as exc:  # noqa: BLE001 - the scores are useful without FiftyOne
        LOG.exception("Could not write the uniqueness of project %s to FiftyOne", project.id)
        return readable_error(exc)
    return ""
