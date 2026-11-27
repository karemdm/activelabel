"""The job that groups the similarity map again with a chosen method (global queue, CPU lane; #605, #614).

The map keeps its points; only the groups change (k-means with a chosen number of groups, HDBSCAN with
groups of free size and noise, or ``auto``, the map's own choice). The state of the job is the
``clustering`` entry of ``similarity.json``: ``{status, job, method, groups, min_size, created, started,
finished, error}``. A new similarity map resets it to ``{}`` (its groups are ``auto`` again). The groups
are written to the FiftyOne dataset as the map's are (field ``similarity_group``). Cancelling stops between
the steps (reading, grouping, writing); a running scikit-learn fit is not interrupted.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

from activelabel import fiftyone_bridge, optional
from activelabel.constants import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from activelabel.curation import fiftyone_runs, similarity, similarity_job
from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Progress, readable_error
from activelabel.projects import Project
from activelabel.scheduler import CPU
from activelabel.storage import now
from activelabel.training.runner import TrainingRunner

LOG = logging.getLogger(__name__)
KIND = "clustering"
ACTIVE = (QUEUED, RUNNING)


def _state(store: similarity.SimilarityStore) -> dict[str, Any]:
    """The ``clustering`` entry of the similarity state.

    Args:
        store: The similarity files of the project.

    Returns:
        The entry (``{}`` when the map was never regrouped).
    """
    entry = store.read().get("clustering")
    return dict(entry) if isinstance(entry, dict) else {}


def _set(store: similarity.SimilarityStore, **fields: Any) -> dict[str, Any]:
    """Change some fields of the ``clustering`` entry.

    Args:
        store: The similarity files of the project.
        **fields: The fields to set.

    Returns:
        The whole entry after the change.
    """
    with store.lock:
        entry = {**_state(store), **fields}
        store.update(clustering=entry)
    return entry


def start(
    runner: TrainingRunner,
    project: Project,
    project_dir: Path,
    method: str,
    groups: int = 0,
    min_size: int = 5,
) -> dict[str, Any]:
    """Queue the regrouping (the caller checked that neither it nor the map is queued or running).

    Args:
        runner: The training runner (its queue holds the heavy jobs of every project).
        project: The project.
        project_dir: Folder of the project.
        method: ``auto``, ``kmeans`` or ``hdbscan``.
        groups: With ``kmeans``: how many groups.
        min_size: With ``hdbscan``: the smallest group.

    Returns:
        The ``clustering`` entry, queued.

    Raises:
        CodedError: ``similarity_map_needed`` without a finished map; ``cluster_method_unknown``,
            ``cluster_groups_range`` or ``cluster_min_size_small`` for a method or number out of range
            (checked before anything is queued).
    """
    store = similarity.SimilarityStore(project_dir)
    if store.read().get("points") is None or store.load_vectors() is None:
        raise CodedError(
            "similarity_map_needed", "Compute the similarity map first: the groups are made on it."
        )
    _check(method, groups, min_size)
    with store.lock:  # the job starts only after its id is saved (it reads the state first)
        _set(
            store,
            status=QUEUED,
            method=method,
            groups=groups,
            min_size=min_size,
            created=now(),
            started="",
            finished="",
            error="",
        )
        job_id = runner.enqueue_tool(
            project.id,
            KIND,
            partial(run, project, project_dir),
            {"method": method},
            device=CPU,
            on_cancel=partial(cancel, runner, project.id, project_dir),
        )
        return _set(store, job=job_id)


def _check(method: str, groups: int, min_size: int) -> None:
    """Refuse a method or a number out of range before anything is queued.

    Args:
        method: The method asked for.
        groups: The number of groups asked for (``kmeans``).
        min_size: The smallest group asked for (``hdbscan``).

    Raises:
        CodedError: ``cluster_method_unknown`` (with ``methods``), ``cluster_groups_range`` (with ``min`` and
            ``max``) or ``cluster_min_size_small`` (with ``min``), each translated in the interface.
    """
    if method not in similarity.CLUSTER_METHODS:
        methods = ", ".join(similarity.CLUSTER_METHODS)
        raise CodedError("cluster_method_unknown", f"The method must be one of: {methods}.", methods=methods)
    if method == "kmeans" and not 2 <= groups <= similarity.MAX_CHOSEN_GROUPS:
        top = similarity.MAX_CHOSEN_GROUPS
        raise CodedError(
            "cluster_groups_range", f"The number of groups must be from 2 to {top}.", min=2, max=top
        )
    if method == "hdbscan" and min_size < 2:
        raise CodedError("cluster_min_size_small", "The smallest group must have at least 2 images.", min=2)


def state(runner: TrainingRunner, project_dir: Path) -> dict[str, Any]:
    """The ``clustering`` entry as the interface shows it (one left by a previous start is interrupted).

    Args:
        runner: The training runner.
        project_dir: Folder of the project.

    Returns:
        The entry (``{}`` when the map was never regrouped).
    """
    store = similarity.SimilarityStore(project_dir)
    entry = _state(store)
    if entry.get("status") in ACTIVE and runner.tool_job(str(entry.get("job") or "")) is None:
        entry = _set(store, status=FAILED, error="interrupted", finished=now())
    return entry


def cancel(runner: TrainingRunner, project_id: int, project_dir: Path) -> dict[str, Any]:
    """Cancel a queued or running regrouping (the groups stay as they were).

    Args:
        runner: The training runner.
        project_id: Id of the project.
        project_dir: Folder of the project.

    Returns:
        The ``clustering`` entry.
    """
    store = similarity.SimilarityStore(project_dir)
    with store.lock:
        entry = _state(store)
        if entry.get("status") not in ACTIVE:
            return entry
        runner.cancel_tool(project_id, str(entry.get("job") or ""), KIND)
        if entry.get("status") == QUEUED:
            return _set(store, status=CANCELLED, finished=now())
    return entry


def _record(project: Project, project_dir: Path, entry: dict[str, Any], images: int) -> None:
    """Record the regrouping as a run of the dataset (#629); a failure there is only logged.

    Args:
        project: The project.
        project_dir: Folder of the project.
        entry: The ``clustering`` entry (``method``, ``groups``, ``min_size``).
        images: How many images were grouped.
    """
    if not optional.FIFTYONE:  # the light install: no dataset to record on
        return
    try:
        dataset = fiftyone_bridge.existing_dataset(project_dir, project.id)
        if dataset is not None:
            fiftyone_runs.record(
                dataset,
                "clustering",
                method=entry.get("method"),
                groups=entry.get("groups"),
                min_size=entry.get("min_size"),
                field="similarity_group",
                images=images,
            )
    except Exception:  # noqa: BLE001 - the groups are kept without the run
        LOG.exception("Could not record the regrouping of project %s as a FiftyOne run", project.id)


def run(project: Project, project_dir: Path, progress: Progress) -> None:
    """Body of the job: group the map's images again, save the groups, write them to FiftyOne.

    Args:
        project: The project.
        project_dir: Folder of the project.
        progress: Progress handle of the job (checked between the steps).
    """
    store = similarity.SimilarityStore(project_dir)
    with store.lock:
        entry = _state(store)
        if entry.get("status") != QUEUED:  # cancelled while waiting
            return
        _set(store, status=RUNNING, started=now())
    try:
        saved = store.load_vectors()
        data = store.read()
        points = data.get("points")
        if saved is None or not isinstance(points, dict):
            raise CodedError("similarity_map_needed", "Compute the similarity map first.")
        ids, vectors, _backbone = saved
        row_of = {image_id: row for row, image_id in enumerate(ids)}
        xy_of = dict(zip(points.get("ids", []), points.get("xy", []), strict=False))
        map_ids = [str(i) for i in points.get("ids", []) if i in row_of]
        progress.update(message="Grouping")
        labels = similarity.cluster(
            vectors[[row_of[i] for i in map_ids]],
            str(entry.get("method")),
            int(entry.get("groups") or 0),
            int(entry.get("min_size") or 5),
        )
        progress.check()
        result = {
            **{k: data[k] for k in ("method",) if k in data},
            "groups": similarity.typical_images(map_ids, vectors[[row_of[i] for i in map_ids]], labels),
            "points": {
                "ids": map_ids,
                "xy": [xy_of[i] for i in map_ids],
                "group": [int(g) for g in labels],
            },
        }
        store.update(groups=result["groups"], points=result["points"])
        progress.update(message="Writing to FiftyOne")
        fiftyone_error = similarity_job._publish(project, Manifest(project_dir), project_dir, result)
        if not fiftyone_error:
            _record(project, project_dir, entry, len(map_ids))
        _set(
            store,
            status=DONE,
            finished=now(),
            noise=int((labels == similarity.NOISE).sum()),
            fiftyone_error=fiftyone_error,
        )
    except Cancelled:
        _set(store, status=CANCELLED, finished=now())
    except Exception as exc:  # noqa: BLE001 - shown to the person
        LOG.exception("The regrouping of project %s failed", project.id)
        _set(store, status=FAILED, error=readable_error(exc), finished=now())
