"""Show a project in the FiftyOne App: images, quality measures, issues and ground truth.

Every project has one persistent dataset. FiftyOne keeps all datasets of the machine in one
database, so the name must not repeat across workspaces: it is ``activelabel-<project id>-<random>``,
chosen once and recorded in ``<project>/fiftyone.json`` (see :func:`dataset_name`). Samples are keyed by
the image id (field ``image_id``). Segmentation polygons become ``Polylines`` with
``filled=True`` (simpler than rasterised masks, and they keep their exact shape).

Each sample also records what ActiveLabel wrote in ``ground_truth`` (``activelabel_written``: its
fingerprint; ``activelabel_labels``: the hash of the project labels it came from). A sample whose
``ground_truth`` no longer matches its fingerprint was edited in FiftyOne (see :func:`edited_images`).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.core.utils import to_slug

from activelabel.annotate import sessions
from activelabel.annotate.fo_labels import fingerprint
from activelabel.curation.issues import ISSUES, NEAR_DUPLICATE, load_duplicate_of
from activelabel.fiftyone_names import record_name, recorded_name
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest
from activelabel.images.video import VideoStore
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.sync.links import label_hash
from activelabel.tasks import Task
from activelabel.tasks_store import ImageTask, TaskStore, jobs

GROUND_TRUTH = "ground_truth"
WRITTEN = "activelabel_written"  # fingerprint of the ground_truth ActiveLabel last wrote
WRITTEN_LABELS = "activelabel_labels"  # label_hash of the project labels it was written from
EXCLUDED_TAG = "excluded"
VIEW_ALL_ISSUES = "curation - quality issues"
VIEW_EXCLUDED = "curation - excluded"
VIEW_DUPLICATES = "curation - near duplicates"

TASK_VIEW_PREFIX = "task-"  # saved view of a task: ``task-<id>`` (stable; the name is its description)

APP_PORT = 5161  # not 5151: another FiftyOne program on the machine may own the default port
PORT_TRIES = 11  # APP_PORT and the next 10 ports

DATASET_PREFIX = "activelabel"  # datasets are ``<prefix>-<project id>[-<random>]``

_session: Any = None
_names_lock = threading.Lock()
log = logging.getLogger(__name__)


def _is_ours(name: str, project_dir: Path) -> bool:
    """Whether the dataset ``name`` holds the images of this project folder (judged by sample paths).

    Args:
        name: name of an existing FiftyOne dataset.
        project_dir: the project folder.

    Returns:
        True when its first sample lies inside the folder; False for an empty dataset.
    """
    sample = fo.load_dataset(name).first()
    if sample is None:  # empty: cannot tell whose it is, so it is left alone
        return False
    return Path(os.path.normcase(sample.filepath)).is_relative_to(os.path.normcase(project_dir.resolve()))


def _legacy_name(project_dir: Path, project_id: int) -> str | None:
    """The old shared name ``activelabel-<id>``, when a dataset of that name holds this project's images.

    Args:
        project_dir: the project folder.
        project_id: id of the project.

    Returns:
        The old name, or None.
    """
    legacy = f"{DATASET_PREFIX}-{project_id}"
    return legacy if fo.dataset_exists(legacy) and _is_ours(legacy, project_dir) else None


def dataset_name(project_dir: Path, project_id: int) -> str:
    """Name of the FiftyOne dataset of a project, chosen once and recorded in the project folder.

    FiftyOne has one database per machine, so two workspaces (or a test server and the real one) must
    not share the name ``activelabel-<id>``. A new project gets ``activelabel-<id>-<8 random hex>``.
    A dataset created by earlier versions under the plain name is kept under that name (renaming it
    would break open FiftyOne App tabs and edit sessions) when its samples point into this project
    folder; one that holds another workspace's images is never adopted.

    Args:
        project_dir: the project folder.
        project_id: id of the project.

    Returns:
        The dataset name.
    """
    with _names_lock:
        name = recorded_name(project_dir)
        if name is None:
            name = (
                _legacy_name(project_dir, project_id)
                or f"{DATASET_PREFIX}-{project_id}-{uuid.uuid4().hex[:8]}"
            )
            record_name(project_dir, name)
        return name


def existing_dataset(project_dir: Path, project_id: int) -> fo.Dataset | None:
    """The project's dataset, or None when it was never created.

    Args:
        project_dir: the project folder.
        project_id: id of the project.

    Returns:
        The dataset, or None.
    """
    name = dataset_name(project_dir, project_id)
    return fo.load_dataset(name) if fo.dataset_exists(name) else None


def delete_dataset(project_dir: Path, project_id: int, recorded: str | None) -> bool:
    """Delete the dataset of a project that is being deleted (``project_dir`` need not exist any more).

    Only a dataset recorded for the project, or the old plain name when it holds this project's
    images, is deleted. Never raises: a failure is logged, the project is gone anyway.

    Args:
        project_dir: the folder the project had.
        project_id: its id.
        recorded: :func:`~activelabel.fiftyone_names.recorded_name` read before the folder was removed.

    Returns:
        Whether a dataset was deleted.
    """
    try:
        name = recorded or _legacy_name(project_dir, project_id)
        if name and fo.dataset_exists(name):
            fo.delete_dataset(name)
            return True
    except Exception:  # noqa: BLE001 - best effort: the project is already deleted
        log.warning("Could not delete the FiftyOne dataset of project %s", project_id, exc_info=True)
    return False


def _label_type(task: Task) -> type:
    """FiftyOne label class used for the ground truth of a task.

    Args:
        task: task of the project.

    Returns:
        ``Classification``, ``Detections``, ``Polylines`` or ``Keypoints``.
    """
    return {
        Task.CLASSIFICATION: fo.Classification,
        Task.DETECTION: fo.Detections,
        Task.SEGMENTATION: fo.Polylines,
        Task.KEYPOINTS: fo.Keypoints,
    }[task]


def _declare_fields(dataset: fo.Dataset, task: Task) -> bool:
    """Make sure every field exists with the right type (also when there are no samples).

    Args:
        dataset: the project's dataset.
        task: task of the project (decides the type of ``ground_truth``; a field of another type is dropped).

    Returns:
        Whether ``ground_truth`` was dropped (the task of the project changed): no sample holds edits now.
    """
    scalars: dict[str, Any] = {
        "image_id": fo.StringField,
        "name": fo.StringField,
        "status": fo.StringField,
        "brightness": fo.FloatField,
        "contrast": fo.FloatField,
        "sharpness": fo.FloatField,
        "duplicate_of": fo.StringField,
        "duplicate_group": fo.StringField,
        "task_name": fo.StringField,
        "video": fo.StringField,
        "video_name": fo.StringField,
        "task": fo.IntField,
        "job": fo.IntField,
        "position": fo.IntField,
        "frame_number": fo.IntField,
        "time": fo.FloatField,
        WRITTEN: fo.StringField,
        WRITTEN_LABELS: fo.StringField,
    }
    for name, kind in scalars.items():
        dataset.add_sample_field(name, kind)
    dataset.add_sample_field("image_size", fo.ListField, subfield=fo.IntField)
    dataset.add_sample_field("quality_issues", fo.ListField, subfield=fo.StringField)
    wanted = _label_type(task)
    dropped = False
    if GROUND_TRUTH in dataset.get_field_schema():
        current = dataset.get_field(GROUND_TRUTH)
        if getattr(current, "document_type", None) is not wanted:  # the task of the project changed
            dataset.delete_sample_field(GROUND_TRUTH)
            dropped = True
    dataset.add_sample_field(GROUND_TRUTH, fo.EmbeddedDocumentField, embedded_doc_type=wanted)
    return dropped


def _visibility(point: list[float]) -> int:
    """Visibility of a keypoint: its third value, or 2 (visible) for a point written as ``[x, y]``.

    Args:
        point: the keypoint ``[x, y]`` or ``[x, y, visibility]``.

    Returns:
        The visibility (0 not labelled, 1 occluded, 2 visible).
    """
    return int(point[2]) if len(point) > 2 else 2


def _ground_truth(task: Task, labels: list[Label]) -> Any:
    """FiftyOne label of one image for the task (None when it has no labels).

    Args:
        task: task of the project.
        labels: the internal labels of the image.

    Returns:
        The FiftyOne label of the task's type, or None when ``labels`` is empty.
    """
    if not labels:
        return None
    if task is Task.CLASSIFICATION:
        return fo.Classification(label=labels[0].label, confidence=labels[0].score)
    if task is Task.DETECTION:
        return fo.Detections(
            detections=[
                fo.Detection(label=lb.label, bounding_box=lb.bbox, index=lb.track, confidence=lb.score)
                for lb in labels
                if lb.bbox
            ]
        )
    if task is Task.SEGMENTATION:
        polylines = [
            fo.Polyline(
                label=lb.label,
                points=[[tuple(p) for p in ring] for ring in lb.polygons],
                filled=True,
                index=lb.track,
            )
            for lb in labels
            if lb.polygons
        ]
        return fo.Polylines(polylines=polylines)
    return fo.Keypoints(
        keypoints=[
            fo.Keypoint(
                label=lb.label,
                index=lb.track,
                confidence=lb.score,
                points=[
                    (p[0], p[1]) if _visibility(p) > 0 else (float("nan"), float("nan")) for p in lb.keypoints
                ],
                # FiftyOne has no visibility nor box for keypoints: kept aside so an edit does not lose them
                visibility=[_visibility(p) for p in lb.keypoints],
                box=list(lb.bbox) if lb.bbox else None,
            )
            for lb in labels
            if lb.keypoints
        ]
    )


def _apply(
    sample: fo.Sample,
    record: ImageRecord,
    manifest: Manifest,
    task: Task,
    annotations: AnnotationSet | None,
    duplicate_of: dict[str, str],
    kept: set[str],
    where: dict[str, dict[str, Any]],
    protected: bool = False,
) -> None:
    """Copy the data of one image record into its FiftyOne sample.

    Args:
        sample: the sample to fill.
        record: the image record.
        manifest: the images of the project (gives the file path).
        task: task of the project.
        annotations: ground truth, or None when there is none yet.
        duplicate_of: near-duplicate image id -> id of the image it duplicates.
        kept: ids of the images that others duplicate.
        where: placement (task, job, position, video name) by image id, see :func:`_placement`.
        protected: the image is in an open FiftyOne session: when its ``ground_truth`` was edited in
            FiftyOne (see :func:`is_edited`) it is left as it is; otherwise it is rewritten like any other.
    """
    sample.filepath = str(manifest.file_path(record).resolve())
    sample["image_id"] = record.id
    sample["name"] = record.name
    sample["status"] = record.status
    for metric in ("brightness", "contrast", "sharpness"):
        sample[metric] = record.metrics.get(metric)
    sample["image_size"] = [record.width, record.height]
    sample["quality_issues"] = list(record.issues)
    sample["duplicate_of"] = duplicate_of.get(record.id)
    sample["duplicate_group"] = duplicate_of.get(record.id) or (record.id if record.id in kept else None)
    place = where.get(record.id, {})
    sample["task"] = place.get("task")
    sample["task_name"] = place.get("task_name")
    sample["job"] = place.get("job")
    sample["position"] = place.get("position")
    sample["video"] = record.video or None
    sample["video_name"] = place.get("video_name") if record.video else None
    sample["frame_number"] = record.frame if record.video else None
    sample["time"] = record.time if record.video else None
    tags = [t for t in sample.tags if t != EXCLUDED_TAG]
    sample.tags = [*tags, EXCLUDED_TAG] if record.status == EXCLUDED else tags
    if protected and is_edited(sample, task):  # edits not brought back yet: FiftyOne keeps its version
        return
    labels = annotations.images.get(record.id, []) if annotations else []
    sample[GROUND_TRUTH] = _ground_truth(task, labels)
    sample[WRITTEN] = fingerprint(sample[GROUND_TRUTH], task)
    sample[WRITTEN_LABELS] = label_hash(labels)


def is_edited(sample: fo.Sample, task: Task) -> bool:
    """Whether the ``ground_truth`` of a sample was changed in FiftyOne since ActiveLabel wrote it.

    Args:
        sample: the sample.
        task: task of the project.

    Returns:
        True when it differs from the recorded fingerprint, or when there is none (a dataset written by
        an earlier version: its edits cannot be told apart, so they are kept).
    """
    written = sample[WRITTEN] if sample.has_field(WRITTEN) else None
    return written is None or fingerprint(sample[GROUND_TRUTH], task) != written


def edited_images(dataset: fo.Dataset, image_ids: list[str], task: Task) -> set[str]:
    """Images whose ``ground_truth`` was changed in FiftyOne since ActiveLabel wrote it (:func:`is_edited`).

    Args:
        dataset: the project's dataset.
        image_ids: the images to look at (those without a sample are left out).
        task: task of the project.

    Returns:
        The ids of the edited images.
    """
    view = dataset.match(F("image_id").is_in(list(image_ids)))
    return {sample["image_id"] for sample in view.iter_samples() if is_edited(sample, task)}


def written_labels(dataset: fo.Dataset, image_ids: list[str]) -> dict[str, str]:
    """Hash of the project labels ActiveLabel last wrote into each sample (``activelabel_labels``).

    Args:
        dataset: the project's dataset.
        image_ids: the images to look at.

    Returns:
        The hash by image id (images without a sample or without a record are left out).
    """
    if WRITTEN_LABELS not in dataset.get_field_schema():
        return {}
    view = dataset.match(F("image_id").is_in(list(image_ids)))
    pairs = zip(view.values("image_id"), view.values(WRITTEN_LABELS), strict=True)
    return {image_id: value for image_id, value in pairs if value}


def _placement(project_dir: Any, tasks: list[ImageTask]) -> dict[str, dict[str, Any]]:
    """Task, job, position and video name of every image that is in a task.

    Args:
        project_dir: the project folder (to read the video names).
        tasks: the tasks of the project.

    Returns:
        ``{image_id: {task, task_name, job, position, video_name}}``.
    """
    try:
        names = {v.id: v.name for v in VideoStore(project_dir).list()}
    except Exception:  # noqa: BLE001 - no video store: the names stay empty
        names = {}
    out: dict[str, dict[str, Any]] = {}
    for task in tasks:
        job_of = {i: j.n for j in jobs(task) for i in j.image_ids}
        for position, image_id in enumerate(task.images, 1):
            out[image_id] = {
                "task": task.id,
                "task_name": task.name,
                "job": job_of.get(image_id),
                "position": position,
                "video_name": names.get(str((task.source or {}).get("video", ""))),
            }
    return out


def task_view_name(task_id: int) -> str:
    """Saved view name of a task (based on the id, so a rename does not change the URL).

    Args:
        task_id: id of the task.

    Returns:
        ``task-<id>``.
    """
    return f"{TASK_VIEW_PREFIX}{task_id}"


def task_view(dataset: fo.Dataset, task_id: int, job: int | None = None, *, video: bool | None = None) -> Any:
    """View of a task (or one of its jobs), in task order; frames of a video task grouped by video.

    Args:
        dataset: the project's dataset.
        task_id: id of the task.
        job: job number to keep only that job, or None for the whole task.
        video: whether the task is made from video; by default it is detected from the samples.

    Returns:
        The view.
    """
    view = dataset.match(F("task") == task_id)
    if job is not None:
        view = view.match(F("job") == job)
    if video is None:
        video = len(view.match(F("video") != None)) > 0  # noqa: E711
    if video:
        return view.group_by("video", order_by="frame_number")
    return view.sort_by("position")


def _save_views(dataset: fo.Dataset, tasks: list[ImageTask]) -> None:
    """Save the curation views and one view per task, and delete the views of deleted tasks.

    Args:
        dataset: the project's dataset.
        tasks: the tasks of the project.
    """
    views: dict[str, fo.DatasetView] = {
        VIEW_ALL_ISSUES: dataset.match(F("quality_issues").length() > 0),
        VIEW_EXCLUDED: dataset.match_tags(EXCLUDED_TAG),
        VIEW_DUPLICATES: dataset.match(F("duplicate_group") != None).sort_by("duplicate_group"),  # noqa: E711
    }
    for issue in ISSUES:
        if issue != NEAR_DUPLICATE:
            views[f"curation - {issue.replace('_', ' ')}"] = dataset.match(
                F("quality_issues").contains(issue)
            )
    for name, view in views.items():
        dataset.save_view(name, view, overwrite=True)
    wanted: set[str] = set()
    for task in tasks:
        name = task_view_name(task.id)
        wanted.add(name)
        view = task_view(dataset, task.id, video=bool((task.source or {}).get("video")))
        dataset.save_view(name, view, description=f"Task: {task.name}", overwrite=True)
    for old in dataset.list_saved_views():
        if old.startswith(TASK_VIEW_PREFIX) and old not in wanted:  # task deleted
            dataset.delete_saved_view(old)


def open_session_images(project_dir: Any) -> set[str]:
    """Images of open FiftyOne annotation sessions: an edited ``ground_truth`` of theirs is left as it is.

    Args:
        project_dir: the project folder.

    Returns:
        The image ids.
    """
    return sessions.protected_images(project_dir)


def sync_dataset(project: Project, manifest: Manifest, annotations: AnnotationSet | None) -> fo.Dataset:
    """Create or refresh the persistent FiftyOne dataset of a project, in place.

    Samples are matched by ``image_id``: re-running updates them, adds new images and removes
    the samples of deleted images. Excluded images stay, tagged ``excluded``. The saved views
    ``curation - quality issues``, ``curation - excluded``, ``curation - near duplicates`` and
    one ``curation - <issue>`` per issue (spaces instead of underscores) are refreshed too, and so is
    one saved view ``task-<id>`` per task (description ``Task: <name>``); views of deleted tasks
    are removed. Every sample gets ``task``, ``task_name``, ``job``, ``position`` and, for video
    frames, ``video``, ``video_name``, ``frame_number`` and ``time``. The ``ground_truth`` of an image in
    an open FiftyOne annotation session that was edited in FiftyOne is not overwritten (the edits are not
    in the project yet); every other sample shows the project's labels and records their fingerprint.

    Args:
        project: the project (name of the dataset, task, classes).
        manifest: images, measures and issues.
        annotations: ground truth, or None when there is none yet.

    Returns:
        The dataset.
    """
    name = dataset_name(manifest.project_dir, project.id)
    dataset = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name, persistent=True)
    dataset.persistent = True
    reset = _declare_fields(dataset, project.task)

    records = {r.id: r for r in manifest.all()}
    duplicate_of = load_duplicate_of(manifest.project_dir)
    kept = set(duplicate_of.values())
    tasks = TaskStore(manifest.project_dir).load()
    where = _placement(manifest.project_dir, tasks)
    protected = set() if reset else open_session_images(manifest.project_dir)

    known: dict[str, str] = {}
    stale: list[str] = []
    for sample_id, image_id in zip(dataset.values("id"), dataset.values("image_id"), strict=True):
        if image_id in records and image_id not in known:
            known[image_id] = sample_id
        else:
            stale.append(sample_id)
    if stale:
        dataset.delete_samples(stale)

    new: list[fo.Sample] = []
    for record in records.values():
        if record.id not in known:
            sample = fo.Sample(filepath=str(manifest.file_path(record).resolve()))
            _apply(sample, record, manifest, project.task, annotations, duplicate_of, kept, where)
            new.append(sample)
    if new:
        dataset.add_samples(new)
    for sample in dataset.select(list(known.values())).iter_samples(autosave=True):
        _apply(
            sample,
            records[sample["image_id"]],
            manifest,
            project.task,
            annotations,
            duplicate_of,
            kept,
            where,
            protected=sample["image_id"] in protected,
        )

    dataset.classes = {GROUND_TRUTH: list(project.classes)}
    dataset.info = {**dataset.info, "project_id": project.id, "project_name": project.name}
    dataset.save()
    _save_views(dataset, tasks)
    return dataset


def resolve_view(dataset: fo.Dataset, requested: str | None) -> str | None:
    """Saved view to open for a request.

    Accepts a saved view name, an issue name (``blurry``, ``near_duplicate``, with or without the
    ``curation - `` prefix) or None. Unknown names fall back to the view of all issues.

    Args:
        dataset: the project's dataset.
        requested: the name asked for, or None.

    Returns:
        The saved view name, or None when nothing was requested or no fallback view exists.
    """
    if not requested:
        return None
    saved = set(dataset.list_saved_views())
    if re.fullmatch(rf"{TASK_VIEW_PREFIX}\d+", requested) and requested in saved:
        return requested
    if requested in saved:
        return requested
    issue = requested.removeprefix("curation - ").strip()
    candidates = [VIEW_DUPLICATES] if issue == NEAR_DUPLICATE else [f"curation - {issue.replace('_', ' ')}"]
    for name in candidates:
        if name in saved:
            return name
    return VIEW_ALL_ISSUES if VIEW_ALL_ISSUES in saved else None


def _slug(dataset: fo.Dataset, view: str) -> str:
    """URL slug of a saved view (FiftyOne's own slug of its name when it is not listed).

    Args:
        dataset: the project's dataset.
        view: name of the saved view.

    Returns:
        The slug.
    """
    for info in dataset.list_saved_views(info=True):
        data = info if isinstance(info, dict) else info.to_dict()
        if data.get("name") == view and data.get("slug"):
            return str(data["slug"])
    return str(to_slug(view))


def app_origin(port: int) -> str:
    """Address of the App on a port, as the person's browser reaches it.

    In a GitHub Codespace opened in the browser, ``localhost`` is the person's own machine, so the
    address is the codespace's forwarded one, ``https://<codespace>-<port>.<forwarding domain>``, built
    from ``CODESPACE_NAME`` and ``GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN`` (the port must be forwarded,
    see the devcontainer). Elsewhere it is ``http://localhost:<port>``.

    Args:
        port: port of the App.

    Returns:
        The scheme, host and port, without a trailing slash.
    """
    name = os.environ.get("CODESPACE_NAME", "").strip()
    domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "").strip()
    if name and domain:
        return f"https://{name}-{port}.{domain}"
    return f"http://localhost:{port}"


def app_url(dataset: fo.Dataset, view: str | None, port: int) -> str:
    """URL that opens the dataset (and saved view) directly, whichever server answers on the port.

    Args:
        dataset: the project's dataset.
        view: name of the saved view to open, or None for the whole dataset.
        port: port of the App.

    Returns:
        The URL (on the codespace's forwarded address inside a GitHub Codespace, see :func:`app_origin`).
    """
    url = f"{app_origin(port)}/datasets/{dataset.name}"
    return f"{url}?view={_slug(dataset, view)}" if view else url


def launch_app(
    dataset: fo.Dataset,
    view: str | None = None,
    port: int = APP_PORT,
    task: int | None = None,
    job: int | None = None,
) -> str:
    """Start the FiftyOne App in the background (no browser opened) and return a URL for the dataset.

    A session of ours that is already running is reused: it just switches to the dataset (and
    saved view, when given). If the port is busy the next ones are tried (up to +10). The URL
    names the dataset (and view), so it opens correctly even if another FiftyOne program
    answers on that port.

    Args:
        dataset: the project's dataset.
        view: saved view (or issue name, see :func:`resolve_view`) to open; None for the whole dataset.
        port: first port to try.
        task: id of a task to open (takes the place of ``view``).
        job: with ``task``, number of the job to show.

    Returns:
        The URL of the dataset (and view) in the App.

    Raises:
        OSError: no port was free.
    """
    global _session
    if task is not None:
        view = task_view_name(task) if task_view_name(task) in dataset.list_saved_views() else None
    else:
        view = resolve_view(dataset, view)
    if view and len(dataset.load_saved_view(view)) == 0:
        view = None  # an empty view looks like a broken App: show the whole dataset instead
    target: Any = dataset.load_saved_view(view) if view else dataset
    if task is not None and job is not None:  # a job is not a saved view: built on demand
        built = task_view(dataset, task, job)
        if len(built):
            view, target = None, built
    if _session is not None:
        try:
            _session.dataset = dataset
            if view or target is not dataset:
                _session.view = target
            return app_url(dataset, view, int(_session.port))
        except Exception:  # noqa: BLE001 - the session was closed: start a new one
            _session = None
    last: Exception | None = None
    for candidate in range(port, port + PORT_TRIES):
        try:
            _session = fo.launch_app(target, port=candidate, remote=True)
        except Exception as exc:  # noqa: BLE001 - port busy: try the next one
            last = exc
            continue
        return app_url(dataset, view, candidate)
    raise OSError(f"No free port for the FiftyOne App ({port}-{port + PORT_TRIES - 1}): {last}")


def pull_exclusions(dataset: fo.Dataset, manifest: Manifest, tag: str = "exclude") -> int:
    """Exclude the images the user tagged in the App, then remove the tag.

    Args:
        dataset: the project's dataset.
        manifest: images of the project.
        tag: the tag marking images to exclude.

    Returns:
        How many images were excluded (images already excluded do not count).
    """
    tagged = dataset.match_tags(tag)
    ids = [i for i in tagged.values("image_id") if i]
    changed = manifest.set_status(ids, EXCLUDED, "tagged in FiftyOne") if ids else 0
    if ids:
        tagged.tag_samples(EXCLUDED_TAG)
        tagged.untag_samples(tag)
    return changed
