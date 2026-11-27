"""HTTP API used by the interface (see docs/DESIGN.md)."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from activelabel import annotations as ann
from activelabel import fiftyone_bridge
from activelabel.backups import make_backup
from activelabel.constants import DATASET_STEPS, PROGRESS_STEPS
from activelabel.curation import issues, quality, similarity, uniqueness
from activelabel.curation import summary as curation_summary
from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.fiftyone_names import recorded_name
from activelabel.images.ingest import check_folder, ingest_files, ingest_folder, thumbnail
from activelabel.images.manifest import ACTIVE, EXCLUDED, ImageRecord, Manifest
from activelabel.images.tags import forget_images
from activelabel.importing.common import read_origin
from activelabel.jobs import Job, Progress
from activelabel.labels import (
    AnnotationSet,
    annotations_lock,
    annotations_path,
    load_annotations,
    save_annotations,
)
from activelabel.optional import FIFTYONE
from activelabel.preannotate import annotated_ids, empty_ids, image_state
from activelabel.projects import Project, ProjectNotFound, check_class_spelling, clean_classes
from activelabel.scheduler import CPU
from activelabel.storage import DamagedFile
from activelabel.tasks import Task
from activelabel.tasks_store import TaskNotFound, TaskStore
from activelabel.tasks_store import jobs as task_jobs
from activelabel.web.errors import ApiError, bad_request, need_fiftyone
from activelabel.web.state import AppState, get_state

LOG = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
STATE = Depends(get_state)
MAX_PAGE = 500
JOB_GONE = "This activity no longer exists (the list of finished jobs was cleared or the server restarted)."
UPLOADS = Annotated[list[UploadFile], File()]


# -- bodies -------------------------------------------------------------------
class NewProject(BaseModel):
    """Body of POST /projects."""

    name: str
    description: str = ""
    task: Task
    classes: list[str] = Field(default_factory=list)


class ProjectChanges(BaseModel):
    """Body of PATCH /projects/{id}; only the given fields change."""

    name: str | None = None
    description: str | None = None
    task: Task | None = None
    classes: list[str] | None = None
    annotations: Literal["convert", "discard"] | None = (
        None  # what to do with the ground truth on a task change
    )
    classes_action: Literal["remove_annotations"] | None = None  # when removed classes are still in use


class DeleteProject(BaseModel):
    """Body of DELETE /projects/{id}: the project name, typed again."""

    confirm_name: str


class FolderImport(BaseModel):
    """Body of POST /projects/{id}/images/folder."""

    path: str


class ImageIds(BaseModel):
    """Body of the exclude / restore / delete calls."""

    ids: list[str] = Field(max_length=100_000)
    reason: str = Field("", max_length=500)
    confirm: bool = False


# -- helpers ------------------------------------------------------------------
def _unreadable(reason: str) -> ApiError:
    """HTTP 400: an annotation file does not follow its format.

    Args:
        reason: The parser's English text (kept in ``params`` for the log, not shown translated).

    Returns:
        The HTTP 400 error to raise (code ``annotation_unreadable``).
    """
    return ApiError(
        400, "annotation_unreadable", f"The annotation file could not be read: {reason}", reason=reason
    )


def _malformed(found: ann.Unreadable) -> ApiError:
    """HTTP 400 for a file that is not well-formed JSON or XML or not UTF-8 text, with where it went wrong.

    Args:
        found: What the parser found wrong (file, kind, line and column).

    Returns:
        The HTTP 400 error to raise: ``annotation_not_json``, ``annotation_not_xml`` or
        ``annotation_not_text``.
    """
    detail = f"The annotation file could not be read: {found.file}: {found.reason}"
    if found.kind == "json":
        return ApiError(
            400, "annotation_not_json", detail, file=found.file, line=found.line, column=found.column
        )
    if found.kind == "xml":
        return ApiError(
            400, "annotation_not_xml", detail, file=found.file, line=found.line, column=found.column
        )
    return ApiError(400, "annotation_not_text", detail, file=found.file)


def project_overview(state: AppState, project: Project) -> dict[str, Any]:
    """Project with counts and the status of each step (what the stepper shows).

    Args:
        state: The shared state of the application.
        project: The project.

    Returns:
        The project with its origin, counts and the status of each step.
    """
    project_dir = state.store.dir(project.id)
    records = Manifest(project_dir).all()
    active = [r for r in records if r.status == ACTIVE]
    annotations = load_annotations(project_dir)
    ids = {r.id for r in active}
    # "annotated" = at least one reviewed label, as in the tasks and jobs; an image reviewed with no
    # objects is counted apart (empty) and an image with only pre-annotations waits for review
    images_map = annotations.images if annotations else {}
    labelled = annotated_ids(images_map, ids)
    empty = empty_ids(images_map, ids)
    preannotated = {i for i in ids if image_state(images_map, i) == "pre"}
    measured = [r for r in active if r.metrics]
    with_issues = [r for r in active if r.issues]
    if not measured:
        curation = "not_analyzed"
    elif len(measured) < len(active):
        curation = "outdated"  # images added after the last analysis
    else:
        curation = "analyzed"
    return {
        **project.as_dict(),
        "origin": read_origin(project_dir),  # where an imported project came from (None otherwise)
        "counts": {
            "images": len(records),
            "active": len(active),
            "excluded": len(records) - len(active),
            "annotated": len(labelled),
            "empty": len(empty),
            "preannotated": len(preannotated),
            "labels": sum(len(annotations.images[i]) for i in labelled) if annotations else 0,
            "with_issues": len(with_issues),
            "analyzed": len(measured),
            "tasks": len(TaskStore(project_dir).load()),
        },
        "steps": {
            "images": "done" if active else "empty",
            "annotations": "loaded" if labelled or empty or preannotated else "none",
            "curation": curation,
        },
    }


def _image_item(record: ImageRecord, labels: int) -> dict[str, Any]:
    """An image record as the image list shows it.

    Args:
        record: The image record.
        labels: How many labels the image has.

    Returns:
        The image as the interface lists it (without its file path).
    """
    item = record.as_dict()
    item.pop("file", None)
    item["labels"] = labels
    return item


# -- projects -----------------------------------------------------------------
@router.get("/projects")
def list_projects(state: AppState = STATE) -> list[dict[str, Any]]:
    """Every project with its counts, newest first.

    A project with a damaged file is still listed, with ``damaged`` (a readable message) and empty counts,
    so one bad file does not hide the others.

    Args:
        state: The shared state of the application.

    Returns:
        The overview of every project, newest first.
    """
    items = [_safe_overview(state, p) for p in state.store.list()]
    items += [_damaged_overview(i, message) for i, message in state.store.damaged().items()]
    return sorted(items, key=lambda item: item["id"], reverse=True)


def _damaged_overview(project_id: int, message: str, project: Project | None = None) -> dict[str, Any]:
    """Stand-in for a project that cannot be read: right shape, empty counts, ``damaged`` set.

    Args:
        project_id: Id of the project.
        message: The readable message about the damaged file.
        project: The project, when it could still be read.

    Returns:
        A project overview with empty counts and ``damaged`` set.
    """
    base = (
        project.as_dict()
        if project
        else {
            "id": project_id,
            "name": f"Project {project_id}",
            "description": "",
            "task": Task.CLASSIFICATION.value,
            "classes": [],
            "created": "",
            "updated": "",
        }
    )
    counts = dict.fromkeys(
        (
            "images",
            "active",
            "excluded",
            "annotated",
            "preannotated",
            "labels",
            "with_issues",
            "analyzed",
            "tasks",
        ),
        0,
    )
    steps = {"images": "empty", "annotations": "none", "curation": "not_analyzed"}
    return {**base, "origin": None, "counts": counts, "steps": steps, "damaged": message}


def _safe_overview(state: AppState, project: Project) -> dict[str, Any]:
    """The overview of a project, even when one of its files is damaged.

    Args:
        state: The shared state of the application.
        project: The project.

    Returns:
        The overview, or a stand-in with ``damaged`` set when a file of the project cannot be read.
    """
    try:
        return project_overview(state, project)
    except DamagedFile as exc:
        return _damaged_overview(project.id, str(exc), project)


@router.post("/projects")
def create_project(body: NewProject, state: AppState = STATE) -> dict[str, Any]:
    """Create a project (the id is automatic).

    Args:
        body: Name, description, task and classes of the new project.
        state: The shared state of the application.

    Returns:
        The project (see :func:`project_overview`).
    """
    try:
        check_class_spelling(body.classes)
        project = state.store.create(body.name, body.description, body.task, body.classes)
    except DamagedFile:
        raise  # a damaged file is not the caller's mistake: answered by the error handler
    except ValueError as exc:
        raise bad_request(exc) from exc
    return project_overview(state, project)


@router.get("/projects/{project_id}")
def get_project(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """One project with counts and step status.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The project (see :func:`project_overview`).
    """
    return project_overview(state, state.project(project_id))


def _counts(annotations: AnnotationSet) -> tuple[int, int]:
    """Images with labels and labels of a ground truth.

    Args:
        annotations: The ground truth.

    Returns:
        ``(images with labels, labels)``.
    """
    return sum(1 for v in annotations.images.values() if v), sum(len(v) for v in annotations.images.values())


def _classes_used(annotations: AnnotationSet, keep: set[str]) -> dict[str, dict[str, Any]]:
    """Classes of the ground truth that are not in ``keep`` (lower case): name -> labels and image ids.

    Args:
        annotations: The ground truth.
        keep: Class names to keep (lower case).

    Returns:
        The other classes the labels use, each with its label count and image ids.
    """
    used: dict[str, dict[str, Any]] = {}
    for image_id, labels in annotations.images.items():
        for lb in labels:
            if lb.label.lower() not in keep:
                entry = used.setdefault(lb.label, {"labels": 0, "ids": set()})
                entry["labels"] += 1
                entry["ids"].add(image_id)
    return used


def _with_classes(annotations: AnnotationSet, project: Project) -> AnnotationSet:
    """The ground truth with the class list of the project (so ``class_names`` never lags behind).

    Labels whose class differs from a project class only in case take the project's spelling: a class
    renamed that way (``cat`` -> ``Cat``) keeps its labels, and a version never gets both names.

    Args:
        annotations: The ground truth; changed in place.
        project: The project, with its class list already updated.

    Returns:
        ``annotations``, with the project's class list and spelling.
    """
    annotations.classes = list(project.classes)
    spelling = {c.lower(): c for c in project.classes}
    for labels in annotations.images.values():
        for label in labels:
            label.label = spelling.get(label.label.lower(), label.label)
    return annotations


@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: ProjectChanges, state: AppState = STATE) -> dict[str, Any]:
    """Change name, description, task or classes.

    Changing the task of a project whose ground truth is of another task needs ``annotations``:
    ``convert`` (when possible) or ``discard`` (a backup copy is kept). Without it the answer is
    HTTP 409 with ``detail`` describing the options, and nothing changes.

    Removing classes that the ground truth still uses needs ``classes_action: "remove_annotations"``
    (a backup copy is kept); otherwise HTTP 409 ``classes_in_use``. Adding or reordering classes never
    needs a decision. The task decision is checked first.

    The change waits for no running job (HTTP 409). Changing the task or the classes also waits for
    training, ONNX export and pre-annotation of the project (HTTP 409 ``busy_training``): they read both.
    It holds the annotations lock from reading
    ``annotations.json`` to writing it (backup included), so no other writer is lost.

    Args:
        project_id: Id of the project.
        body: The fields to change (name, description, task, classes and what to do with the annotations).
        state: The shared state of the application.

    Returns:
        The project (see :func:`project_overview`).
    """
    project = state.project(project_id)
    if body.classes is not None:
        try:
            check_class_spelling(body.classes, project.classes)
        except ValueError as exc:
            raise bad_request(exc) from exc
    changes_labels = (body.task is not None and body.task is not project.task) or (
        body.classes is not None and clean_classes(body.classes) != project.classes
    )
    state.busy(project_id, training=changes_labels)  # a run or pre-annotation reads the type and classes
    with annotations_lock(state.store.dir(project_id)):
        return _update_project_locked(project_id, body, state)


def _update_project_locked(project_id: int, body: ProjectChanges, state: AppState) -> dict[str, Any]:
    """Change a project while the annotations lock is held (see :func:`update_project`).

    Args:
        project_id: Id of the project.
        body: The fields to change.
        state: The shared state of the application.

    Returns:
        The project (see :func:`project_overview`).
    """
    project = state.project(project_id)
    changes = body.model_dump(exclude_none=True, exclude={"annotations", "classes_action"})
    project_dir = state.store.dir(project_id)
    ground_truth = load_annotations(project_dir) if "task" in changes else None
    result: dict[str, Any] | None = None
    converted: AnnotationSet | None = None
    if ground_truth and ground_truth.images and ground_truth.task is not body.task and body.task is not None:
        images, labels = _counts(ground_truth)
        old, new = ground_truth.task, body.task
        possible = ann.conversion_possible(old, new)
        if body.annotations is None:
            raise HTTPException(
                409,
                {
                    "code": "task_change_annotations",
                    "message": (
                        f"This project has annotations for {old.value.capitalize()} ({images} images). "
                        f"Changing the task to {new.value.capitalize()} needs a decision about them."
                    ),
                    "from_task": old.value,
                    "to_task": new.value,
                    "images": images,
                    "labels": labels,
                    "options": ["convert", "discard"] if possible else ["discard"],
                },
            )
        if body.annotations == "convert":
            converted = ann.convert_annotations(ground_truth, new)
            if converted is None:
                raise ApiError(
                    400,
                    "annotations_not_convertible",
                    f"Annotations for {old.value} cannot be converted to {new.value}.",
                    from_task=old.value,
                    to_task=new.value,
                )
            result = dict(zip(("images", "labels"), _counts(converted), strict=True))
        else:
            result = {"images": images, "labels": labels}
        result = {"action": body.annotations, **result}
    classes_result: dict[str, int] | None = None
    cleaned: AnnotationSet | None = None
    if body.classes is not None and not (result is not None and converted is None):  # not after a discard
        current = converted or load_annotations(project_dir)
        keep = {c.lower() for c in body.classes}
        used = _classes_used(current, keep) if current else {}
        if used:
            if body.classes_action is None:
                names = ", ".join(used)
                n_labels = sum(u["labels"] for u in used.values())
                n_images = len({i for u in used.values() for i in u["ids"]})
                raise HTTPException(
                    409,
                    {
                        "code": "classes_in_use",
                        "message": (
                            f"The classes {names} are used by {n_labels} annotations in {n_images} images."
                            if len(used) > 1
                            else f"The class {names} is used by {n_labels} annotations in {n_images} images."
                        ),
                        "classes": [
                            {"name": n, "labels": u["labels"], "images": len(u["ids"])}
                            for n, u in used.items()
                        ],
                        "options": ["remove_annotations"],
                    },
                )
            assert current is not None
            kept_images = {k: [lb for lb in v if lb.label.lower() in keep] for k, v in current.images.items()}
            removed = sum(len(v) for v in current.images.values()) - sum(len(v) for v in kept_images.values())
            touched = sum(1 for k, v in kept_images.items() if len(v) != len(current.images[k]))
            cleaned = AnnotationSet(current.task, current.classes, kept_images, current.source)
            classes_result = {"removed_labels": removed, "images": touched}
    try:
        project = state.store.update(project_id, **changes)
    except ValueError as exc:
        raise bad_request(exc) from exc
    if converted is not None:
        save_annotations(project_dir, _with_classes(cleaned or converted, project))
    elif result is not None:
        make_backup(project_dir, "task")
        annotations_path(project_dir).unlink()
    elif cleaned is not None:
        make_backup(project_dir, "classes")
        save_annotations(project_dir, _with_classes(cleaned, project))
    elif body.classes is not None or body.task is not None:
        current = load_annotations(project_dir)  # one without images needed no decision: it just follows
        if current is not None and (current.classes != project.classes or current.task is not project.task):
            current.task = project.task
            save_annotations(project_dir, _with_classes(current, project))
    overview = project_overview(state, project)
    if result is not None:
        overview["annotations_result"] = result
    if classes_result is not None:
        overview["classes_result"] = classes_result
    return overview


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, body: DeleteProject, state: AppState = STATE) -> dict[str, Any]:
    """Delete a project with all its files. The name must be typed again.

    FiftyOne is only touched when the project has a recorded dataset (``fiftyone.json``).

    Args:
        project_id: Id of the project.
        body: The project name, typed again to confirm.
        state: The shared state of the application.

    Returns:
        ``{deleted: project_id}``.
    """
    project = state.project(project_id)
    if body.confirm_name.strip() != project.name:
        raise ApiError(400, "name_mismatch", "The name does not match; nothing was deleted.")
    state.busy(project_id, training=True)
    project_dir = state.store.dir(project_id)
    recorded = recorded_name(project_dir)
    state.store.delete(project_id)
    if recorded and FIFTYONE:  # a project that never used FiftyOne has nothing there: do not start it
        drop_fiftyone_dataset(project_dir, project_id, recorded)
    return {"deleted": project_id}


def drop_fiftyone_dataset(project_dir: Path, project_id: int, recorded: str) -> None:
    """Delete the project's recorded FiftyOne dataset in the background (starting it can take seconds).

    Args:
        project_dir: Folder of the project.
        project_id: Id of the project.
        recorded: Name of the FiftyOne dataset recorded for the project.
    """
    threading.Thread(
        target=fiftyone_bridge.delete_dataset,
        args=(project_dir, project_id, recorded),
        name="activelabel-drop-dataset",
        daemon=True,
    ).start()


# -- images -------------------------------------------------------------------
def ingest_uploads(manifest: Manifest, files: list[UploadFile]) -> tuple[dict[str, Any], list[str]]:
    """Store uploaded files with the ingest code; returns the report and the ids actually added, in order.

    Args:
        manifest: The image manifest of the project.
        files: The uploaded files.

    Returns:
        ``(report, ids)``: the ingest report and the ids actually added, in order.
    """
    report = ingest_files(manifest, [(f.filename or "image", f.file) for f in files])
    return report.as_dict(), report.ids


@router.post("/projects/{project_id}/images")
def upload_images(project_id: int, files: UPLOADS, state: AppState = STATE) -> dict[str, Any]:
    """Add uploaded images and .zip archives as a new task (the interface sends them in chunks).

    Args:
        project_id: Id of the project.
        files: The uploaded files.
        state: The shared state of the application.

    Returns:
        The ingest report.
    """
    manifest = state.manifest(project_id)
    tasks = TaskStore(state.store.dir(project_id))
    with tasks.locked():  # ingest and registration are atomic
        tasks.ensure()  # migrate before the new images exist, so they do not end up in "Task 1"
        report, added = ingest_uploads(manifest, files)
        if added:
            task = tasks.create(images=added)
            report["task"] = task.id
    return report


def _import_folder_job(
    manifest: Manifest, tasks: TaskStore, folder: Path, root: Path | None, progress: Progress
) -> dict[str, Any]:
    """Body of the ``import_folder`` job: ingest the folder and put its images in a new task.

    Args:
        manifest: The image manifest of the project.
        tasks: The task store of the project.
        folder: The folder to import (already checked).
        root: The import root the folder must be inside of (``None`` when not restricted).
        progress: Where the job reports its progress.

    Returns:
        The ingest report, plus ``task`` (the new task id) when any image was added.
    """
    with tasks.locked():
        tasks.ensure()
        ingested = ingest_folder(manifest, folder, root, progress)
        result = ingested.as_dict()
        if ingested.ids:
            result["task"] = tasks.create(images=ingested.ids).id
        return result


@router.post("/projects/{project_id}/images/folder")
def import_folder(project_id: int, body: FolderImport, state: AppState = STATE) -> dict[str, Any]:
    """Import a folder of this machine (only inside ACTIVELABEL_IMPORT_ROOT) as a new task, as a job.

    A folder that is off limits or missing is refused at once (HTTP 400).

    Args:
        project_id: Id of the project.
        body: The folder to import.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    manifest = state.manifest(project_id)
    root = state.settings.import_root
    try:
        folder = check_folder(Path(body.path).expanduser(), root)  # refused now, not later inside the job
    except ValueError as exc:
        raise bad_request(exc) from exc
    tasks = TaskStore(state.store.dir(project_id))
    job = partial(_import_folder_job, manifest, tasks, folder, root)
    return state.jobs.submit(project_id, "import_folder", job).as_dict()


def _uniqueness_scores(state: AppState, project_id: int) -> dict[str, float]:
    """The saved uniqueness scores of a project (empty before the first scoring).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        Image id to score (0 to 1).
    """
    scores = uniqueness.UniquenessStore(state.store.dir(project_id)).read().get("scores")
    return dict(scores) if isinstance(scores, dict) else {}


def _map_group(state: AppState, project_id: int) -> dict[str, int]:
    """The group of each image on the similarity map (empty without a map).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        Image id to group (``-1``: no group).
    """
    points = similarity.SimilarityStore(state.store.dir(project_id)).read().get("points")
    if not isinstance(points, dict):
        return {}
    return {str(i): int(g) for i, g in zip(points.get("ids", []), points.get("group", []), strict=False)}


@router.get("/projects/{project_id}/images")
def list_images(
    project_id: int,
    status: str = Query("active", pattern="^(active|excluded|all)$"),
    issue: str = "",
    ids: str = "",
    task: int | None = None,
    job: int | None = None,
    dataset: int | None = None,
    part: str | None = None,
    fold: int | None = None,
    sort: str = Query("", pattern="^(|uniqueness_desc|uniqueness_asc)$"),
    unique_below: float | None = Query(None, ge=0, le=1),
    group: int | None = Query(None, ge=-1),
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=MAX_PAGE),
    state: AppState = STATE,
) -> dict[str, Any]:
    """A page of images, filtered by status, issue, task, job or a comma-separated list of ids.

    ``issue`` is one flag (blurry, dark, ...) or ``any`` for every image with at least one flag. In a
    classification project every item also has ``classes``: the class names of its ground truth.

    With ``task`` the items are in task order; ``job`` (1-based) needs ``task``. ``dataset`` keeps the
    images of a dataset version, of one ``part`` (train, val, test) or ``fold`` (0-based).

    Args:
        project_id: Id of the project.
        status: ``active``, ``excluded`` or ``all``.
        issue: Keep the images flagged with this issue.
        ids: Comma-separated image ids to keep.
        task: Keep the images of this task.
        job: Keep the images of this job (1-based; needs ``task``).
        dataset: Keep the images of this dataset version.
        part: With ``dataset``: ``train``, ``val`` or ``test``.
        fold: With ``dataset``: keep this fold (0-based).
        sort: ``uniqueness_desc`` (most unique first) or ``uniqueness_asc`` (most repeated first), by the
            saved uniqueness scores; images without a score come last. Empty keeps the order of the project.
        unique_below: Keep the images whose uniqueness is below this value (0 to 1); images without a score
            are left out.
        group: Keep the images of this group of the similarity map (``-1``: the ones HDBSCAN left without a
            group); an image that is not on the map is left out.
        offset: Items to skip.
        limit: Items to return.
        state: The shared state of the application.

    Returns:
        ``{total, items}``: how many images match and the requested page; an item has ``uniqueness`` when
        it was scored.
    """
    manifest = state.manifest(project_id)
    records = manifest.all()
    if job is not None and task is None:
        raise ApiError(400, "task_required", "Give the task of the job.")
    if (part is not None or fold is not None) and dataset is None:
        raise ApiError(400, "dataset_required", "Give the dataset version of the part or fold.")
    if task is not None:
        try:
            chosen = TaskStore(state.store.dir(project_id)).get(task)
        except TaskNotFound:
            raise ApiError(404, "task_not_found", "This task does not exist.") from None
        order = chosen.images
        if job is not None:
            found = [j for j in task_jobs(chosen) if j.n == job]
            if not found:
                raise ApiError(404, "job_not_found", "This job does not exist.")
            order = found[0].image_ids
        by_id = {r.id: r for r in records}
        records = [by_id[i] for i in order if i in by_id]
    if dataset is not None:
        try:
            members = DatasetStore(state.store.dir(project_id)).image_ids(dataset, part, fold)
        except DatasetNotFound:
            raise ApiError(404, "dataset_not_found", "This dataset version does not exist.") from None
        except ValueError as error:
            raise bad_request(error) from None
        if task is None:
            by_id = {r.id: r for r in records}
            records = [by_id[i] for i in members if i in by_id]
        else:
            keep = set(members)
            records = [r for r in records if r.id in keep]
    if status != "all":
        records = [r for r in records if r.status == status]
    if issue == "any":
        records = [r for r in records if r.issues]
    elif issue:
        records = [r for r in records if issue in r.issues]
    if ids:
        wanted = set(ids.split(","))
        records = [r for r in records if r.id in wanted]
    if group is not None:
        in_group = _map_group(state, project_id)
        records = [r for r in records if in_group.get(r.id) == group]
    scores = _uniqueness_scores(state, project_id)
    if unique_below is not None:
        records = [r for r in records if r.id in scores and scores[r.id] < unique_below]
    if sort:
        sign = -1 if sort == "uniqueness_desc" else 1
        records = sorted(records, key=lambda r: (r.id not in scores, sign * scores.get(r.id, 0.0)))
    annotations = load_annotations(state.store.dir(project_id))
    counts = {k: len(v) for k, v in annotations.images.items()} if annotations else {}
    classified = annotations is not None and annotations.task == Task.CLASSIFICATION
    duplicate_of = issues.load_duplicate_of(state.store.dir(project_id))
    names = {r.id: r.name for r in manifest.all()} if duplicate_of else {}
    page = records[offset : offset + limit]
    items = []
    for record in page:
        item = _image_item(record, counts.get(record.id, 0))
        if classified and annotations:
            item["classes"] = sorted({lb.label for lb in annotations.images.get(record.id, [])})
        if record.id in scores:
            item["uniqueness"] = scores[record.id]
        original = duplicate_of.get(record.id)
        if original and "near_duplicate" in record.issues:
            item["duplicate_of"], item["duplicate_of_name"] = original, names.get(original, "")
        items.append(item)
    return {"total": len(records), "items": items}


def _record(state: AppState, project_id: int, image_id: str) -> tuple[Any, ImageRecord]:
    """The manifest and the record of an image, or HTTP 404.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        image_id: Id of the image.

    Returns:
        ``(manifest, record)`` of the image.
    """
    manifest = state.manifest(project_id)
    try:
        return manifest, manifest.get(image_id)
    except KeyError:
        raise ApiError(404, "image_not_found", "This image is not in the project.") from None


@router.get("/projects/{project_id}/images/{image_id}/thumb")
def image_thumb(project_id: int, image_id: str, state: AppState = STATE) -> FileResponse:
    """Thumbnail (JPEG, made on first request).

    Args:
        project_id: Id of the project.
        image_id: Id of the image.
        state: The shared state of the application.

    Returns:
        The thumbnail as a JPEG file response.
    """
    manifest, record = _record(state, project_id, image_id)
    return FileResponse(thumbnail(manifest, record), headers={"Cache-Control": "max-age=86400"})


@router.get("/projects/{project_id}/images/{image_id}/file")
def image_file(project_id: int, image_id: str, state: AppState = STATE) -> FileResponse:
    """The original file.

    Args:
        project_id: Id of the project.
        image_id: Id of the image.
        state: The shared state of the application.

    Returns:
        The original file.
    """
    manifest, record = _record(state, project_id, image_id)
    return FileResponse(manifest.file_path(record), filename=Path(record.name).name)


@router.post("/projects/{project_id}/images/exclude")
def exclude_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Leave images out of the project (reversible).

    Args:
        project_id: Id of the project.
        body: The image ids and the reason.
        state: The shared state of the application.

    Returns:
        ``{changed}``: how many images changed.
    """
    return {"changed": state.manifest(project_id).set_status(body.ids, EXCLUDED, body.reason)}


@router.post("/projects/{project_id}/images/restore")
def restore_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Bring excluded images back.

    Args:
        project_id: Id of the project.
        body: The image ids.
        state: The shared state of the application.

    Returns:
        ``{changed}``: how many images changed.
    """
    return {"changed": state.manifest(project_id).set_status(body.ids, ACTIVE)}


@router.post("/projects/{project_id}/images/delete")
def delete_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Delete images and their files for good (also their annotations and thumbnails).

    Args:
        project_id: Id of the project.
        body: The image ids and the confirmation.
        state: The shared state of the application.

    Returns:
        ``{deleted}``: how many images were deleted.
    """
    if not body.confirm:
        raise ApiError(400, "confirm_required", "Deleting cannot be undone: confirm it first.")
    return {"deleted": delete_image_files(state, project_id, body.ids)}


def delete_image_files(state: AppState, project_id: int, ids: list[str]) -> int:
    """Delete images for good: files, thumbnails, annotations, tags and their place in a task.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        ids: The images to delete.

    Returns:
        How many images were deleted.
    """
    state.busy(project_id, training=True)
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    tasks = TaskStore(project_dir)
    with tasks.locked():
        tasks.load()  # migrate first, then take the images out
        known = manifest.load()
        ids = [i for i in dict.fromkeys(ids) if i in known]  # only this project's images reach the disk
        deleted = manifest.delete(ids)
        tasks.remove_images(ids)
    for image_id in ids:
        (project_dir / "thumbs" / f"{image_id}.jpg").unlink(missing_ok=True)
    with annotations_lock(project_dir):
        annotations = load_annotations(project_dir)
        if annotations and any(i in annotations.images for i in ids):
            for image_id in ids:
                annotations.images.pop(image_id, None)
            save_annotations(project_dir, annotations)
    forget_images(project_dir, ids)
    return deleted


# -- jobs -----------------------------------------------------------------------
@router.get("/projects/{project_id}/jobs")
def project_jobs(project_id: int, state: AppState = STATE) -> list[dict[str, Any]]:
    """Running and recent jobs of a project, newest first.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The jobs of the project, newest first.
    """
    state.project(project_id)
    return [j.as_dict() for j in state.jobs.list(project_id)]


@router.delete("/projects/{project_id}/jobs")
def clear_jobs(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Remove the finished jobs of a project from the list (running ones stay).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{removed}``: how many finished jobs were removed from the list.
    """
    state.project(project_id)
    return {"removed": state.jobs.clear(project_id)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: AppState = STATE) -> dict[str, Any]:
    """One job, of any project (including imports, which belong to project 0).

    Args:
        job_id: Id of the job.
        state: The shared state of the application.

    Returns:
        The job.
    """
    try:
        return state.jobs.get(job_id).as_dict()
    except KeyError:
        raise ApiError(404, "activity_not_found", JOB_GONE) from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Ask a job to stop.

    Args:
        job_id: Id of the job.
        state: The shared state of the application.

    Returns:
        The job, asked to stop.
    """
    try:
        job: Job = state.jobs.cancel(job_id)
    except KeyError:
        raise ApiError(404, "activity_not_found", JOB_GONE) from None
    return job.as_dict()


# -- annotations ----------------------------------------------------------------
class ApplyAnnotations(BaseModel):
    """Body of POST /projects/{id}/annotations/{token}/apply."""

    action: str
    mode: str = "replace"


@router.post("/projects/{project_id}/annotations")
def check_annotations(
    project_id: int,
    files: UPLOADS,
    task: Annotated[int | None, Form()] = None,
    class_name: Annotated[str, Form()] = "",
    state: AppState = STATE,
) -> dict[str, Any]:
    """Keep the uploaded annotation files aside and say what is in them (nothing is imported yet).

    ``task`` is a video task: its images are then matched by frame number (docs/VIDEO.md). Without it,
    the project's only video task is used by the formats made for video (CVAT for video, MOT).
    ``class_name`` names the objects of MOT files that come without a labels file. When the files cannot be
    imported at all (``can_import`` false) they are not kept: the report still comes back, and discarding
    its token is harmless.

    Args:
        project_id: Id of the project.
        files: The uploaded files.
        task: Id of the video task the files belong to, when they are for a video.
        class_name: Name of the class of the objects, for formats without class names.
        state: The shared state of the application.

    Returns:
        The report of what is in the files, with the ``token`` that identifies them and whether the
        project already has annotations.
    """
    project = state.project(project_id)
    manifest = state.manifest_with_images(project_id)
    project_dir = state.store.dir(project_id)
    try:
        target = ann.build_target(project_dir, manifest, task, class_name)
    except ValueError as exc:
        raise bad_request(exc) from exc
    if task is not None and target is None:
        raise bad_request(ValueError("Choose a task made from a video."))
    try:
        token = ann.stage(project_dir, [(f.filename or "annotations", f.file) for f in files])
    except ValueError as exc:
        raise bad_request(exc) from exc
    staged = ann.load_staged(project_dir, token)
    try:
        fmt = ann.detect_format(staged)
    except (ValueError, ArithmeticError) as exc:  # a parser tripping on a malformed file
        ann.discard_staged(project_dir, token)
        raise _unreadable(str(exc)) from exc
    if fmt is None:
        broken = ann.unreadable_file(staged)
        ann.discard_staged(project_dir, token)
        if broken is not None:
            raise _malformed(broken)
        raise ApiError(
            400,
            "annotation_format_unknown",
            "The annotation format was not recognised. Supported: COCO JSON, YOLO txt (with data.yaml or "
            "classes.txt), Pascal VOC XML, CVAT for images or video XML, MOT txt and CSV or JSON values for "
            "regression.",
        )
    if target is not None and task is None and fmt not in ann.VIDEO_ONLY:
        target = None  # image files are matched by name unless the task was chosen
    if fmt in ann.VIDEO_ONLY and target is None:
        ann.discard_staged(project_dir, token)
        raise ApiError(
            400,
            "video_task_required",
            "This file is made for a video: choose the video task it belongs to (none or several found).",
        )
    if target is not None:
        ann.set_target(project_dir, token, {"task": target.task, "class_name": target.class_name})
    try:
        result = ann.parse(
            staged, fmt, target.images() if target else manifest.active(), target, manifest.excluded()
        )
        report = ann.validate(result, project)
    except (ValueError, ArithmeticError) as exc:  # a parser tripping on a malformed file
        ann.discard_staged(project_dir, token)
        raise _unreadable(str(exc)) from exc
    if not report.can_import:  # the only option left is to discard: nothing to keep aside
        ann.discard_staged(project_dir, token)
    out = {
        **report.as_dict(),
        "token": token,
        "has_annotations": load_annotations(project_dir) is not None,
        "task": target.task if target else None,
    }
    if report.can_import:  # the files are kept aside: the report stays until they are applied or discarded
        ann.set_report(project_dir, token, out)
    return out


@router.get("/projects/{project_id}/annotations/pending")
def pending_annotations_report(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """The report of the last annotation check that was neither applied nor discarded.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{report: <the report as POST /annotations returned it, or null>}``.
    """
    state.project(project_id)
    return {"report": ann.pending_report(state.store.dir(project_id))}


@router.post("/projects/{project_id}/annotations/{token}/apply")
def apply_annotations(
    project_id: int, token: str, body: ApplyAnnotations, state: AppState = STATE
) -> dict[str, Any]:
    """Import (or discard) staged annotation files as the user decided.

    Args:
        project_id: Id of the project.
        token: Token of the staged files.
        body: What to do (action and mode).
        state: The shared state of the application.

    Returns:
        The counts of what was imported.
    """
    manifest = state.manifest(project_id)
    state.busy(project_id)
    try:
        return ann.apply(state.store, project_id, manifest, token, body.action, body.mode)
    except FileNotFoundError as exc:  # unknown, discarded or expired staging
        raise ApiError(404, "staging_not_found", str(exc)) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


@router.delete("/projects/{project_id}/annotations/{token}")
def discard_annotations(project_id: int, token: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw away staged annotation files.

    Args:
        project_id: Id of the project.
        token: Token of the staged files.
        state: The shared state of the application.

    Returns:
        ``{discarded: true}``.
    """
    state.project(project_id)
    try:
        ann.discard_staged(state.store.dir(project_id), token)
    except ValueError as exc:
        raise bad_request(exc) from exc
    return {"discarded": True}


@router.get("/projects/{project_id}/annotations")
def annotations_summary(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """What the project ground truth has: task, classes, labels per class.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The task, classes and counts of the ground truth, or ``{loaded: false}``.
    """
    project = state.project(project_id)
    ground_truth = load_annotations(state.store.dir(project_id))
    if ground_truth is None:
        return {"loaded": False}
    active = {r.id for r in state.manifest(project_id).active()}
    per_class: dict[str, int] = {}
    images = 0
    waiting = 0
    empty = 0
    for image_id, labels in ground_truth.images.items():
        if image_id not in active:
            continue
        if image_state(ground_truth.images, image_id) == "pre":
            waiting += 1  # pre-annotations: not counted until reviewed
            continue
        if not labels:
            empty += 1  # reviewed with no objects: counted apart
            continue
        images += 1
        for label in labels:
            per_class[label.label] = per_class.get(label.label, 0) + 1
    return {
        "loaded": True,
        "task": ground_truth.task.value,
        "task_matches": ground_truth.task is project.task,
        "class_names": ground_truth.classes,
        "images_annotated": images,
        "images_empty": empty,
        "images_preannotated": waiting,
        "labels_total": sum(per_class.values()),
        "labels_per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "source": {
            k: v for k, v in ground_truth.source.items() if k in ("format", "saved", "files", "last_session")
        },
    }


# -- curation -------------------------------------------------------------------
class Analyze(BaseModel):
    """Body of POST /projects/{id}/curation/analyze."""

    recompute: bool = False


def _curation(state: AppState, project_id: int) -> dict[str, Any]:
    """The curation summary of a project with its thresholds and duplicates.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The curation summary with the thresholds, their defaults and the duplicates.
    """
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    return {
        **curation_summary.summary(manifest),
        "thresholds": asdict(issues.load_thresholds(project_dir)),
        "defaults": asdict(issues.Thresholds()),
        "duplicate_of": issues.load_duplicate_of(project_dir),
    }


def _analyze_job(
    manifest: Manifest, project_dir: Path, recompute: bool, progress: Progress
) -> dict[str, Any]:
    """Body of the ``analyze`` job: check the files, measure the images, then flag the issues.

    Args:
        manifest: The image manifest of the project.
        project_dir: Folder of the project (where the thresholds are saved).
        recompute: Measure every image again, also those already measured.
        progress: Where the job reports its progress.

    Returns:
        ``{measured, issues, step}``: how many images were measured and the count per issue (with the
        images whose file is missing or changed on disk, #620).
    """
    issues.flag_files(manifest, progress)
    measured = quality.measure_images(manifest, progress, recompute=recompute)
    progress.update(message="Flagging issues")
    counts = issues.flag_issues(manifest, issues.load_thresholds(project_dir))
    return {"measured": measured, "issues": counts, "step": "curation"}


@router.post("/projects/{project_id}/curation/analyze")
def analyze(project_id: int, body: Analyze | None = None, state: AppState = STATE) -> dict[str, Any]:
    """Measure the images and flag issues, as a job (it can be cancelled).

    Args:
        project_id: Id of the project.
        body: Whether to measure every image again.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    manifest = state.manifest_with_images(project_id)
    return _queue_analysis(state, project_id, manifest, bool(body and body.recompute)).as_dict()


def _queue_analysis(state: AppState, project_id: int, manifest: Manifest, recompute: bool) -> Job:
    """Queue the curation analysis of a project on the processor lane.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        manifest: The image manifest of the project.
        recompute: Measure every image again, also those already measured.

    Returns:
        The queued job, or the analysis of the project already waiting or running (asking twice does not
        queue the same work twice, #603).
    """
    waiting = [j for j in state.jobs.active(project_id) if j.kind == "analyze"]
    if waiting:
        return waiting[0]
    job = partial(_analyze_job, manifest, state.store.dir(project_id), recompute)
    return state.jobs.submit(project_id, "analyze", job, {"recompute": recompute}, device=CPU)


def restore_analyze(state: AppState, entry: dict[str, Any]) -> None:
    """Queue again an analysis saved in ``queue.json`` (skipped when the project or its images are gone).

    Args:
        state: The shared state of the application.
        entry: The saved queue entry (``project_id`` and ``params`` with ``recompute``).
    """
    project_id = int(entry.get("project_id", 0))
    try:
        manifest = state.manifest_with_images(project_id)
    except (ApiError, ProjectNotFound):
        LOG.info(
            "The saved analysis of project %s was dropped: the project or its images are gone", project_id
        )
        return
    params = entry.get("params")
    recompute = bool(params.get("recompute")) if isinstance(params, dict) else False
    _queue_analysis(state, project_id, manifest, recompute)


@router.delete("/projects/{project_id}/curation")
def reset_curation(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Forget the analysis (measures, issues, hashes, duplicates); thresholds stay.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{reset}``: how many images were reset.
    """
    manifest = state.manifest(project_id)
    state.busy(project_id)
    return {"reset": issues.reset_curation(manifest)}


@router.get("/projects/{project_id}/curation")
def curation_overview(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Counts per issue, ids per issue, histograms and thresholds.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        The curation summary (see :func:`_curation`).
    """
    return _curation(state, project_id)


@router.put("/projects/{project_id}/curation/thresholds")
def set_thresholds(project_id: int, body: dict[str, float], state: AppState = STATE) -> dict[str, Any]:
    """Change thresholds and flag the images again (no new measuring).

    Args:
        project_id: Id of the project.
        body: The thresholds to change.
        state: The shared state of the application.

    Returns:
        The curation summary (see :func:`_curation`).
    """
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    current = asdict(issues.load_thresholds(project_dir))
    unknown = set(body) - set(current)
    if unknown:
        raise ApiError(
            400,
            "unknown_thresholds",
            f"Unknown thresholds: {', '.join(sorted(unknown))}",
            names=sorted(unknown),
        )
    try:
        thresholds = issues.Thresholds(**{**current, **body})
    except (TypeError, ValueError) as exc:
        raise bad_request(exc) from exc
    issues.save_thresholds(project_dir, thresholds)
    issues.flag_issues(manifest, thresholds)
    return _curation(state, project_id)


@router.get("/projects/{project_id}/curation/suggested")
def suggested(project_id: int, state: AppState = STATE) -> dict[str, list[str]]:
    """Images that are probably worth excluding (never excluded automatically).

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{ids}``: the images that are probably worth excluding.
    """
    return {"ids": curation_summary.suggest_exclusions(state.manifest(project_id))}


# -- FiftyOne -------------------------------------------------------------------
class OpenFiftyOne(BaseModel):
    """Body of POST /projects/{id}/fiftyone."""

    view: str | None = None
    task: int | None = None  # open this task (saved view ``task-<id>``)
    job: int | None = None  # with ``task``: only this job (view built on demand)


def _open_fiftyone_job(
    project: Project,
    manifest: Manifest,
    project_dir: Path,
    view: str | None,
    task_id: int | None,
    job_no: int | None,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``fiftyone`` job: refresh the project's dataset and launch the FiftyOne App.

    Args:
        project: The project to open.
        manifest: The image manifest of the project.
        project_dir: Folder of the project.
        view: Name of the view to open (the whole dataset when ``None``).
        task_id: Restrict the view to this task, when given.
        job_no: Restrict the view to this job of the task, when given.
        progress: Where the job reports its progress.

    Returns:
        ``{url, view, step}``: the address of the App and the view opened.
    """
    progress.update(done=0, total=PROGRESS_STEPS, message="Preparing the FiftyOne dataset")
    dataset = fiftyone_bridge.sync_dataset(
        project,
        manifest,
        load_annotations(project_dir),
        on_progress=progress.share(0, DATASET_STEPS, "Preparing the FiftyOne dataset"),
    )
    progress.update(done=DATASET_STEPS + 5, message="Opening the FiftyOne App")
    url = fiftyone_bridge.launch_app(dataset, view, task=task_id, job=job_no)
    progress.update(done=PROGRESS_STEPS)
    return {"url": url, "view": view or "", "step": "curation"}


@router.post("/projects/{project_id}/fiftyone")
def open_fiftyone(
    project_id: int, body: OpenFiftyOne | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Refresh the project's FiftyOne dataset and open the App (job; returns the URL as its result).

    Args:
        project_id: Id of the project.
        body: The view, task and job to open.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    project = state.project(project_id)
    need_fiftyone()
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    view = body.view if body else None
    task_id = body.task if body else None
    job_no = body.job if body else None
    if task_id is not None:
        try:
            TaskStore(project_dir).get(task_id)
        except TaskNotFound:
            raise ApiError(404, "task_not_found", "This task does not exist.") from None
    job = partial(_open_fiftyone_job, project, manifest, project_dir, view, task_id, job_no)
    return state.jobs.submit(project_id, "fiftyone", job).as_dict()


@router.post("/projects/{project_id}/fiftyone/pull-exclusions")
def pull_exclusions(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Exclude the images tagged ``exclude`` in the FiftyOne App.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{excluded}``: how many images were excluded.
    """
    state.project(project_id)
    need_fiftyone()
    manifest = state.manifest(project_id)
    dataset = fiftyone_bridge.existing_dataset(state.store.dir(project_id), project_id)
    if dataset is None:
        raise ApiError(
            400,
            "fiftyone_not_opened",
            "Open the project in FiftyOne first, tag images with 'exclude', then try again.",
        )
    return {"excluded": fiftyone_bridge.pull_exclusions(dataset, manifest)}
