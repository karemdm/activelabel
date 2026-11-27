"""HTTP API of tasks and jobs (see docs/TASKS.md)."""

from __future__ import annotations

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from activelabel.annotate import credentials
from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.images.video import VideoStore
from activelabel.importing.common import RemoteNotFound
from activelabel.labels import load_annotations
from activelabel.preannotate import annotated_ids
from activelabel.sync import links
from activelabel.sync.registry import get_remote
from activelabel.tasks_store import (
    STALE_UPLOAD_SECONDS,
    ImageTask,
    TaskNotFound,
    TaskStore,
    check_job_size,
    clean_name,
    jobs,
)
from activelabel.web.api import delete_image_files, ingest_uploads
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
FILES = Annotated[list[UploadFile], File()]


class TaskChanges(BaseModel):
    """Body of PATCH /projects/{id}/tasks/{tid}: only the given fields change."""

    name: str | None = None
    open: bool | None = None


class DeleteTask(BaseModel):
    """Body of DELETE /projects/{id}/tasks/{tid}."""

    confirm: bool = False


def task_view(task: ImageTask, active: set[str], annotated: set[str]) -> dict[str, Any]:
    """A task for the interface: counts and its jobs (``first``/``last`` are 1-based positions).

    Args:
        task: The task.
        active: Ids of the active images.
        annotated: Ids of the images with a reviewed label.

    Returns:
        The task with its counts and jobs.
    """
    ids = set(task.images)
    return {
        "id": task.id,
        "name": task.name,
        "created": task.created,
        "job_size": task.job_size,
        "open": task.open,
        "incomplete": task.incomplete,
        "source": task.source,
        "images": len(task.images),
        "active": len(ids & active),
        "annotated": len(ids & annotated),
        "jobs": [
            {
                "n": j.n,
                "first": j.first,
                "last": j.last,
                "images": len(j.image_ids),
                "active": len(set(j.image_ids) & active),
                "annotated": len(set(j.image_ids) & annotated),
            }
            for j in jobs(task)
        ],
    }


def _sets(state: AppState, project_id: int) -> tuple[set[str], set[str]]:
    """Ids of active images and of images with at least one reviewed label (pre-annotations wait).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        ``(active image ids, annotated image ids)``.
    """
    project_dir = state.store.dir(project_id)
    active = {r.id for r in Manifest(project_dir).all() if r.status == ACTIVE}
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    return active, annotated_ids(images, active)


def _view(state: AppState, project_id: int, task: ImageTask) -> dict[str, Any]:
    """A task of the project as the interface shows it.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        task: The task.

    Returns:
        The task as the interface shows it.
    """
    return task_view(task, *_sets(state, project_id))


def _store(state: AppState, project_id: int) -> TaskStore:
    """The task store of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The task store of the project.
    """
    state.project(project_id)
    return TaskStore(state.store.dir(project_id))


def _find(store: TaskStore, task_id: int) -> ImageTask:
    """A task by id, or HTTP 404.

    Args:
        store: The task store of the project.
        task_id: Id of the task.

    Returns:
        The task.
    """
    try:
        return store.get(task_id)
    except TaskNotFound:
        raise ApiError(404, "task_not_found", "This task does not exist.") from None


@router.get("/projects/{project_id}/tasks")
def list_tasks(project_id: int, refresh: bool = False, state: AppState = STATE) -> dict[str, Any]:
    """Every task of the project, in creation order.

    Args:
        project_id: Id of the project.
        refresh: Read the state of the CVAT jobs again instead of reusing a read of the last minute.
        state: The shared state of the application.

    Returns:
        ``{tasks}`` in creation order.
    """
    store = _store(state, project_id)
    store.close_stale(STALE_UPLOAD_SECONDS)  # uploads cut off since the last look are closed and marked
    active, annotated = _sets(state, project_id)
    views = [task_view(t, active, annotated) for t in store.load()]
    remote = remote_status(state, project_id, refresh)
    configured = credentials.is_configured(credentials.load_settings(state.settings.workspace, "cvat"))
    link = links.get_link(state.store.dir(project_id), "cvat")
    for view in views:
        found = remote.get(str(view["id"]))
        if found and found.get("missing"):  # CVAT answered without it: deleted there (#684)
            view["cvat"] = found["missing"]
            continue
        if found:
            view["remote"] = found
        view["cvat"] = "ready" if "remote" in view else cvat_reason(view["id"], configured, link)
    return {"tasks": views}


def cvat_reason(task_id: int, configured: bool, link: dict[str, Any] | None) -> str:
    """Why a task has no CVAT task to open.

    Args:
        task_id: the local task.
        configured: whether the CVAT connection is set up.
        link: the project's link with CVAT, or None.

    Returns:
        ``not_configured`` (no CVAT connection), ``not_linked`` (the project is not linked with a CVAT
        project), ``not_synchronised`` (the task is not in CVAT yet) or ``unreachable`` (it is, but CVAT
        did not answer). A task CVAT answered without is ``removed_remote`` (``remote_gone`` when the CVAT
        project itself is gone): :func:`remote_status` says so.
    """
    if not configured:
        return "not_configured"
    if not link:
        return "not_linked"
    return "unreachable" if str(task_id) in ((link.get("tasks")) or {}) else "not_synchronised"


REMOTE_TTL = 60.0  # seconds a read of the CVAT tasks/jobs (or its failure) is reused
REMOTE_TIMEOUT = 8.0  # seconds the task list waits for CVAT before showing the tasks without it
_remote_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="activelabel-cvat-status")
_remote_cache: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}


def remote_status(state: AppState, project_id: int, refresh: bool = False) -> dict[str, Any]:
    """CVAT task and jobs (state, stage, assignee) of each linked local task: ``{task id: {...}}``.

    Read from CVAT at most once a minute per project; empty when the project is not linked, CVAT is
    not configured or cannot be reached (the task list must never fail or hang because of CVAT: any
    failure, and a wait longer than ``REMOTE_TIMEOUT``, is remembered for ``REMOTE_TTL`` seconds).

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        refresh: Ignore what was read in the last minute and ask CVAT again.

    Returns:
        ``{task id: {...}}``: the remote state of each linked task (empty when unknown). A linked task CVAT
        answered without is ``{"missing": "removed_remote"}``, or ``{"missing": "remote_gone"}`` when the
        CVAT project itself is gone.
    """
    link = links.get_link(state.store.dir(project_id), "cvat")
    tmap: dict[str, Any] = (link or {}).get("tasks") or {}
    if not link or not tmap:
        return {}
    key = (project_id, f"{link['remote_id']}@{link.get('last_sync', '')}")  # a new sync reads again
    cached = _remote_cache.get(key)
    if cached and not refresh and time.monotonic() - cached[0] < REMOTE_TTL:
        return cached[1]
    settings = credentials.load_settings(state.settings.workspace, "cvat")
    if not credentials.is_configured(settings):
        return {}
    remote = get_remote("cvat")
    try:
        future = _remote_pool.submit(remote.tasks, settings, str(link["remote_id"]))
        found = {rt.id: rt for rt in future.result(timeout=REMOTE_TIMEOUT)}
    except Exception:  # noqa: BLE001 - unreachable, slow, login refused...: the list shows without CVAT
        _remote_cache[key] = (time.monotonic(), {})
        return {}
    missing = "removed_remote" if found else _missing_reason(remote, settings, str(link["remote_id"]))
    out: dict[str, Any] = {}
    for local_id, entry in tmap.items():
        rt = found.get(str((entry or {}).get("remote_task", "")))
        if rt is None:
            if missing:
                out[str(local_id)] = {"missing": missing}
            continue
        out[str(local_id)] = {
            "task": rt.id,
            "url": rt.url,
            "jobs": [
                {
                    "n": j.n,
                    "id": j.id,
                    "url": j.url,
                    "state": j.state,
                    "stage": j.stage,
                    "assignee": j.assignee,
                }
                for j in rt.jobs
            ],
        }
    _remote_cache[key] = (time.monotonic(), out)
    return out


def _missing_reason(remote: Any, settings: Any, remote_id: str) -> str:
    """Why CVAT listed no task of the linked project: the project is gone, or its tasks were deleted.

    Args:
        remote: The CVAT adapter.
        settings: Address and login of the CVAT server.
        remote_id: The CVAT project id.

    Returns:
        ``remote_gone`` (CVAT answered 404 for the project), ``removed_remote`` (it is there), or an empty
        text when CVAT did not answer (nothing is known).
    """
    try:
        _remote_pool.submit(remote.get, settings, remote_id).result(timeout=REMOTE_TIMEOUT)
    except RemoteNotFound:
        return "remote_gone"
    except Exception:  # noqa: BLE001 - unreachable or slow: say nothing rather than something wrong
        return ""
    return "removed_remote"


def _whole_number(text: str) -> int | None:
    """``text`` as an integer (None when empty).

    Args:
        text: The text to read.

    Returns:
        The number, or ``None`` when the text is empty.

    Raises:
        ValueError: not a whole number (readable message).
    """
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError("The job size must be a whole number.") from None


@router.post("/projects/{project_id}/tasks")
def create_task(
    project_id: int,
    files: FILES,
    name: Annotated[str, Form()] = "",
    job_size: Annotated[str, Form()] = "",
    open: Annotated[bool, Form()] = True,
    state: AppState = STATE,
) -> dict[str, Any]:
    """Create a task from uploaded images and .zip archives.

    The task stays ``open`` (later chunks can be appended) until it is closed with PATCH, unless
    ``open`` is false; a task created closed with no image added is refused (HTTP 400). Name and
    images per job are checked before any image is stored.

    Args:
        project_id: Id of the project.
        files: The uploaded files.
        name: Name of the task.
        job_size: Images per job, as text (empty for the default).
        open: Keep the task open for later chunks.
        state: The shared state of the application.

    Returns:
        ``{task, report}``.
    """
    manifest = state.manifest(project_id)
    store = TaskStore(state.store.dir(project_id))
    try:
        size = check_job_size(_whole_number(job_size))
        task_name = clean_name(name) if name.strip() else None
        if task_name:
            store.check_name_free(task_name)  # before any image is stored
    except ValueError as exc:
        raise bad_request(exc) from None
    with store.locked():  # ingest and registration are atomic
        store.ensure()  # migrate before the new images exist
        report, added = ingest_uploads(manifest, files)
        if not added and not open:
            raise HTTPException(
                400,
                {"code": "no_images_added", "message": "No image was added to the task.", "report": report},
            )
        task = store.create(task_name, size, added, open=open)
    return {"task": _view(state, project_id, task), "report": report}


@router.post("/projects/{project_id}/tasks/{task_id}/images")
def append_images(project_id: int, task_id: int, files: FILES, state: AppState = STATE) -> dict[str, Any]:
    """Append uploaded images to an open task (a later chunk of the same upload).

    Args:
        project_id: Id of the project.
        task_id: Id of the task.
        files: The uploaded files.
        state: The shared state of the application.

    Returns:
        The ingest report.
    """
    manifest = state.manifest(project_id)
    store = TaskStore(state.store.dir(project_id))
    with store.locked():  # nothing is stored unless the task exists and is open
        if not _find(store, task_id).open:
            raise ApiError(409, "task_closed", "This task is closed: its images are fixed.")
        report, added = ingest_uploads(manifest, files)
        store.append(task_id, added)
    return report


@router.patch("/projects/{project_id}/tasks/{task_id}")
def change_task(project_id: int, task_id: int, body: TaskChanges, state: AppState = STATE) -> dict[str, Any]:
    """Rename a task and/or close it (``open: false``). Reopening a closed task is refused (HTTP 409).

    Closing a task that holds no image (an upload that added nothing) removes it: the answer then has
    ``removed: true``.

    Args:
        project_id: Id of the project.
        task_id: Id of the task.
        body: The new name and/or ``open: false``.
        state: The shared state of the application.

    Returns:
        The task (with ``removed: true`` when it was empty and got removed).
    """
    store = _store(state, project_id)
    with store.locked():
        task = _find(store, task_id)
        if body.open and not task.open:
            raise ApiError(409, "task_closed", "A closed task cannot be reopened.")
        try:
            if body.name is not None:
                task = store.rename(task_id, body.name)
            if body.open is False:
                task = store.close(task_id)
                if not task.images:  # an upload that added nothing (all duplicates) leaves no empty task
                    store.delete(task_id)
                    return {**_view(state, project_id, task), "removed": True}
        except ValueError as exc:
            raise bad_request(exc) from exc
    return _view(state, project_id, task)


@router.delete("/projects/{project_id}/tasks/{task_id}")
def delete_task(project_id: int, task_id: int, body: DeleteTask, state: AppState = STATE) -> dict[str, int]:
    """Delete a task and its images (files, thumbnails, annotations). Needs ``confirm: true``.

    Args:
        project_id: Id of the project.
        task_id: Id of the task.
        body: The confirmation.
        state: The shared state of the application.

    Returns:
        ``{deleted}``: how many images were deleted.
    """
    store = _store(state, project_id)
    with store.locked():
        task = _find(store, task_id)
        if not body.confirm:
            raise ApiError(
                400,
                "confirm_required",
                "Deleting a task deletes its images and cannot be undone: confirm it first.",
            )
        state.busy(project_id, training=True)  # checked before anything changes (deleting images needs it)
        # images first: their task goes with its last image; deleting the task first would leave
        # them without a task for a moment, and reading the tasks would rebuild one with a new id
        deleted = delete_image_files(state, project_id, list(task.images))
        with contextlib.suppress(TaskNotFound):
            store.delete(task_id)  # a task that had no images
        video = (task.source or {}).get("video")
        if video:
            VideoStore(state.store.dir(project_id)).drop_if_unused(str(video), store.load())
        return {"deleted": deleted}
