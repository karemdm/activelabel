"""HTTP API of tasks and jobs (see docs/TASKS.md)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.labels import load_annotations
from activelabel.tasks_store import (
    ImageTask,
    TaskNotFound,
    TaskStore,
    check_job_size,
    clean_name,
    jobs,
)
from activelabel.web.api import delete_image_files, ingest_uploads
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
    """A task for the interface: counts and its jobs (``first``/``last`` are 1-based positions)."""
    ids = set(task.images)
    return {
        "id": task.id,
        "name": task.name,
        "created": task.created,
        "job_size": task.job_size,
        "open": task.open,
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
    """Ids of active images and of images with at least one label."""
    project_dir = state.store.dir(project_id)
    active = {r.id for r in Manifest(project_dir).all() if r.status == ACTIVE}
    annotations = load_annotations(project_dir)
    labelled = {i for i, labels in (annotations.images if annotations else {}).items() if labels}
    return active, labelled


def _view(state: AppState, project_id: int, task: ImageTask) -> dict[str, Any]:
    return task_view(task, *_sets(state, project_id))


def _store(state: AppState, project_id: int) -> TaskStore:
    state.project(project_id)
    return TaskStore(state.store.dir(project_id))


def _find(store: TaskStore, task_id: int) -> ImageTask:
    try:
        return store.get(task_id)
    except TaskNotFound:
        raise HTTPException(404, "This task does not exist.") from None


@router.get("/projects/{project_id}/tasks")
def list_tasks(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Every task of the project, in creation order."""
    store = _store(state, project_id)
    active, annotated = _sets(state, project_id)
    return {"tasks": [task_view(t, active, annotated) for t in store.load()]}


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
    """
    manifest = state.manifest(project_id)
    store = TaskStore(state.store.dir(project_id))
    try:
        size = check_job_size(int(job_size) if job_size.strip() else None)
        task_name = clean_name(name) if name.strip() else None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    with store.locked():  # ingest and registration are atomic
        store.ensure()  # migrate before the new images exist
        report, added = ingest_uploads(manifest, files)
        if not added and not open:
            raise HTTPException(400, {"message": "No image was added to the task.", "report": report})
        task = store.create(task_name, size, added, open=open)
    return {"task": _view(state, project_id, task), "report": report}


@router.post("/projects/{project_id}/tasks/{task_id}/images")
def append_images(project_id: int, task_id: int, files: FILES, state: AppState = STATE) -> dict[str, Any]:
    """Append uploaded images to an open task (a later chunk of the same upload)."""
    manifest = state.manifest(project_id)
    store = TaskStore(state.store.dir(project_id))
    with store.locked():  # nothing is stored unless the task exists and is open
        if not _find(store, task_id).open:
            raise HTTPException(409, "This task is closed: its images are fixed.")
        report, added = ingest_uploads(manifest, files)
        store.append(task_id, added)
    return report


@router.patch("/projects/{project_id}/tasks/{task_id}")
def change_task(project_id: int, task_id: int, body: TaskChanges, state: AppState = STATE) -> dict[str, Any]:
    """Rename a task and/or close it (``open: false``). Reopening a closed task is refused."""
    store = _store(state, project_id)
    with store.locked():
        task = _find(store, task_id)
        if body.open and not task.open:
            raise HTTPException(400, "A closed task cannot be reopened.")
        try:
            if body.name is not None:
                task = store.rename(task_id, body.name)
            if body.open is False:
                task = store.close(task_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return _view(state, project_id, task)


@router.delete("/projects/{project_id}/tasks/{task_id}")
def delete_task(project_id: int, task_id: int, body: DeleteTask, state: AppState = STATE) -> dict[str, int]:
    """Delete a task and its images (files, thumbnails, annotations). Needs ``confirm: true``."""
    store = _store(state, project_id)
    with store.locked():
        _find(store, task_id)
        if not body.confirm:
            raise HTTPException(
                400, "Deleting a task deletes its images and cannot be undone: confirm it first."
            )
        state.busy(project_id)
        ids = store.delete(task_id)
        return {"deleted": delete_image_files(state, project_id, ids)}
