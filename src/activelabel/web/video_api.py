"""HTTP API of videos and tasks made from them (see docs/VIDEO.md)."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from activelabel.images.ingest import ingest_frames
from activelabel.images.video import (
    VIDEO_EXTENSIONS,
    VideoInfo,
    VideoStore,
    check_extraction,
    extract_frames,
    frame_jpeg,
    plan,
    resolve_selection,
)
from activelabel.jobs import Progress
from activelabel.tasks_store import TaskStore, check_job_size, clean_name
from activelabel.web.state import AppState, get_state
from activelabel.web.tasks_api import _view

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
CHUNK = 1024 * 1024


class Selection(BaseModel):
    """Frames of a video: range, and a step in frames or an interval in seconds."""

    start: int = 0
    stop: int | None = None
    step: int = 1
    every_seconds: float | None = None


class VideoTask(Selection):
    """Body of POST /projects/{id}/tasks/from-video."""

    video: str
    name: str | None = None
    job_size: int | None = None


def _videos(state: AppState, project_id: int) -> VideoStore:
    state.project(project_id)
    return VideoStore(state.store.dir(project_id))


def _find(videos: VideoStore, video_id: str) -> VideoInfo:
    try:
        return videos.get(video_id)
    except KeyError:
        raise HTTPException(404, "This video does not exist.") from None


def _select(info: VideoInfo, body: Selection) -> tuple[int, int, int]:
    try:
        return resolve_selection(info, body.start, body.stop, body.step, body.every_seconds)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/projects/{project_id}/videos")
def upload_video(
    project_id: int, file: Annotated[UploadFile, File()], state: AppState = STATE
) -> dict[str, Any]:
    """Store one video (streamed to disk, hashed while written) and read its properties."""
    videos = _videos(state, project_id)
    name = file.filename or "video"
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported video type; use one of: {', '.join(sorted(VIDEO_EXTENSIONS))}."
        )
    videos.folder.mkdir(parents=True, exist_ok=True)
    tmp = videos.folder / f".{uuid.uuid4().hex}.part"
    digest, size = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                digest.update(chunk)
                out.write(chunk)
        info = videos.add(Path(name).name, ext, tmp, digest.hexdigest()[:16], size)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        tmp.unlink(missing_ok=True)
    return info.public()


@router.get("/projects/{project_id}/videos")
def list_videos(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Videos of the project."""
    return {"videos": [v.public() for v in _videos(state, project_id).list()]}


@router.delete("/projects/{project_id}/videos/{video_id}")
def delete_video(project_id: int, video_id: str, state: AppState = STATE) -> dict[str, str]:
    """Delete a video file, only when no task uses it."""
    videos = _videos(state, project_id)
    _find(videos, video_id)
    if any(j.kind == "video_task" for j in state.jobs.active(project_id)):
        raise HTTPException(409, "A task is being made from a video: wait for it to finish, then delete.")
    tasks = TaskStore(state.store.dir(project_id))
    with tasks.locked():
        if not videos.drop_if_unused(video_id, tasks.load()):
            raise HTTPException(409, "A task uses this video: delete the task first.")
    return {"deleted": video_id}


@router.get("/projects/{project_id}/videos/{video_id}/frame")
def video_frame(
    project_id: int,
    video_id: str,
    n: Annotated[int, Query(ge=0)],
    w: Annotated[int, Query(ge=16, le=4096)] = 320,
    state: AppState = STATE,
) -> Response:
    """JPEG of one frame (preview strip)."""
    videos = _videos(state, project_id)
    info = _find(videos, video_id)
    if n >= info.frames:
        raise HTTPException(400, f"The video has {info.frames} frames (0 to {info.frames - 1}).")
    try:
        data = frame_jpeg(videos.file(info), n, w)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return Response(data, media_type="image/jpeg")


@router.post("/projects/{project_id}/videos/{video_id}/plan")
def video_plan(project_id: int, video_id: str, body: Selection, state: AppState = STATE) -> dict[str, int]:
    """How many frames a selection gives (no extraction)."""
    info = _find(_videos(state, project_id), video_id)
    return plan(*_select(info, body))


@router.post("/projects/{project_id}/tasks/from-video")
def task_from_video(project_id: int, body: VideoTask, state: AppState = STATE) -> dict[str, Any]:
    """Create a (closed) task with the selected frames of a video, as a background job."""
    videos = _videos(state, project_id)
    info = _find(videos, body.video)
    start, stop, step = _select(info, body)
    try:
        size = check_job_size(body.job_size)
        task_name = clean_name(body.name) if body.name and body.name.strip() else None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    manifest = state.manifest(project_id)
    tasks = TaskStore(state.store.dir(project_id))
    video_file = videos.file(info)
    planned = plan(start, stop, step)["frames"]
    try:
        check_extraction(info, planned, videos.folder)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    stem = (Path(info.name).stem or "video").replace("/", "_").replace("\\", "_")

    def run(progress: Progress) -> dict[str, Any]:
        temp = videos.folder / f".frames-{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            frames = extract_frames(video_file, start, stop, step, temp, progress)
            if not frames:
                raise ValueError("No frame could be read from the video.")
            real_stop = frames[-1][0]  # the file may end before the planned last frame
            with tasks.locked():  # ingest and registration are atomic
                tasks.ensure()
                items = [
                    (f"{stem}/frame_{index:06d}.jpg", path, info.id, index, index / info.fps)
                    for index, path in frames
                ]
                ids = ingest_frames(manifest, items, progress, salt=str(tasks.next_id()))
                source = {"video": info.id, "start": start, "stop": real_stop, "step": step, "fps": info.fps}
                try:
                    task = tasks.create(task_name or stem[:100], size, ids, source=source)
                except Exception:
                    manifest.delete(ids)
                    raise
            return {
                "task": _view(state, project_id, task),
                "frames": len(ids),
                "planned": planned,
                "stopped_early": len(ids) < planned,
                "step": step,
            }
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    return state.jobs.submit(project_id, "video_task", run).as_dict()
