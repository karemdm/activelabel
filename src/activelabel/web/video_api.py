"""HTTP API of videos and tasks made from them (see docs/VIDEO.md)."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from pydantic import BaseModel

from activelabel.images.ingest import ingest_frames
from activelabel.images.manifest import Manifest
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
from activelabel.web.errors import ApiError, bad_request
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
    """The video store of a project.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.

    Returns:
        The video store of the project.
    """
    state.project(project_id)
    return VideoStore(state.store.dir(project_id))


def _find(videos: VideoStore, video_id: str) -> VideoInfo:
    """A video by id, or HTTP 404.

    Args:
        videos: The video store.
        video_id: Id of the video.

    Returns:
        The video.
    """
    try:
        return videos.get(video_id)
    except KeyError:
        raise ApiError(404, "video_not_found", "This video does not exist.") from None


def _select(info: VideoInfo, body: Selection) -> tuple[int, int, int]:
    """The frame range a selection means, or HTTP 400 when it is wrong.

    Args:
        info: The video.
        body: The selection (start, stop, step or seconds).

    Returns:
        ``(start, stop, step)`` frame numbers.
    """
    try:
        return resolve_selection(info, body.start, body.stop, body.step, body.every_seconds)
    except ValueError as exc:
        raise bad_request(exc) from None


@router.post("/projects/{project_id}/videos")
def upload_video(
    project_id: int, files: Annotated[list[UploadFile], File(alias="file")], state: AppState = STATE
) -> dict[str, Any]:
    """Store one video (streamed to disk, hashed while written) and read its properties.

    The request has exactly one ``file`` part; a second one is refused (400 ``one_file_only``).

    Args:
        project_id: Id of the project.
        files: The uploaded files.
        state: The shared state of the application.

    Returns:
        The video with its properties.
    """
    if len(files) != 1:
        raise ApiError(400, "one_file_only", "Send one video per request.")
    file = files[0]
    videos = _videos(state, project_id)
    name = file.filename or "video"
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ApiError(
            400,
            "video_type_unsupported",
            f"Unsupported video type; use one of: {', '.join(sorted(VIDEO_EXTENSIONS))}.",
            allowed=sorted(VIDEO_EXTENSIONS),
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
        raise bad_request(exc) from None
    finally:
        tmp.unlink(missing_ok=True)
    return info.public()


@router.get("/projects/{project_id}/videos")
def list_videos(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Videos of the project.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{videos}``.
    """
    return {"videos": [v.public() for v in _videos(state, project_id).list()]}


@router.delete("/projects/{project_id}/videos/{video_id}")
def delete_video(project_id: int, video_id: str, state: AppState = STATE) -> dict[str, str]:
    """Delete a video file, only when no task uses it.

    Args:
        project_id: Id of the project.
        video_id: Id of the video.
        state: The shared state of the application.

    Returns:
        ``{deleted: video_id}``.
    """
    videos = _videos(state, project_id)
    _find(videos, video_id)
    if any(j.kind == "video_task" for j in state.jobs.active(project_id)):
        raise ApiError(
            409,
            "video_task_running",
            "A task is being made from a video: wait for it to finish, then delete.",
        )
    tasks = TaskStore(state.store.dir(project_id))
    with tasks.locked():
        if not videos.drop_if_unused(video_id, tasks.load()):
            raise ApiError(409, "video_in_use", "A task uses this video: delete the task first.")
    return {"deleted": video_id}


@router.get("/projects/{project_id}/videos/{video_id}/frame")
def video_frame(
    project_id: int,
    video_id: str,
    n: Annotated[int, Query(ge=0)],
    w: Annotated[int, Query(ge=16, le=4096)] = 320,
    state: AppState = STATE,
) -> Response:
    """JPEG of one frame (preview strip).

    Args:
        project_id: Id of the project.
        video_id: Id of the video.
        n: Frame number (0-based).
        w: Width of the image, in pixels.
        state: The shared state of the application.

    Returns:
        The frame as a JPEG.
    """
    videos = _videos(state, project_id)
    info = _find(videos, video_id)
    if n >= info.frames:
        raise ApiError(
            400,
            "frame_out_of_range",
            f"The video has {info.frames} frames (0 to {info.frames - 1}).",
            frames=info.frames,
        )
    try:
        data = frame_jpeg(videos.file(info), n, w)
    except ValueError as exc:
        raise bad_request(exc) from None
    return Response(data, media_type="image/jpeg")


@router.post("/projects/{project_id}/videos/{video_id}/plan")
def video_plan(project_id: int, video_id: str, body: Selection, state: AppState = STATE) -> dict[str, int]:
    """How many frames a selection gives (no extraction).

    Args:
        project_id: Id of the project.
        video_id: Id of the video.
        body: The selection.
        state: The shared state of the application.

    Returns:
        ``{frames}``: how many frames the selection gives.
    """
    info = _find(_videos(state, project_id), video_id)
    return plan(*_select(info, body))


def _video_task_job(
    state: AppState,
    project_id: int,
    videos: VideoStore,
    info: VideoInfo,
    manifest: Manifest,
    tasks: TaskStore,
    frames_range: tuple[int, int, int],
    size: int | None,
    task_name: str | None,
    stem: str,
    planned: int,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``video_task`` job: extract the frames, ingest them and create the task.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        videos: The video store of the project.
        info: The video.
        manifest: The image manifest of the project.
        tasks: The task store of the project.
        frames_range: ``(start, stop, step)``: first frame, last frame and the gap between frames taken.
        size: Images per job of the new task (the default when ``None``).
        task_name: Name of the task (the video name when ``None``).
        stem: Name of the video without extension, used in the image names.
        planned: How many frames the selection should give.
        progress: Where the job reports its progress.

    Returns:
        ``{task, frames, planned, stopped_early, step}``.

    Raises:
        ValueError: No frame could be read from the video.
    """
    start, stop, step = frames_range
    temp = videos.folder / f".frames-{uuid.uuid4().hex}"
    temp.mkdir(parents=True)
    try:
        frames = extract_frames(videos.file(info), start, stop, step, temp, progress)
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


@router.post("/projects/{project_id}/tasks/from-video")
def task_from_video(project_id: int, body: VideoTask, state: AppState = STATE) -> dict[str, Any]:
    """Create a (closed) task with the selected frames of a video, as a background job.

    Args:
        project_id: Id of the project.
        body: The video, the selection, the task name and the job size.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    videos = _videos(state, project_id)
    info = _find(videos, body.video)
    start, stop, step = _select(info, body)
    try:
        size = check_job_size(body.job_size)
        task_name = clean_name(body.name) if body.name and body.name.strip() else None
    except ValueError as exc:
        raise bad_request(exc) from None
    manifest = state.manifest(project_id)
    tasks = TaskStore(state.store.dir(project_id))
    planned = plan(start, stop, step)["frames"]
    try:
        check_extraction(info, planned, videos.folder)
    except ValueError as exc:
        raise bad_request(exc) from None
    stem = (Path(info.name).stem or "video").replace("/", "_").replace("\\", "_")
    job = partial(
        _video_task_job,
        state,
        project_id,
        videos,
        info,
        manifest,
        tasks,
        (start, stop, step),
        size,
        task_name,
        stem,
        planned,
    )
    return state.jobs.submit(project_id, "video_task", job).as_dict()
