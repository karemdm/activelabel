"""Videos of a project and the extraction of their frames (see docs/VIDEO.md).

Videos are kept as uploaded in ``<project>/videos/<id>.<ext>`` (id = first 16 hex digits of the
SHA-256) with their metadata in ``videos.json``. OpenCV reads them; no other tool is needed.
"""

from __future__ import annotations

import math
import shutil
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from activelabel.errors import CodedError
from activelabel.jobs import Progress
from activelabel.storage import known_fields, read_json, write_json

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
JPEG_QUALITY = 95
MAX_TASK_FRAMES = 20_000  # frames one task may take from a video
JPEG_BYTES_PER_PIXEL = 0.6  # typical size of a quality-95 frame, to estimate the disk it needs

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per file, shared by every store object that uses it.

    Args:
        path: the file to lock (resolved, so every spelling of it shares one lock).

    Returns:
        The re-entrant lock of that file.
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


@dataclass
class VideoInfo:
    """Metadata of a stored video."""

    id: str
    name: str
    fps: float
    frames: int
    duration: float
    width: int
    height: int
    size_bytes: int
    ext: str = ".mp4"

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            All the fields, as a dict.
        """
        return asdict(self)

    def public(self) -> dict[str, Any]:
        """The form returned by the API (without the file extension).

        Returns:
            All the fields but ``ext``, as a dict.
        """
        data = asdict(self)
        data.pop("ext")
        return data


def probe(path: Path) -> tuple[float, int, int, int]:
    """Read ``(fps, frames, width, height)`` of a video.

    Args:
        path: the video file.

    Returns:
        The frame rate, the number of frames, the width and the height in pixels.

    Raises:
        CodedError: OpenCV cannot read it, or it has no frames or no frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise CodedError("video_unreadable", "This file is not a video that can be read.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok:
        raise CodedError("video_unreadable", "This file is not a video that can be read.")
    if frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise CodedError("video_no_frames", "The video has no frames or no frame rate.")
    return fps, frames, width, height


class VideoStore:
    """Videos of one project (``videos.json`` and the ``videos`` folder)."""

    def __init__(self, project_dir: Path) -> None:
        """Point the store at ``<project_dir>/videos`` and ``<project_dir>/videos.json``.

        Args:
            project_dir: folder of the project.
        """
        self.project_dir = project_dir
        self.folder = project_dir / "videos"
        self.path = project_dir / "videos.json"
        self._lock = _lock_for(self.path)

    def list(self) -> list[VideoInfo]:
        """Every video, in upload order.

        Returns:
            The metadata of each stored video.
        """
        return [VideoInfo(**known_fields(VideoInfo, item)) for item in read_json(self.path, [])]

    def get(self, video_id: str) -> VideoInfo:
        """One video.

        Args:
            video_id: id of the video.

        Returns:
            Its metadata.

        Raises:
            KeyError: unknown id.
        """
        for info in self.list():
            if info.id == video_id:
                return info
        raise KeyError(video_id)

    def file(self, info: VideoInfo) -> Path:
        """Absolute path of the stored video.

        Args:
            info: metadata of the video.

        Returns:
            Its path in the videos folder.
        """
        return self.folder / f"{info.id}{info.ext}"

    def add(self, name: str, ext: str, tmp: Path, video_id: str, size: int) -> VideoInfo:
        """Register a fully written upload (``tmp``, inside the videos folder); moves it into place.

        The same content again returns the existing record and discards ``tmp``.

        Args:
            name: original file name.
            ext: file extension with the dot (it decides the decoder OpenCV uses).
            tmp: the uploaded file, already complete, inside the videos folder.
            video_id: id of the video (the first 16 hex digits of the SHA-256 of its content).
            size: size of the file in bytes.

        Returns:
            The metadata of the video (the existing one when it was already stored).

        Raises:
            ValueError: the file cannot be read as a video (a ``CodedError`` from :func:`probe`; the
                file is removed).
        """
        with self._lock:
            for info in self.list():
                if info.id == video_id:
                    tmp.unlink(missing_ok=True)
                    return info
            # OpenCV picks the decoder from the extension: probe a copy under the final name
            final = self.folder / f"{video_id}{ext}"
            tmp.replace(final)
            try:
                fps, frames, width, height = probe(final)
            except ValueError:
                final.unlink(missing_ok=True)
                raise
            info = VideoInfo(video_id, name, fps, frames, frames / fps, width, height, size, ext)
            write_json(self.path, [i.as_dict() for i in [*self.list(), info]])
            return info

    def delete(self, video_id: str) -> None:
        """Remove a video and its record.

        Args:
            video_id: id of the video (an unknown id changes nothing).
        """
        with self._lock:
            kept = []
            for info in self.list():
                if info.id == video_id:
                    self.file(info).unlink(missing_ok=True)
                else:
                    kept.append(info)
            write_json(self.path, [i.as_dict() for i in kept])

    def drop_if_unused(self, video_id: str, tasks: Iterable[Any]) -> bool:
        """Delete the video when no task uses it.

        Args:
            video_id: id of the video.
            tasks: the tasks of the project (their ``source`` says which video they use).

        Returns:
            Whether the video was deleted.
        """
        if any((t.source or {}).get("video") == video_id for t in tasks):
            return False
        self.delete(video_id)
        return True


def resolve_selection(
    info: VideoInfo,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    every_seconds: float | None = None,
) -> tuple[int, int, int]:
    """Validate a frame selection.

    Args:
        info: the video.
        start: first frame.
        stop: last frame, included (None = the last frame of the video).
        step: take one frame every ``step`` frames.
        every_seconds: take one frame every this many seconds; when given it replaces ``step``
            (rounded to whole frames, at least 1).

    Returns:
        ``(start, stop, step)`` with the defaults filled in.

    Raises:
        CodedError: ``0 <= start <= stop < frames`` or ``step >= 1`` does not hold.
    """
    stop = info.frames - 1 if stop is None else stop
    if every_seconds is not None:
        if not 0 < every_seconds <= 86_400:  # also refuses NaN and infinity
            raise CodedError(
                "video_seconds_range", "Seconds between frames must be greater than 0 and at most one day."
            )
        step = max(1, round(every_seconds * info.fps))
    if step < 1:
        raise CodedError("video_step_range", "Take one frame every 1 or more frames.")
    if not 0 <= start <= stop < info.frames:
        raise CodedError(
            "video_range_invalid",
            f"The range must satisfy 0 <= start <= end < {info.frames} (frames of the video).",
            frames=info.frames,
        )
    return start, stop, step


def plan(start: int, stop: int, step: int) -> dict[str, int]:
    """How many frames a selection has: ``{frames, step, first, last}``.

    Args:
        start: first frame.
        stop: last frame, included.
        step: take one frame every ``step`` frames.

    Returns:
        The number of frames taken, the step, and the first and last frame taken.
    """
    count = (stop - start) // step + 1
    return {"frames": count, "step": step, "first": start, "last": start + (count - 1) * step}


def check_extraction(info: VideoInfo, frames: int, folder: Path) -> None:
    """Refuse a selection that is too big or would not fit on the disk.

    Frames are written twice while a task is made (a temporary copy, then the project's images),
    so twice the estimated size must be free in ``folder``.

    Args:
        info: the video (its size sets the estimate).
        frames: number of frames the selection takes.
        folder: where the frames will be written (created when missing).

    Raises:
        CodedError: more than ``MAX_TASK_FRAMES`` frames, or not enough free space.
    """
    if frames > MAX_TASK_FRAMES:
        raise CodedError(
            "video_too_many_frames",
            f"This selection has {frames} frames (at most {MAX_TASK_FRAMES} per task): "
            "take one frame every few frames, or a shorter range.",
            frames=frames,
            max=MAX_TASK_FRAMES,
        )
    needed = round(2 * frames * info.width * info.height * JPEG_BYTES_PER_PIXEL)
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if needed > free:
        raise CodedError(
            "video_not_enough_disk",
            f"The frames need about {needed // 2**20} MB of free disk space and only {free // 2**20} MB "
            "are free: take fewer frames.",
            needed_mb=needed // 2**20,
            free_mb=free // 2**20,
        )


def extract_frames(
    video: Path, start: int, stop: int, step: int, target: Path, progress: Progress | None = None
) -> list[tuple[int, Path]]:
    """Write frames ``start, start+step, ... <= stop`` as JPEG files in ``target``.

    Frames are read in sequence (``grab``/``retrieve``), which is exact and faster than seeking.

    Args:
        video: the video file.
        start: first frame.
        stop: last frame, included.
        step: take one frame every ``step`` frames.
        target: existing folder for the files (named by the frame index, ``<index:08d>.jpg``).
        progress: reports progress and checks for cancellation (None = neither).

    Returns:
        ``(frame index, file)`` in order; fewer than planned when the file ends early.

    Raises:
        Cancelled: the job was cancelled (the caller removes ``target``).
        ValueError: a frame cannot be encoded as JPEG.
    """
    total = plan(start, stop, step)["frames"]
    out: list[tuple[int, Path]] = []
    cap = cv2.VideoCapture(str(video))
    try:
        for index in range(stop + 1):
            if not cap.grab():
                break
            if index < start or (index - start) % step:
                if progress is not None and index % 100 == 0:
                    progress.check()
                continue
            ok, image = cap.retrieve()
            if not ok:
                break
            path = target / f"{index:08d}.jpg"
            # encode in memory and write with Python: cv2.imwrite fails on non-ASCII paths on Windows
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise ValueError(f"Frame {index} could not be saved.")
            path.write_bytes(encoded.tobytes())
            out.append((index, path))
            if progress is not None:
                progress.update(len(out), total, "Extracting frames")
    finally:
        cap.release()
    return out


def frame_jpeg(video: Path, index: int, width: int = 320) -> bytes:
    """One frame as JPEG, at most ``width`` pixels wide.

    Args:
        video: the video file.
        index: the frame number.
        width: the largest width in pixels (a wider frame is shrunk, keeping its proportions).

    Returns:
        The JPEG bytes (quality 80).

    Raises:
        CodedError: the frame cannot be read or encoded (a ``ValueError``).
    """
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok:
        raise CodedError("frame_unreadable", "This frame cannot be read.")
    rows, cols = image.shape[:2]
    if cols > width:
        image = cv2.resize(image, (width, max(1, round(rows * width / cols))), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise CodedError("frame_unreadable", "This frame cannot be encoded.")
    return bytes(data.tobytes())
