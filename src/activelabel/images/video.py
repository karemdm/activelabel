"""Videos of a project and the extraction of their frames (see docs/VIDEO.md).

Videos are kept as uploaded in ``<project>/videos/<id>.<ext>`` (id = first 16 hex digits of the
SHA-256) with their metadata in ``videos.json``. OpenCV reads them; no other tool is needed.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from activelabel.jobs import Progress
from activelabel.storage import read_json, write_json

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
JPEG_QUALITY = 95

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
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
        """JSON-ready form."""
        return asdict(self)

    def public(self) -> dict[str, Any]:
        """The form returned by the API (without the file extension)."""
        data = asdict(self)
        data.pop("ext")
        return data


def probe(path: Path) -> tuple[float, int, int, int]:
    """Read ``(fps, frames, width, height)`` of a video.

    Raises:
        ValueError: OpenCV cannot read it, or it has no frames or no frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("This file is not a video that can be read.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("This file is not a video that can be read.")
    if frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The video has no frames or no frame rate.")
    return fps, frames, width, height


class VideoStore:
    """Videos of one project (``videos.json`` and the ``videos`` folder)."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.folder = project_dir / "videos"
        self.path = project_dir / "videos.json"
        self._lock = _lock_for(self.path)

    def list(self) -> list[VideoInfo]:
        """Every video, in upload order."""
        return [VideoInfo(**item) for item in read_json(self.path, [])]

    def get(self, video_id: str) -> VideoInfo:
        """One video.

        Raises:
            KeyError: unknown id.
        """
        for info in self.list():
            if info.id == video_id:
                return info
        raise KeyError(video_id)

    def file(self, info: VideoInfo) -> Path:
        """Absolute path of the stored video."""
        return self.folder / f"{info.id}{info.ext}"

    def add(self, name: str, ext: str, tmp: Path, video_id: str, size: int) -> VideoInfo:
        """Register a fully written upload (``tmp``, inside the videos folder); moves it into place.

        The same content again returns the existing record and discards ``tmp``.

        Raises:
            ValueError: the file cannot be read as a video.
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
        """Remove a video and its record."""
        with self._lock:
            kept = []
            for info in self.list():
                if info.id == video_id:
                    self.file(info).unlink(missing_ok=True)
                else:
                    kept.append(info)
            write_json(self.path, [i.as_dict() for i in kept])

    def drop_if_unused(self, video_id: str, tasks: Iterable[Any]) -> bool:
        """Delete the video when no task uses it. Returns whether it was deleted."""
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
    """Validate a frame selection. Returns ``(start, stop, step)``.

    Raises:
        ValueError: ``0 <= start <= stop < frames`` or ``step >= 1`` does not hold.
    """
    stop = info.frames - 1 if stop is None else stop
    if every_seconds is not None:
        if not every_seconds > 0:
            raise ValueError("Seconds between frames must be greater than 0.")
        step = max(1, round(every_seconds * info.fps))
    if step < 1:
        raise ValueError("Take one frame every 1 or more frames.")
    if not 0 <= start <= stop < info.frames:
        raise ValueError(f"The range must satisfy 0 <= start <= end < {info.frames} (frames of the video).")
    return start, stop, step


def plan(start: int, stop: int, step: int) -> dict[str, int]:
    """How many frames a selection has: ``{frames, step, first, last}``."""
    count = (stop - start) // step + 1
    return {"frames": count, "step": step, "first": start, "last": start + (count - 1) * step}


def extract_frames(
    video: Path, start: int, stop: int, step: int, target: Path, progress: Progress | None = None
) -> list[tuple[int, Path]]:
    """Write frames ``start, start+step, ... <= stop`` as JPEG files in ``target``.

    Frames are read in sequence (``grab``/``retrieve``), which is exact and faster than seeking.

    Returns:
        ``(frame index, file)`` in order; fewer than planned when the file ends early.

    Raises:
        Cancelled: the job was cancelled (the caller removes ``target``).
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
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise ValueError(f"Frame {index} could not be saved.")
            out.append((index, path))
            if progress is not None:
                progress.update(len(out), total, "Extracting frames")
    finally:
        cap.release()
    return out


def frame_jpeg(video: Path, index: int, width: int = 320) -> bytes:
    """One frame as JPEG, at most ``width`` pixels wide.

    Raises:
        ValueError: the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("This frame cannot be read.")
    rows, cols = image.shape[:2]
    if cols > width:
        image = cv2.resize(image, (width, max(1, round(rows * width / cols))), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("This frame cannot be encoded.")
    return bytes(data.tobytes())
