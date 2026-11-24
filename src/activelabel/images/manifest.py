"""The list of images of a project (``images.json``).

Images are stored by content: the id is a hash of the file (salted for video frames), so the same
image uploaded twice is detected. Excluding an image keeps its file (it can be restored); deleting it
removes the file.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.constants import ACTIVE, EXCLUDED
from activelabel.storage import read_json, write_json

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per file, shared by every Manifest object of the same project."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


MAX_CACHED = 6
_PARSED_LOCK = threading.Lock()
_PARSED: dict[Path, tuple[tuple[int, int], list[dict[str, Any]]]] = {}  # file -> (mtime, size), its JSON


def _remember(key: Path, stamp: tuple[int, int], data: list[dict[str, Any]]) -> None:
    """Keep the parsed list of a manifest (the last few only: a big project holds tens of MB)."""
    with _PARSED_LOCK:
        _PARSED.pop(key, None)
        _PARSED[key] = (stamp, data)
        while len(_PARSED) > MAX_CACHED:
            _PARSED.pop(next(iter(_PARSED)))


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


@dataclass
class ImageRecord:
    """One image of a project.

    Attributes:
        id: first 16 hex digits of the SHA-256 of the file (content id); for a video frame the hash also
            covers the video, frame index and task, so identical frames of different tasks stay apart.
        name: original file name, with its sub-folder inside the upload (``a/b.jpg``).
        file: stored file, relative to the project folder (``images/<id>.<ext>``).
        width: pixels.
        height: pixels.
        size_bytes: file size.
        added: ISO time (UTC).
        status: ``active`` or ``excluded`` (left out of annotation and training).
        metrics: curation measures (brightness, contrast, sharpness, ...).
        issues: curation flags (dark, bright, low_contrast, blurry, small, odd_aspect, grayscale,
            near_duplicate).
        phash: perceptual hash (hex) for near duplicates, empty until measured.
        excluded_reason: why it was excluded (free text or an issue name).
        video: id of the source video for a frame extracted from a video (docs/VIDEO.md), else empty.
        frame: index of the frame in that video.
        time: position of the frame in the video, in seconds.
    """

    id: str
    name: str
    file: str
    width: int
    height: int
    size_bytes: int
    added: str
    status: str = ACTIVE
    metrics: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    phash: str = ""
    excluded_reason: str = ""
    video: str = ""
    frame: int | None = None
    time: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (not ``dataclasses.asdict``: that is several times slower on 50 000 images)."""
        return {**self.__dict__, "metrics": dict(self.metrics), "issues": list(self.issues)}


class Manifest:
    """Images of one project, saved in ``<project>/images.json``."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.path = project_dir / "images.json"
        self._lock = _lock_for(self.path)

    def _items(self) -> list[dict[str, Any]]:
        """The JSON list, parsed again only when the file changed (the parse is most of the cost)."""
        key, stamp = self.path.resolve(), _stamp(self.path)
        with _PARSED_LOCK:
            cached = _PARSED.get(key)
        if stamp is not None and cached is not None and cached[0] == stamp:
            return cached[1]
        data: list[dict[str, Any]] = read_json(self.path, [])
        if stamp is not None:
            _remember(key, stamp, data)
        return data

    def load(self) -> dict[str, ImageRecord]:
        """Every image by id, in the order they were added (new objects: change them freely)."""
        records = {}
        for item in self._items():
            records[item["id"]] = ImageRecord(
                **{**item, "metrics": dict(item.get("metrics", {})), "issues": list(item.get("issues", []))}
            )
        return records

    def save(self, records: dict[str, ImageRecord]) -> None:
        """Replace the list."""
        data = [r.as_dict() for r in records.values()]
        with self._lock:
            write_json(self.path, data, indent=None)  # long list: compact
            stamp = _stamp(self.path)
            if stamp is not None:
                _remember(self.path.resolve(), stamp, data)

    def all(self) -> list[ImageRecord]:
        """Every image."""
        return list(self.load().values())

    def active(self) -> list[ImageRecord]:
        """Images that are not excluded."""
        return [r for r in self.load().values() if r.status == ACTIVE]

    def get(self, image_id: str) -> ImageRecord:
        """One image.

        Raises:
            KeyError: unknown id.
        """
        return self.load()[image_id]

    def file_path(self, record: ImageRecord) -> Path:
        """Absolute path of the stored file."""
        return self.project_dir / record.file

    def add(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """Add images; ids already present are skipped. Returns the ones added."""
        with self._lock:
            current = self.load()
            added = [r for r in records if r.id not in current]
            for record in added:
                current[record.id] = record
            self.save(current)
            return added

    def update(self, changes: dict[str, dict[str, Any]]) -> None:
        """Change fields of several images at once: ``{id: {field: value}}``. Unknown ids are ignored."""
        with self._lock:
            current = self.load()
            for image_id, values in changes.items():
                record = current.get(image_id)
                if record is None:
                    continue
                for key, value in values.items():
                    if not hasattr(record, key) or key in ("id", "file"):
                        raise ValueError(f"Field '{key}' cannot be changed.")
                    setattr(record, key, value)
            self.save(current)

    def set_status(self, ids: list[str], status: str, reason: str = "") -> int:
        """Exclude or restore images. Returns how many changed."""
        if status not in (ACTIVE, EXCLUDED):
            raise ValueError(f"Unknown status: {status}")
        with self._lock:
            current = self.load()
            changed = 0
            for image_id in ids:
                record = current.get(image_id)
                if record is not None and record.status != status:
                    record.status = status
                    record.excluded_reason = reason if status == EXCLUDED else ""
                    changed += 1
            self.save(current)
            return changed

    def delete(self, ids: list[str]) -> int:
        """Remove images and their files for good. Returns how many were removed."""
        with self._lock:
            current = self.load()
            gone = [r for r in (current.pop(i, None) for i in ids) if r is not None]
            self.save(current)  # first: a file that cannot be removed now does not leave a stale manifest
        for record in gone:
            with contextlib.suppress(OSError):  # e.g. held open by a download on Windows
                self.file_path(record).unlink(missing_ok=True)
        return len(gone)
