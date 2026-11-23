"""The list of images of a project (``images.json``).

Images are stored by content: the id is a hash of the file, so the same image uploaded
twice is detected. Excluding an image keeps its file (it can be restored); deleting it
removes the file.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from activelabel.storage import read_json, write_json

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per file, shared by every Manifest object of the same project."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


ACTIVE = "active"
EXCLUDED = "excluded"


@dataclass
class ImageRecord:
    """One image of a project.

    Attributes:
        id: first 16 hex digits of the SHA-256 of the file (content id).
        name: original file name, with its sub-folder inside the upload (``a/b.jpg``).
        file: stored file, relative to the project folder (``images/<id>.<ext>``).
        width: pixels.
        height: pixels.
        size_bytes: file size.
        added: ISO time (UTC).
        status: ``active`` or ``excluded`` (left out of annotation and training).
        metrics: curation measures (brightness, contrast, sharpness, ...).
        issues: curation flags (dark, blurry, duplicate, ...).
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
        """JSON-ready form."""
        return asdict(self)


class Manifest:
    """Images of one project, saved in ``<project>/images.json``."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.path = project_dir / "images.json"
        self._lock = _lock_for(self.path)

    def load(self) -> dict[str, ImageRecord]:
        """Every image by id, in the order they were added."""
        data = read_json(self.path, [])
        return {item["id"]: ImageRecord(**item) for item in data}

    def save(self, records: dict[str, ImageRecord]) -> None:
        """Replace the list."""
        with self._lock:
            write_json(self.path, [r.as_dict() for r in records.values()])

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
