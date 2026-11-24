"""Small helpers to keep files consistent on disk."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
RETRIES = 8
RETRY_DELAY = 0.02  # seconds, growing with each attempt (about 0.7 s in all)


class DamagedFile(ValueError):
    """A JSON file of the workspace cannot be read (cut off or edited by hand).

    A ``ValueError`` so code that already handles unreadable data keeps working; the web layer answers
    it with a readable message instead of a bare server error.

    Attributes:
        file: name of the damaged file (``project.json``).
    """

    def __init__(self, message: str, file: str = "") -> None:
        super().__init__(message)
        self.file = file


def now() -> str:
    """Current time, ISO 8601 in UTC (the interface shows it in local time)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _retry(action: Callable[[], T]) -> T:
    """Run a file operation, retrying briefly when Windows reports the file as in use.

    On Windows, replacing a file fails while any process (another thread, an antivirus, the search
    indexer) has it open; the condition lasts milliseconds.
    """
    for attempt in range(RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise AssertionError("unreachable")


def write_json(path: Path, data: Any, indent: int | None = 1) -> None:
    """Write JSON atomically: a reader never sees a half-written file, and a power cut keeps the old one.

    The data is flushed to disk before the file replaces the old one.

    Args:
        path: file to write (its folder is created).
        data: JSON-ready data.
        indent: spaces per level for a readable file; None for a compact one, about five times faster to
            write (used for the long lists of images).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=indent)
            handle.flush()
            os.fsync(handle.fileno())
        _retry(lambda: os.replace(tmp, path))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, or ``default`` when it does not exist.

    Raises:
        DamagedFile: the file is damaged (the message names the file and is readable).
    """
    if not path.exists():
        return default
    try:
        return json.loads(_retry(lambda: path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DamagedFile(
            f"The file {path.name} is damaged and cannot be read ({exc}). "
            f"Restore it from a backup or remove it: {path}",
            path.name,
        ) from exc
