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


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
        _retry(lambda: os.replace(tmp, path))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, or ``default`` when it does not exist."""
    if not path.exists():
        return default
    return json.loads(_retry(lambda: path.read_text(encoding="utf-8")))
