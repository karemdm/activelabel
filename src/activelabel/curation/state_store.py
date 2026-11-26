"""A small JSON state file per project and analysis (uniqueness, label checks...), with its own lock."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from activelabel.storage import read_json, write_json

_locks: dict[Path, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock(path: Path) -> threading.RLock:
    """The lock of one state file (shared by every store of the same file).

    Args:
        path: The state file.

    Returns:
        Its lock.
    """
    with _locks_guard:
        return _locks.setdefault(path.resolve(), threading.RLock())


class StateStore:
    """The state file of one analysis of one project."""

    def __init__(self, project_dir: Path, file_name: str) -> None:
        """Bind the store to a project folder.

        Args:
            project_dir: Folder of the project.
            file_name: Name of the state file in it (``uniqueness.json``...).
        """
        self.path = Path(project_dir) / file_name
        self.lock = _lock(self.path)  # held by whoever reads, changes and writes the state together

    def read(self) -> dict[str, Any]:
        """The state of the last computation.

        Returns:
            The saved state; ``{}`` when it was never computed.
        """
        data = read_json(self.path, {})
        return data if isinstance(data, dict) else {}

    def update(self, **fields: Any) -> dict[str, Any]:
        """Change some fields of the state and save it.

        Args:
            **fields: The fields to set.

        Returns:
            The whole state after the change.
        """
        with self.lock:
            data = {**self.read(), **fields}
            write_json(self.path, data, indent=None)
        return data

    def clear(self) -> None:
        """Delete the state."""
        self.path.unlink(missing_ok=True)
