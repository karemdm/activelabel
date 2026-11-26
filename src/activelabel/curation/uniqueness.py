"""Uniqueness of the images: how far each one is from its nearest neighbours in the embeddings (#605, #610).

The embeddings are the ones the similarity map saved (:mod:`activelabel.curation.similarity`), so nothing
runs on the GPU here. The score follows FiftyOne Brain's ``compute_uniqueness``: the cosine distances to
the ``K`` nearest images, weighted (``WEIGHTS``), divided by the largest value, so the most unique image
scores 1 and near duplicates score close to 0. It is computed in chunks of rows, so a job can stop between
two chunks. The state and the scores are kept in ``uniqueness.json``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from activelabel.storage import read_json, write_json

STATE_FILE = "uniqueness.json"
#: Weights of the distances to the 1st, 2nd and 3rd nearest images (as in FiftyOne Brain).
WEIGHTS = (0.6, 0.3, 0.1)
#: Largest weighted distance still taken as "the same image" (float32 noise of the cosine distance).
EPSILON = 1e-5
#: Floats of the distance block computed at once (rows x images), about 16 MB.
BLOCK = 4_000_000

_locks: dict[Path, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock(path: Path) -> threading.RLock:
    """The lock of one state file (shared by every store of the same project).

    Args:
        path: The state file.

    Returns:
        Its lock.
    """
    with _locks_guard:
        return _locks.setdefault(path.resolve(), threading.RLock())


def scores(vectors: np.ndarray, check: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Uniqueness of each row of the embeddings, from 0 (has a twin) to 1 (the most unique).

    Args:
        vectors: Embeddings, one row per image.
        check: Called with ``(rows done, rows)`` after every chunk (it may raise to stop).

    Returns:
        One score per row; all zeros for fewer than two rows or when every row is the same image.
    """
    rows = len(vectors)
    if rows < 2:
        return np.zeros(rows, dtype=np.float64)
    unit = vectors.astype(np.float32)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    k = min(len(WEIGHTS), rows - 1)
    weights = np.array(WEIGHTS[:k], dtype=np.float64)
    weights /= weights.sum()
    raw = np.empty(rows, dtype=np.float64)
    chunk = max(1, BLOCK // rows)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        distances = 1.0 - unit[start:stop] @ unit.T
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf  # not its own neighbour
        nearest = np.sort(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)
        raw[start:stop] = np.clip(nearest, 0.0, None) @ weights
        if check is not None:
            check(stop, rows)
    top = float(raw.max())
    if top < EPSILON:  # all the images are the same: float noise must not become a score of 1
        return np.zeros(rows, dtype=np.float64)
    return raw / top


class UniquenessStore:
    """The uniqueness file of one project."""

    def __init__(self, project_dir: Path) -> None:
        """Bind the store to a project folder.

        Args:
            project_dir: Folder of the project.
        """
        self.path = Path(project_dir) / STATE_FILE
        self.lock = _lock(self.path)  # held by whoever reads, changes and writes the state together

    def read(self) -> dict[str, Any]:
        """The state of the last computation.

        Returns:
            The saved state (``scores``: image id to score); ``{}`` when it was never computed.
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
        """Delete the scores."""
        self.path.unlink(missing_ok=True)
