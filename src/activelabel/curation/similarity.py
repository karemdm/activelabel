"""Similarity map of the curation: embeddings of the images, a 2-D map of them and groups of similar images.

The embeddings are computed in the training environment (``training/workers/embed.py``); everything here
runs in the app with NumPy and scikit-learn (dependencies of FiftyOne), so the light install needs no
PyTorch. Files of a project:

- ``<project>/similarity.npz``: ``ids`` (image ids) and ``vectors`` (one embedding per image, length 1),
  with the ``backbone`` that made them;
- ``<project>/similarity.json``: the state of the last computation (``status``, progress, ``error``) and
  its result: the ``method`` of the map, the ``groups`` (size and most typical image of each) and the
  ``points`` (``ids``, ``xy`` scaled to 0..1, ``group`` of each image).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from activelabel.storage import read_json, write_json

STATE_FILE = "similarity.json"
VECTORS_FILE = "similarity.npz"
MAX_GROUPS = 8  # more groups than this are hard to read on a map
TSNE_LIMIT = 3000  # above this many images t-SNE is slow on a CPU: the map uses PCA
REDUCED_DIMS = 50  # the groups are found on the embeddings reduced to this many dimensions
SILHOUETTE_SAMPLE = 2000  # images used to compare the numbers of groups
SEED = 0
#: Ways to regroup the map (#614): ``auto`` (k-means, best silhouette), ``kmeans`` (a given number of
#: groups) and ``hdbscan`` (groups of free size; images that fit no group are noise, group -1).
CLUSTER_METHODS = ("auto", "kmeans", "hdbscan")
MAX_CHOSEN_GROUPS = 50  # most groups a person may ask k-means for
NOISE = -1

_LOCKS: dict[Path, threading.RLock] = {}
_GUARD = threading.Lock()


def _lock(path: Path) -> threading.RLock:
    """One lock per state file, shared by every store object of the project.

    Args:
        path: The state file (resolved, so every spelling of it shares one lock).

    Returns:
        The re-entrant lock of that file.
    """
    with _GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


def project_points(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    """Place every embedding on a 2-D map: similar images end up close together.

    t-SNE keeps neighbourhoods (groups stand out); PCA, used for very small or very large sets, keeps the
    main directions and is fast.

    Args:
        vectors: Embeddings, one row per image.

    Returns:
        The points (``N x 2``, each axis scaled to 0..1) and the method used (``tsne``, ``pca`` or ``none``
        for fewer than two images).
    """
    count = len(vectors)
    if count < 2:
        return np.full((count, 2), 0.5), "none"
    if 5 <= count <= TSNE_LIMIT:
        perplexity = float(min(30.0, (count - 1) / 3))
        points = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=SEED).fit_transform(
            vectors
        )
        method = "tsne"
    else:
        points = PCA(n_components=min(2, vectors.shape[1]), random_state=SEED).fit_transform(vectors)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros((count, 1))])
        method = "pca"
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return (points - low) / span, method


def find_groups(vectors: np.ndarray) -> np.ndarray:
    """Groups of similar images: k-means, with the number of groups (2 to ``MAX_GROUPS``) of best silhouette.

    Args:
        vectors: Embeddings, one row per image.

    Returns:
        The group of each image (0, 1, …); a single group 0 when there are fewer than four images or the
        images are all the same.
    """
    count = len(vectors)
    if count < 4 or np.allclose(vectors, vectors[0]):
        return np.zeros(count, dtype=int)
    dims = min(REDUCED_DIMS, count, vectors.shape[1])
    reduced = (
        PCA(n_components=dims, random_state=SEED).fit_transform(vectors)
        if dims < vectors.shape[1]
        else vectors
    )
    best_labels, best_score = np.zeros(count, dtype=int), -1.0
    for k in range(2, min(MAX_GROUPS, count - 1) + 1):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=SEED).fit_predict(reduced)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(
            silhouette_score(reduced, labels, sample_size=min(SILHOUETTE_SAMPLE, count), random_state=SEED)
        )
        if score > best_score:
            best_labels, best_score = labels, score
    return _by_size(best_labels)


def _reduced(vectors: np.ndarray) -> np.ndarray:
    """The embeddings reduced to at most ``REDUCED_DIMS`` dimensions (PCA), as the groups are found on them.

    Args:
        vectors: Embeddings, one row per image.

    Returns:
        The reduced rows (the embeddings themselves when they are already small).
    """
    dims = min(REDUCED_DIMS, len(vectors), vectors.shape[1])
    if dims >= vectors.shape[1]:
        return vectors
    return np.asarray(PCA(n_components=dims, random_state=SEED).fit_transform(vectors))


def cluster(vectors: np.ndarray, method: str = "auto", groups: int = 0, min_size: int = 5) -> np.ndarray:
    """Group the images again with a chosen method (#614).

    Args:
        vectors: Embeddings, one row per image.
        method: ``auto``, ``kmeans`` or ``hdbscan`` (see :data:`CLUSTER_METHODS`).
        groups: With ``kmeans``: how many groups (2 to :data:`MAX_CHOSEN_GROUPS`, at most one per image).
        min_size: With ``hdbscan``: the smallest group (2 or more).

    Returns:
        The group of each image, 0 the largest; ``hdbscan`` gives :data:`NOISE` to images of no group.

    Raises:
        ValueError: unknown method, or a number out of range.
    """
    if method not in CLUSTER_METHODS:
        raise ValueError(f"The method must be one of: {', '.join(CLUSTER_METHODS)}.")
    count = len(vectors)
    if method == "auto":
        return find_groups(vectors)
    if count < 4 or np.allclose(vectors, vectors[0]):
        return np.zeros(count, dtype=int)
    reduced = _reduced(vectors)
    if method == "kmeans":
        if not 2 <= groups <= MAX_CHOSEN_GROUPS:
            raise ValueError(f"The number of groups must be from 2 to {MAX_CHOSEN_GROUPS}.")
        labels = KMeans(n_clusters=min(groups, count), n_init="auto", random_state=SEED).fit_predict(reduced)
        return _by_size(labels)
    if min_size < 2:
        raise ValueError("The smallest group must have at least 2 images.")
    labels = HDBSCAN(min_cluster_size=min(min_size, count), copy=True).fit_predict(reduced)
    kept = labels != NOISE
    out = np.full(count, NOISE, dtype=int)
    if kept.any():
        out[kept] = _by_size(labels[kept])
    return out


def _by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber the groups from the largest (0) to the smallest.

    Args:
        labels: The group of each image.

    Returns:
        The same partition, group 0 being the largest (ties: the lower old number first).
    """
    groups, sizes = np.unique(labels, return_counts=True)
    order = sorted(range(len(groups)), key=lambda i: (-sizes[i], groups[i]))
    renumber = {int(groups[i]): new for new, i in enumerate(order)}
    return np.array([renumber[int(x)] for x in labels], dtype=int)


def typical_images(ids: list[str], vectors: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    """Size and most typical image (the closest to the mean of the group) of every group.

    Args:
        ids: Image id of each row.
        vectors: Embeddings, one row per image.
        labels: Group of each image.

    Returns:
        ``[{"group", "size", "typical"}]`` ordered by group number (the noise of HDBSCAN is not a group).
    """
    out = []
    for group in sorted(set(labels.tolist()) - {NOISE}):
        rows = np.flatnonzero(labels == group)
        centre = vectors[rows].mean(axis=0)
        closest = rows[int(np.argmin(np.linalg.norm(vectors[rows] - centre, axis=1)))]
        out.append({"group": int(group), "size": int(len(rows)), "typical": ids[int(closest)]})
    return out


def build_map(ids: list[str], vectors: np.ndarray) -> dict[str, Any]:
    """The map and the groups of a set of embeddings.

    Args:
        ids: Image id of each row.
        vectors: Embeddings, one row per image.

    Returns:
        ``{"method", "groups", "points": {"ids", "xy", "group"}}`` (see the module docstring).
    """
    points, method = project_points(vectors)
    labels = find_groups(vectors)
    return {
        "method": method,
        "groups": typical_images(ids, vectors, labels),
        "points": {
            "ids": list(ids),
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in points],
            "group": [int(g) for g in labels],
        },
    }


class SimilarityStore:
    """The similarity files of one project."""

    def __init__(self, project_dir: Path) -> None:
        """Bind the store to a project folder.

        Args:
            project_dir: Folder of the project.
        """
        self.project_dir = Path(project_dir)
        self.state_path = self.project_dir / STATE_FILE
        self.vectors_path = self.project_dir / VECTORS_FILE
        self.lock = _lock(self.state_path)  # held by whoever reads, changes and writes the state together

    def read(self) -> dict[str, Any]:
        """The state of the last computation.

        Returns:
            The saved state; ``{}`` when the map was never computed.
        """
        data = read_json(self.state_path, {})
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
            write_json(self.state_path, data, indent=None)
        return data

    def save_vectors(self, ids: list[str], vectors: np.ndarray, backbone: str) -> None:
        """Keep the embeddings for the next features that use them (duplicates, leaks, similar images).

        Args:
            ids: Image id of each row.
            vectors: Embeddings, one row per image.
            backbone: The network that made them.
        """
        with self.vectors_path.open("wb") as handle:
            np.savez_compressed(
                handle, ids=np.array(ids), vectors=vectors.astype(np.float32), backbone=backbone
            )

    def load_vectors(self) -> tuple[list[str], np.ndarray, str] | None:
        """The saved embeddings.

        Returns:
            ``(ids, vectors, backbone)``, or None when there are none.
        """
        if not self.vectors_path.exists():
            return None
        with np.load(self.vectors_path) as data:
            return [str(x) for x in data["ids"]], np.asarray(data["vectors"]), str(data["backbone"])

    def clear(self) -> None:
        """Delete the map and the embeddings."""
        self.state_path.unlink(missing_ok=True)
        self.vectors_path.unlink(missing_ok=True)


def is_outdated(state: dict[str, Any], active_ids: set[str]) -> bool:
    """Whether the images changed since the map was made (an image added, excluded or deleted).

    Args:
        state: The saved state (see :class:`SimilarityStore`).
        active_ids: Ids of the active images of the project now.

    Returns:
        True when a finished map does not cover exactly the active images (``False`` without a map).
    """
    points = state.get("points") or {}
    if not points.get("ids"):
        return False
    return set(points["ids"]) != active_ids - set(state.get("unreadable") or [])
