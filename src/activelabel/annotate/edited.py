"""Images edited in the FiftyOne App and not brought back to the project yet (#700).

An edit session lends images to the App; what is edited there stays in the dataset until the session is
pulled and applied. This module counts those images for the Annotations step and the project header.

The count is cheap to ask for again and again: the images of the sessions are fixed by the session
list, and :func:`activelabel.fiftyone_bridge.edit_stamp` tells with one database aggregation whether any of
their samples was saved since the last count. Only then are the samples read (the same
:func:`~activelabel.fiftyone_bridge.edited_images` the pull uses, so there is one definition of "edited").
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from activelabel import fiftyone_bridge
from activelabel.annotate import sessions
from activelabel.optional import FIFTYONE
from activelabel.tasks import Task

log = logging.getLogger(__name__)
#: Sessions whose images may hold edits not in the project: still open, pulled but not applied, or applied
#: and edited again (the App's "send" button can bring those back, see ``annotate_api.pull``).
TRACKED = (sessions.OPEN, sessions.PULLED, sessions.APPLIED)
_lock = threading.Lock()
#: Last answer per project folder: ``(images looked at, stamp of their samples, images edited)``.
_cache: dict[Path, tuple[frozenset[str], str, frozenset[str]]] = {}


def tracked(project_dir: Path) -> dict[str, list[str]]:
    """The images of the FiftyOne sessions that may hold edits made in the App.

    Args:
        project_dir: the project folder.

    Returns:
        The image ids by session id (sessions of other tools, and discarded ones, are left out).
    """
    return {
        s["id"]: list(s.get("image_ids", []))
        for s in sessions.list_sessions(project_dir)
        if s.get("tool") == "fiftyone" and s.get("status") in TRACKED
    }


def _edited(project_dir: Path, dataset: Any, image_ids: frozenset[str], task: Task) -> frozenset[str]:
    """The edited images among ``image_ids``, read again only when a sample was saved since last time.

    Args:
        project_dir: the project folder (the cache key).
        dataset: the project's FiftyOne dataset.
        image_ids: the images to look at.
        task: task of the project.

    Returns:
        The ids of the images whose ``ground_truth`` was changed in FiftyOne since ActiveLabel wrote it.
    """
    stamp = fiftyone_bridge.edit_stamp(dataset, list(image_ids))
    with _lock:
        known = _cache.get(project_dir)
    if known and known[0] == image_ids and known[1] == stamp:
        return known[2]
    edited = frozenset(fiftyone_bridge.edited_images(dataset, list(image_ids), task))
    with _lock:
        _cache[project_dir] = (image_ids, stamp, edited)
    return edited


def summary(project_dir: Path, open_dataset: Callable[[], Any], task: Task) -> dict[str, Any]:
    """How many images were edited in FiftyOne and are not in the project yet.

    The dataset is opened only when FiftyOne is installed and a session may hold edits. Best effort: the
    Annotations step must load even when the dataset cannot be read, so a failure is logged and counts as
    nothing edited.

    Args:
        project_dir: the project folder.
        open_dataset: gives the project's FiftyOne dataset, or None when it does not exist (yet).
        task: task of the project.

    Returns:
        ``{edited_count, edited}``: the number of edited images, and how many of them each session holds
        (sessions with none are left out; an image of two sessions counts in both, once in the total).
    """
    by_session = tracked(project_dir)
    image_ids = frozenset(i for ids in by_session.values() for i in ids)
    if not FIFTYONE or not image_ids:
        return {"edited_count": 0, "edited": {}}
    try:
        dataset = open_dataset()
        if dataset is None:
            return {"edited_count": 0, "edited": {}}
        edited = _edited(project_dir, dataset, image_ids, task)
    except Exception:  # noqa: BLE001 - a broken count must not break the list of sessions
        log.warning("Could not count the images edited in FiftyOne", exc_info=True)
        return {"edited_count": 0, "edited": {}}
    per_session = {sid: n for sid, ids in by_session.items() if (n := len(edited.intersection(ids)))}
    return {"edited_count": len(edited), "edited": per_session}
