"""Tags of the images of a project, brought back from FiftyOne (``<project>/tags.json``).

Format::

    {"images": {"<image id>": ["blurry", "check"], ...}}

An image without tags is not listed. The tags of the labels themselves are kept with each label in
``annotations.json`` (``Label.tags``). The tags ActiveLabel uses itself in FiftyOne (``excluded``, and
``exclude``, which asks for an exclusion) are never stored here.
"""

from __future__ import annotations

import threading
from pathlib import Path

from activelabel.storage import read_json, write_json

TAGS_FILE = "tags.json"
SYSTEM_TAGS = frozenset({"excluded", "exclude"})  # tags with a meaning of their own in FiftyOne
_lock = threading.RLock()


def image_tags(tags: list[str] | tuple[str, ...]) -> list[str]:
    """The tags worth keeping: not empty, not a system tag, each once, in their order.

    Args:
        tags: tags as FiftyOne or the person gives them.

    Returns:
        The cleaned list.
    """
    out: list[str] = []
    for tag in tags:
        name = str(tag).strip()
        if name and name not in SYSTEM_TAGS and name not in out:
            out.append(name)
    return out


def load_tags(project_dir: Path) -> dict[str, list[str]]:
    """The tags of every image that has some.

    Args:
        project_dir: the project folder.

    Returns:
        Tags by image id (an empty dict when the file does not exist or holds something else; a file that
        is not JSON raises :class:`~activelabel.storage.DamagedFile`, like every data file).
    """
    data = read_json(project_dir / TAGS_FILE, {})
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, dict):
        return {}
    return {k: [str(t) for t in v] for k, v in images.items() if isinstance(v, list) and v}


def set_tags(project_dir: Path, tags: dict[str, list[str]]) -> int:
    """Replace the tags of some images (an empty list removes them); the other images keep theirs.

    Args:
        project_dir: the project folder.
        tags: the new tags by image id.

    Returns:
        How many images got other tags than they had.
    """
    with _lock:
        current = load_tags(project_dir)
        changed = 0
        for image_id, values in tags.items():
            cleaned = image_tags(values)
            if cleaned != current.get(image_id, []):
                changed += 1
            if cleaned:
                current[image_id] = cleaned
            else:
                current.pop(image_id, None)
        if changed:
            write_json(project_dir / TAGS_FILE, {"images": current})
    return changed


def forget_images(project_dir: Path, image_ids: list[str] | set[str]) -> None:
    """Remove the tags of deleted images.

    Args:
        project_dir: the project folder.
        image_ids: the images deleted.
    """
    set_tags(project_dir, dict.fromkeys(image_ids, []))
