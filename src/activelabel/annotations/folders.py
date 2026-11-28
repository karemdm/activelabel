"""Image classification from a folder tree (``<class>/<image>``), derived from the image names."""

from __future__ import annotations

from activelabel.annotations.context import ParseContext, norm_name
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label

MIN_CLASSES = 2
MIN_SHARE = 0.9
#: Folder names that split a dataset (``train/cat/a.jpg``, ``cat/val/a.jpg``): never a class (#568).
SPLIT_FOLDERS = frozenset({"train", "training", "val", "valid", "validation", "test", "testing"})


def _folders(name: str) -> list[str]:
    """Folders of an image name, outermost first.

    Args:
        name: ``ImageRecord.name`` (``cat/a.jpg``).

    Returns:
        The folder parts of the name, without the file name and without empty parts.
    """
    return [p for p in norm_name(name).split("/")[:-1] if p]


def _is_split(folder: str) -> bool:
    """Whether a folder name is a split (train, val, test, ...), ignoring case.

    Args:
        folder: a folder name.

    Returns:
        True when the folder names a split of the dataset.
    """
    return folder.lower() in SPLIT_FOLDERS


def split_folders(names: list[str]) -> list[str]:
    """The split folders (train, val, test, ...) found in the names, sorted ignoring case.

    Args:
        names: ``ImageRecord.name`` values.

    Returns:
        The distinct split folder names as written, sorted; empty when there are none.
    """
    found = {f for name in names for f in _folders(name) if _is_split(f)}
    return sorted(found, key=str.lower)


def folder_classes(names: list[str]) -> dict[str, str] | None:
    """Class per image name when the names sit inside class folders, else None.

    The class is the nearest parent folder that is not a split (``train``, ``val``, ``test``, ...), so
    ``train/cat/a.jpg`` and ``cat/train/a.jpg`` are both ``cat``. Needs at least 2 different classes and
    at least 90% of the images with a class. Images at the top level (or only inside split folders) get
    no class.

    Args:
        names: ``ImageRecord.name`` values (``cat/a.jpg``).

    Returns:
        ``{name: class}`` for the images that have a class folder, or None.
    """
    found: dict[str, str] = {}
    for name in names:
        for folder in reversed(_folders(name)):
            if not _is_split(folder):
                found[name] = folder
                break
    if not names or len(found) < MIN_SHARE * len(names) or len(set(found.values())) < MIN_CLASSES:
        return None
    return found


def parse_folders(ctx: ParseContext, images: list[ImageRecord]) -> None:
    """Label each image with its parent folder (nothing when the names are not in class folders).

    Args:
        ctx: parse context; receives the classes (sorted ignoring case) and one label per image.
        images: image records whose names hold the class folders.
    """
    mapping = folder_classes([r.name for r in images])
    if mapping is None:
        return
    for cls in sorted(set(mapping.values()), key=str.lower):
        ctx.add_class(cls)
    for rec in images:
        if rec.name in mapping:
            ctx.claim(rec)
            ctx.add_label(rec, Label(mapping[rec.name]))
