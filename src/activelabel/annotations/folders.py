"""Image classification from a folder tree (``<class>/<image>``), derived from the image names."""

from __future__ import annotations

from activelabel.annotations.context import ParseContext, norm_name
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label

MIN_CLASSES = 2
MIN_SHARE = 0.9


def folder_classes(names: list[str]) -> dict[str, str] | None:
    """Class per image name when the names sit inside class folders, else None.

    The class is the immediate parent folder. Needs at least 2 different folders and at least 90% of
    the images inside a folder. Images at the top level get no class.

    Args:
        names: ``ImageRecord.name`` values (``cat/a.jpg``).

    Returns:
        ``{name: class}`` for the images that are inside a folder, or None.
    """
    found: dict[str, str] = {}
    for name in names:
        parts = norm_name(name).split("/")
        if len(parts) >= 2 and parts[-2]:
            found[name] = parts[-2]
    if not names or len(found) < MIN_SHARE * len(names) or len(set(found.values())) < MIN_CLASSES:
        return None
    return found


def parse_folders(ctx: ParseContext, images: list[ImageRecord]) -> None:
    """Label each image with its parent folder (nothing when the names are not in class folders)."""
    mapping = folder_classes([r.name for r in images])
    if mapping is None:
        return
    for cls in sorted(set(mapping.values()), key=str.lower):
        ctx.add_class(cls)
    for rec in images:
        if rec.name in mapping:
            ctx.claim(rec)
            ctx.add_label(rec, Label(mapping[rec.name]))
