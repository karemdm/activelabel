"""Format detection and the ``parse`` entry point."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from visionloop.annotations.coco import parse_coco
from visionloop.annotations.context import ParseContext
from visionloop.annotations.folders import parse_folders
from visionloop.annotations.labelstudio import parse_labelstudio
from visionloop.annotations.models import ParseResult
from visionloop.annotations.xmlformats import parse_cvat, parse_voc
from visionloop.annotations.yolo import is_label_text, is_names_file, parse_yolo
from visionloop.images.manifest import ImageRecord

FORMATS = ("coco", "yolo", "voc", "cvat", "labelstudio", "folders")


def _json_format(path: Path) -> str | None:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("images"), list) and "annotations" in data:
        return "coco"
    if isinstance(data, list) and data and all(isinstance(t, dict) for t in data):
        if any(isinstance(t.get("data"), dict) for t in data):
            return "labelstudio"
    return None


def _xml_format(path: Path) -> str | None:
    try:
        for _event, element in ET.iterparse(str(path), events=("start",)):
            return {"annotation": "voc", "annotations": "cvat"}.get(element.tag)
    except (ET.ParseError, OSError):
        return None
    return None


def _yolo_part(path: Path) -> str | None:
    """Return labels for a label txt, names for data.yaml or classes files, None when unrelated."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            return "names" if isinstance(data, dict) and "names" in data else None
        if is_names_file(path):
            return "names"
        if suffix == ".txt":
            return "labels" if is_label_text(path.read_text(encoding="utf-8-sig")) is not False else None
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return None


def detect_format(files: list[Path]) -> str | None:
    """Annotation format of these files by content: coco, yolo, voc, cvat, labelstudio or None.

    Several YOLO txt files plus data.yaml count as one dataset. Unrelated files are ignored; files of
    two different formats give None. Class folders (``folders``) come from image names, see
    :func:`visionloop.annotations.folders.folder_classes`.
    """
    found: set[str] = set()
    yolo_labels = 0
    for path in files:
        suffix = path.suffix.lower()
        fmt = _json_format(path) if suffix == ".json" else _xml_format(path) if suffix == ".xml" else None
        if fmt:
            found.add(fmt)
        elif suffix in (".txt", ".yaml", ".yml", ".names"):
            if _yolo_part(path) == "labels" and path.read_text(encoding="utf-8-sig").strip():
                yolo_labels += 1
    if yolo_labels:
        found.add("yolo")
    return next(iter(found)) if len(found) == 1 else None


def parse(files: list[Path], fmt: str, images: list[ImageRecord]) -> ParseResult:
    """Read annotation files and match them to images.

    Images match by exact relative name, then by case-insensitive file name; a file name shared by
    several images is reported (``ambiguous_image_name``) and skipped. Coordinates become relative using
    the image size of the record, or the size written in the file (COCO, VOC, CVAT) when it differs
    (``size_mismatch``).

    Args:
        files: annotation files (see :func:`detect_format`).
        fmt: one of coco, yolo, voc, cvat, labelstudio, folders (folders ignores ``files``).
        images: images of the project.

    Returns:
        The labels per image id and what was found; nothing is written to disk.

    Raises:
        ValueError: unknown format.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown annotation format: {fmt}")
    ctx = ParseContext(fmt, images)
    if fmt == "folders":
        parse_folders(ctx, images)
    else:
        {
            "coco": parse_coco,
            "yolo": parse_yolo,
            "voc": parse_voc,
            "cvat": parse_cvat,
            "labelstudio": parse_labelstudio,
        }[fmt](ctx, files)
    return ctx.result()
