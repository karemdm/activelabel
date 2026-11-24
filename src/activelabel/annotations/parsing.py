"""Format detection and the ``parse`` entry point."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from activelabel.annotations.coco import parse_coco
from activelabel.annotations.context import ParseContext
from activelabel.annotations.folders import parse_folders
from activelabel.annotations.labelstudio import parse_labelstudio
from activelabel.annotations.models import ParseResult
from activelabel.annotations.target import VideoTarget
from activelabel.annotations.videoformats import (
    is_cvat_video,
    is_mot_text,
    parse_coco_frames,
    parse_cvat_video,
    parse_mot,
    parse_yolo_frames,
)
from activelabel.annotations.xmlformats import parse_cvat, parse_voc
from activelabel.annotations.yolo import is_label_text, is_names_file, parse_yolo
from activelabel.images.manifest import ImageRecord

FORMATS = ("coco", "yolo", "voc", "cvat", "labelstudio", "folders", "cvatvideo", "mot")
VIDEO_ONLY = ("cvatvideo", "mot")  # formats that only make sense for a video task


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
            if element.tag == "annotations" and is_cvat_video(path):
                return "cvatvideo"
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
            text = path.read_text(encoding="utf-8-sig")
            if is_mot_text(text):
                return "mot"
            return "labels" if is_label_text(text) is not False else None
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return None


def detect_format(files: list[Path]) -> str | None:
    """Annotation format by content: coco, yolo, voc, cvat, labelstudio, cvatvideo (CVAT video), mot or None.

    Several YOLO txt files plus data.yaml count as one dataset. Unrelated files are ignored; files of
    two different formats give None. Class folders (``folders``) come from image names, see
    :func:`activelabel.annotations.folders.folder_classes`.
    """
    found: set[str] = set()
    yolo_labels = 0
    mot = False
    for path in files:
        suffix = path.suffix.lower()
        fmt = _json_format(path) if suffix == ".json" else _xml_format(path) if suffix == ".xml" else None
        if fmt:
            found.add(fmt)
        elif suffix in (".txt", ".yaml", ".yml", ".names"):
            part = _yolo_part(path)
            if part == "labels" and path.read_text(encoding="utf-8-sig").strip():
                yolo_labels += 1
            elif part == "mot":
                mot = True
    if yolo_labels:
        found.add("yolo")
    if mot:
        found.add("mot")
    return next(iter(found)) if len(found) == 1 else None


def unreadable_file(files: list[Path]) -> tuple[str, str] | None:
    """The first JSON or XML file that is not well-formed, as ``(file name, parser message)``, or None.

    Used to say what is wrong when :func:`detect_format` finds no format: a cut-off file is not "an
    unknown format".
    """
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                json.loads(path.read_text(encoding="utf-8-sig"))
            elif suffix == ".xml":
                ET.parse(path)
        except (ValueError, ET.ParseError) as exc:  # UnicodeDecodeError is a ValueError
            return path.name, str(exc)
        except OSError:
            continue
    return None


def parse(
    files: list[Path], fmt: str, images: list[ImageRecord], target: VideoTarget | None = None
) -> ParseResult:
    """Read annotation files and match them to images.

    Images match by exact relative name, then by the same name ignoring case, then by file name alone
    ignoring case; a name shared by several images is reported (``ambiguous_image_name``) and skipped.
    Coordinates become relative using the image size of the record, or the size written in the file
    (COCO, VOC, CVAT) when it differs (``size_mismatch``).

    Args:
        files: annotation files (see :func:`detect_format`).
        fmt: one of ``FORMATS``: coco, yolo, voc, cvat, labelstudio, folders (ignores ``files``), cvatvideo
            and mot (these two need ``target``).
        images: images of the project (of the video task, when ``target`` is given).
        target: a video task. COCO and YOLO files are then matched by frame number instead of by file
            name (see :mod:`activelabel.annotations.videoformats`); cvatvideo and mot need it.

    Returns:
        The labels per image id and what was found; nothing is written to disk.

    Raises:
        ValueError: unknown format, or a video format without a video task.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown annotation format: {fmt}")
    if fmt in VIDEO_ONLY and target is None:
        raise ValueError("This file is made for a video: choose the video task it belongs to.")
    ctx = ParseContext(fmt, images, target)
    if fmt == "folders":
        parse_folders(ctx, images)
    else:
        parsers = {
            "coco": parse_coco_frames if target else parse_coco,
            "yolo": parse_yolo_frames if target else parse_yolo,
            "voc": parse_voc,
            "cvat": parse_cvat,
            "labelstudio": parse_labelstudio,
            "cvatvideo": parse_cvat_video,
            "mot": parse_mot,
        }
        parsers[fmt](ctx, files)
    return ctx.result()
