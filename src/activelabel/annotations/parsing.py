"""Format detection and the ``parse`` entry point."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from activelabel.annotations.coco import parse_coco
from activelabel.annotations.context import ParseContext
from activelabel.annotations.folders import folder_classes, parse_folders
from activelabel.annotations.models import ParseResult
from activelabel.annotations.regression import is_regression_json, parse_regression
from activelabel.annotations.staging import IMAGE_SUFFIXES
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

FORMATS = ("coco", "yolo", "voc", "cvat", "folders", "cvatvideo", "mot", "regression")
VIDEO_ONLY = ("cvatvideo", "mot")  # formats that only make sense for a video task


def _json_format(path: Path) -> str | None:
    """Format of a JSON file.

    Args:
        path: the file.

    Returns:
        ``"coco"`` when it looks like COCO, ``"regression"`` when it holds values per image and target
        (see :mod:`activelabel.annotations.regression`), else None (also when it cannot be read).
    """
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("images"), list) and "annotations" in data:
        return "coco"
    return "regression" if is_regression_json(data) else None


def _xml_format(path: Path) -> str | None:
    """Format of an XML file, from its root element.

    Args:
        path: the file.

    Returns:
        ``"cvatvideo"``, ``"voc"`` or ``"cvat"``, else None (also when it cannot be read).
    """
    try:
        for _event, element in ET.iterparse(str(path), events=("start",)):
            if element.tag == "annotations" and is_cvat_video(path):
                return "cvatvideo"
            return {"annotation": "voc", "annotations": "cvat"}.get(element.tag)
    except (ET.ParseError, OSError):
        return None
    return None


def _yolo_part(path: Path) -> str | None:
    """Return labels for a label txt, names for data.yaml or classes files, None when unrelated.

    Args:
        path: the file.

    Returns:
        ``"labels"``, ``"names"`` or ``"mot"``, or None when it is unrelated.
    """
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
    """Annotation format by content: coco, yolo, voc, cvat, cvatvideo (CVAT video), mot, regression or None.

    Several YOLO txt files plus data.yaml count as one dataset. Unrelated files are ignored; files of
    two different formats give None. Class folders (``folders``) are recognised when the files are
    only images (see :func:`activelabel.annotations.staging.stage`) in at least two class folders,
    see :func:`activelabel.annotations.folders.folder_classes`.

    Args:
        files: the staged files.

    Returns:
        The format name, or None.
    """
    found: set[str] = set()
    yolo_labels = 0
    mot = False
    for path in files:
        suffix = path.suffix.lower()
        fmt = (
            _json_format(path)
            if suffix == ".json"
            else _xml_format(path)
            if suffix == ".xml"
            else "regression"
            if suffix == ".csv"
            else None
        )
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
    if not found and files and all(p.suffix.lower() in IMAGE_SUFFIXES for p in files):
        return "folders" if folder_classes([f"{p.parent.name}/{p.name}" for p in files]) else None
    return next(iter(found)) if len(found) == 1 else None


@dataclass(frozen=True)
class Unreadable:
    """Where and how an annotation file is malformed (the place is 0 when the parser does not say).

    Attributes:
        file: file name.
        kind: ``json``, ``xml`` or ``text`` (not valid UTF-8).
        line: line of the first error.
        column: column of the first error.
        reason: the parser's own message (English; for the logs and the API ``detail``).
    """

    file: str
    kind: str
    line: int = 0
    column: int = 0
    reason: str = ""


def unreadable_file(files: list[Path]) -> Unreadable | None:
    """The first JSON or XML file that is not well-formed, or None.

    Used to say what is wrong when :func:`detect_format` finds no format: a cut-off file is not "an
    unknown format".

    Args:
        files: the staged files.

    Returns:
        What is wrong with the first malformed file, or None.
    """
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                json.loads(path.read_text(encoding="utf-8-sig"))
            elif suffix == ".xml":
                ET.parse(path)
        except json.JSONDecodeError as exc:
            return Unreadable(path.name, "json", exc.lineno, exc.colno, str(exc))
        except ET.ParseError as exc:
            line, column = exc.position
            return Unreadable(path.name, "xml", line, column, str(exc))
        except ValueError as exc:  # UnicodeDecodeError
            return Unreadable(path.name, "text", reason=str(exc))
        except OSError:
            continue
    return None


def parse(
    files: list[Path],
    fmt: str,
    images: list[ImageRecord],
    target: VideoTarget | None = None,
    excluded: list[ImageRecord] | None = None,
) -> ParseResult:
    """Read annotation files and match them to images.

    Images match by exact relative name, then by the same name ignoring case, then by file name alone
    ignoring case; a name shared by several images is reported (``ambiguous_image_name``) and skipped.
    Coordinates become relative using the image size of the record, or the size written in the file
    (COCO, VOC, CVAT) when it differs (``size_mismatch``).

    Args:
        files: annotation files (see :func:`detect_format`).
        fmt: one of ``FORMATS``: coco, yolo, voc, cvat, folders (ignores ``files``), cvatvideo
            and mot (these two need ``target``), regression (CSV or JSON values).
        images: images of the project (of the video task, when ``target`` is given).
        target: a video task. COCO and YOLO files are then matched by frame number instead of by file
            name (see :mod:`activelabel.annotations.videoformats`); cvatvideo and mot need it.
        excluded: excluded images of the project; entries for them are reported (``excluded_image_entries``)
            and skipped, not counted as unmatched.

    Returns:
        The labels per image id and what was found; nothing is written to disk.

    Raises:
        ValueError: unknown format, or a video format without a video task.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown annotation format: {fmt}")
    if fmt in VIDEO_ONLY and target is None:
        raise ValueError("This file is made for a video: choose the video task it belongs to.")
    ctx = ParseContext(fmt, images, target, excluded)
    if fmt == "folders":
        parse_folders(ctx, images)
    else:
        parsers = {
            "coco": parse_coco_frames if target else parse_coco,
            "yolo": parse_yolo_frames if target else parse_yolo,
            "voc": parse_voc,
            "cvat": parse_cvat,
            "cvatvideo": parse_cvat_video,
            "mot": parse_mot,
            "regression": parse_regression,
        }
        parsers[fmt](ctx, files)
    return ctx.result()
