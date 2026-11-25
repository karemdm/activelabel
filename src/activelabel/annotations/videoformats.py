"""Annotation files made for a video, matched to the frames of a video task by frame number.

Frame numbers are video frame indexes (0-based) unless the format says otherwise (MOT is 1-based).
A label on a frame that the task does not use is counted (``frames_not_in_task``), never an error.

* CVAT for video 1.1 XML: tracks (``box``, ``polygon``, ``points``; ``outside="1"`` entries are not
  visible and skipped) and non-tracked ``<image id=frame>`` entries. CVAT's own export writes video
  frame numbers (``0, 15, 30...`` for ``frame_filter = step=15``); a file with ``start_frame`` /
  ``frame_filter`` whose numbers are not all on that grid (``start_frame + k * step``) is read as task
  frames ``0..size-1`` and converted to video frames ``start + i * step``.
* MOT ``gt.txt`` / ``det.txt``: ``frame,id,left,top,width,height,conf,class,...``; frames are 1-based.
* YOLO txt per frame (``frame_000120.txt``): the trailing integer of the file name.
* COCO with ``frame_id`` / ``frame_index`` / ``frame`` in ``images``, or a trailing integer in
  ``file_name`` (COCO-VID, TAO).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from activelabel.annotations.coco import _annotation
from activelabel.annotations.context import ParseContext, declared_size
from activelabel.annotations.xmlformats import UNSUPPORTED_SHAPES, _f, _shape, read_xml
from activelabel.annotations.yolo import _line, is_names_file, load_names, majority_is_pose, read_label_texts
from activelabel.labels import Label

TRAILING_NUMBER = re.compile(r"(\d+)$")
TRACK_SHAPES = {"box", "polygon", "points"}


def trailing_frame(name: str) -> int | None:
    """Integer at the end of a file name without extension (``video_120.txt`` -> 120), else None.

    Args:
        name: file name; only its stem is read.

    Returns:
        The number, or None when there is none.
    """
    match = TRAILING_NUMBER.search(Path(name).stem)
    return int(match.group(1)) if match else None


def _int(value: Any) -> int | None:
    """Whole number held by a value.

    Args:
        value: a number or text.

    Returns:
        The value as int, or None when it is not a finite whole number.
    """
    try:
        number = float(value)
        return int(number) if number == int(number) else None
    except (TypeError, ValueError, OverflowError):  # not a number, NaN or infinity
        return None


def _cover(ctx: ParseContext, first: int | None, last: int | None) -> None:
    """Give every task image between the first and the last annotated frame an entry (maybe empty).

    Args:
        ctx: parse context; its target's images get the empty entries.
        first: first annotated frame; None does nothing.
        last: last annotated frame; None does nothing.
    """
    if ctx.target is None or first is None or last is None:
        return
    for frame, rec in ctx.target.frames.items():
        if first <= frame <= last:
            ctx.labels.setdefault(rec.id, [])


# -- CVAT for video 1.1 --------------------------------------------------------------
def is_cvat_video(path: Path) -> bool:
    """True for a CVAT XML that has ``<track>`` elements.

    Args:
        path: the file.

    Returns:
        True when it is a CVAT XML with ``<track>`` elements.
    """
    try:
        return path.suffix.lower() == ".xml" and read_xml(path).find("track") is not None
    except ValueError:
        return False


def _filter_step(text: str) -> int:
    """Step of a CVAT ``frame_filter`` text such as ``step=15``.

    Args:
        text: the text; may be empty.

    Returns:
        The step, 1 when there is none.
    """
    match = re.search(r"step\s*=\s*(\d+)", text or "")
    return max(1, int(match.group(1))) if match else 1


def _task_numbered(root: ET.Element, start: int, step: int, last: int | None) -> bool:
    """Whether the frame numbers of a CVAT video file count the task's frames (0..size-1).

    CVAT's export uses video frame numbers: they all lie on ``start + k * step``. Otherwise they are
    task frames.

    Args:
        root: root of the CVAT video file.
        start: first video frame of the task.
        step: the task takes one frame every ``step`` frames.
        last: last video frame of the task; None when unknown.

    Returns:
        True when the numbers are task frames, False when they are video frames.
    """
    numbers = [
        n
        for n in (
            *(_int(sh.get("frame")) for tr in root.findall("track") for sh in tr),
            *(_int(im.get("id")) for im in root.findall("image")),
        )
        if n is not None
    ]
    if not numbers:
        return False
    on_grid = all(n >= start and (n - start) % step == 0 and (last is None or n <= last) for n in numbers)
    return not (on_grid and (step > 1 or start > 0))


def parse_cvat_video(ctx: ParseContext, files: list[Path]) -> None:
    """Read CVAT for video 1.1 XML files into ``ctx``.

    Args:
        ctx: parse context; receives the classes, labels (tracks keep their id), files read and problems.
        files: staged files; only ``.xml`` files are read.
    """
    for path in files:
        if path.suffix.lower() != ".xml":
            continue
        try:
            root = read_xml(path)
            if root.tag != "annotations" or (root.find("track") is None and root.find("image") is None):
                raise ValueError("not a CVAT for video file")
        except ValueError as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)
        _read_cvat_video(ctx, root)


@dataclass
class _Grid:
    """How a CVAT video file numbers its frames and how big it says the frames are."""

    start: int = 0
    step: int = 1
    relative: bool = False  # numbers are task frames 0..size-1, converted to video frames
    width: float = 0.0
    height: float = 0.0
    last: int | None = None
    ranged: bool = False  # the meta gives the task's first and last video frame

    def frame(self, ctx: ParseContext, value: Any) -> int | None:
        """Video frame of a number in the file; None (and a problem) when it is not a frame number.

        Args:
            ctx: parse context that records the problem.
            value: frame number as written in the file.

        Returns:
            The video frame index, or None when the value is not a frame number.
        """
        number = _int(value)
        if number is None or number < 0:
            ctx.problem("invalid_annotation", f"frame {value}")
            return None
        return self.start + number * self.step if self.relative else number

    def size(self, rec_width: int, rec_height: int) -> tuple[float, float]:
        """The size written in the file, else the stored size of the image.

        Args:
            rec_width: stored width of the image.
            rec_height: stored height of the image.

        Returns:
            ``(width, height)`` in pixels.
        """
        if self.width > 0 and self.height > 0:
            return self.width, self.height
        return float(rec_width), float(rec_height)


def _read_grid(root: ET.Element) -> _Grid:
    """Read the frame numbering and size from the ``meta/task`` of a CVAT video file.

    Args:
        root: root of the CVAT video file.

    Returns:
        The numbering and size the file declares.
    """
    task = root.find("meta/task")
    grid = _Grid()
    if task is not None:
        first_text = (task.findtext("start_frame") or "").strip()
        filter_text = (task.findtext("frame_filter") or "").strip()
        grid.start = _int(first_text) or 0
        grid.step = _filter_step(filter_text)
        grid.last = _int((task.findtext("stop_frame") or "").strip())
        grid.ranged = bool(first_text) and grid.last is not None
        grid.relative = bool(first_text or filter_text) and _task_numbered(
            root, grid.start, grid.step, grid.last
        )
        grid.width = _f(task.findtext("original_size/width"))
        grid.height = _f(task.findtext("original_size/height"))
    return grid


def _read_cvat_video(ctx: ParseContext, root: ET.Element) -> None:
    """Read the classes, tracks and per-frame shapes of one CVAT video file into ``ctx``.

    Args:
        ctx: parse context; receives classes and labels (tracked shapes keep their track id).
        root: root of the CVAT video file.
    """
    grid = _read_grid(root)
    for labels in root.iter("labels"):
        for declared in labels.findall("label"):
            name = (declared.findtext("name") or "").strip()
            if name:
                ctx.add_class(name)

    seen: list[int] = []
    for track in root.findall("track"):
        name, track_id = track.get("label") or "", _int(track.get("id"))
        for shape in track:
            if shape.tag not in TRACK_SHAPES | UNSUPPORTED_SHAPES | {"skeleton"}:
                continue
            frame = grid.frame(ctx, shape.get("frame"))
            if frame is None or shape.get("outside") == "1":
                continue
            seen.append(frame)
            if shape.tag not in TRACK_SHAPES:
                ctx.problem("unsupported_shape", f"track {track_id}: {shape.tag}")
                continue
            rec = ctx.at_frame(frame)
            if rec is None:
                continue
            shape.set("label", name)
            w, h = grid.size(rec.width, rec.height)
            label = _shape(ctx, shape, w, h, f"track {track_id} frame {frame}")
            if label is not None:
                ctx.add_label(rec, replace(label, track=track_id))
    for image in root.findall("image"):
        frame = grid.frame(ctx, image.get("id"))
        if frame is None:
            continue
        seen.append(frame)
        rec = ctx.at_frame(frame)
        if rec is None:
            continue
        ctx.labels.setdefault(rec.id, [])
        w, h = grid.size(rec.width, rec.height)
        for shape in image:
            label = _shape(ctx, shape, w, h, f"frame {frame}")
            if label is not None:
                ctx.add_label(rec, label)
    if seen:
        if (grid.relative or grid.ranged) and grid.last is not None:
            _cover(ctx, grid.start, grid.last)
        else:
            _cover(ctx, min(seen), max(seen))


# -- MOT ------------------------------------------------------------------------------
def is_mot_text(text: str) -> bool:
    """True when every non-blank line is comma separated numbers with at least frame, id and a box.

    Args:
        text: content of a txt file.

    Returns:
        True when it looks like MOT.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    for line in lines:
        parts = line.split(",")
        if len(parts) < 6:
            return False
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            return False
        if _int(numbers[0]) is None:
            return False
    return True


def parse_mot(ctx: ParseContext, files: list[Path]) -> None:
    """Read MOT ``gt.txt`` / ``det.txt`` files; class ids are 1-based, names come from a labels file.

    Args:
        ctx: parse context; its target is required. Receives labels with their track id, files read and
            problems.
        files: staged files; names files give the classes, other ``.txt`` files with MOT lines are read.
    """
    names, _ = load_names(ctx, files)
    assert ctx.target is not None
    seen: list[int] = []
    for path in files:
        if path.suffix.lower() != ".txt" or is_names_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        if not is_mot_text(text):
            continue
        ctx.files.append(path.name)
        for line in text.splitlines():
            if not line.strip():
                continue
            v = [float(p) for p in line.split(",")]
            where = f"{path.name}: {line[:40]}"
            frame = int(v[0]) - 1  # MOT frames start at 1
            if frame < 0:
                ctx.problem("invalid_annotation", where)
                continue
            seen.append(frame)
            if names:
                cls = _int(v[7]) if len(v) > 7 else None
                if cls is None or cls - 1 not in names:
                    ctx.problem("unknown_class_id", f"{where} (class {cls})")
                    continue
                name = names[cls - 1]
            else:
                name = ctx.target.class_name
            rec = ctx.at_frame(frame, f"frame {frame + 1} (MOT)")
            if rec is None:
                continue
            box = ctx.box(v[2], v[3], v[4], v[5], rec.width, rec.height, where)
            if box is not None:
                ctx.add_label(rec, Label(name, bbox=box, track=_int(v[1])))
    if seen:
        _cover(ctx, min(seen), max(seen))


# -- YOLO per frame -----------------------------------------------------------------------
def parse_yolo_frames(ctx: ParseContext, files: list[Path]) -> None:
    """Read YOLO txt files named ``<anything><frame>.txt`` (the number is the video frame).

    Args:
        ctx: parse context; its target is required. Receives labels, files read and problems.
        files: staged files: label ``.txt`` files and the files with class names.
    """
    names, kpt = load_names(ctx, files)
    missing: set[int] = set()
    texts = read_label_texts(ctx, [p for p in files if p.suffix.lower() == ".txt" and not is_names_file(p)])
    pose_hint = majority_is_pose(texts.values())
    for path, text in texts.items():
        frame = trailing_frame(path.name)
        if frame is None:
            ctx.problem("frame_number_missing", path.name)
            continue
        rec = ctx.at_frame(
            frame, f"{path.name} (frame {frame})", n=max(1, sum(1 for ln in text.splitlines() if ln.strip()))
        )
        if rec is None or not ctx.claim(rec, path.name):
            continue
        ctx.files.append(path.name)
        for line in text.splitlines():
            if line.strip():
                label = _line(ctx, line, names, kpt, missing, pose_hint, f"{path.name}: {line[:40]}")
                if label is not None:
                    ctx.add_label(rec, label)
    if missing:
        ctx.problem("missing_class_names", ", ".join(f"class_{i}" for i in sorted(missing)[:5]))


# -- COCO with frames --------------------------------------------------------------------
def _coco_frame(image: dict[str, Any]) -> int | None:
    """Video frame of a COCO image entry.

    Args:
        image: the entry from ``images``.

    Returns:
        The frame number, or None when there is none.
    """
    for key in ("frame_id", "frame_index", "frame"):
        if image.get(key) is not None:
            return _int(image[key])
    return trailing_frame(str(image.get("file_name") or ""))


def parse_coco_frames(ctx: ParseContext, files: list[Path]) -> None:
    """Read COCO files whose images are frames of the video (``frame_id`` or a number in ``file_name``).

    Args:
        ctx: parse context; its target is required. Receives labels with their track id, files read and
            problems.
        files: staged files; only ``.json`` files are read.
    """
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("images"), list):
                raise ValueError("not a COCO file")
            _read_coco_frames(ctx, data)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)


def _own_video(ctx: ParseContext, data: dict[str, Any]) -> Any:
    """``video_id`` of this task's video when the file has several videos (matched by name), else None.

    Args:
        ctx: parse context; its target gives the video name.
        data: the decoded COCO JSON.

    Returns:
        The video id, or None when the file has a single video or none matches.
    """
    ids = {im.get("video_id") for im in data["images"] if im.get("video_id") is not None}
    if len(ids) < 2 or ctx.target is None:
        return None
    stem = Path(ctx.target.name).stem.lower()
    for video in data.get("videos") or []:
        label = str(video.get("name") or video.get("file_name") or "")
        if label and Path(label).stem.lower() == stem and video.get("id") in ids:
            return video["id"]
    return None


def _read_coco_frames(ctx: ParseContext, data: dict[str, Any]) -> None:
    """Read one decoded COCO document whose images are frames into ``ctx``.

    Args:
        ctx: parse context; receives classes and labels (with ``track_id`` as track).
        data: the decoded COCO JSON.
    """
    cats: dict[Any, dict[str, Any]] = {}
    for cat in data.get("categories") or []:
        cats[cat["id"]] = cat
        ctx.add_class(str(cat["name"]))
    own = _own_video(ctx, data)
    per_image = Counter(a.get("image_id") for a in data.get("annotations") or [])
    images: dict[Any, tuple[Any, float, float]] = {}
    for im in data["images"]:
        if own is not None and im.get("video_id") != own:
            continue
        name = str(im.get("file_name") or im.get("id"))
        frame = _coco_frame(im)
        # without a frame number the name is tried, as for images
        rec = (
            ctx.match(name)
            if frame is None
            else ctx.at_frame(frame, f"{name} (frame {frame})", n=max(1, per_image[im.get("id")]))
        )
        if rec is None or not ctx.claim(rec, name):
            continue
        width, height = declared_size(ctx, rec, im.get("width"), im.get("height"), name)
        images[im["id"]] = (rec, width, height)
    for ann in data.get("annotations") or []:
        entry = images.get(ann.get("image_id"))
        if entry is None:
            continue
        rec, width, height = entry
        label = _annotation(ctx, ann, cats, width, height, rec.name)
        if label is not None:
            track = _int(ann.get("track_id", ann.get("instance_id")))
            ctx.add_label(rec, replace(label, track=track))
