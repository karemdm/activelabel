"""YOLO (Ultralytics) txt labels: detection, segmentation and pose.

Line shapes: ``cls cx cy w h`` (5 values) is a box; an odd count above 5 is a polygon; pose lines
(``cls cx cy w h`` plus ``x y [v]`` per point) are recognised when ``kpt_shape`` is in data.yaml, or when
the count is even and ``(n - 5)`` is a multiple of 3. Class names come from ``data.yaml`` (``names`` list
or dict), else ``classes.txt`` / ``obj.names``; without names, ``class_<n>`` is used.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml

from activelabel.annotations.context import ParseContext, bbox_of_points, stem_parts
from activelabel.labels import Label

NAME_FILES = {"classes.txt", "obj.names", "names.txt", "labels.txt"}


MIN_LABEL_VALUES = 5  # a label line: class and a box (``cls cx cy w h``)


def _numbers(line: str) -> list[float] | None:
    """Numbers of a line.

    Args:
        line: whitespace separated text.

    Returns:
        The values, or None when one is not a number.
    """
    try:
        return [float(t) for t in line.split()]
    except ValueError:
        return None


def _is_label_line(line: str) -> bool:
    """Whether a line has the values of a label.

    Args:
        line: one line of a txt file.

    Returns:
        True when all values are numbers and there are at least 5.
    """
    values = _numbers(line)
    return values is not None and len(values) >= MIN_LABEL_VALUES


def is_names_file(path: Path) -> bool:
    """True for ``classes.txt`` style files that list class names.

    A file called ``labels.txt`` (or ``classes.txt``, ``names.txt``) that holds label lines is the label
    file of an image of that name, not a list of names; ``.names`` files always list names.

    Args:
        path: the file.

    Returns:
        True when the file lists class names.
    """
    if path.suffix.lower() == ".names" or path.name.lower() == "obj.names":
        return True
    if path.name.lower() not in NAME_FILES:
        return False
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return True
    return not any(_is_label_line(ln) for ln in text.splitlines())


def is_label_text(text: str) -> bool | None:
    """Whether a txt file holds YOLO labels: True when most non-blank lines are numeric label lines.

    None when the file is empty; False for files that are not labels (``train.txt`` lists paths). A label
    file with a few malformed lines is still a label file: those lines are reported by the parser.

    Args:
        text: content of a txt file.

    Returns:
        True for labels, False for another kind of file, None when the file is empty.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    numeric = sum(1 for ln in lines if _numbers(ln) is not None)
    return numeric * 2 >= len(lines) and any(_is_label_line(ln) for ln in lines)


def read_names(data: Any) -> dict[int, str] | None:
    """Class names from a data.yaml ``names`` value (list or dict).

    Args:
        data: the ``names`` value: a list, or a dict from class id to name.

    Returns:
        The names by class id, or None when ``data`` is neither.
    """
    if isinstance(data, list):
        return {i: str(n) for i, n in enumerate(data)}
    if isinstance(data, dict):
        return {int(k): str(v) for k, v in data.items()}
    return None


def load_names(ctx: ParseContext, files: list[Path]) -> tuple[dict[int, str] | None, tuple[int, int] | None]:
    """Class names (and ``kpt_shape``) from data.yaml or classes.txt files; declares them in ``ctx``.

    Args:
        ctx: parse context; receives the class names, the files read and problems.
        files: staged files: ``data.yaml`` (preferred) or a names file.

    Returns:
        ``(names, kpt_shape)``: names by class id (None when none were found) and ``(points, values per
        point)`` (None when not set).
    """
    names: dict[int, str] | None = None
    kpt: tuple[int, int] | None = None
    for path in files:
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict) and "names" in data:
                    names = read_names(data["names"])
                    shape = data.get("kpt_shape")
                    if isinstance(shape, list) and len(shape) == 2:
                        kpt = (int(shape[0]), int(shape[1]))
                        if kpt[0] < 1 or kpt[1] not in (2, 3):
                            raise ValueError(f"kpt_shape must be [points, 2 or 3], not {shape}")
                    ctx.files.append(path.name)
            except (OSError, ValueError, TypeError, OverflowError, yaml.YAMLError) as exc:
                ctx.problem("invalid_file", f"{path.name}: {exc}")
    if names is None:
        for path in files:
            if is_names_file(path):
                try:
                    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
                    names = dict(enumerate(ln for ln in lines if ln))
                    ctx.files.append(path.name)
                except OSError as exc:
                    ctx.problem("invalid_file", f"{path.name}: {exc}")
                break
    if names:
        for index in sorted(names):
            ctx.add_class(names[index])
    return names, kpt


def parse_yolo(ctx: ParseContext, files: list[Path]) -> None:
    """Read YOLO label files (and data.yaml / classes.txt) into ``ctx``.

    Args:
        ctx: parse context; receives the classes, labels, files read and problems.
        files: staged files: label ``.txt`` files and the files with class names.
    """
    names, kpt = load_names(ctx, files)
    labels = [p for p in files if p.suffix.lower() == ".txt" and not is_names_file(p)]
    try:
        root = Path(os.path.commonpath([str(p.parent) for p in labels])) if labels else Path()
    except ValueError:
        root = Path()
    missing: set[int] = set()
    for path in labels:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        if is_label_text(text) is False:
            continue  # train.txt, readme.txt, ...
        try:
            rel = path.relative_to(root).as_posix() if root != Path() else path.name
        except ValueError:
            rel = path.name
        rec = ctx.match_stem(stem_parts(rel), rel)
        if rec is None or not ctx.claim(rec, rel):
            continue
        ctx.files.append(rel)
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                label = _line(ctx, line, names, kpt, missing, f"{rel}:{number}: {line[:40]}")
                if label is not None:
                    ctx.add_label(rec, label)
    if missing:
        ctx.problem("missing_class_names", ", ".join(f"class_{i}" for i in sorted(missing)[:5]))


def _line(
    ctx: ParseContext,
    line: str,
    names: dict[int, str] | None,
    kpt: tuple[int, int] | None,
    missing: set[int],
    where: str,
) -> Label | None:
    """Label for one YOLO line, or None when it cannot be used.

    Args:
        ctx: parse context that records the problems found.
        line: the line: class id and relative values.
        names: class names by id, or None to use ``class_<id>``.
        kpt: ``(points, values per point)`` from ``kpt_shape``, or None.
        missing: set that receives the class ids used without a name.
        where: text shown as the example of any problem.

    Returns:
        A box, polygon or pose label with relative geometry, or None.
    """
    vals = _numbers(line)
    if not vals or not all(math.isfinite(v) for v in vals) or vals[0] != int(vals[0]) or vals[0] < 0:
        ctx.problem("invalid_annotation", where)
        return None
    cls = int(vals[0])
    if names:
        if cls not in names:
            ctx.problem("unknown_class_id", f"{where} (class {cls})")
            return None
        name = names[cls]
    else:
        name = f"class_{cls}"
        missing.add(cls)
    n = len(vals)
    if n == 5:
        box = ctx.box(vals[1] - vals[3] / 2, vals[2] - vals[4] / 2, vals[3], vals[4], 1, 1, where)
        return Label(name, bbox=box) if box else None
    dim = 0
    if kpt is not None and n > 5 and (n - 5) % kpt[1] == 0:
        dim = kpt[1]
    elif kpt is None and n > 5 and n % 2 == 0 and (n - 5) % 3 == 0:
        dim = 3
    if dim:
        return _pose(ctx, name, vals, dim, kpt[0] if kpt else None, where)
    if n >= 7 and n % 2 == 1:
        ring = ctx.ring([(vals[i], vals[i + 1]) for i in range(1, n, 2)], 1, 1, where)
        if ring is None:
            return None
        return Label(name, bbox=bbox_of_points(ring), polygons=[ring])
    ctx.problem("polygon_too_few_points" if n < 7 else "invalid_annotation", where)
    return None


def _pose(
    ctx: ParseContext, name: str, vals: list[float], dim: int, expected: int | None, where: str
) -> Label | None:
    """Label for a YOLO pose line.

    Args:
        ctx: parse context that records the problems found.
        name: class name.
        vals: all numbers of the line, the class id first.
        dim: values per keypoint: 2 (x, y) or 3 (x, y, v).
        expected: keypoint count from ``kpt_shape``, or None.
        where: text shown as the example of any problem.

    Returns:
        The label with box and relative keypoints, or None when a point or the box is unusable.
    """
    box = ctx.box(vals[1] - vals[3] / 2, vals[2] - vals[4] / 2, vals[3], vals[4], 1, 1, where)
    count = (len(vals) - 5) // dim
    if expected is not None and expected != count:
        ctx.problem("keypoint_count_mismatch", f"{where} ({count} of {expected})")
    points = []
    for i in range(count):
        base = 5 + i * dim
        x, y = vals[base], vals[base + 1]
        v = vals[base + 2] if dim == 3 else (2.0 if (x or y) else 0.0)
        pt = ctx.point(x, y, v, 1, 1, where)
        if pt is None:
            return None
        points.append(pt)
    if box is None:
        return None
    return Label(name, bbox=box, keypoints=points)
