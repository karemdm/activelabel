"""Regression values from a CSV or a JSON file: one number per target and image (#619).

CSV: a header row; one column names the image (``image``, ``file``, ``filename`` or ``path``, else the first
column) and every other column is a target. The delimiter is a comma, a semicolon or a tab (read from the
header row); a decimal comma is read when the value is quoted or the delimiter is not a comma. An empty cell
is "no value for this target".

JSON: ``{"<image>": {"<target>": number, ...}, ...}`` or ``[{"image": "<image>", "<target>": number}, ...]``.

A value that is not a finite number is skipped and reported (``value_invalid``).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

from activelabel.annotations.context import ParseContext
from activelabel.labels import Label

IMAGE_COLUMNS = ("image", "file", "filename", "path")
DELIMITERS = (",", ";", "\t")
_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def image_key(keys: list[str]) -> str | None:
    """The key or column that names the image: ``image``, ``file``, ``filename`` or ``path`` (any case).

    Args:
        keys: the column names or the keys of one JSON entry.

    Returns:
        The first of the keys that is one of ``IMAGE_COLUMNS`` ignoring case, or None.
    """
    for key in keys:
        if str(key).strip().lower() in IMAGE_COLUMNS:
            return key
    return None


def _scalars(entry: Any, skip: str | None = None) -> bool:
    """Whether an object only holds numbers, strings or nulls (except the key ``skip``).

    Args:
        entry: a decoded JSON value.
        skip: a key that is not checked (the image name).

    Returns:
        True for an object whose other values are numbers, strings or null.
    """
    return isinstance(entry, dict) and all(
        isinstance(v, (int, float, str)) or v is None for k, v in entry.items() if k != skip
    )


def _list_entry(entry: Any) -> bool:
    """Whether a JSON list item is ``{"image": name, target: value, ...}`` with at least one target.

    Args:
        entry: one item of the list.

    Returns:
        True when it has an image key, another key and only simple values.
    """
    if not isinstance(entry, dict):
        return False
    key = image_key(list(entry))
    return key is not None and len(entry) > 1 and _scalars(entry, key)


def is_regression_json(data: Any) -> bool:
    """Whether decoded JSON has the shape of regression values (and is not COCO).

    Args:
        data: the decoded JSON document.

    Returns:
        True for ``{image: {target: value}}`` or a list of objects that each have an image key and at least
        one target; every value must be a number, a string or null.
    """
    if isinstance(data, dict):
        return bool(data) and "images" not in data and all(_scalars(e) for e in data.values())
    if isinstance(data, list):
        return bool(data) and all(_list_entry(e) for e in data)
    return False


def _number(raw: Any) -> tuple[float | None, bool]:
    """A cell or JSON value as a number.

    Args:
        raw: the value as read: a number, text (a decimal comma is accepted) or None.

    Returns:
        ``(value, valid)``: ``(None, True)`` for an empty cell, ``(None, False)`` when it is not a finite
        number, else ``(number, True)``.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, True
    if isinstance(raw, bool):
        return None, False
    if isinstance(raw, str):
        text = raw.strip()
        if not _NUMBER.match(text):
            return None, False
        raw = text.replace(",", ".")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, False
    return (value, True) if math.isfinite(value) else (None, False)


def _add_entry(ctx: ParseContext, name: str, values: list[tuple[str, Any]]) -> None:
    """Label one image with the values of its targets.

    Args:
        ctx: parse context; receives the targets as classes, the labels and the problems.
        name: the image as the file writes it.
        values: ``(target, raw value)`` pairs (a CSV may repeat a column: the importer keeps the first); empty
            values are skipped, invalid ones are reported.
    """
    for target, _ in values:
        ctx.add_class(target)
    rec = ctx.match(name)
    if rec is None or not ctx.claim(rec, name):
        return
    for target, raw in values:
        value, valid = _number(raw)
        if not valid:
            ctx.problem("value_invalid", f"{name}: {target} = {raw}")
        elif value is not None:
            ctx.add_label(rec, Label(target, value=value))


def _delimiter(text: str) -> str:
    """The delimiter of a CSV text, from its header row.

    Args:
        text: the whole file.

    Returns:
        The one of ``DELIMITERS`` that the header row uses most (a comma when it has none).
    """
    header = next((ln for ln in text.splitlines() if ln.strip()), "")
    return max(DELIMITERS, key=lambda d: (header.count(d), d == ","))


def _parse_csv(ctx: ParseContext, text: str) -> None:
    """Read the rows of a CSV text.

    Args:
        ctx: parse context.
        text: the whole file.

    Raises:
        ValueError: the file has no header row or no target column.
    """
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=_delimiter(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("the file is empty")
    header = [c.strip() for c in rows[0]]
    column = header.index(image_key(header) or header[0])
    targets = [(i, h) for i, h in enumerate(header) if i != column and h]
    if not targets:
        raise ValueError("no target column")
    for row in rows[1:]:
        name = row[column].strip() if column < len(row) else ""
        if not name or len(row) > len(header):
            ctx.problem("invalid_annotation", name or "(row without an image)")
            continue
        _add_entry(ctx, name, [(h, row[i] if i < len(row) else None) for i, h in targets])


def _parse_json(ctx: ParseContext, data: Any) -> None:
    """Read decoded JSON values.

    Args:
        ctx: parse context.
        data: ``{image: {target: value}}`` or a list of ``{"image": ..., target: value}``.

    Raises:
        ValueError: the document does not have one of those shapes.
    """
    if not is_regression_json(data):
        raise ValueError("not a file of regression values")
    if isinstance(data, dict):
        for name, values in data.items():
            _add_entry(ctx, str(name), [(str(k), v) for k, v in values.items()])
        return
    for entry in data:
        key = image_key(list(entry))
        _add_entry(ctx, str(entry[key]), [(str(k), v) for k, v in entry.items() if k != key])


def parse_regression(ctx: ParseContext, files: list[Path]) -> None:
    """Read regression value files into ``ctx``.

    Args:
        ctx: parse context; receives the targets (as classes, in column or first-seen order), the labels
            (``Label(target, value=number)``), the names of the files read and the problems.
        files: staged files; only ``.csv`` and ``.json`` files are read, and one that cannot be read is
            reported as ``invalid_file``.
    """
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".json"):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            if suffix == ".csv":
                _parse_csv(ctx, text)
            else:
                _parse_json(ctx, json.loads(text))
        except (OSError, ValueError, csv.Error) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)
