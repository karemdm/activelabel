"""The values of a regression project in CVAT (#619): a tag per target with an attribute :data:`VALUE`.

Each target of a regression project is a CVAT label; the value of a target for an image is a *tag* of
that label on the frame, whose attribute :data:`VALUE` holds the number as text (``"31"``, ``"2.75"``).

The attribute is a mutable **text** attribute, not a CVAT ``number``: CVAT's number attribute needs a
minimum, a maximum and a step, and the CVAT interface refuses a value that is not a multiple of the step,
which with decimal values depends on floating point noise. A text attribute takes any number. Reading
accepts a decimal comma (``"2,75"``) and any attribute called ``value`` (ignoring case) of the label, so a
CVAT project whose labels have a ``number`` attribute of that name is read too.

The spec is added to a CVAT label the first time a value of its target is written (like the visibility
attribute of poses), so a label of another project type never gets it.
"""

from __future__ import annotations

import math
from typing import Any

from activelabel.labels import Label

VALUE = "value"  # text attribute of a CVAT label of a regression target: the value of the target


def value_spec() -> dict[str, Any]:
    """The attribute spec :data:`VALUE` added to the CVAT label of a target (a mutable text).

    Returns:
        The attribute as CVAT takes it in the ``attributes`` of a label.
    """
    return {"name": VALUE, "mutable": True, "input_type": "text", "default_value": "", "values": [""]}


def value_specs(rows: list[dict[str, Any]]) -> dict[int, int]:
    """The id of the :data:`VALUE` attribute of each CVAT label that has one (the name ignores case).

    Args:
        rows: Labels as ``GET /api/labels`` lists them (with their ``attributes``).

    Returns:
        ``{label id: attribute spec id}``.
    """
    out: dict[int, int] = {}
    for row in rows:
        for attr in row.get("attributes") or []:
            if str(attr.get("name", "")).casefold() == VALUE and attr.get("id") is not None:
                out.setdefault(int(row["id"]), int(attr["id"]))
    return out


def format_value(value: float) -> str:
    """The text of a value in CVAT: a whole number without decimals, else the shortest exact form.

    Args:
        value: A finite number.

    Returns:
        ``"31"`` for 31.0, ``"2.75"`` for 2.75.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() and abs(number) < 1e15 else repr(number)


def parse_value(text: Any) -> float | None:
    """The number written in an attribute (a decimal comma is accepted).

    Args:
        text: The attribute value as CVAT gives it (text, or a number).

    Returns:
        The finite number, or None when the text is empty or not a finite number.
    """
    cleaned = str(text if text is not None else "").strip().replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def tag_value(tag: dict[str, Any], spec: int | None) -> float | None:
    """The value a CVAT tag holds in its :data:`VALUE` attribute.

    Args:
        tag: A CVAT tag (``attributes`` of ``{spec_id, value}``).
        spec: Id of the :data:`VALUE` attribute of the tag's label, or None when it has none.

    Returns:
        The number, or None when the tag has no readable value.
    """
    if spec is None:
        return None
    text = next((a.get("value") for a in tag.get("attributes") or [] if a.get("spec_id") == spec), None)
    return parse_value(text)


def value_attributes(label: Label, spec: int | None) -> list[dict[str, Any]]:
    """The ``attributes`` of the CVAT tag of a label: its value, when it has one and the label the spec.

    Args:
        label: A label (a regression label has a ``value``).
        spec: Id of the :data:`VALUE` attribute of its CVAT label, or None.

    Returns:
        ``[{spec_id, value}]``, or an empty list.
    """
    if label.value is None or spec is None:
        return []
    return [{"spec_id": spec, "value": format_value(label.value)}]


def with_value(tag: dict[str, Any], value: float, spec: int) -> dict[str, Any]:
    """A copy of a CVAT tag with another value: its id and its other attributes are kept.

    Args:
        tag: The tag as CVAT has it.
        value: The new value.
        spec: Id of the :data:`VALUE` attribute of its label.

    Returns:
        The edited copy.
    """
    others = [a for a in tag.get("attributes") or [] if a.get("spec_id") != spec]
    return {**tag, "attributes": [*others, {"spec_id": spec, "value": format_value(value)}]}
