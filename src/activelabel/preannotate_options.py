"""Settings of a pre-annotation: which classes to write and the inference parameters of each kind of model.

Every kind of model describes its own parameters (:func:`schema`): name, type, default, range, step and
whether it is ``basic`` (shown first) or ``advanced``. A parameter that does not apply to a kind is not
listed, so the interface never shows it. :func:`resolve` checks the values a person sent against that
description and fills the rest with the defaults: the result is what the run stores and uses, so it can be
repeated exactly. A kind not listed here (a future regression model, for example) has only the parameters
that apply to every kind.

The model process (``training/workers/predict.py``) applies the parameters of the network: confidence,
NMS, objects per image, smallest object, mask threshold and classification margin. ActiveLabel applies the
parameters of the mask outlines (``min_region``, ``simplify``) and, again, the class filter, so a class that
was not chosen is never written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from activelabel.errors import CodedError

DETECTION_KINDS = ("detection", "keypoints", "instance")
MASK_KINDS = ("instance", "semantic")
ALL_KINDS = ("classification", "detection", "keypoints", "instance", "semantic")
#: Default minimum confidence: a classifier's top class needs more than a detector's box.
CONFIDENCE_DEFAULTS = {"classification": 0.5}
CONFIDENCE_DEFAULT = 0.25


@dataclass(frozen=True)
class OptionSpec:
    """One inference parameter of a pre-annotation.

    Attributes:
        name: key in ``options`` (``confidence``, ``nms_iou``…).
        type: ``float``, ``int`` or ``bool``.
        default: value used when the person does not set one.
        minimum: lowest value accepted (numbers only).
        maximum: highest value accepted (numbers only).
        step: step of the input field (numbers only).
        kinds: kinds of model it applies to (empty: every kind).
        level: ``basic`` (shown first) or ``advanced`` (under "more settings").
    """

    name: str
    type: str
    default: float | int | bool
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    kinds: tuple[str, ...] = ()
    level: str = "advanced"


OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("confidence", "float", CONFIDENCE_DEFAULT, 0.001, 0.999, 0.01, level="basic"),
    OptionSpec("mask_threshold", "float", 0.5, 0.1, 0.9, 0.05, ("instance",), "basic"),
    OptionSpec("margin", "float", 0.0, 0.0, 0.9, 0.05, ("classification",)),
    OptionSpec("nms_iou", "float", 0.5, 0.05, 0.95, 0.05, DETECTION_KINDS),
    OptionSpec("max_detections", "int", 100, 1, 1000, 1, DETECTION_KINDS),
    OptionSpec("class_agnostic_nms", "bool", False, kinds=DETECTION_KINDS),
    OptionSpec("min_size", "float", 0.0, 0.0, 0.5, 0.005, DETECTION_KINDS),
    OptionSpec("min_region", "float", 0.0002, 0.0, 0.05, 0.0001, MASK_KINDS),
    OptionSpec("simplify", "float", 0.004, 0.0, 0.05, 0.001, MASK_KINDS),
)


def default_of(spec: OptionSpec, kind: str) -> float | int | bool:
    """Default of a parameter for a kind of model (only the confidence depends on the kind).

    Args:
        spec: the parameter.
        kind: the kind of model.

    Returns:
        The default value.
    """
    if spec.name == "confidence":
        return CONFIDENCE_DEFAULTS.get(kind, CONFIDENCE_DEFAULT)
    return spec.default


def specs_for(kind: str) -> list[OptionSpec]:
    """Parameters that apply to a kind of model.

    Args:
        kind: the kind of model (``classification``, ``detection``…; any other name gets the common ones).

    Returns:
        The parameters, in display order.
    """
    return [s for s in OPTIONS if not s.kinds or kind in s.kinds]


def schema(kind: str) -> list[dict[str, Any]]:
    """Self-description of the parameters of a kind of model, for the interface.

    Args:
        kind: the kind of model.

    Returns:
        One dict per parameter: ``name``, ``type``, ``default``, ``min``, ``max``, ``step``, ``level``.
    """
    return [
        {
            "name": s.name,
            "type": s.type,
            "default": default_of(s, kind),
            "min": s.minimum,
            "max": s.maximum,
            "step": s.step,
            "level": s.level,
        }
        for s in specs_for(kind)
    ]


def _check(spec: OptionSpec, value: Any) -> float | int | bool:
    """One value checked against its parameter.

    Args:
        spec: the parameter.
        value: the value sent.

    Returns:
        The value with the parameter's type (floats rounded to 6 decimals after the range check).

    Raises:
        CodedError: ``option_not_bool``, ``option_not_number``, ``option_not_whole`` or ``option_range``.
    """
    if spec.type == "bool":
        if not isinstance(value, bool):
            raise CodedError("option_not_bool", f"'{spec.name}' must be true or false.", option=spec.name)
        return value
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CodedError("option_not_number", f"'{spec.name}' must be a number.", option=spec.name)
    if spec.type == "int" and value != int(value):
        raise CodedError("option_not_whole", f"'{spec.name}' must be a whole number.", option=spec.name)
    assert spec.minimum is not None and spec.maximum is not None
    if (
        not spec.minimum <= value <= spec.maximum
    ):  # the value sent, before rounding: as the form checks it (#663)
        raise CodedError(
            "option_range",
            f"'{spec.name}' must be between {spec.minimum} and {spec.maximum}.",
            option=spec.name,
            min=spec.minimum,
            max=spec.maximum,
        )
    return int(value) if spec.type == "int" else round(float(value), 6)


def resolve(kind: str, given: dict[str, Any] | None) -> dict[str, Any]:
    """The parameters a run uses: the values sent, checked, and the defaults for the rest.

    Args:
        kind: the kind of model.
        given: values sent by the person (``None`` or missing keys: the defaults).

    Returns:
        Every parameter of the kind with its value.

    Raises:
        CodedError: ``option_unknown`` for a parameter the kind does not have, or an error of a value
            (see :func:`_check`).
    """
    given = dict(given or {})
    specs = specs_for(kind)
    unknown = sorted(set(given) - {s.name for s in specs})
    if unknown:
        raise CodedError(
            "option_unknown",
            f"This model does not have these settings: {', '.join(unknown)}.",
            options=", ".join(unknown),
        )
    return {s.name: _check(s, given[s.name]) if s.name in given else default_of(s, kind) for s in specs}


def model_classes(card: dict[str, Any], project_classes: list[str]) -> list[dict[str, Any]]:
    """Classes a saved model can write, with whether the project still has each one.

    A trained model knows the project's classes at its training; a public model only the project classes
    its categories are mapped to.

    Args:
        card: the saved model's card (``classes``, or ``source: public`` and ``mapping``).
        project_classes: the project's current classes.

    Returns:
        ``[{"name", "in_project"}]``, without repeats, in the model's order.
    """
    if card.get("source") == "public":
        names = list((card.get("mapping") or {}).values())
    else:
        names = list(card.get("classes") or [])
    return [
        {"name": n, "in_project": n in project_classes} for n in dict.fromkeys(str(n) for n in names if n)
    ]


def resolve_classes(available: list[dict[str, Any]], chosen: list[str] | None) -> list[str]:
    """Classes a run writes: the ones chosen, or every class of the model the project has.

    Args:
        available: the model's classes, from :func:`model_classes`.
        chosen: names chosen by the person (``None``: all).

    Returns:
        The class names, in the model's order.

    Raises:
        CodedError: ``class_not_in_model`` (a name the model cannot write into this project) or
            ``no_class_selected``.
    """
    usable = [c["name"] for c in available if c["in_project"]]
    if chosen is None:
        return usable
    wrong = [c for c in chosen if c not in usable]
    if wrong:
        raise CodedError(
            "class_not_in_model",
            f"This model cannot write these classes into this project: {', '.join(wrong[:5])}.",
            classes=", ".join(wrong[:5]),
        )
    picked = [c for c in usable if c in set(chosen)]
    if not picked:
        raise CodedError("no_class_selected", "Choose at least one class to pre-annotate.")
    return picked
