"""Data augmentation techniques a training can use, per kind of task, as data (see docs/TRAINING.md).

The ``augmentation`` hyperparameter picks a level: ``none``, ``light``, ``standard`` and ``strong`` are
presets (each one a list of techniques, the same the workers applied before techniques could be chosen) and
``custom`` takes the list from the ``augmentations`` option of the run configuration. Either way the runner
hands the worker the resolved list (:func:`resolve`) and the worker applies exactly those techniques, in the
order of :data:`TECHNIQUES`, to the training images only.

Geometric techniques move boxes, masks and keypoints with the image; a horizontal flip of keypoints also swaps
the left/right pairs (``flip_pairs``), which therefore must be given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from activelabel.errors import CodedError
from activelabel.textcheck import message_of
from activelabel.training.catalog import ParamSpec

PRESETS_NAMES = ("none", "light", "standard", "strong")
CUSTOM = "custom"
LEVELS = (*PRESETS_NAMES, CUSTOM)
CLASSIFICATION = ("classification",)
OBJECTS = ("detection", "instance", "keypoints", "semantic")
ALL_KINDS = CLASSIFICATION + OBJECTS


class AugmentationError(CodedError):
    """The list of techniques is not valid (a :class:`~activelabel.errors.CodedError`: readable message)."""


def _p(
    name: str, default: float | int, low: float, high: float, help_: str, kind: str = "float"
) -> ParamSpec:
    """A numeric parameter of a technique.

    Args:
        name: key in the technique's ``params``.
        default: safe default value.
        low: lowest value.
        high: highest value.
        help_: one line for the form.
        kind: ``float`` or ``int``.

    Returns:
        The parameter spec.
    """
    return ParamSpec(name, kind, default, low, high, help=help_, group="augmentation")


_PROB = _p("p", 0.5, 0.0, 1.0, "Chance of applying it to an image.")


@dataclass(frozen=True)
class Technique:
    """One augmentation technique.

    Attributes:
        name: stable id (the key of the list entries).
        kinds: kinds of task that offer it.
        params: its parameters, with safe defaults.
        help: one line for the form.
        geometric: it moves pixels, so boxes, masks and keypoints move with them.
        batch: it mixes the images of a batch (mixup, cutmix) instead of changing one image.
        conflicts: techniques that cannot be chosen together with this one.
    """

    name: str
    kinds: tuple[str, ...]
    params: tuple[ParamSpec, ...] = ()
    help: str = ""
    geometric: bool = False
    batch: bool = False
    conflicts: tuple[str, ...] = ()

    def needs_flip_pairs(self, kind: str) -> bool:
        """Whether an entry of this technique must give the keypoint ``flip_pairs``.

        Args:
            kind: kind of task.

        Returns:
            True for a horizontal flip of keypoints.
        """
        return self.name == "hflip" and kind == "keypoints"

    def as_dict(self, kind: str) -> dict[str, Any]:
        """JSON-ready form for one kind of task.

        Args:
            kind: kind of task (decides ``needs_flip_pairs``).

        Returns:
            The technique as plain data for the API.
        """
        return {
            "name": self.name,
            "help": self.help,
            "geometric": self.geometric,
            "batch": self.batch,
            "conflicts": list(self.conflicts),
            "needs_flip_pairs": self.needs_flip_pairs(kind),
            "params": [p.as_dict() for p in self.params],
        }


#: Every technique, in the order the workers apply them.
TECHNIQUES: tuple[Technique, ...] = (
    Technique(
        "random_resized_crop",
        ("classification", "semantic"),
        (_p("scale_min", 0.6, 0.05, 1.0, "Smallest share of the image area the crop keeps."),),
        "Crops a random part of the image and resizes it to the input size.",
        geometric=True,
    ),
    Technique("hflip", ALL_KINDS, (_PROB,), "Mirrors the image left to right.", geometric=True),
    Technique(
        "vflip",
        tuple(k for k in ALL_KINDS if k != "keypoints"),
        (_PROB,),
        "Mirrors the image top to bottom (aerial, microscope and document images).",
        geometric=True,
    ),
    Technique(
        "rotation",
        ALL_KINDS,
        (_p("degrees", 15.0, 0.0, 180.0, "Largest angle, either way."),),
        "Rotates the image by a random angle; the corners become black (boxes get the rotated box's extent).",
        geometric=True,
    ),
    Technique(
        "scale_jitter",
        OBJECTS,
        (
            _p("scale_min", 0.75, 0.1, 1.0, "Smallest zoom (below 1 shows more, with a black border)."),
            _p("scale_max", 1.25, 1.0, 4.0, "Largest zoom (above 1 crops in)."),
        ),
        "Zooms in or out by a random factor around the centre.",
        geometric=True,
    ),
    Technique(
        "translate",
        OBJECTS,
        (_p("fraction", 0.1, 0.0, 0.5, "Largest shift, as a share of the width and height."),),
        "Shifts the image by a random offset.",
        geometric=True,
    ),
    Technique(
        "randaugment",
        CLASSIFICATION,
        (
            _p("num_ops", 2, 1, 10, "Operations applied to each image.", "int"),
            _p("magnitude", 9, 0, 30, "Strength of the operations.", "int"),
        ),
        "Applies random photo operations (RandAugment).",
        conflicts=("trivialaugment",),
    ),
    Technique(
        "trivialaugment",
        CLASSIFICATION,
        (),
        "Applies one random photo operation of random strength (TrivialAugment, no settings).",
        conflicts=("randaugment",),
    ),
    Technique(
        "color_jitter",
        ALL_KINDS,
        (
            _p("brightness", 0.2, 0.0, 1.0, "Brightness change, up to this share."),
            _p("contrast", 0.2, 0.0, 1.0, "Contrast change, up to this share."),
            _p("saturation", 0.2, 0.0, 1.0, "Colour saturation change, up to this share."),
            _p("hue", 0.0, 0.0, 0.5, "Hue shift, up to this share of the colour wheel."),
        ),
        "Changes brightness, contrast, saturation and hue at random.",
    ),
    Technique(
        "blur",
        ALL_KINDS,
        (
            _p("p", 0.2, 0.0, 1.0, "Chance of applying it to an image."),
            _p("sigma_max", 1.5, 0.1, 5.0, "Strongest blur (Gaussian sigma, in pixels)."),
        ),
        "Blurs the image (out-of-focus photos).",
    ),
    Technique(
        "grayscale",
        ALL_KINDS,
        (_p("p", 0.1, 0.0, 1.0, "Chance of applying it to an image."),),
        "Turns the image grey (the model relies less on colour).",
    ),
    Technique(
        "random_erasing",
        CLASSIFICATION,
        (_p("p", 0.25, 0.0, 1.0, "Chance of applying it to an image."),),
        "Erases a random rectangle (the model looks at more of the object).",
    ),
    Technique(
        "mixup",
        CLASSIFICATION,
        (_p("alpha", 0.2, 0.01, 10.0, "Beta distribution parameter: larger mixes more."),),
        "Blends pairs of images and their classes (with cutmix, one of the two per batch).",
        batch=True,
    ),
    Technique(
        "cutmix",
        CLASSIFICATION,
        (_p("alpha", 1.0, 0.01, 10.0, "Beta distribution parameter: larger mixes more."),),
        "Pastes a patch of one image on another; the classes mix by area.",
        batch=True,
    ),
)
_BY_NAME = {t.name: t for t in TECHNIQUES}
_ORDER = {t.name: i for i, t in enumerate(TECHNIQUES)}

_CLS_PRESETS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "none": [],
    "light": [("random_resized_crop", {"scale_min": 0.8}), ("hflip", {})],
    "standard": [
        ("random_resized_crop", {"scale_min": 0.6}),
        ("hflip", {}),
        ("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}),
    ],
    "strong": [
        ("random_resized_crop", {"scale_min": 0.4}),
        ("hflip", {}),
        ("rotation", {"degrees": 15.0}),
        ("trivialaugment", {}),
        ("color_jitter", {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}),
    ],
}
_OBJECT_PRESETS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "none": [],
    "light": [("hflip", {})],
    "standard": [("hflip", {}), ("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.0})],
    "strong": [("hflip", {}), ("color_jitter", {"brightness": 0.4, "contrast": 0.4, "saturation": 0.0})],
}
#: Keypoints have no flip in the presets: which points swap places is only known from ``flip_pairs``. Without
#: the flip, "light" would be empty (the same as "none"), so it varies the light a little instead (#635).
_KEYPOINT_PRESETS = {k: [e for e in v if e[0] != "hflip"] for k, v in _OBJECT_PRESETS.items()}
_KEYPOINT_PRESETS["light"] = [("color_jitter", {"brightness": 0.1, "contrast": 0.1, "saturation": 0.0})]
_SEMANTIC_PRESETS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "none": [],
    "light": [("random_resized_crop", {"scale_min": 0.8}), ("hflip", {})],
    "standard": [
        ("random_resized_crop", {"scale_min": 0.6}),
        ("hflip", {}),
        ("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}),
    ],
    "strong": [
        ("random_resized_crop", {"scale_min": 0.4}),
        ("hflip", {}),
        ("vflip", {"p": 0.2}),
        ("rotation", {"degrees": 15.0}),
        ("color_jitter", {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}),
    ],
}
_PRESETS = {
    "classification": _CLS_PRESETS,
    "detection": _OBJECT_PRESETS,
    "instance": _OBJECT_PRESETS,
    "keypoints": _KEYPOINT_PRESETS,
    "semantic": _SEMANTIC_PRESETS,
}


def techniques_for(kind: str) -> list[Technique]:
    """Techniques a kind of task offers.

    Args:
        kind: kind of task (``classification``, ``detection``, ``instance``, ``keypoints`` or ``semantic``).

    Returns:
        The techniques, in the order the workers apply them.
    """
    return [t for t in TECHNIQUES if kind in t.kinds]


def _entry(technique: Technique, given: dict[str, Any]) -> dict[str, Any]:
    """A list entry with every parameter (default unless given), without checking.

    Args:
        technique: the technique.
        given: parameters that replace the defaults.

    Returns:
        ``{"name", "params"}``.
    """
    return {"name": technique.name, "params": {**{p.name: p.default for p in technique.params}, **given}}


def preset(level: str, kind: str) -> list[dict[str, Any]]:
    """The techniques of a preset level for a kind of task.

    Args:
        level: ``none``, ``light``, ``standard`` or ``strong``.
        kind: kind of task.

    Returns:
        The entries ``{"name", "params"}`` with every parameter filled in.

    Raises:
        AugmentationError: the level is not a preset.
    """
    if level not in PRESETS_NAMES:
        raise AugmentationError(
            "augmentation_level_invalid",
            f"The augmentation level must be one of: {', '.join(PRESETS_NAMES)}.",
            choices=list(PRESETS_NAMES),
        )
    return [_entry(_BY_NAME[name], given) for name, given in _PRESETS[kind][level]]


def catalogue(kind: str) -> dict[str, Any]:
    """What the form needs for a kind of task: its techniques and the lists of the presets.

    Args:
        kind: kind of task.

    Returns:
        ``{"techniques": [...], "presets": {level: [entries]}}``.
    """
    return {
        "techniques": [t.as_dict(kind) for t in techniques_for(kind)],
        "presets": {level: preset(level, kind) for level in PRESETS_NAMES},
    }


def _flip_pairs(value: Any) -> list[list[int]]:
    """Check the keypoint flip pairs: ``[[left, right], ...]`` of distinct point indexes.

    Args:
        value: what the configuration gives.

    Returns:
        The pairs as lists of two ints.

    Raises:
        AugmentationError: not a list of pairs of whole numbers, a negative index, or an index used twice.
    """
    message = "The keypoint flip pairs must be a list of [left, right] point numbers, each point used once."
    if not isinstance(value, list):
        raise AugmentationError("augmentation_flip_pairs_invalid", message)
    seen: set[int] = set()
    pairs = []
    for pair in value:
        if (
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in pair)
            or pair[0] == pair[1]
            or seen & set(pair)
        ):
            raise AugmentationError("augmentation_flip_pairs_invalid", message)
        seen.update(pair)
        pairs.append([int(pair[0]), int(pair[1])])
    return pairs


def _check_entry(entry: Any, kind: str) -> dict[str, Any]:
    """Check one entry of the list against the techniques of a kind of task.

    Args:
        entry: ``{"name": ..., "params": {...}}`` (``params`` may be left out).
        kind: kind of task.

    Returns:
        The entry with every parameter in its right type (defaults filled in).

    Raises:
        AugmentationError: not a mapping, a technique this kind of task does not offer, an unknown or invalid
            parameter, or missing keypoint flip pairs.
    """
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("name"), str)
        or set(entry) - {"name", "params"}
    ):
        raise AugmentationError(
            "augmentation_entry_invalid", "Each augmentation technique must be {name, params}."
        )
    name = entry["name"]
    technique = _BY_NAME.get(name)
    if technique is None or kind not in technique.kinds:
        raise AugmentationError(
            "augmentation_unknown",
            f"'{name}' is not an augmentation technique for {kind} models.",
            name=name,
            kind=kind,
        )
    given = entry.get("params") or {}
    if not isinstance(given, dict):
        raise AugmentationError(
            "augmentation_entry_invalid", "Each augmentation technique must be {name, params}."
        )
    specs = {p.name: p for p in technique.params}
    allowed = set(specs) | ({"flip_pairs"} if technique.needs_flip_pairs(kind) else set())
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise AugmentationError(
            "augmentation_param_unknown",
            f"'{unknown[0]}' is not a setting of the augmentation technique '{name}'.",
            name=name,
            param=unknown[0],
        )
    params: dict[str, Any] = {}
    for spec in technique.params:
        try:
            params[spec.name] = spec.check(given.get(spec.name, spec.default))
        except CodedError as error:
            params_ = {**error.params, "name": f"{name}.{spec.name}"}
            raise AugmentationError(error.code, message_of(error), **params_) from None
    if technique.needs_flip_pairs(kind):
        if "flip_pairs" not in given:
            raise AugmentationError(
                "augmentation_flip_pairs_required",
                "Flipping keypoints needs the left/right pairs "
                "(an empty list when no point has a mirror twin).",
            )
        params["flip_pairs"] = _flip_pairs(given["flip_pairs"])
    return {"name": name, "params": params}


def validate(entries: Any, kind: str) -> list[dict[str, Any]]:
    """Check a list of techniques for a kind of task.

    Args:
        entries: the ``augmentations`` option of a run configuration.
        kind: kind of task of the model.

    Returns:
        The checked entries in the order the workers apply them (an empty list trains without augmentation).

    Raises:
        AugmentationError: not a list, an invalid entry, a technique given twice, or two that conflict.
    """
    if not isinstance(entries, list):
        raise AugmentationError("augmentations_not_list", "The augmentation techniques must be a list.")
    checked = [_check_entry(e, kind) for e in entries]
    names = [e["name"] for e in checked]
    for name in names:
        if names.count(name) > 1:
            raise AugmentationError(
                "augmentation_repeated", f"The augmentation technique '{name}' is given twice.", name=name
            )
        clash = next((c for c in _BY_NAME[name].conflicts if c in names), None)
        if clash:
            raise AugmentationError(
                "augmentation_conflict",
                f"The augmentation techniques '{name}' and '{clash}' cannot be used together.",
                name=name,
                other=clash,
            )
    return sorted(checked, key=lambda e: _ORDER[e["name"]])


def resolve(level: str, entries: list[dict[str, Any]] | None, kind: str) -> list[dict[str, Any]]:
    """The techniques a training applies.

    Args:
        level: the ``augmentation`` hyperparameter (a preset or ``custom``).
        entries: the checked ``augmentations`` of the configuration (used with ``custom``).
        kind: kind of task of the model.

    Returns:
        The entries ``{"name", "params"}``, in the order the workers apply them.

    Raises:
        AugmentationError: ``custom`` without a list, or an unknown level.
    """
    if level == CUSTOM:
        if entries is None:
            raise AugmentationError(
                "augmentation_custom_needs_list",
                "The augmentation level 'custom' needs the list of techniques.",
            )
        return validate(entries, kind)
    return preset(level, kind)


def max_flip_index(entries: list[dict[str, Any]]) -> int:
    """The highest keypoint index the flip pairs use.

    Args:
        entries: resolved entries.

    Returns:
        The index, or -1 when no pair is given.
    """
    indexes = [i for e in entries for pair in e["params"].get("flip_pairs", []) for i in pair]
    return max(indexes, default=-1)
