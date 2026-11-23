"""The model catalogue, as data (see docs/TRAINING.md).

Each model lists its family, packages, default input size and hyperparameters; the screen builds
its form from this, so adding a model does not change the interface. Defaults are chosen for a
CPU: a small batch and few epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

FAMILIES = ("torchvision", "timm", "ultralytics")
TASK_KINDS = ("classification", "detection", "semantic", "instance", "keypoints")
STARTS = ("transfer", "finetune", "scratch")
PROJECT_TYPES = ("classification", "detection", "segmentation", "keypoints")
KINDS = ("int", "float", "choice", "bool")

#: Selection metrics per kind of task: (allowed, default).
METRICS: dict[str, tuple[tuple[str, ...], str]] = {
    "classification": (("accuracy", "macro_f1"), "accuracy"),
    "detection": (("map50_95", "map50"), "map50_95"),
    "semantic": (("miou", "pixel_accuracy"), "miou"),
    "instance": (("map50_95", "map50"), "map50_95"),
    "keypoints": (("map50_95", "map50"), "map50_95"),
}


@dataclass(frozen=True)
class ParamSpec:
    """A hyperparameter: its type, default, limits and a one-line help.

    Attributes:
        name: key in ``params`` of a run configuration.
        kind: ``int``, ``float``, ``choice`` or ``bool``.
        default: value used when the person does not set it.
        min: lowest value (numbers only).
        max: highest value (numbers only).
        choices: allowed values (``choice`` only).
        help: one line for the form.
        group: ``common`` (every model) or ``model`` (this model only).
    """

    name: str
    kind: str
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    group: str = "common"

    def check(self, value: Any) -> Any:
        """Validate ``value`` and return it in the right Python type.

        Raises:
            ValueError: with a readable message.
        """
        name = self.name
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"'{name}' must be true or false.")
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"'{name}' must be one of: {', '.join(self.choices)}.")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"'{name}' must be a number.")
        if self.kind == "int":
            if int(value) != value:
                raise ValueError(f"'{name}' must be a whole number.")
            value = int(value)
        else:
            value = float(value)
        if self.min is not None and value < self.min:
            raise ValueError(f"'{name}' must be at least {self.min:g}.")
        if self.max is not None and value > self.max:
            raise ValueError(f"'{name}' must be at most {self.max:g}.")
        return value

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "choices": list(self.choices),
            "help": self.help,
            "group": self.group,
        }


@dataclass(frozen=True)
class ModelSpec:
    """A model a project can train.

    Attributes:
        id: stable identifier (also the weights name where there is one).
        label: name shown to people.
        types: project types that offer it.
        family: ``torchvision``, ``timm`` or ``ultralytics``.
        task_kind: classification, detection, semantic, instance or keypoints.
        packages: pip requirement names the training environment needs.
        image_size: default input size (pixels, square).
        supports_pretrained: whether transfer learning / fine-tuning are possible.
        notes: one line (licence, caveats).
        params: model-specific hyperparameters.
        defaults: replacements for the default of common parameters (for example ``lr``).
    """

    id: str
    label: str
    types: tuple[str, ...]
    family: str
    task_kind: str
    packages: tuple[str, ...]
    image_size: int
    supports_pretrained: bool = True
    notes: str = ""
    params: tuple[ParamSpec, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)

    def param_specs(self) -> list[ParamSpec]:
        """Common parameters (with this model's defaults) followed by the model's own."""
        out = []
        for spec in COMMON_PARAMS:
            default = (
                self.image_size if spec.name == "image_size" else self.defaults.get(spec.name, spec.default)
            )
            out.append(replace(spec, default=default))
        return out + list(self.params)

    def param(self, name: str) -> ParamSpec | None:
        """One parameter by name (common or model-specific)."""
        return next((p for p in self.param_specs() if p.name == name), None)

    def default_metric(self) -> str:
        """Default selection metric of this model's kind of task."""
        return METRICS[self.task_kind][1]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, with the effective parameter specs and selection metrics."""
        metrics, default = METRICS[self.task_kind]
        return {
            "id": self.id,
            "label": self.label,
            "types": list(self.types),
            "family": self.family,
            "task_kind": self.task_kind,
            "packages": list(self.packages),
            "image_size": self.image_size,
            "supports_pretrained": self.supports_pretrained,
            "notes": self.notes,
            "params": [p.as_dict() for p in self.param_specs()],
            "metrics": list(metrics),
            "default_metric": default,
        }


COMMON_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("epochs", "int", 10, 1, 1000, help="Passes over the training images."),
    ParamSpec("batch_size", "int", 8, 1, 512, help="Images per step (small suits a CPU)."),
    ParamSpec("lr", "float", 0.001, 1e-6, 1.0, help="Learning rate."),
    ParamSpec("optimizer", "choice", "adamw", choices=("sgd", "adam", "adamw"), help="Optimizer."),
    ParamSpec("weight_decay", "float", 0.0001, 0.0, 1.0, help="Weight decay (regularisation)."),
    ParamSpec(
        "scheduler", "choice", "cosine", choices=("none", "cosine", "step"), help="Learning rate schedule."
    ),
    ParamSpec("image_size", "int", 224, 32, 2048, help="Side of the square input, in pixels."),
    ParamSpec(
        "augmentation",
        "choice",
        "light",
        choices=("none", "light", "standard", "strong"),
        help="How much the training images are varied (flips, colour, scale).",
    ),
    ParamSpec(
        "patience", "int", 0, 0, 100, help="Stop after this many epochs without improvement (0 = off)."
    ),
    ParamSpec("seed", "int", 42, 0, 2**31 - 1, help="Random seed (same seed, same data order)."),
    ParamSpec(
        "start",
        "choice",
        "transfer",
        choices=STARTS,
        help="transfer: pretrained, only the head trains; finetune: pretrained, all trains; scratch: random.",
    ),
    ParamSpec(
        "backbone_lr_factor",
        "float",
        0.1,
        0.001,
        1.0,
        help="Fine-tuning: the backbone learns at lr times this factor.",
    ),
)

_LABEL_SMOOTHING = ParamSpec(
    "label_smoothing", "float", 0.0, 0.0, 0.5, help="Softens the target classes.", group="model"
)
_TRAINABLE_LAYERS = ParamSpec(
    "trainable_backbone_layers",
    "int",
    3,
    0,
    5,
    help="Backbone stages that train in fine-tuning.",
    group="model",
)
_SCORE_THRESH = ParamSpec(
    "score_thresh", "float", 0.05, 0.0, 1.0, help="Lowest score a prediction needs to be kept.", group="model"
)
_MOSAIC = ParamSpec("mosaic", "bool", True, help="Mosaic augmentation (four images in one).", group="model")
_CLOSE_MOSAIC = ParamSpec(
    "close_mosaic", "int", 2, 0, 100, help="Turn mosaic off for the last N epochs.", group="model"
)

_TV = ("torch", "torchvision")
_TIMM = ("torch", "torchvision", "timm")
_ULTRA = ("ultralytics",)
_TRANSFORMER = {"lr": 0.0001}
_TV_DET = {"optimizer": "sgd", "lr": 0.005, "batch_size": 4, "weight_decay": 0.0005}
_TV_SEG = {"lr": 0.0001, "batch_size": 4}
_YOLO = {"optimizer": "sgd", "lr": 0.01, "batch_size": 8, "weight_decay": 0.0005}


def _classification() -> list[ModelSpec]:
    def m(
        id_: str, label: str, family: str, size: int = 224, defaults: dict[str, Any] | None = None
    ) -> ModelSpec:
        packages = _TV if family == "torchvision" else _TIMM
        return ModelSpec(
            id_,
            label,
            ("classification",),
            family,
            "classification",
            packages,
            size,
            params=(_LABEL_SMOOTHING,),
            defaults=defaults or {},
        )

    return [
        m("resnet18", "ResNet-18", "torchvision"),
        m("resnet50", "ResNet-50", "torchvision"),
        m("efficientnet_b0", "EfficientNet-B0", "timm"),
        m("efficientnet_b3", "EfficientNet-B3", "timm", 288),
        m("convnext_tiny", "ConvNeXt-T", "torchvision", defaults=_TRANSFORMER),
        m("vit_b_16", "ViT-B/16", "torchvision", defaults=_TRANSFORMER),
        m("swin_t", "Swin-T", "torchvision", defaults=_TRANSFORMER),
        m("mobilenet_v3_large", "MobileNetV3 Large", "torchvision"),
    ]


def _tv(id_: str, label: str, ptype: str, kind: str, size: int, defaults: dict[str, Any]) -> ModelSpec:
    extra = (_TRAINABLE_LAYERS, _SCORE_THRESH) if kind in ("detection", "instance", "keypoints") else ()
    return ModelSpec(id_, label, (ptype,), "torchvision", kind, _TV, size, params=extra, defaults=defaults)


def _torchvision_models() -> list[ModelSpec]:
    det, seg = "detection", "segmentation"
    return [
        _tv("fasterrcnn_resnet50_fpn", "Faster R-CNN ResNet-50 FPN", det, det, 640, _TV_DET),
        _tv("fasterrcnn_mobilenet_v3_large_fpn", "Faster R-CNN MobileNetV3 FPN", det, det, 640, _TV_DET),
        _tv("retinanet_resnet50_fpn", "RetinaNet ResNet-50 FPN", det, det, 640, _TV_DET),
        _tv("ssdlite320_mobilenet_v3_large", "SSDlite320 MobileNetV3", det, det, 320, _TV_DET),
        _tv("fcos_resnet50_fpn", "FCOS ResNet-50 FPN", det, det, 640, _TV_DET),
        _tv("deeplabv3_resnet50", "DeepLabV3 ResNet-50", seg, "semantic", 512, _TV_SEG),
        _tv("deeplabv3_mobilenet_v3_large", "DeepLabV3 MobileNetV3", seg, "semantic", 512, _TV_SEG),
        _tv("fcn_resnet50", "FCN ResNet-50", seg, "semantic", 512, _TV_SEG),
        _tv("lraspp_mobilenet_v3_large", "LR-ASPP MobileNetV3", seg, "semantic", 512, _TV_SEG),
        _tv("maskrcnn_resnet50_fpn", "Mask R-CNN ResNet-50 FPN", seg, "instance", 640, _TV_DET),
        _tv(
            "keypointrcnn_resnet50_fpn",
            "Keypoint R-CNN ResNet-50 FPN",
            "keypoints",
            "keypoints",
            640,
            _TV_DET,
        ),
    ]


def _yolo_models() -> list[ModelSpec]:
    out = []
    variants = (
        ("", "detection", "detection"),
        ("-seg", "segmentation", "instance"),
        ("-pose", "keypoints", "keypoints"),
    )
    for suffix, ptype, kind in variants:
        for gen, name in (("yolov8", "YOLOv8"), ("yolo11", "YOLO11")):
            for size in "nsml":
                out.append(
                    ModelSpec(
                        f"{gen}{size}{suffix}",
                        f"{name}{size}{suffix}",
                        (ptype,),
                        "ultralytics",
                        kind,
                        _ULTRA,
                        640,
                        notes="Ultralytics is licensed AGPL-3.0.",
                        params=(_MOSAIC, _CLOSE_MOSAIC),
                        defaults=_YOLO,
                    )
                )
    return out


MODELS: tuple[ModelSpec, ...] = tuple(_classification() + _torchvision_models() + _yolo_models())
_BY_ID = {m.id: m for m in MODELS}


def get_model(model_id: str) -> ModelSpec | None:
    """One model by id, or ``None``."""
    return _BY_ID.get(model_id)


def models_for(project_type: str | None = None) -> list[ModelSpec]:
    """Models a project of this type offers (every model without a type)."""
    return [m for m in MODELS if project_type is None or project_type in m.types]
