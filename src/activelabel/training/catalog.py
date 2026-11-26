"""The model catalogue, as data (see docs/TRAINING.md).

Each model lists its family, packages, default input size and hyperparameters; the screen builds
its form from this, so adding a model does not change the interface. Defaults are chosen for a
CPU: a small batch and few epochs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from activelabel.errors import CodedError

FAMILIES = ("torchvision", "timm", "ultralytics")
TASK_KINDS = ("classification", "detection", "semantic", "instance", "keypoints")
STARTS = ("transfer", "finetune", "scratch")
#: How a photo becomes the square input of the network (see ``workers/options.py``).
RESIZE_POLICIES = ("stretch", "letterbox", "center_crop")
#: Learning rate schedules every worker applies (see ``workers/options.py``).
SCHEDULERS = ("none", "cosine", "step", "onecycle", "plateau")
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

        Args:
            value: the value a configuration gives.

        Returns:
            The value as ``bool``, ``str``, ``int`` or ``float``.

        Raises:
            CodedError: with a readable message.
        """
        name = self.name
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise CodedError("param_not_bool", f"'{name}' must be true or false.", name=name)
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise CodedError(
                    "param_not_choice",
                    f"'{name}' must be one of: {', '.join(self.choices)}.",
                    name=name,
                    choices=list(self.choices),
                )
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CodedError("param_not_number", f"'{name}' must be a number.", name=name)
        try:
            if isinstance(value, float) and not math.isfinite(value):
                raise CodedError("param_not_finite", f"'{name}' must be a finite number.", name=name)
            if self.kind == "int":
                if int(value) != value:
                    raise CodedError("param_not_whole", f"'{name}' must be a whole number.", name=name)
                value = int(value)
            else:
                value = float(value)
        except OverflowError:
            raise CodedError("param_too_large", f"'{name}' is too large.", name=name) from None
        if isinstance(value, float) and not math.isfinite(value):
            raise CodedError("param_not_finite", f"'{name}' must be a finite number.", name=name)
        if self.min is not None and value < self.min:
            raise CodedError(
                "param_min", f"'{name}' must be at least {self.min:.12g}.", name=name, min=self.min
            )
        if self.max is not None and value > self.max:
            raise CodedError(
                "param_max", f"'{name}' must be at most {self.max:.12g}.", name=name, max=self.max
            )
        return value

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            The parameter as plain data for the API.
        """
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
        limits: replacements for the (min, max) of common parameters, where the model accepts less
            (for example a fixed input size).
        choices: replacements for the choices of common parameters, where the model's library offers
            fewer (Ultralytics has only the cosine and the constant learning-rate schedules).
        arch: architecture or generation the dialog groups by (``ResNet``, ``YOLO11``...).
        variant: size or variant inside the architecture (``50``, ``n``...); empty when it has only one.
        experimental: offered, but not as reliable as the others (the dialog says so).
        params_m: parameters of the network, in millions (``None`` when not known).
        builder: the name the library builds it by, when it differs from ``id`` (two families have a
            network of the same name: torchvision's EfficientNet-B0 is ``tv_efficientnet_b0`` here).
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
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    arch: str = ""
    variant: str = ""
    experimental: bool = False
    params_m: float | None = None
    builder: str = ""

    @property
    def library_name(self) -> str:
        """The name the library (torchvision, timm, Ultralytics) builds this model by.

        Returns:
            ``builder``, or the id when the two are the same.
        """
        return self.builder or self.id

    def param_specs(self) -> list[ParamSpec]:
        """Common parameters (with this model's defaults, limits and choices) followed by the model's own.

        Returns:
            The effective specs, in the order the form shows them.
        """
        out = []
        for spec in COMMON_PARAMS:
            default = (
                self.image_size if spec.name == "image_size" else self.defaults.get(spec.name, spec.default)
            )
            low, high = self.limits.get(spec.name, (spec.min, spec.max))
            choices = self.choices.get(spec.name, spec.choices)
            out.append(replace(spec, default=default, min=low, max=high, choices=choices))
        return out + list(self.params)

    def param(self, name: str) -> ParamSpec | None:
        """One parameter by name (common or model-specific).

        Args:
            name: the parameter's key in ``params``.

        Returns:
            Its effective spec, or ``None`` when this model has no such parameter.
        """
        return next((p for p in self.param_specs() if p.name == name), None)

    def default_metric(self) -> str:
        """Default selection metric of this model's kind of task.

        Returns:
            A metric name of :data:`METRICS`.
        """
        return METRICS[self.task_kind][1]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, with the effective parameter specs and selection metrics.

        Returns:
            The model as plain data for the API.
        """
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
            "arch": self.arch or self.label,
            "variant": self.variant,
            "experimental": self.experimental,
            "params_m": self.params_m,
            # float32 weights: four bytes per parameter (the file of a trained model is about this size)
            "file_mb": None if self.params_m is None else round(self.params_m * 4 * 1e6 / 2**20, 1),
        }


COMMON_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("epochs", "int", 10, 1, 1000, help="Passes over the training images."),
    ParamSpec("batch_size", "int", 8, 1, 512, help="Images per step (small suits a CPU)."),
    ParamSpec("lr", "float", 0.001, 1e-6, 1.0, help="Learning rate."),
    ParamSpec("optimizer", "choice", "adamw", choices=("sgd", "adam", "adamw"), help="Optimizer."),
    ParamSpec("weight_decay", "float", 0.0001, 0.0, 1.0, help="Weight decay (regularisation)."),
    ParamSpec(
        "scheduler",
        "choice",
        "cosine",
        choices=SCHEDULERS,
        help="Learning rate schedule: none, cosine, step (/10 every third of the epochs), onecycle (up, then "
        "down; it has its own warmup) or plateau (/10 when the selection metric stops improving).",
    ),
    ParamSpec(
        "warmup_epochs",
        "int",
        0,
        0,
        50,
        help="Epochs over which the learning rate rises from almost 0 (0 = off); steadies fine-tuning.",
    ),
    ParamSpec("image_size", "int", 224, 32, 2048, help="Side of the square input, in pixels."),
    ParamSpec(
        "resize",
        "choice",
        "stretch",
        choices=RESIZE_POLICIES,
        help="How a photo becomes the square input: stretch (the proportions change), letterbox (bands "
        "fill the rest) or center_crop (the edges are cut). Evaluation, pre-annotation and export use the "
        "same rule.",
    ),
    ParamSpec(
        "augmentation",
        "choice",
        "light",
        choices=("none", "light", "standard", "strong", "custom"),
        help="How much the training images are varied: a preset list of techniques, or custom (the list "
        "of the run's 'augmentations' option).",
    ),
    ParamSpec(
        "patience", "int", 0, 0, 100, help="Stop after this many epochs without improvement (0 = off)."
    ),
    ParamSpec(
        "amp",
        "choice",
        "auto",
        choices=("auto", "on", "off"),
        help="Mixed precision (float16) on a CUDA GPU: faster and lighter. auto and on use it on CUDA only; "
        "the CPU and Apple GPUs train in full precision.",
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
_CLASS_WEIGHTS = ParamSpec(
    "class_weights",
    "choice",
    "none",
    choices=("none", "inverse"),
    help="inverse: rare classes weigh more in the loss (inverse of their frequency in the training part).",
    group="model",
)
_CLS_LOSS = ParamSpec(
    "loss",
    "choice",
    "cross_entropy",
    choices=("cross_entropy", "focal"),
    help="focal: counts the easy examples less, for imbalanced classes.",
    group="model",
)
_FOCAL_GAMMA = ParamSpec(
    "focal_gamma",
    "float",
    2.0,
    0.0,
    5.0,
    help="Focal loss: how much easy examples are discounted.",
    group="model",
)
_SEG_LOSS = ParamSpec(
    "loss",
    "choice",
    "cross_entropy",
    choices=("cross_entropy", "dice", "cross_entropy_dice"),
    help="dice and cross_entropy_dice make small classes count (a model that only predicts background).",
    group="model",
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
#: Detectors keep the proportions of the photo by default (torchvision's own resize did too).
_TV_DET = {
    "optimizer": "sgd",
    "lr": 0.005,
    "batch_size": 4,
    "weight_decay": 0.0005,
    "resize": "letterbox",
    "warmup_epochs": 1,  # detectors from COCO weights need a ramp (they had a fixed one before)
}
_TV_SEG = {"lr": 0.0001, "batch_size": 4}
_YOLO = {"optimizer": "sgd", "lr": 0.01, "batch_size": 8, "weight_decay": 0.0005}
#: Ultralytics decays the learning rate linearly or with a cosine (``cos_lr``); ``none`` keeps it constant.
_YOLO_CHOICES: dict[str, tuple[str, ...]] = {"scheduler": ("none", "cosine")}
_FIXED_224: dict[str, tuple[float, float]] = {"image_size": (224, 224)}
_FIXED_320: dict[str, tuple[float, float]] = {"image_size": (320, 320)}


#: torchvision classification networks (#618): ``(id, architecture, variant, label, parameters in millions,
#: input size)``, grouped by architecture; the input size is the one of the default ImageNet weights.
_TV_CLASSIFIERS: tuple[tuple[str, str, str, str, float, int], ...] = (
    ("resnet18", "ResNet", "18", "ResNet-18", 11.7, 224),
    ("resnet34", "ResNet", "34", "ResNet-34", 21.8, 224),
    ("resnet50", "ResNet", "50", "ResNet-50", 25.6, 224),
    ("resnet101", "ResNet", "101", "ResNet-101", 44.5, 224),
    ("resnet152", "ResNet", "152", "ResNet-152", 60.2, 224),
    ("resnext50_32x4d", "ResNeXt", "50 32x4d", "ResNeXt-50 32x4d", 25.0, 224),
    ("resnext101_32x8d", "ResNeXt", "101 32x8d", "ResNeXt-101 32x8d", 88.8, 224),
    ("resnext101_64x4d", "ResNeXt", "101 64x4d", "ResNeXt-101 64x4d", 83.5, 224),
    ("wide_resnet50_2", "Wide ResNet", "50-2", "Wide ResNet-50-2", 68.9, 224),
    ("wide_resnet101_2", "Wide ResNet", "101-2", "Wide ResNet-101-2", 126.9, 224),
    ("densenet121", "DenseNet", "121", "DenseNet-121", 8.0, 224),
    ("densenet161", "DenseNet", "161", "DenseNet-161", 28.7, 224),
    ("densenet169", "DenseNet", "169", "DenseNet-169", 14.1, 224),
    ("densenet201", "DenseNet", "201", "DenseNet-201", 20.0, 224),
    ("vgg11", "VGG", "11", "VGG-11", 132.9, 224),
    ("vgg11_bn", "VGG", "11 BN", "VGG-11 BN", 132.9, 224),
    ("vgg13", "VGG", "13", "VGG-13", 133.0, 224),
    ("vgg13_bn", "VGG", "13 BN", "VGG-13 BN", 133.1, 224),
    ("vgg16", "VGG", "16", "VGG-16", 138.4, 224),
    ("vgg16_bn", "VGG", "16 BN", "VGG-16 BN", 138.4, 224),
    ("vgg19", "VGG", "19", "VGG-19", 143.7, 224),
    ("vgg19_bn", "VGG", "19 BN", "VGG-19 BN", 143.7, 224),
    ("mobilenet_v2", "MobileNet", "V2", "MobileNetV2", 3.5, 224),
    ("mobilenet_v3_small", "MobileNet", "V3 Small", "MobileNetV3 Small", 2.5, 224),
    ("mobilenet_v3_large", "MobileNet", "V3 Large", "MobileNetV3 Large", 5.5, 224),
    ("efficientnet_b0", "EfficientNet", "B0", "EfficientNet-B0", 5.3, 224),
    ("efficientnet_b1", "EfficientNet", "B1", "EfficientNet-B1", 7.8, 240),
    ("efficientnet_b2", "EfficientNet", "B2", "EfficientNet-B2", 9.1, 288),
    ("efficientnet_b3", "EfficientNet", "B3", "EfficientNet-B3", 12.2, 300),
    ("efficientnet_b4", "EfficientNet", "B4", "EfficientNet-B4", 19.3, 380),
    ("efficientnet_b5", "EfficientNet", "B5", "EfficientNet-B5", 30.4, 456),
    ("efficientnet_b6", "EfficientNet", "B6", "EfficientNet-B6", 43.0, 528),
    ("efficientnet_b7", "EfficientNet", "B7", "EfficientNet-B7", 66.3, 600),
    ("efficientnet_v2_s", "EfficientNetV2", "S", "EfficientNetV2-S", 21.5, 384),
    ("efficientnet_v2_m", "EfficientNetV2", "M", "EfficientNetV2-M", 54.1, 480),
    ("efficientnet_v2_l", "EfficientNetV2", "L", "EfficientNetV2-L", 118.5, 480),
    ("regnet_x_400mf", "RegNetX", "400MF", "RegNetX-400MF", 5.5, 224),
    ("regnet_x_800mf", "RegNetX", "800MF", "RegNetX-800MF", 7.3, 224),
    ("regnet_x_1_6gf", "RegNetX", "1.6GF", "RegNetX-1.6GF", 9.2, 224),
    ("regnet_x_3_2gf", "RegNetX", "3.2GF", "RegNetX-3.2GF", 15.3, 224),
    ("regnet_x_8gf", "RegNetX", "8GF", "RegNetX-8GF", 39.6, 224),
    ("regnet_x_16gf", "RegNetX", "16GF", "RegNetX-16GF", 54.3, 224),
    ("regnet_x_32gf", "RegNetX", "32GF", "RegNetX-32GF", 107.8, 224),
    ("regnet_y_400mf", "RegNetY", "400MF", "RegNetY-400MF", 4.3, 224),
    ("regnet_y_800mf", "RegNetY", "800MF", "RegNetY-800MF", 6.4, 224),
    ("regnet_y_1_6gf", "RegNetY", "1.6GF", "RegNetY-1.6GF", 11.2, 224),
    ("regnet_y_3_2gf", "RegNetY", "3.2GF", "RegNetY-3.2GF", 19.4, 224),
    ("regnet_y_8gf", "RegNetY", "8GF", "RegNetY-8GF", 39.4, 224),
    ("regnet_y_16gf", "RegNetY", "16GF", "RegNetY-16GF", 83.6, 224),
    ("regnet_y_32gf", "RegNetY", "32GF", "RegNetY-32GF", 145.0, 224),
    ("regnet_y_128gf", "RegNetY", "128GF", "RegNetY-128GF", 644.8, 384),
    ("convnext_tiny", "ConvNeXt", "T", "ConvNeXt-T", 28.6, 224),
    ("convnext_small", "ConvNeXt", "S", "ConvNeXt-S", 50.2, 224),
    ("convnext_base", "ConvNeXt", "B", "ConvNeXt-B", 88.6, 224),
    ("convnext_large", "ConvNeXt", "L", "ConvNeXt-L", 197.8, 224),
    ("vit_b_16", "ViT", "B/16", "ViT-B/16", 86.6, 224),
    ("vit_b_32", "ViT", "B/32", "ViT-B/32", 88.2, 224),
    ("vit_l_16", "ViT", "L/16", "ViT-L/16", 304.3, 224),
    ("vit_l_32", "ViT", "L/32", "ViT-L/32", 306.5, 224),
    ("vit_h_14", "ViT", "H/14", "ViT-H/14", 633.5, 518),
    ("swin_t", "Swin", "T", "Swin-T", 28.3, 224),
    ("swin_s", "Swin", "S", "Swin-S", 49.6, 224),
    ("swin_b", "Swin", "B", "Swin-B", 87.8, 224),
    ("swin_v2_t", "Swin V2", "T", "Swin V2-T", 28.4, 256),
    ("swin_v2_s", "Swin V2", "S", "Swin V2-S", 49.7, 256),
    ("swin_v2_b", "Swin V2", "B", "Swin V2-B", 87.9, 256),
    ("maxvit_t", "MaxViT", "T", "MaxViT-T", 30.9, 224),
    ("shufflenet_v2_x0_5", "ShuffleNet V2", "0.5x", "ShuffleNet V2 0.5x", 1.4, 224),
    ("shufflenet_v2_x1_0", "ShuffleNet V2", "1.0x", "ShuffleNet V2 1.0x", 2.3, 224),
    ("shufflenet_v2_x1_5", "ShuffleNet V2", "1.5x", "ShuffleNet V2 1.5x", 3.5, 224),
    ("shufflenet_v2_x2_0", "ShuffleNet V2", "2.0x", "ShuffleNet V2 2.0x", 7.4, 224),
    ("squeezenet1_0", "SqueezeNet", "1.0", "SqueezeNet 1.0", 1.2, 224),
    ("squeezenet1_1", "SqueezeNet", "1.1", "SqueezeNet 1.1", 1.2, 224),
    ("mnasnet0_5", "MNASNet", "0.5", "MNASNet 0.5", 2.2, 224),
    ("mnasnet0_75", "MNASNet", "0.75", "MNASNet 0.75", 3.2, 224),
    ("mnasnet1_0", "MNASNet", "1.0", "MNASNet 1.0", 4.4, 224),
    ("mnasnet1_3", "MNASNet", "1.3", "MNASNet 1.3", 6.3, 224),
    ("googlenet", "GoogLeNet", "", "GoogLeNet", 6.6, 224),
    ("inception_v3", "Inception", "V3", "Inception V3", 27.2, 299),
    ("alexnet", "AlexNet", "", "AlexNet", 61.1, 224),
)
#: Architectures whose position embeddings or windows fix the input size (pretrained weights).
_FIXED_INPUT = ("ViT", "MaxViT")
#: Smallest input of architectures that cannot take the common minimum (their pooling would reach zero).
_MIN_INPUT = {"Inception": 75, "AlexNet": 64}
#: Transformers (and ConvNeXt) train better with a lower learning rate.
_LOW_LR = ("ViT", "Swin", "Swin V2", "MaxViT", "ConvNeXt")


def _classifier_defaults(arch: str, millions: float) -> dict[str, Any]:
    """Per-variant defaults of a classification network: learning rate by architecture, batch by size.

    Args:
        arch: the architecture (``ResNet``, ``ViT``...).
        millions: its parameters, in millions.

    Returns:
        The replacements for the defaults of the common parameters.
    """
    out: dict[str, Any] = dict(_TRANSFORMER) if arch in _LOW_LR else {}
    if millions > 300:
        out["batch_size"] = 2
    elif millions > 100:
        out["batch_size"] = 4
    return out


def _classifier_limits(arch: str, size: int) -> dict[str, tuple[float, float]]:
    """Input size limits of a classification network.

    Args:
        arch: the architecture.
        size: its input size.

    Returns:
        The replacements for the (min, max) of the common parameters.
    """
    if arch in _FIXED_INPUT:
        return {"image_size": (size, size)}
    if arch in _MIN_INPUT:
        return {"image_size": (_MIN_INPUT[arch], 2048)}
    return {}


def _classifier(entry: tuple[str, str, str, str, float, int]) -> ModelSpec:
    """A classification model of torchvision.

    Args:
        entry: ``(id, architecture, variant, label, parameters in millions, input size)``, as in
            :data:`_TV_CLASSIFIERS`.

    Returns:
        The model spec.
    """
    name, arch, variant, label, millions, size = entry
    # timm has EfficientNets of the same names (efficientnet_b0, efficientnet_b3): torchvision's get a prefix
    id_ = f"tv_{name}" if name.startswith("efficientnet_b") else name
    return ModelSpec(
        id_,
        label,
        ("classification",),
        "torchvision",
        "classification",
        _TV,
        size,
        params=(_LABEL_SMOOTHING, _CLASS_WEIGHTS, _CLS_LOSS, _FOCAL_GAMMA),
        defaults=_classifier_defaults(arch, millions),
        limits=_classifier_limits(arch, size),
        arch=arch,
        variant=variant,
        params_m=millions,
        builder=name if id_ != name else "",
    )


def _timm_classifier(id_: str, arch: str, variant: str, label: str, millions: float, size: int) -> ModelSpec:
    """A classification model of timm.

    Args:
        id_: model id (the timm model name).
        arch: the architecture the dialog groups by.
        variant: its size inside the architecture.
        label: name shown to people.
        millions: parameters, in millions.
        size: input size of its pretrained weights.

    Returns:
        The model spec.
    """
    return ModelSpec(
        id_,
        label,
        ("classification",),
        "timm",
        "classification",
        _TIMM,
        size,
        params=(_LABEL_SMOOTHING, _CLASS_WEIGHTS, _CLS_LOSS, _FOCAL_GAMMA),
        defaults=_classifier_defaults(arch, millions),
        arch=arch,
        variant=variant,
        params_m=millions,
    )


def _classification() -> list[ModelSpec]:
    """The classification models: every torchvision network, grouped by architecture, and those of timm.

    Returns:
        One spec per model, torchvision and timm.
    """
    timm = [
        _timm_classifier("efficientnet_b0", "EfficientNet", "B0", "EfficientNet-B0", 5.3, 224),
        _timm_classifier("efficientnet_b3", "EfficientNet", "B3", "EfficientNet-B3", 12.2, 288),
    ]
    return [_classifier(entry) for entry in _TV_CLASSIFIERS] + timm


def _tv(
    id_: str,
    label: str,
    ptype: str,
    kind: str,
    size: int,
    defaults: dict[str, Any],
    arch: str = "",
    variant: str = "",
    millions: float | None = None,
) -> ModelSpec:
    """A detection or segmentation model of torchvision.

    Args:
        id_: torchvision builder name.
        label: name shown to people.
        ptype: project type that offers it.
        kind: kind of task (detection, semantic, instance or keypoints).
        size: default input size, in pixels.
        defaults: replacements for the default of common parameters.
        arch: the architecture the dialog groups by (``Faster R-CNN``...).
        variant: its backbone (``ResNet-50 FPN``...).
        millions: parameters of the network (COCO head), in millions.

    Returns:
        The model spec; object models add the backbone layers and score threshold parameters, semantic models
        the class weights and loss.
    """
    extra = (_TRAINABLE_LAYERS, _SCORE_THRESH) if kind in ("detection", "instance", "keypoints") else ()
    if kind == "semantic":
        extra = (_CLASS_WEIGHTS, _SEG_LOSS)
    limits = _FIXED_320 if id_.startswith("ssdlite320") else {}  # SSDlite has a fixed input
    if id_.startswith("ssdlite320"):
        defaults = {**defaults, "resize": "stretch"}  # what SSDlite's own resize did
    return ModelSpec(
        id_,
        label,
        (ptype,),
        "torchvision",
        kind,
        _TV,
        size,
        params=extra,
        defaults=defaults,
        limits=limits,
        arch=arch,
        variant=variant,
        params_m=millions,
    )


def _torchvision_models() -> list[ModelSpec]:
    """The torchvision detection, segmentation and keypoint models.

    Returns:
        One spec per model.
    """
    det, seg = "detection", "segmentation"
    return [
        _tv(
            "fasterrcnn_resnet50_fpn",
            "Faster R-CNN ResNet-50 FPN",
            det,
            det,
            640,
            _TV_DET,
            "Faster R-CNN",
            "ResNet-50 FPN",
            41.8,
        ),
        _tv(
            "fasterrcnn_mobilenet_v3_large_fpn",
            "Faster R-CNN MobileNetV3 FPN",
            det,
            det,
            640,
            _TV_DET,
            "Faster R-CNN",
            "MobileNetV3 FPN",
            19.4,
        ),
        _tv(
            "retinanet_resnet50_fpn",
            "RetinaNet ResNet-50 FPN",
            det,
            det,
            640,
            _TV_DET,
            "RetinaNet",
            "ResNet-50 FPN",
            34.0,
        ),
        _tv(
            "ssdlite320_mobilenet_v3_large",
            "SSDlite320 MobileNetV3",
            det,
            det,
            320,
            _TV_DET,
            "SSDlite",
            "320 MobileNetV3",
            3.4,
        ),
        _tv("fcos_resnet50_fpn", "FCOS ResNet-50 FPN", det, det, 640, _TV_DET, "FCOS", "ResNet-50 FPN", 32.3),
        _tv(
            "deeplabv3_resnet50",
            "DeepLabV3 ResNet-50",
            seg,
            "semantic",
            512,
            _TV_SEG,
            "DeepLabV3",
            "ResNet-50",
            42.0,
        ),
        _tv(
            "deeplabv3_mobilenet_v3_large",
            "DeepLabV3 MobileNetV3",
            seg,
            "semantic",
            512,
            _TV_SEG,
            "DeepLabV3",
            "MobileNetV3",
            11.0,
        ),
        _tv("fcn_resnet50", "FCN ResNet-50", seg, "semantic", 512, _TV_SEG, "FCN", "ResNet-50", 35.3),
        _tv(
            "lraspp_mobilenet_v3_large",
            "LR-ASPP MobileNetV3",
            seg,
            "semantic",
            512,
            _TV_SEG,
            "LR-ASPP",
            "MobileNetV3",
            3.2,
        ),
        _tv(
            "maskrcnn_resnet50_fpn",
            "Mask R-CNN ResNet-50 FPN",
            seg,
            "instance",
            640,
            _TV_DET,
            "Mask R-CNN",
            "ResNet-50 FPN",
            44.4,
        ),
        _tv(
            "keypointrcnn_resnet50_fpn",
            "Keypoint R-CNN ResNet-50 FPN",
            "keypoints",
            "keypoints",
            640,
            _TV_DET,
            "Keypoint R-CNN",
            "ResNet-50 FPN",
            59.1,
        ),
    ]


def _yolo_models() -> list[ModelSpec]:
    """The Ultralytics YOLO models (trained by ``workers/yolo.py`` once the AGPL licence is accepted).

    Returns:
        One spec per generation, size and task.
    """
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
                        notes="Ultralytics is licensed AGPL-3.0: accept the licence before it is installed.",
                        params=(_MOSAIC, _CLOSE_MOSAIC),
                        defaults=_YOLO,
                        choices=_YOLO_CHOICES,
                    )
                )
    return out


MODELS: tuple[ModelSpec, ...] = tuple(_classification() + _torchvision_models() + _yolo_models())
_BY_ID = {m.id: m for m in MODELS}


def get_model(model_id: str) -> ModelSpec | None:
    """One model by id.

    Args:
        model_id: the model's id.

    Returns:
        Its spec, or ``None`` when the catalogue has no such model.
    """
    return _BY_ID.get(model_id)


def models_for(project_type: str | None = None) -> list[ModelSpec]:
    """Models a project of this type offers.

    Args:
        project_type: a project type, or ``None`` for every model.

    Returns:
        The specs, in catalogue order.
    """
    return [m for m in MODELS if project_type is None or project_type in m.types]
