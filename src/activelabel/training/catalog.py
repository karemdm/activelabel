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

FAMILIES = ("torchvision", "timm", "smp", "ultralytics")
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
        family: ``torchvision``, ``timm``, ``smp`` (segmentation_models_pytorch) or ``ultralytics``.
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
_SMP = ("torch", "torchvision", "segmentation-models-pytorch")
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
_LOW_LR = (
    "ViT",
    "Swin",
    "Swin V2",
    "MaxViT",
    "ConvNeXt",
    "ConvNeXt V2",
    "DeiT",
    "BEiT",
    "EVA-02",
    "CaiT",
    "CoAtNet",
)


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


#: timm classification networks (#617), from the default pretrained configuration of timm 1.0.31:
#: ``(timm name, architecture, variant, label, parameters in millions, input size, fixed input size)``. The
#: weights are the first (default) pretrained tag of each name.
_TIMM_CLASSIFIERS: tuple[tuple[str, str, str, str, float, int, bool], ...] = (
    ("resnet18", "ResNet", "18", "ResNet 18", 11.69, 224, False),
    ("resnet34", "ResNet", "34", "ResNet 34", 21.8, 224, False),
    ("resnet50", "ResNet", "50", "ResNet 50", 25.56, 224, False),
    ("resnet101", "ResNet", "101", "ResNet 101", 44.55, 224, False),
    ("resnet152", "ResNet", "152", "ResNet 152", 60.19, 224, False),
    ("resnext50_32x4d", "ResNeXt", "50 32x4d", "ResNeXt 50 32x4d", 25.03, 224, False),
    ("resnext101_32x8d", "ResNeXt", "101 32x8d", "ResNeXt 101 32x8d", 88.79, 224, False),
    ("resnext101_64x4d", "ResNeXt", "101 64x4d", "ResNeXt 101 64x4d", 83.46, 224, False),
    ("regnety_008", "RegNet", "Y-800MF", "RegNet Y-800MF", 6.26, 224, False),
    ("regnety_016", "RegNet", "Y-1.6GF", "RegNet Y-1.6GF", 11.2, 224, False),
    ("regnety_032", "RegNet", "Y-3.2GF", "RegNet Y-3.2GF", 19.44, 224, False),
    ("regnety_080", "RegNet", "Y-8GF", "RegNet Y-8GF", 39.18, 224, False),
    ("regnety_160", "RegNet", "Y-16GF", "RegNet Y-16GF", 83.59, 224, False),
    ("densenet121", "DenseNet", "121", "DenseNet 121", 7.98, 224, False),
    ("densenet169", "DenseNet", "169", "DenseNet 169", 14.15, 224, False),
    ("densenet201", "DenseNet", "201", "DenseNet 201", 20.01, 224, False),
    ("mobilenetv3_small_100", "MobileNetV3", "Small 1.0", "MobileNetV3 Small 1.0", 2.54, 224, False),
    ("mobilenetv3_large_100", "MobileNetV3", "Large 1.0", "MobileNetV3 Large 1.0", 5.48, 224, False),
    ("mobilenetv3_large_150d", "MobileNetV3", "Large 1.5d", "MobileNetV3 Large 1.5d", 14.62, 256, False),
    ("mobilenetv4_conv_small", "MobileNetV4", "Conv S", "MobileNetV4 Conv S", 3.77, 224, False),
    ("mobilenetv4_conv_medium", "MobileNetV4", "Conv M", "MobileNetV4 Conv M", 9.72, 256, False),
    ("mobilenetv4_hybrid_medium", "MobileNetV4", "Hybrid M", "MobileNetV4 Hybrid M", 11.07, 256, False),
    ("mobilenetv4_conv_large", "MobileNetV4", "Conv L", "MobileNetV4 Conv L", 32.59, 384, False),
    ("mobilenetv4_hybrid_large", "MobileNetV4", "Hybrid L", "MobileNetV4 Hybrid L", 37.76, 384, False),
    ("efficientnet_b0", "EfficientNet", "B0", "EfficientNet B0", 5.29, 224, False),
    ("efficientnet_b2", "EfficientNet", "B2", "EfficientNet B2", 9.11, 256, False),
    ("efficientnet_b3", "EfficientNet", "B3", "EfficientNet B3", 12.23, 288, False),
    ("efficientnet_b4", "EfficientNet", "B4", "EfficientNet B4", 19.34, 320, False),
    ("efficientnet_b5", "EfficientNet", "B5", "EfficientNet B5", 30.39, 448, False),
    ("tf_efficientnet_b6", "EfficientNet", "B6", "EfficientNet B6", 43.04, 528, False),
    ("tf_efficientnet_b7", "EfficientNet", "B7", "EfficientNet B7", 66.35, 600, False),
    ("tf_efficientnetv2_s", "EfficientNetV2", "S", "EfficientNetV2 S", 21.46, 300, False),
    ("tf_efficientnetv2_m", "EfficientNetV2", "M", "EfficientNetV2 M", 54.14, 384, False),
    ("tf_efficientnetv2_l", "EfficientNetV2", "L", "EfficientNetV2 L", 118.52, 384, False),
    ("ghostnet_100", "GhostNet", "1.0", "GhostNet 1.0", 5.18, 224, False),
    ("ghostnetv2_100", "GhostNet", "V2 1.0", "GhostNet V2 1.0", 6.16, 224, False),
    ("ghostnetv2_160", "GhostNet", "V2 1.6", "GhostNet V2 1.6", 12.39, 224, False),
    ("mobilevit_xxs", "MobileViT", "XXS", "MobileViT XXS", 1.27, 256, False),
    ("mobilevit_xs", "MobileViT", "XS", "MobileViT XS", 2.32, 256, False),
    ("mobilevit_s", "MobileViT", "S", "MobileViT S", 5.58, 256, False),
    ("mobilevitv2_100", "MobileViTv2", "1.0", "MobileViTv2 1.0", 4.9, 256, False),
    ("mobilevitv2_150", "MobileViTv2", "1.5", "MobileViTv2 1.5", 10.59, 256, False),
    ("mobilevitv2_200", "MobileViTv2", "2.0", "MobileViTv2 2.0", 18.45, 256, False),
    ("convnext_atto", "ConvNeXt", "A", "ConvNeXt A", 3.7, 224, False),
    ("convnext_tiny", "ConvNeXt", "T", "ConvNeXt T", 28.59, 224, False),
    ("convnext_small", "ConvNeXt", "S", "ConvNeXt S", 50.22, 224, False),
    ("convnext_base", "ConvNeXt", "B", "ConvNeXt B", 88.59, 224, False),
    ("convnext_large", "ConvNeXt", "L", "ConvNeXt L", 197.77, 224, False),
    ("convnextv2_atto", "ConvNeXt V2", "A", "ConvNeXt V2 A", 3.71, 224, False),
    ("convnextv2_nano", "ConvNeXt V2", "N", "ConvNeXt V2 N", 15.62, 224, False),
    ("convnextv2_tiny", "ConvNeXt V2", "T", "ConvNeXt V2 T", 28.64, 224, False),
    ("convnextv2_base", "ConvNeXt V2", "B", "ConvNeXt V2 B", 88.72, 224, False),
    ("convnextv2_large", "ConvNeXt V2", "L", "ConvNeXt V2 L", 197.96, 224, False),
    ("eca_nfnet_l0", "NFNet", "ECA-L0", "NFNet ECA-L0", 24.14, 224, False),
    ("eca_nfnet_l1", "NFNet", "ECA-L1", "NFNet ECA-L1", 41.41, 256, False),
    ("eca_nfnet_l2", "NFNet", "ECA-L2", "NFNet ECA-L2", 56.72, 320, False),
    ("dm_nfnet_f0", "NFNet", "F0", "NFNet F0", 71.49, 192, False),
    ("dm_nfnet_f1", "NFNet", "F1", "NFNet F1", 132.63, 224, False),
    ("dm_nfnet_f3", "NFNet", "F3", "NFNet F3", 254.92, 320, False),
    ("vit_tiny_patch16_224", "ViT", "Ti/16", "ViT Ti/16", 5.72, 224, True),
    ("vit_small_patch16_224", "ViT", "S/16", "ViT S/16", 22.05, 224, True),
    ("vit_base_patch16_224", "ViT", "B/16", "ViT B/16", 86.57, 224, True),
    ("vit_base_patch16_384", "ViT", "B/16 384", "ViT B/16 384", 86.86, 384, True),
    ("vit_large_patch16_224", "ViT", "L/16", "ViT L/16", 304.33, 224, True),
    ("deit_small_patch16_224", "DeiT", "S/16", "DeiT S/16", 22.05, 224, True),
    ("deit_base_patch16_224", "DeiT", "B/16", "DeiT B/16", 86.57, 224, True),
    ("deit3_small_patch16_224", "DeiT", "DeiT3 S/16", "DeiT DeiT3 S/16", 22.06, 224, True),
    ("deit3_base_patch16_224", "DeiT", "DeiT3 B/16", "DeiT DeiT3 B/16", 86.59, 224, True),
    ("deit3_large_patch16_224", "DeiT", "DeiT3 L/16", "DeiT DeiT3 L/16", 304.37, 224, True),
    ("swin_tiny_patch4_window7_224", "Swin", "T", "Swin T", 28.29, 224, True),
    ("swin_small_patch4_window7_224", "Swin", "S", "Swin S", 49.61, 224, True),
    ("swin_base_patch4_window7_224", "Swin", "B", "Swin B", 87.77, 224, True),
    ("swin_base_patch4_window12_384", "Swin", "B 384", "Swin B 384", 87.9, 384, True),
    ("swin_large_patch4_window7_224", "Swin", "L", "Swin L", 196.53, 224, True),
    ("beit_base_patch16_224", "BEiT", "B/16", "BEiT B/16", 86.53, 224, True),
    ("beit_base_patch16_384", "BEiT", "B/16 384", "BEiT B/16 384", 86.74, 384, True),
    ("beit_large_patch16_224", "BEiT", "L/16", "BEiT L/16", 304.43, 224, True),
    ("beitv2_base_patch16_224", "BEiT", "BEiTv2 B/16", "BEiT BEiTv2 B/16", 86.53, 224, True),
    ("beitv2_large_patch16_224", "BEiT", "BEiTv2 L/16", "BEiT BEiTv2 L/16", 304.43, 224, True),
    ("eva02_tiny_patch14_336", "EVA-02", "Ti/14 336", "EVA-02 Ti/14 336", 5.76, 336, True),
    ("eva02_small_patch14_336", "EVA-02", "S/14 336", "EVA-02 S/14 336", 22.13, 336, True),
    ("eva02_base_patch14_448", "EVA-02", "B/14 448", "EVA-02 B/14 448", 87.12, 448, True),
    ("eva02_large_patch14_448", "EVA-02", "L/14 448", "EVA-02 L/14 448", 305.08, 448, True),
    ("cait_xxs24_224", "CaiT", "XXS24", "CaiT XXS24", 11.96, 224, True),
    ("cait_xxs36_224", "CaiT", "XXS36", "CaiT XXS36", 17.3, 224, True),
    ("cait_s24_224", "CaiT", "S24", "CaiT S24", 46.92, 224, True),
    ("cait_s36_384", "CaiT", "S36 384", "CaiT S36 384", 68.37, 384, True),
    ("cait_m36_384", "CaiT", "M36 384", "CaiT M36 384", 271.22, 384, True),
    ("coatnet_nano_rw_224", "CoAtNet", "N", "CoAtNet N", 15.14, 224, True),
    ("coatnet_0_rw_224", "CoAtNet", "0", "CoAtNet 0", 27.44, 224, True),
    ("coatnet_1_rw_224", "CoAtNet", "1", "CoAtNet 1", 41.72, 224, True),
    ("coatnet_2_rw_224", "CoAtNet", "2", "CoAtNet 2", 73.87, 224, True),
    ("maxvit_nano_rw_256", "MaxViT", "N", "MaxViT N", 15.45, 256, True),
    ("maxvit_tiny_tf_224", "MaxViT", "T", "MaxViT T", 30.92, 224, True),
    ("maxvit_small_tf_224", "MaxViT", "S", "MaxViT S", 68.93, 224, True),
    ("maxvit_base_tf_224", "MaxViT", "B", "MaxViT B", 119.47, 224, True),
    ("maxvit_large_tf_224", "MaxViT", "L", "MaxViT L", 211.79, 224, True),
    ("fastvit_t8", "FastViT", "T8", "FastViT T8", 4.03, 256, False),
    ("fastvit_t12", "FastViT", "T12", "FastViT T12", 7.55, 256, False),
    ("fastvit_s12", "FastViT", "S12", "FastViT S12", 9.47, 256, False),
    ("fastvit_sa24", "FastViT", "SA24", "FastViT SA24", 21.55, 256, False),
    ("fastvit_ma36", "FastViT", "MA36", "FastViT MA36", 44.07, 256, False),
)


def _timm_classifier(entry: tuple[str, str, str, str, float, int, bool], taken: set[str]) -> ModelSpec:
    """A classification model of timm.

    Args:
        entry: a row of :data:`_TIMM_CLASSIFIERS`.
        taken: ids of the torchvision models: a timm network of the same name gets the id ``timm_<name>``
            (``efficientnet_b0`` and ``efficientnet_b3``, which older runs used, keep theirs).

    Returns:
        The model spec.
    """
    name, arch, variant, label, millions, size, fixed = entry
    id_ = f"timm_{name}" if name in taken else name
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
        limits={"image_size": (size, size)} if fixed else {},
        arch=arch,
        variant=variant,
        params_m=millions,
        builder=name if id_ != name else "",
    )


def _classification() -> list[ModelSpec]:
    """The classification models: every torchvision network, grouped by architecture, and those of timm.

    Returns:
        One spec per model, torchvision and timm.
    """
    torchvision = [_classifier(entry) for entry in _TV_CLASSIFIERS]
    taken = {m.id for m in torchvision}
    return torchvision + [_timm_classifier(entry, taken) for entry in _TIMM_CLASSIFIERS]


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


#: Ultralytics generations (#632), the default first; YOLO12 is not offered (owner's decision). Every
#: generation has pretrained weights for every task and size (release v8.4.0 of ``ultralytics/assets``).
YOLO_GENERATIONS = (("yolo26", "YOLO26"), ("yolo11", "YOLO11"), ("yolov8", "YOLOv8"))
YOLO_SIZES = "nsmlx"
#: ``(id suffix, project type, kind of task)`` of the Ultralytics tasks ActiveLabel trains.
YOLO_TASKS = (
    ("", "detection", "detection"),
    ("-seg", "segmentation", "instance"),
    ("-pose", "keypoints", "keypoints"),
    ("-cls", "classification", "classification"),
)
#: Batch by size (n/s small, l/x large): what fits a common GPU at the default input size.
YOLO_BATCH = {"n": 16, "s": 16, "m": 8, "l": 4, "x": 4}
#: Parameters in millions, from Ultralytics' published tables, by generation and task (sizes n s m l x).
YOLO_PARAMS: dict[tuple[str, str], tuple[float, ...]] = {
    ("yolov8", ""): (3.2, 11.2, 25.9, 43.7, 68.2),
    ("yolov8", "-seg"): (3.4, 11.8, 27.3, 46.0, 71.8),
    ("yolov8", "-pose"): (3.3, 11.6, 26.4, 44.4, 69.4),
    ("yolov8", "-cls"): (2.7, 6.4, 17.0, 37.5, 57.4),
    ("yolo11", ""): (2.6, 9.4, 20.1, 25.3, 56.9),
    ("yolo11", "-seg"): (2.9, 10.1, 22.4, 27.6, 62.1),
    ("yolo11", "-pose"): (2.9, 9.9, 20.9, 26.2, 58.8),
    ("yolo11", "-cls"): (1.6, 5.5, 10.4, 12.9, 28.4),
}
#: The architecture each family proposes first (the dialog's default).
DEFAULT_ARCH = {"ultralytics": "YOLO26"}
#: GPU memory from which the dialog proposes a medium model rather than a small one.
LARGE_GPU_MB = 8 * 1024


def device_size(gpu_memory_mb: int | None, on_gpu: bool) -> str:
    """The size the dialog proposes first for this computer: n on the CPU, s on a small GPU, m on a large one.

    Args:
        gpu_memory_mb: memory of the GPU, in MiB (``None`` when not known).
        on_gpu: whether the training runs on a GPU.

    Returns:
        ``n``, ``s`` or ``m``.
    """
    if not on_gpu:
        return "n"
    return "m" if gpu_memory_mb is not None and gpu_memory_mb >= LARGE_GPU_MB else "s"


def _yolo_models() -> list[ModelSpec]:
    """The Ultralytics YOLO models (trained by ``workers/yolo.py`` once the AGPL licence is accepted).

    Returns:
        One spec per generation, size and task.
    """
    out = []
    for suffix, ptype, kind in YOLO_TASKS:
        for gen, name in YOLO_GENERATIONS:
            params = YOLO_PARAMS.get((gen, suffix))
            for k, size in enumerate(YOLO_SIZES):
                out.append(
                    ModelSpec(
                        f"{gen}{size}{suffix}",
                        f"{name}{size}{suffix}",
                        (ptype,),
                        "ultralytics",
                        kind,
                        _ULTRA,
                        224 if kind == "classification" else 640,
                        notes="Ultralytics is licensed AGPL-3.0: accept the licence before it is installed.",
                        params=(_MOSAIC, _CLOSE_MOSAIC),
                        defaults={**_YOLO, "batch_size": YOLO_BATCH[size]},
                        arch=name,
                        variant=size,
                        params_m=params[k] if params else None,
                        choices=_YOLO_CHOICES,
                    )
                )
    return out


#: segmentation_models_pytorch decoders (#606): ``(class name, label)``; each is offered with every encoder of
#: :data:`SMP_ENCODERS` (Segformer with its own Mix Transformer encoders).
SMP_DECODERS = (
    ("Unet", "U-Net"),
    ("UnetPlusPlus", "U-Net++"),
    ("DeepLabV3Plus", "DeepLabV3+"),
    ("FPN", "FPN"),
    ("PSPNet", "PSPNet"),
    ("Linknet", "LinkNet"),
    ("MAnet", "MA-Net"),
    ("PAN", "PAN"),
    ("UPerNet", "UPerNet"),
)
#: Encoders with ImageNet weights in segmentation_models_pytorch: ``(encoder name, label)``.
SMP_ENCODERS = (("resnet34", "ResNet-34"), ("resnet50", "ResNet-50"), ("mobilenet_v2", "MobileNetV2"))
SMP_SEGFORMER_ENCODERS = (("mit_b0", "MiT-B0"), ("mit_b1", "MiT-B1"), ("mit_b2", "MiT-B2"))
#: Sides must be a multiple of 32 (the worker checks it); at least 128 pixels (an old limit of PAN, kept).
_SMP_LIMITS: dict[str, tuple[float, float]] = {"image_size": (128, 2048)}


def _smp(decoder: str, decoder_label: str, encoder: str, encoder_label: str) -> ModelSpec:
    """A semantic segmentation model of segmentation_models_pytorch: a decoder on an encoder.

    Args:
        decoder: the smp class name (``Unet``...).
        decoder_label: the decoder's name shown to people (``U-Net``...).
        encoder: the smp encoder name (``resnet34``...).
        encoder_label: the encoder's name shown to people.

    Returns:
        The model spec: id ``smp_<decoder>_<encoder>``, built by smp as ``<decoder>:<encoder>``.
    """
    return ModelSpec(
        f"smp_{decoder.lower()}_{encoder}",
        f"{decoder_label} {encoder_label}",
        ("segmentation",),
        "smp",
        "semantic",
        _SMP,
        512,
        params=(_CLASS_WEIGHTS, _SEG_LOSS),
        defaults=_TV_SEG,
        limits=_SMP_LIMITS,
        arch=decoder_label,
        variant=encoder_label,
        builder=f"{decoder}:{encoder}",
    )


def _smp_models() -> list[ModelSpec]:
    """The segmentation_models_pytorch models (trained by the segmentation worker).

    Returns:
        One spec per decoder and encoder.
    """
    out = [_smp(d, dl, e, el) for d, dl in SMP_DECODERS for e, el in SMP_ENCODERS]
    return out + [_smp("Segformer", "SegFormer", e, el) for e, el in SMP_SEGFORMER_ENCODERS]


MODELS: tuple[ModelSpec, ...] = tuple(
    _classification() + _torchvision_models() + _smp_models() + _yolo_models()
)
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
