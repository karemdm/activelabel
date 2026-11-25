"""Semantic segmentation worker (docs/TRAINING.md): ``python segmentation.py spec.json``.

Same protocol as the classification worker: one JSON object per line on stdout (``log``,
``warning``, ``setup``, ``progress``, ``epoch``, ``done``, ``error``), and ``best.pt``, ``last.pt`` and
``results.json`` in ``output_dir``.

Standalone: it imports only the standard library and, inside the functions that need them, torch,
torchvision, numpy and PIL, so it runs in the training environment (which has no ActiveLabel) and
its pure helpers (rasterization, augmentation, metrics, spec parsing) can be tested without torch.

Spec data (``parts[name]`` items)::

    {"path": "/abs/image.jpg",
     "objects": [{"label": 0, "polygons": [[[x, y], ...], ...], "bbox": [x, y, w, h]}, ...]}

``label`` is the index into ``spec["classes"]`` (a class name is accepted too); coordinates are
relative (0..1). In the mask, 0 is the background and class ``i`` is ``i + 1``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import numpy
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

try:
    from activelabel.training.workers import augment, options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling files
    import augment  # type: ignore[no-redef]
    import options  # type: ignore[no-redef]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
METRICS = ("miou", "dice", "pixel_accuracy")
MODELS = ("deeplabv3_resnet50", "deeplabv3_mobilenet_v3_large", "fcn_resnet50", "lraspp_mobilenet_v3_large")
AUX_WEIGHT = 0.5
BACKGROUND = "background"
ProgressFn = Any


class SpecError(ValueError):
    """The spec cannot be used (the message is readable)."""


# ---------------------------------------------------------------------------
# Pure helpers (no torch)


def emit(event: str, **fields: Any) -> None:
    """Print one event line.

    Args:
        event: Event name (``log``, ``warning``, ``setup``, ``progress``, ``epoch``, ``done`` or ``error``).
        **fields: Extra fields of the JSON object.
    """
    print(json.dumps({"event": event, **fields}), flush=True)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read and check ``spec.json``.

    Args:
        path: Path of the spec file.

    Returns:
        The spec as a dict.

    Raises:
        SpecError: missing keys, an unsupported task or model, no images or a fold ``"all"``.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task", "model", "family", "start", "params", "classes", "parts", "output_dir", "metric"):
        if key not in spec:
            raise SpecError(f"spec.json has no '{key}'.")
    if spec["task"] not in ("semantic", "segmentation"):
        raise SpecError(f"This worker trains semantic segmentation, not {spec['task']}.")
    if spec.get("fold") == "all":
        raise SpecError("Training all folds comes later: choose one fold.")
    if spec["family"] != "torchvision" or spec["model"] not in MODELS:
        raise SpecError(f"Unknown semantic segmentation model '{spec['model']}'.")
    if spec["start"] not in ("transfer", "finetune", "scratch"):
        raise SpecError(f"Unknown start '{spec['start']}'.")
    if spec["metric"] not in METRICS:
        raise SpecError(f"The selection metric must be one of: {', '.join(METRICS)}.")
    if not spec["classes"] or len(spec["classes"]) > 254:
        raise SpecError("Segmentation needs between 1 and 254 classes.")
    parts = spec["parts"]
    if not parts.get("train") or not parts.get("val"):
        raise SpecError("Training needs images in the train and validation parts.")
    if len(parts["train"]) < 2:
        raise SpecError("Training needs at least 2 training images (batch normalisation).")
    return dict(spec)


def class_id(label: Any, classes: list[str]) -> int:
    """Mask value of an object: its class index + 1 (0 is the background).

    Args:
        label: Class name, or index into ``classes``.
        classes: The project's classes (without the background).

    Returns:
        The mask value, from 1.

    Raises:
        SpecError: unknown class.
    """
    if isinstance(label, str):
        if label not in classes:
            raise SpecError(f"Unknown class '{label}'.")
        return classes.index(label) + 1
    index = int(label)
    if not 0 <= index < len(classes):
        raise SpecError(f"Class index {index} is out of range.")
    return index + 1


def _has_polygons(obj: dict[str, Any]) -> bool:
    """Whether ``obj`` has at least one ring with three or more points.

    Args:
        obj: Spec object with optional ``polygons``.

    Returns:
        True when it has a usable polygon.
    """
    return any(len(ring) >= 3 for ring in obj.get("polygons") or [])


def count_box_only(parts: dict[str, list[dict[str, Any]]]) -> int:
    """Objects that have a box but no polygon (they are drawn as filled rectangles).

    Args:
        parts: The spec parts: image items by part name.

    Returns:
        The number of such objects over all parts.
    """
    return sum(
        1
        for items in parts.values()
        for item in items
        for obj in item.get("objects") or []
        if not _has_polygons(obj) and obj.get("bbox")
    )


def rasterize_mask(objects: list[dict[str, Any]], width: int, height: int, classes: list[str]) -> Any:
    """Class mask (PIL ``L``, 0 = background) of the objects; later objects paint over earlier ones.

    Every ring of an object is filled with its class (rings are not treated as holes). An object
    without a usable polygon is drawn as its filled bounding box; one with neither is skipped.

    Args:
        objects: Spec objects (``label``, ``polygons`` and ``bbox``, relative coordinates).
        width: Image width in pixels.
        height: Image height in pixels.
        classes: The project's classes (without the background).

    Returns:
        The PIL ``L`` mask.

    Raises:
        SpecError: An object has an unknown class.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for obj in objects:
        value = class_id(obj["label"], classes)
        if _has_polygons(obj):
            for ring in obj["polygons"]:
                if len(ring) >= 3:
                    draw.polygon([(x * width, y * height) for x, y in ring], fill=value)
        elif obj.get("bbox"):
            x, y, w, h = obj["bbox"]
            x0, y0 = round(x * width), round(y * height)
            x1, y1 = max(x0, round((x + w) * width) - 1), max(y0, round((y + h) * height) - 1)
            draw.rectangle([x0, y0, x1, y1], fill=value)
    return mask


def resize_pair(image: Any, mask: Any, size: int, policy: str = options.LEGACY_RESIZE) -> tuple[Any, Any]:
    """Image (bilinear) and mask (nearest) turned into the ``size`` x ``size`` input by the resize policy.

    Letterbox bands are the background in the mask.

    Args:
        image: PIL image.
        mask: PIL class mask.
        size: Side of the square, in pixels.
        policy: The resize policy (one of ``options.RESIZE_POLICIES``).

    Returns:
        ``(image, mask)`` resized.
    """
    return (
        options.fit_image(image, size, policy),
        options.fit_image(mask, size, policy, Image.Resampling.NEAREST, 0),
    )


def _ratio(a: float, b: float) -> float:
    """``a / b``, or 0.0 when ``b`` is zero.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        The quotient, or 0.0.
    """
    return a / b if b else 0.0


def confusion_from_lists(truth: list[int], predicted: list[int], n: int) -> list[list[int]]:
    """``matrix[true][predicted]`` pixel counts over ``n`` values (0 = background).

    Args:
        truth: True mask value of each pixel.
        predicted: Predicted mask value of each pixel.
        n: Number of values (classes plus the background).

    Returns:
        The ``n`` by ``n`` matrix of counts.
    """
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(truth, predicted, strict=True):
        matrix[t][p] += 1
    return matrix


def class_scores(matrix: list[list[int]], index: int) -> tuple[float, float, int, int]:
    """IoU, Dice, true pixels and predicted pixels of one class.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``).
        index: Mask value of the class (0 is the background).

    Returns:
        ``(iou, dice, support, predicted)``.
    """
    tp = matrix[index][index]
    support = sum(matrix[index])
    predicted = sum(row[index] for row in matrix)
    return _ratio(tp, support + predicted - tp), _ratio(2 * tp, support + predicted), support, predicted


def mean_scores(matrix: list[list[int]]) -> tuple[float, float]:
    """Mean IoU and Dice over the object classes (not the background) present in truth or prediction.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``) with the background at index 0.

    Returns:
        ``(mean IoU, mean Dice)``.
    """
    ious, dices = [], []
    for i in range(1, len(matrix)):
        iou, dice, support, predicted = class_scores(matrix, i)
        if support or predicted:
            ious.append(iou)
            dices.append(dice)
    return _ratio(sum(ious), len(ious)), _ratio(sum(dices), len(dices))


def pixel_accuracy(matrix: list[list[int]]) -> float:
    """Share of correctly classified pixels (background included).

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``).

    Returns:
        The pixel accuracy, 0..1.
    """
    return _ratio(sum(matrix[i][i] for i in range(len(matrix))), sum(map(sum, matrix)))


def evaluation(matrix: list[list[int]], classes: list[str]) -> dict[str, Any]:
    """mIoU, Dice, pixel accuracy, per-class IoU / Dice and the confusion matrix.

    ``classes`` are the project classes; the matrix has the background as index 0 before them.
    The background is reported (``background_iou``, ``per_class``) but not in the means.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``) with the background at index 0.
        classes: The project's classes (without the background).

    Returns:
        A dict with ``miou``, ``dice``, ``pixel_accuracy``, ``background_iou``, ``per_class``,
        ``confusion_matrix`` and ``classes`` (the background first).
    """
    names = [BACKGROUND, *classes]
    miou, dice = mean_scores(matrix)
    per_class = {}
    for i, name in enumerate(names):
        iou, d, support, _ = class_scores(matrix, i)
        per_class[name] = {"iou": iou, "dice": d, "support": support}
    return {
        "miou": miou,
        "dice": dice,
        "pixel_accuracy": pixel_accuracy(matrix),
        "background_iou": per_class[BACKGROUND]["iou"],
        "per_class": per_class,
        "confusion_matrix": matrix,
        "classes": names,
    }


def split_folds_note(spec: dict[str, Any]) -> str:
    """One-line description of the data, for the log.

    Args:
        spec: The job spec (``parts``, ``classes`` and optionally ``fold``).

    Returns:
        The sentence with image counts per part, the fold and the number of classes.
    """
    sizes = ", ".join(f"{k} {len(v)}" for k, v in spec["parts"].items())
    fold = f", fold {spec['fold']}" if spec.get("fold") is not None else ""
    return f"Images: {sizes}{fold}. Classes: {len(spec['classes'])} + background."


class SegmentationDataset:
    """Map-style dataset of image + class mask pairs (works with ``torch.utils.data.DataLoader``)."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        classes: list[str],
        size: int,
        augmentations: list[dict[str, Any]] | None,
        resize: str = options.LEGACY_RESIZE,
    ) -> None:
        """Keep the items and the settings.

        Args:
            items: The images of one part (``path`` and ``objects`` each).
            classes: The project's classes (without the background).
            size: Side of the square the images are resized to.
            augmentations: Techniques applied to image and mask together; ``None`` only resizes
                (evaluation).
            resize: The resize policy (one of ``options.RESIZE_POLICIES``).
        """
        self.items = items
        self.classes = classes
        self.size = size
        self.resize = resize
        self.augmentations = augmentations  # None = evaluation (resize only)

    def __len__(self) -> int:
        """Number of images.

        Returns:
            The number of items in the part.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """The normalised image and its class mask at ``index``.

        Args:
            index: Position in the part.

        Returns:
            ``(image tensor 3 x size x size, int64 mask tensor size x size)``.
        """
        item = self.items[index]
        with Image.open(item["path"]) as opened:
            # labels are drawn on the photo as shown (EXIF rotation applied)
            image = (ImageOps.exif_transpose(opened) or opened).convert("RGB")
        mask = rasterize_mask(item.get("objects") or [], image.width, image.height, self.classes)
        if self.augmentations:
            image, mask = augment.augment_pair(image, mask, self.augmentations)
        image, mask = resize_pair(image, mask, self.size, self.resize)
        mean, std = numpy.array(MEAN, dtype="float32"), numpy.array(STD, dtype="float32")
        array = (numpy.asarray(image, dtype="float32") / 255.0 - mean) / std
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), torch.from_numpy(
            numpy.array(mask, dtype="int64")
        )


# ---------------------------------------------------------------------------
# Training (needs torch)


def seed_everything(seed: int, torch: Any) -> None:
    """Seed Python, numpy and torch.

    Args:
        seed: The seed.
        torch: The torch module.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(spec: dict[str, Any], pretrained: bool) -> Any:
    """The model with a new final layer for the classes + background (also in the aux classifier).

    Args:
        spec: The job spec (``model`` and ``classes``).
        pretrained: Start from pretrained weights (downloads them when needed) instead of random ones.

    Returns:
        The model.

    Raises:
        SpecError: unknown model.
    """
    model_id = spec["model"]
    builder = getattr(torchvision.models.segmentation, model_id, None)
    if builder is None or model_id not in MODELS:
        raise SpecError(f"torchvision has no segmentation model '{model_id}'.")
    n = len(spec["classes"]) + 1
    if pretrained:
        model = builder(weights="DEFAULT")
    else:
        model = builder(weights=None, weights_backbone=None)
    if model_id.startswith("lraspp"):
        head = model.classifier
        head.low_classifier = torch.nn.Conv2d(head.low_classifier.in_channels, n, 1)
        head.high_classifier = torch.nn.Conv2d(head.high_classifier.in_channels, n, 1)
    else:
        model.classifier[4] = torch.nn.Conv2d(model.classifier[4].in_channels, n, 1)
        if getattr(model, "aux_classifier", None) is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(model.aux_classifier[4].in_channels, n, 1)
    return model


def head_modules(model: Any) -> list[Any]:
    """The parts that are not the backbone: the classifier and the aux classifier when present.

    Args:
        model: The segmentation model.

    Returns:
        The head modules.
    """
    aux = getattr(model, "aux_classifier", None)
    return [model.classifier] + ([aux] if aux is not None else [])


def make_optimizer(spec: dict[str, Any], model: Any, torch: Any) -> Any:
    """Optimizer for the start mode: head only, backbone at a lower rate, or everything.

    Args:
        spec: The job spec (``start`` and ``params``).
        model: The network; in ``transfer`` mode its parameters are frozen except the heads.
        torch: The torch module.

    Returns:
        The optimizer (SGD, Adam or AdamW as in ``params.optimizer``).
    """
    p = spec["params"]
    lr, start = float(p["lr"]), spec["start"]
    head_params = [x for module in head_modules(model) for x in module.parameters()]
    if start == "transfer":
        for param in model.parameters():
            param.requires_grad = False
        for param in head_params:
            param.requires_grad = True
        groups = [{"params": head_params, "lr": lr}]
    elif start == "finetune":
        ids = {id(x) for x in head_params}
        backbone = [x for x in model.parameters() if id(x) not in ids]
        groups = [
            {"params": backbone, "lr": lr * float(p["backbone_lr_factor"])},
            {"params": head_params, "lr": lr},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr}]
    kwargs: dict[str, Any] = {"weight_decay": float(p["weight_decay"])}
    name = p["optimizer"]
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, **kwargs)
    if name == "adam":
        return torch.optim.Adam(groups, **kwargs)
    return torch.optim.AdamW(groups, **kwargs)


def segmentation_loss(output: Any, masks: Any, loss_fn: Any) -> Any:
    """Cross-entropy of the main output plus ``AUX_WEIGHT`` times that of the aux output, if any.

    Args:
        output: The model output dict (``out`` and optionally ``aux``).
        masks: Class masks (N x H x W).
        loss_fn: Loss function ``(logits, masks)``.

    Returns:
        The loss tensor.
    """
    loss = loss_fn(output["out"], masks)
    if "aux" in output:
        loss = loss + AUX_WEIGHT * loss_fn(output["aux"], masks)
    return loss


def run_epoch(
    model: Any,
    loader: Any,
    device: str,
    torch: Any,
    loss_fn: Any,
    n_classes: int,
    optimizer: Any = None,
    on_step: ProgressFn = None,
    schedule: Any = None,
    precision: Any = None,
) -> tuple[float, list[list[int]]]:
    """One pass over a loader. Trains when ``optimizer`` is given.

    Args:
        model: The network.
        loader: Data loader of ``(images, masks)`` batches.
        device: Torch device name.
        torch: The torch module.
        loss_fn: Loss function ``(logits, masks)``.
        n_classes: Number of mask values (classes plus the background).
        optimizer: The optimizer; ``None`` evaluates without gradients.
        on_step: Called with ``(step, steps)`` after each batch (optional).
        schedule: The learning rate schedule (``options.LrSchedule``), advanced before each training step.
        precision: Mixed precision of the training steps (``options.Precision``); float32 by default.

    Returns:
        ``(mean loss per image, confusion matrix[true][predicted] of pixel counts)``.

    Raises:
        RuntimeError: A training loss is not finite.
    """
    training = optimizer is not None
    amp = precision or options.Precision(False)
    total, count = 0.0, 0
    matrix = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    with torch.set_grad_enabled(training):
        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(device), masks.to(device)
            with amp.autocast(training):
                output = model(images)
                loss = segmentation_loss(output, masks, loss_fn)
            if training:
                if schedule is not None:
                    schedule.advance()
                amp.step(loss, optimizer)
            total += float(loss) * len(masks)
            count += len(masks)
            predicted = output["out"].argmax(1)
            matrix += torch.bincount((masks * n_classes + predicted).flatten(), minlength=n_classes**2)
            if on_step:
                on_step(step, len(loader))
    return _ratio(total, count), matrix.reshape(n_classes, n_classes).tolist()


def batch_loss(model: Any, batch: Any, device: str, loss_fn: Any) -> Any:
    """Loss of one ``(images, masks)`` batch (used to find the batch size that fits the GPU).

    Args:
        model: The network.
        batch: ``(images, masks)`` tensors.
        device: Torch device name.
        loss_fn: Loss function ``(logits, masks)``.

    Returns:
        The loss tensor.
    """
    images, masks = batch
    return segmentation_loss(model(images.to(device)), masks.to(device), loss_fn)


def make_loader(
    items: list[dict[str, Any]],
    augmentations: list[dict[str, Any]] | None,
    shuffle: bool,
    classes: list[str],
    size: int,
    resize: str,
    batch_size: int,
    workers: int,
    generator: Any,
) -> Any:
    """A data loader over ``items``.

    Args:
        items: the images of one part.
        augmentations: the techniques (``None`` reads without augmentation: evaluation).
        shuffle: shuffle with ``generator`` (the training part).
        classes: the project's classes (without the background).
        size: side of the square the images are resized to.
        resize: the resize policy (one of ``options.RESIZE_POLICIES``).
        batch_size: images per batch.
        workers: loader processes (0 loads in the main process).
        generator: the seeded random generator used when shuffling.

    Returns:
        A ``DataLoader``.
    """
    return DataLoader(
        SegmentationDataset(items, classes, size, augmentations, resize),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
        drop_last=shuffle and len(items) % batch_size == 1,  # batch norm needs 2+ samples
    )


def report_step(epoch: int, epochs: int, step: int, steps: int) -> None:
    """Tell the runner about the progress of a training epoch (about twenty times per epoch).

    Args:
        epoch: the epoch (from 1).
        epochs: how many epochs the run has.
        step: the batch just done (from 1).
        steps: batches in the epoch.
    """
    if step == steps or step % max(1, steps // 20) == 0:
        emit("progress", phase="training", epoch=epoch, epochs=epochs, step=step, steps=steps)


def save_checkpoint(path: Path, model: Any, meta: dict[str, Any], epoch: int) -> None:
    """Write the weights of ``model`` with what is needed to rebuild it.

    Args:
        path: the ``.pt`` file.
        model: the network.
        meta: classes, model name, family, image size... saved next to the weights.
        epoch: the epoch the weights come from.
    """
    torch.save({"state_dict": model.state_dict(), **meta, "epoch": epoch}, path)


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """Train, keep the best epoch by the selection metric, evaluate the test part.

    Writes ``best.pt``, ``last.pt`` and ``results.json`` to ``output_dir``.

    Args:
        spec: The checked job spec.

    Returns:
        The results: ``metric``, ``best_epoch``, ``epochs_run``, ``stopped_early``, ``val`` and
        ``test`` (``None`` without a test part).

    Raises:
        SpecError: An unknown model or class.
        options.DeviceError: The device the run asked for is not available.
        options.CodedFailure: Nothing to resume from, or the training changed since it stopped.
        options.OutOfMemory: Not even a batch of 2 images fits in the GPU memory.
        augment.AugmentError: A technique the worker cannot apply.
    """
    p, classes, metric = spec["params"], spec["classes"], spec["metric"]
    n_classes = len(classes) + 1
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(int(spec.get("seed", p["seed"])), torch)
    device = options.pick_device(torch, str(spec.get("device") or "auto"))
    emit("log", message=f"Device: {device}. {split_folds_note(spec)}")
    use_amp, amp_note = options.amp_enabled(str(p["amp"]), device)
    if amp_note:
        emit("warning", message=amp_note, code="amp_needs_cuda", params={})
    precision = options.Precision(use_amp)
    emit("setup", device=device, amp=use_amp)
    boxes = count_box_only(spec["parts"])
    if boxes:
        emit(
            "warning",
            message=f"{boxes} object(s) have a box but no polygon: they are drawn as filled rectangles.",
            code="box_only_objects",
            params={"n": boxes},
        )
    size, batch, policy = int(p["image_size"]), int(p["batch_size"]), str(p["resize"])
    generator = torch.Generator().manual_seed(int(p["seed"]))
    workers = 4 if device.startswith("cuda") else 0
    parts = spec["parts"]
    # a resumed training goes on with the batch of the interrupted one (no new probe of the GPU)
    resumed = (
        options.load_resume(out / "last.pt", str(spec.get("digest") or "")) if spec.get("resume") else None
    )
    if resumed is not None:
        batch = int(resumed["resume"]["batch"])

    loader = partial(
        make_loader,
        classes=classes,
        size=size,
        resize=policy,
        batch_size=batch,
        workers=workers,
        generator=generator,
    )
    techniques = spec["augmentations"]
    augment.check_objects(techniques)
    emit("log", message=f"Augmentation: {augment.describe(techniques)}. Resize: {policy}.")
    train_loader = loader(parts["train"], techniques, True)
    val_loader = loader(parts["val"], None, False)
    model = build_model(spec, pretrained=spec["start"] != "scratch")
    model.to(device)
    optimizer = make_optimizer(spec, model, torch)
    loss_fn = torch.nn.CrossEntropyLoss()
    if device.startswith("cuda") and resumed is None:  # the largest batch up to the one asked for that fits
        try_batch = partial(
            options.probe,
            model=model,
            loader_of=partial(loader, parts["train"], None, False),
            loss_of=partial(batch_loss, device=device, loss_fn=loss_fn),
            precision=precision,
            optimizer=optimizer,
        )
        batch = options.fit_batch(try_batch, batch, emit, 2, torch.cuda.empty_cache)
        seed_everything(int(spec.get("seed", p["seed"])), torch)  # the probe drew random numbers
        loader = partial(loader, batch_size=batch)
        train_loader, val_loader = loader(parts["train"], techniques, True), loader(parts["val"], None, False)
    emit("setup", batch_size=batch)

    epochs, patience = int(p["epochs"]), int(p["patience"])
    schedule = options.LrSchedule(
        optimizer, str(p["scheduler"]), epochs, len(train_loader), int(p["warmup_epochs"])
    )
    best: dict[str, Any] = {}
    since_best, ran = 0, 0
    if resumed is not None:
        model.load_state_dict(resumed["state_dict"])
        loop = options.restore(resumed["resume"], optimizer, schedule, precision, generator)
        best, since_best, ran = loop["best"], int(loop["since_best"]), int(loop["epoch"])
        emit("resumed", epoch=ran)
        emit("log", message=f"Resumed after epoch {ran} of {int(p['epochs'])}.")

    meta = {
        "classes": [BACKGROUND, *classes],
        "model": spec["model"],
        "family": spec["family"],
        "image_size": size,
        "resize": policy,
        "mean": MEAN,
        "std": STD,
    }

    for epoch in range(ran + 1, epochs + 1):
        started = time.monotonic()

        model.train()
        if spec["start"] == "transfer":
            model.eval()  # frozen backbone: no batch-norm statistics changes
            for module in head_modules(model):
                module.train()
        on_step = partial(report_step, epoch, epochs)
        train_loss, _ = run_epoch(
            model, train_loader, device, torch, loss_fn, n_classes, optimizer, on_step, schedule, precision
        )
        model.eval()
        val_loss, matrix = run_epoch(model, val_loader, device, torch, loss_fn, n_classes)
        scores = evaluation(matrix, classes)
        schedule.end_epoch(scores[metric])
        ran = epoch
        improved = not best or scores[metric] > best["scores"][metric]
        if improved:
            best = {"epoch": epoch, "scores": scores, "val_loss": val_loss}
            since_best = 0
            save_checkpoint(out / "best.pt", model, meta, epoch)
        else:
            since_best += 1
        resume = options.resume_state(
            optimizer,
            schedule,
            precision,
            generator,
            epoch=epoch,
            best=best,
            since_best=since_best,
            batch=batch,
            digest=str(spec.get("digest") or ""),
        )
        save_checkpoint(out / "last.pt", model, {**meta, "resume": resume}, epoch)
        emit(
            "epoch",
            epoch=epoch,
            checkpoint=True,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=scores["miou"],
            val_dice=scores["dice"],
            val_pixel_accuracy=scores["pixel_accuracy"],
            val_background_iou=scores["background_iou"],
            lr=schedule.lr(),
            seconds=round(time.monotonic() - started, 2),
            best=improved,
        )
        if patience and since_best >= patience:
            emit("log", message=f"Early stopping: no improvement for {patience} epoch(s).")
            break

    results: dict[str, Any] = {
        "metric": metric,
        "best_epoch": best["epoch"],
        "epochs_run": ran,
        "stopped_early": ran < epochs,
        "val": {"loss": best["val_loss"], **best["scores"]},
        "test": None,
    }
    if parts.get("test"):
        emit("progress", phase="evaluating", epoch=ran, epochs=epochs)
        state = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        test_loss, matrix = run_epoch(
            model, loader(parts["test"], None, False), device, torch, loss_fn, n_classes
        )
        results["test"] = {"loss": test_loss, **evaluation(matrix, classes)}
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return results


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command line; ``argv[1]`` is the path of the spec JSON file.

    Returns:
        Process exit code: 0 on success, 1 after an ``error`` event.
    """
    try:
        spec = load_spec(argv[1])
        results = train(spec)
    except Exception as exc:  # noqa: BLE001 - reported to the runner as an event
        emit("error", message=str(exc) or type(exc).__name__, **options.coded(exc))
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=results)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
