"""Image classification worker (docs/TRAINING.md): ``python classification.py spec.json``.

Standalone: it imports only the standard library and, inside the functions that need them, torch,
torchvision, timm, PIL and numpy, so it runs in the training environment (which has no
ActiveLabel) and its pure helpers (metrics, spec parsing) can be tested without torch.

Output: one JSON object per line on stdout (``log``, ``warning``, ``setup``, ``progress``, ``epoch``,
``done``, ``error``).
Files written to ``output_dir``: ``best.pt``, ``last.pt``, ``results.json``.
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
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from activelabel.training.workers import augment, options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling files
    import augment  # type: ignore[no-redef]
    import options  # type: ignore[no-redef]

try:
    import timm
except ImportError:  # timm is optional: only the timm families need it
    timm = None

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
METRICS = ("accuracy", "macro_f1")
#: Head (the layer that is replaced and trained alone in transfer learning) of torchvision models.
TV_HEADS = {
    "resnet": "fc",
    "convnext": "classifier.2",
    "vit": "heads.head",
    "swin": "head",
    "mobilenet": "classifier.3",
}
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
        SpecError: missing keys, an unsupported task, no images or a fold ``"all"``.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task", "model", "family", "start", "params", "classes", "parts", "output_dir", "metric"):
        if key not in spec:
            raise SpecError(f"spec.json has no '{key}'.")
    if spec["task"] != "classification":
        raise SpecError(f"This worker trains classification, not {spec['task']}.")
    if spec.get("fold") == "all":
        raise SpecError("Training all folds comes later: choose one fold.")
    if spec["family"] not in ("timm", "torchvision"):
        raise SpecError(f"Unknown model family '{spec['family']}'.")
    if spec["start"] not in ("transfer", "finetune", "scratch"):
        raise SpecError(f"Unknown start '{spec['start']}'.")
    if spec["metric"] not in METRICS:
        raise SpecError(f"The selection metric must be one of: {', '.join(METRICS)}.")
    parts = spec["parts"]
    if not parts.get("train") or not parts.get("val"):
        raise SpecError("Training needs images in the train and validation parts.")
    return dict(spec)


def head_path(model_id: str) -> str:
    """Dotted path of the classifier layer of a torchvision model.

    Args:
        model_id: The torchvision model name (e.g. ``resnet50``).

    Returns:
        The dotted attribute path of the layer.

    Raises:
        SpecError: unknown model.
    """
    for prefix, path in TV_HEADS.items():
        if model_id.startswith(prefix):
            return path
    raise SpecError(f"Unknown torchvision classification model '{model_id}'.")


def confusion_matrix(truth: list[int], predicted: list[int], n: int) -> list[list[int]]:
    """``matrix[true][predicted]`` counts.

    Args:
        truth: True class index of each image.
        predicted: Predicted class index of each image.
        n: Number of classes.

    Returns:
        The ``n`` by ``n`` matrix of counts.
    """
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(truth, predicted, strict=True):
        matrix[t][p] += 1
    return matrix


def _ratio(a: float, b: float) -> float:
    """``a / b``, or 0.0 when ``b`` is zero.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        The quotient, or 0.0.
    """
    return a / b if b else 0.0


def per_class(matrix: list[list[int]], classes: list[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support of every class.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``).
        classes: Class names, in the order of the matrix.

    Returns:
        ``{class: {"precision", "recall", "f1", "support"}}``.
    """
    out = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision, recall = _ratio(tp, predicted), _ratio(tp, support)
        f1 = _ratio(2 * precision * recall, precision + recall)
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def accuracy(matrix: list[list[int]]) -> float:
    """Share of correct predictions.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``).

    Returns:
        The accuracy, 0..1.
    """
    total = sum(map(sum, matrix))
    return _ratio(sum(matrix[i][i] for i in range(len(matrix))), total)


def macro_f1(matrix: list[list[int]]) -> float:
    """Mean F1 over the classes that appear in the truth or in the predictions.

    Args:
        matrix: Confusion matrix (``matrix[true][predicted]``).

    Returns:
        The macro F1, 0..1.
    """
    n = len(matrix)
    scores = []
    for i in range(n):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        if support == 0 and predicted == 0:
            continue
        tp = matrix[i][i]
        precision, recall = _ratio(tp, predicted), _ratio(tp, support)
        scores.append(_ratio(2 * precision * recall, precision + recall))
    return _ratio(sum(scores), len(scores))


def evaluation(truth: list[int], predicted: list[int], classes: list[str]) -> dict[str, Any]:
    """Accuracy, macro F1, per-class metrics and the confusion matrix.

    Args:
        truth: True class index of each image.
        predicted: Predicted class index of each image.
        classes: Class names.

    Returns:
        A dict with ``accuracy``, ``macro_f1``, ``per_class``, ``confusion_matrix``, ``classes`` and
        ``images``.
    """
    matrix = confusion_matrix(truth, predicted, len(classes))
    return {
        "accuracy": accuracy(matrix),
        "macro_f1": macro_f1(matrix),
        "per_class": per_class(matrix, classes),
        "confusion_matrix": matrix,
        "classes": list(classes),
        "images": len(truth),
    }


def split_folds_note(spec: dict[str, Any]) -> str:
    """One-line description of the data, for the log.

    Args:
        spec: The job spec (``parts``, ``classes`` and optionally ``fold``).

    Returns:
        The sentence with image counts per part, the fold and the number of classes.
    """
    parts = spec["parts"]
    sizes = ", ".join(f"{k} {len(v)}" for k, v in parts.items())
    fold = f", fold {spec['fold']}" if spec.get("fold") is not None else ""
    return f"Images: {sizes}{fold}. Classes: {len(spec['classes'])}."


def label_counts(items: list[dict[str, Any]], n_classes: int) -> list[int]:
    """Images of each class in a part.

    Args:
        items: The images of the part (``label`` each, the class index).
        n_classes: Number of classes.

    Returns:
        One count per class, in class order.
    """
    counts = [0] * n_classes
    for item in items:
        counts[int(item["label"])] += 1
    return counts


class ImageDataset:
    """Map-style dataset of ``{"path", "label"}`` items (works with ``torch.utils.data.DataLoader``)."""

    def __init__(self, items: list[dict[str, Any]], transform: Any) -> None:
        """Keep the items and the transform.

        Args:
            items: The images of one part (``path`` and ``label`` each).
            transform: The transforms applied to every image.
        """
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        """Number of images.

        Returns:
            The number of items in the part.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """The transformed image and class index at ``index``.

        Args:
            index: Position in the part.

        Returns:
            ``(image tensor, class index)``.
        """
        item = self.items[index]
        with Image.open(item["path"]) as image:
            # labels are drawn on the photo as shown (EXIF rotation applied)
            upright = ImageOps.exif_transpose(image) or image
            return self.transform(upright.convert("RGB")), int(item["label"])


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


def eval_transform(size: int, policy: str) -> Any:
    """Transform of validation, test and pre-annotated images: the resize policy, then normalisation.

    Args:
        size: Side of the square input.
        policy: The resize policy (one of ``options.RESIZE_POLICIES``).

    Returns:
        The composed transform (PIL image in, normalised tensor out).
    """
    return transforms.Compose(
        [options.Fit(size, policy), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def _get(module: Any, path: str) -> Any:
    """The attribute (or sequence item, for numeric parts) of ``module`` at a dotted ``path``.

    Args:
        module: The object to start from.
        path: Dotted path such as ``classifier.2``.

    Returns:
        The object at the path.
    """
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set(module: Any, path: str, value: Any) -> None:
    """Replace the attribute (or sequence item) of ``module`` at a dotted ``path`` with ``value``.

    Args:
        module: The object to start from.
        path: Dotted path such as ``classifier.2``.
        value: The new object.
    """
    *parents, last = path.split(".")
    holder = _get(module, ".".join(parents)) if parents else module
    if last.isdigit():
        holder[int(last)] = value
    else:
        setattr(holder, last, value)


def build_model(spec: dict[str, Any], pretrained: bool) -> tuple[Any, Any]:
    """The model with a new classifier for the classes, and that classifier layer.

    Args:
        spec: The job spec (``classes``, ``model``, ``family`` and ``params.image_size``).
        pretrained: Load pretrained weights (downloads them when needed).

    Returns:
        ``(model, classifier layer)``.

    Raises:
        SpecError: a size the model cannot take, or an unknown torchvision model.
        ImportError: a timm model was asked for and timm is not installed.
    """
    n = len(spec["classes"])
    model_id, size = spec["model"], int(spec["params"]["image_size"])
    if spec["family"] == "timm":
        if timm is None:
            raise ImportError("timm is not installed in the training environment.")
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=n)
        return model, model.get_classifier()

    if model_id.startswith("vit") and size != 224:
        raise SpecError("ViT-B/16 only takes 224 pixel images.")
    builder = getattr(torchvision.models, model_id, None)
    if builder is None:
        raise SpecError(f"torchvision has no model '{model_id}'.")
    model = builder(weights="DEFAULT" if pretrained else None)
    path = head_path(model_id)
    old = _get(model, path)
    _set(model, path, torch.nn.Linear(old.in_features, n))
    return model, _get(model, path)


def make_optimizer(spec: dict[str, Any], model: Any, head: Any, torch: Any) -> Any:
    """Optimizer for the start mode: head only, backbone at a lower rate, or everything.

    Args:
        spec: The job spec (``start`` and ``params``).
        model: The network; in ``transfer`` mode its parameters are frozen except the head.
        head: The classifier layer of ``model``.
        torch: The torch module.

    Returns:
        The optimizer (SGD, Adam or AdamW as in ``params.optimizer``).
    """
    p = spec["params"]
    lr, start = float(p["lr"]), spec["start"]
    head_params = list(head.parameters())
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


def run_epoch(
    model: Any,
    loader: Any,
    device: str,
    torch: Any,
    loss_fn: Any,
    optimizer: Any = None,
    on_step: ProgressFn = None,
    mixes: list[tuple[str, float]] | None = None,
    schedule: Any = None,
    precision: Any = None,
) -> tuple[float, list[int], list[int]]:
    """One pass over a loader. Trains when ``optimizer`` is given.

    Args:
        model: The network.
        loader: Data loader of ``(images, labels)`` batches.
        device: Torch device name.
        torch: The torch module.
        loss_fn: Loss function ``(output, labels)``.
        optimizer: The optimizer; ``None`` evaluates without gradients.
        on_step: Called with ``(step, steps)`` after each batch (optional).
        mixes: Mixup/cutmix ``(name, alpha)`` pairs for training batches (``augment.mix_settings``).
        schedule: The learning rate schedule (``options.LrSchedule``), advanced before each training step.
        precision: Mixed precision of the training steps (``options.Precision``); float32 by default.

    Returns:
        ``(mean loss per image, true class indices, predicted class indices)``.

    Raises:
        RuntimeError: A training loss is not finite.
    """
    training = optimizer is not None
    amp = precision or options.Precision(False)
    total, count = 0.0, 0
    truth: list[int] = []
    predicted: list[int] = []
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            with amp.autocast(training):
                if training and mixes:
                    images, order, lam = augment.mix_batch(images, mixes)
                    output = model(images)
                    loss = lam * loss_fn(output, labels) + (1.0 - lam) * loss_fn(output, labels[order])
                else:
                    output = model(images)
                    loss = loss_fn(output, labels)
            if training:
                if schedule is not None:
                    schedule.advance()
                amp.step(loss, optimizer)
            total += float(loss) * len(labels)
            count += len(labels)
            truth += labels.tolist()
            predicted += output.argmax(1).tolist()
            if on_step:
                on_step(step, len(loader))
    return _ratio(total, count), truth, predicted


def batch_loss(model: Any, batch: Any, device: str, loss_fn: Any) -> Any:
    """Loss of one ``(images, labels)`` batch (used to find the batch size that fits the GPU).

    Args:
        model: The network.
        batch: ``(images, labels)`` tensors.
        device: Torch device name.
        loss_fn: Loss function ``(output, labels)``.

    Returns:
        The loss tensor.
    """
    images, labels = batch
    return loss_fn(model(images.to(device)), labels.to(device))


def make_loader(
    items: list[dict[str, Any]], transform: Any, shuffle: bool, batch_size: int, workers: int, generator: Any
) -> Any:
    """A data loader over ``items``.

    Args:
        items: the images of one part (``path`` and class index each).
        transform: the transforms applied to every image.
        shuffle: shuffle with ``generator`` (the training part).
        batch_size: images per batch.
        workers: loader processes (0 loads in the main process).
        generator: the seeded random generator used when shuffling.

    Returns:
        A ``DataLoader``.
    """
    return DataLoader(
        ImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
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
        SpecError: A model or size the worker cannot use.
        options.DeviceError: The device the run asked for is not available.
        options.CodedFailure: Nothing to resume from, or the training changed since it stopped.
        options.OutOfMemory: Not even a batch of 1 image fits in the GPU memory.
        augment.AugmentError: A technique the worker cannot apply.
        ImportError: A timm model was asked for and timm is not installed.
    """
    p, classes, metric = spec["params"], spec["classes"], spec["metric"]
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
    size, batch, policy = int(p["image_size"]), int(p["batch_size"]), str(p["resize"])

    techniques = spec["augmentations"]
    emit("log", message=f"Augmentation: {augment.describe(techniques)}. Resize: {policy}.")
    train_tf = augment.classification_transform(techniques, size, MEAN, STD, options.Fit(size, policy))
    mixes = augment.mix_settings(techniques)
    eval_tf = eval_transform(size, policy)
    generator = torch.Generator().manual_seed(int(p["seed"]))
    workers = 4 if device.startswith("cuda") else 0
    parts = spec["parts"]
    # a resumed training goes on with the batch of the interrupted one (no new probe of the GPU)
    resumed = (
        options.load_resume(out / "last.pt", str(spec.get("digest") or "")) if spec.get("resume") else None
    )
    if resumed is not None:
        batch = int(resumed["resume"]["batch"])

    loader = partial(make_loader, batch_size=batch, workers=workers, generator=generator)
    train_loader, val_loader = loader(parts["train"], train_tf, True), loader(parts["val"], eval_tf, False)
    model, head = build_model(spec, pretrained=spec["start"] != "scratch")
    model.to(device)
    optimizer = make_optimizer(spec, model, head, torch)
    counts = label_counts(parts["train"], len(classes))
    weights = options.class_weights(counts, str(p.get("class_weights", "none")))
    if weights is not None:
        emit("log", message=f"Class weights (inverse frequency): {[round(w, 2) for w in weights]}.")
    loss_fn = options.build_loss(
        str(p.get("loss", "cross_entropy")),
        weights,
        float(p.get("label_smoothing", 0.0)),
        float(p.get("focal_gamma", 2.0)),
    ).to(device)
    if device.startswith("cuda") and resumed is None:  # the largest batch up to the one asked for that fits
        try_batch = partial(
            options.probe,
            model=model,
            loader_of=partial(loader, parts["train"], eval_tf, False),
            loss_of=partial(batch_loss, device=device, loss_fn=loss_fn),
            precision=precision,
            optimizer=optimizer,
        )
        batch = options.fit_batch(try_batch, batch, emit, 1, torch.cuda.empty_cache)
        seed_everything(int(spec.get("seed", p["seed"])), torch)  # the probe drew random numbers
        loader = partial(loader, batch_size=batch)
        train_loader, val_loader = (
            loader(parts["train"], train_tf, True),
            loader(parts["val"], eval_tf, False),
        )
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
        "classes": classes,
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
            model.eval()  # frozen backbone: no batch-norm statistics or dropout changes
            head.train()
        on_step = partial(report_step, epoch, epochs)
        train_loss, _, _ = run_epoch(
            model, train_loader, device, torch, loss_fn, optimizer, on_step, mixes, schedule, precision
        )
        model.eval()
        val_loss, truth, predicted = run_epoch(model, val_loader, device, torch, loss_fn)
        scores = evaluation(truth, predicted, classes)
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
            val_accuracy=scores["accuracy"],
            val_macro_f1=scores["macro_f1"],
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
        "val": {
            "loss": best["val_loss"],
            **{k: best["scores"][k] for k in ("accuracy", "macro_f1", "per_class", "confusion_matrix")},
            "classes": classes,
        },
        "test": None,
    }
    if parts.get("test"):
        emit("progress", phase="evaluating", epoch=ran, epochs=epochs)
        state = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        test_loss, truth, predicted = run_epoch(
            model, loader(parts["test"], eval_tf, False), device, torch, loss_fn
        )
        results["test"] = {"loss": test_loss, **evaluation(truth, predicted, classes)}
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
