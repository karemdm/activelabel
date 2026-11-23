"""Image classification worker (docs/TRAINING.md): ``python classification.py spec.json``.

Standalone: it imports only the standard library and, inside the functions that need them, torch,
torchvision, timm, PIL and numpy, so it runs in the training environment (which has no
ActiveLabel) and its pure helpers (metrics, spec parsing) can be tested without torch.

Output: one JSON object per line on stdout (``log``, ``progress``, ``epoch``, ``done``, ``error``).
Files written to ``output_dir``: ``best.pt``, ``last.pt``, ``results.json``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
    """Print one event line."""
    print(json.dumps({"event": event, **fields}), flush=True)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read and check ``spec.json``.

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


def augmentation_plan(preset: str, size: int) -> list[tuple[str, dict[str, Any]]]:
    """Training transforms of a preset as ``(torchvision.transforms name, kwargs)`` pairs.

    Raises:
        SpecError: unknown preset.
    """
    if preset == "none":
        return [("Resize", {"size": (size, size)})]
    crop = {"light": 0.8, "standard": 0.6, "strong": 0.4}.get(preset)
    if crop is None:
        raise SpecError(f"Unknown augmentation preset '{preset}'.")
    plan: list[tuple[str, dict[str, Any]]] = [
        ("RandomResizedCrop", {"size": (size, size), "scale": (crop, 1.0)}),
        ("RandomHorizontalFlip", {}),
    ]
    if preset == "standard":
        plan.append(("ColorJitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}))
    if preset == "strong":
        plan.append(("RandomRotation", {"degrees": 15}))
        plan.append(("TrivialAugmentWide", {}))
        plan.append(("ColorJitter", {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}))
    return plan


def eval_plan(size: int) -> list[tuple[str, dict[str, Any]]]:
    """Transforms of validation and test images."""
    return [("Resize", {"size": (size, size)})]


def head_path(model_id: str) -> str:
    """Dotted path of the classifier layer of a torchvision model.

    Raises:
        SpecError: unknown model.
    """
    for prefix, path in TV_HEADS.items():
        if model_id.startswith(prefix):
            return path
    raise SpecError(f"Unknown torchvision classification model '{model_id}'.")


def confusion_matrix(truth: list[int], predicted: list[int], n: int) -> list[list[int]]:
    """``matrix[true][predicted]`` counts."""
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(truth, predicted, strict=True):
        matrix[t][p] += 1
    return matrix


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def per_class(matrix: list[list[int]], classes: list[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support of every class."""
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
    """Share of correct predictions."""
    total = sum(map(sum, matrix))
    return _ratio(sum(matrix[i][i] for i in range(len(matrix))), total)


def macro_f1(matrix: list[list[int]]) -> float:
    """Mean F1 over the classes that appear in the truth or in the predictions."""
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
    """Accuracy, macro F1, per-class metrics and the confusion matrix."""
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
    """One-line description of the data, for the log."""
    parts = spec["parts"]
    sizes = ", ".join(f"{k} {len(v)}" for k, v in parts.items())
    fold = f", fold {spec['fold']}" if spec.get("fold") is not None else ""
    return f"Images: {sizes}{fold}. Classes: {len(spec['classes'])}."


class ImageDataset:
    """Map-style dataset of ``{"path", "label"}`` items (works with ``torch.utils.data.DataLoader``)."""

    def __init__(self, items: list[dict[str, Any]], transform: Any) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        item = self.items[index]
        with Image.open(item["path"]) as image:
            return self.transform(image.convert("RGB")), int(item["label"])


# ---------------------------------------------------------------------------
# Training (needs torch)


def pick_device(torch: Any) -> str:
    """``cuda`` when there is a GPU, ``mps`` on Apple silicon, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, torch: Any) -> None:
    """Seed Python, numpy and torch."""
    import numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(plan: list[tuple[str, dict[str, Any]]]) -> Any:
    """Compose the planned transforms, then tensor conversion and normalisation."""
    from torchvision import transforms

    steps = [getattr(transforms, name)(**kwargs) for name, kwargs in plan]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def _get(module: Any, path: str) -> Any:
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set(module: Any, path: str, value: Any) -> None:
    *parents, last = path.split(".")
    holder = _get(module, ".".join(parents)) if parents else module
    if last.isdigit():
        holder[int(last)] = value
    else:
        setattr(holder, last, value)


def build_model(spec: dict[str, Any], pretrained: bool) -> tuple[Any, Any]:
    """The model with a new classifier for the classes, and that classifier layer.

    Raises:
        SpecError: a size the model cannot take.
    """
    import torch

    n = len(spec["classes"])
    model_id, size = spec["model"], int(spec["params"]["image_size"])
    if spec["family"] == "timm":
        import timm

        model = timm.create_model(model_id, pretrained=pretrained, num_classes=n)
        return model, model.get_classifier()
    import torchvision

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
    """Optimizer for the start mode: head only, backbone at a lower rate, or everything."""
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


def make_scheduler(spec: dict[str, Any], optimizer: Any, torch: Any) -> Any:
    """Learning rate schedule, or ``None``."""
    name, epochs = spec["params"]["scheduler"], int(spec["params"]["epochs"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


def run_epoch(
    model: Any,
    loader: Any,
    device: str,
    torch: Any,
    loss_fn: Any,
    optimizer: Any = None,
    on_step: ProgressFn = None,
) -> tuple[float, list[int], list[int]]:
    """One pass over a loader. Trains when ``optimizer`` is given. Returns loss, truth, predictions."""
    training = optimizer is not None
    total, count = 0.0, 0
    truth: list[int] = []
    predicted: list[int] = []
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss) * len(labels)
            count += len(labels)
            truth += labels.tolist()
            predicted += output.argmax(1).tolist()
            if on_step:
                on_step(step, len(loader))
    return _ratio(total, count), truth, predicted


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """Train, keep the best epoch by the selection metric, evaluate the test part. Returns the results."""
    import torch
    from torch.utils.data import DataLoader

    p, classes, metric = spec["params"], spec["classes"], spec["metric"]
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(int(spec.get("seed", p["seed"])), torch)
    device = pick_device(torch)
    emit("log", message=f"Device: {device}. {split_folds_note(spec)}")
    size, batch = int(p["image_size"]), int(p["batch_size"])

    train_tf = build_transforms(augmentation_plan(p["augmentation"], size))
    eval_tf = build_transforms(eval_plan(size))
    generator = torch.Generator().manual_seed(int(p["seed"]))
    workers = 4 if device == "cuda" else 0
    parts = spec["parts"]

    def loader(name: str, tf: Any, shuffle: bool) -> Any:
        return DataLoader(
            ImageDataset(parts[name], tf),
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            generator=generator if shuffle else None,
        )

    train_loader, val_loader = loader("train", train_tf, True), loader("val", eval_tf, False)
    model, head = build_model(spec, pretrained=spec["start"] != "scratch")
    model.to(device)
    optimizer = make_optimizer(spec, model, head, torch)
    scheduler = make_scheduler(spec, optimizer, torch)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(p.get("label_smoothing", 0.0)))

    epochs, patience = int(p["epochs"]), int(p["patience"])
    best: dict[str, Any] = {}
    since_best, ran = 0, 0

    def checkpoint(file: str, epoch: int) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "model": spec["model"],
                "family": spec["family"],
                "image_size": size,
                "mean": MEAN,
                "std": STD,
                "epoch": epoch,
            },
            out / file,
        )

    for epoch in range(1, epochs + 1):
        started = time.monotonic()

        def on_step(step: int, steps: int, epoch: int = epoch) -> None:
            if step == steps or step % max(1, steps // 20) == 0:
                emit("progress", phase="training", epoch=epoch, epochs=epochs, step=step, steps=steps)

        model.train()
        if spec["start"] == "transfer":
            model.eval()  # frozen backbone: no batch-norm statistics or dropout changes
            head.train()
        train_loss, _, _ = run_epoch(model, train_loader, device, torch, loss_fn, optimizer, on_step)
        model.eval()
        val_loss, truth, predicted = run_epoch(model, val_loader, device, torch, loss_fn)
        scores = evaluation(truth, predicted, classes)
        if scheduler is not None:
            scheduler.step()
        ran = epoch
        improved = not best or scores[metric] > best["scores"][metric]
        if improved:
            best = {"epoch": epoch, "scores": scores, "val_loss": val_loss}
            since_best = 0
            checkpoint("best.pt", epoch)
        else:
            since_best += 1
        checkpoint("last.pt", epoch)
        emit(
            "epoch",
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=scores["accuracy"],
            val_macro_f1=scores["macro_f1"],
            lr=optimizer.param_groups[-1]["lr"],
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
        test_loss, truth, predicted = run_epoch(model, loader("test", eval_tf, False), device, torch, loss_fn)
        results["test"] = {"loss": test_loss, **evaluation(truth, predicted, classes)}
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return results


def main(argv: list[str]) -> int:
    """Entry point."""
    try:
        spec = load_spec(argv[1])
        results = train(spec)
    except Exception as exc:  # noqa: BLE001 - reported to the runner as an event
        emit("error", message=str(exc) or type(exc).__name__)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=results)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
