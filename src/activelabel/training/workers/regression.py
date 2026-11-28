"""Image regression worker (#619, docs/TRAINING.md): ``python regression.py spec.json``.

A regression model predicts one number per target of the project for each image (an age, a count, a
grade). It is a classification network (torchvision or timm, built by ``classification.build_model``) whose
last layer gives one output per target instead of one per class. The targets are normalised with the mean
and standard deviation of the training part (kept in the checkpoint), so targets of very different scales
weigh alike in the loss; the metrics are computed on the real values.

Standalone like the other workers: the standard library, numpy, torch, torchvision, PIL and (optional)
timm, all imported at the top, plus the sibling files.

Output: one JSON object per line on stdout (``log``, ``warning``, ``setup``, ``progress``, ``epoch``,
``done``, ``error``). Files written to ``output_dir``: ``best.pt``, ``last.pt``, ``results.json``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

try:
    from activelabel.training.workers import augment, classification, options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling files
    import augment  # type: ignore[no-redef]
    import classification  # type: ignore[no-redef]
    import options  # type: ignore[no-redef]

METRICS = ("rmse", "mae", "r2")
LOSSES = ("mse", "mae", "huber")
MAX_POINTS = 2000  # predicted-vs-actual pairs kept in results.json, over all targets


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
        SpecError: missing keys, another task, an unknown family, start, metric or loss, no images, or a
            fold ``"all"``.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task", "model", "family", "start", "params", "classes", "parts", "output_dir", "metric"):
        if key not in spec:
            raise SpecError(f"spec.json has no '{key}'.")
    if spec["task"] != "regression":
        raise SpecError(f"This worker trains regression, not {spec['task']}.")
    if spec.get("fold") == "all":
        raise SpecError("Training all folds comes later: choose one fold.")
    if spec["family"] not in ("timm", "torchvision"):
        raise SpecError(f"Unknown model family '{spec['family']}'.")
    if spec["start"] not in ("transfer", "finetune", "scratch"):
        raise SpecError(f"Unknown start '{spec['start']}'.")
    if spec["metric"] not in METRICS:
        raise SpecError(f"The selection metric must be one of: {', '.join(METRICS)}.")
    if str(spec["params"].get("loss", "mse")) not in LOSSES:
        raise SpecError(f"The loss must be one of: {', '.join(LOSSES)}.")
    if not spec["classes"]:
        raise SpecError("A regression model needs at least one target.")
    parts = spec["parts"]
    if not parts.get("train") or not parts.get("val"):
        raise SpecError("Training needs images in the train and validation parts.")
    return dict(spec)


def target_stats(items: list[dict[str, Any]], n: int) -> tuple[list[float], list[float]]:
    """Mean and standard deviation of each target over the images that have a value of it.

    Args:
        items: The images of the training part (``values``: one number or None per target).
        n: Number of targets.

    Returns:
        ``(means, stds)``; a target without values has mean 0, and a target without spread (or without
        values) standard deviation 1, so normalising it never divides by zero.
    """
    means: list[float] = []
    stds: list[float] = []
    for k in range(n):
        found = [float(i["values"][k]) for i in items if i["values"][k] is not None]
        mean = sum(found) / len(found) if found else 0.0
        var = sum((v - mean) ** 2 for v in found) / len(found) if found else 0.0
        means.append(mean)
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    return means, stds


def _one_target(truth: list[float], predicted: list[float]) -> dict[str, float]:
    """MAE, RMSE and R² of one target.

    Args:
        truth: The real values.
        predicted: The predictions, in the same order.

    Returns:
        ``mae``, ``rmse``, ``r2`` and ``count``; R² is 0 when the real values do not vary (it is not
        defined then) and all are 0 without values.
    """
    count = len(truth)
    if not count:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "count": 0}
    errors = [p - t for t, p in zip(truth, predicted, strict=True)]
    mean = sum(truth) / count
    total = sum((t - mean) ** 2 for t in truth)
    squared = sum(e * e for e in errors)
    return {
        "mae": sum(abs(e) for e in errors) / count,
        "rmse": math.sqrt(squared / count),
        "r2": 1.0 - squared / total if total > 0 else 0.0,
        "count": count,
    }


def evaluation(
    truth: list[list[float | None]], predicted: list[list[float]], targets: list[str]
) -> dict[str, Any]:
    """Regression metrics per target and over the targets.

    Args:
        truth: Real values per image (None where the image has no value of a target).
        predicted: Predictions per image, one per target.
        targets: The target names, in output order.

    Returns:
        ``mae``, ``rmse`` and ``r2`` (the mean over the targets that have values), ``per_target``
        (``{target: {mae, rmse, r2, count}}``) and ``points`` (see :func:`sample_points`).
    """
    per_target: dict[str, dict[str, float]] = {}
    for k, name in enumerate(targets):
        pairs = [(row[k], pred[k]) for row, pred in zip(truth, predicted, strict=True)]
        known = [(float(t), float(p)) for t, p in pairs if t is not None]
        per_target[name] = _one_target([t for t, _ in known], [p for _, p in known])
    scored = [m for m in per_target.values() if m["count"]]
    overall = {
        key: (sum(m[key] for m in scored) / len(scored) if scored else 0.0) for key in ("mae", "rmse", "r2")
    }
    return {**overall, "per_target": per_target, "points": sample_points(truth, predicted, targets)}


def sample_points(
    truth: list[list[float | None]], predicted: list[list[float]], targets: list[str], limit: int = MAX_POINTS
) -> dict[str, list[list[float]]]:
    """``[actual, predicted]`` pairs of each target for the charts, at most ``limit`` in all.

    Args:
        truth: Real values per image (None where there is none).
        predicted: Predictions per image.
        targets: The target names.
        limit: The most pairs kept over every target (evenly spaced when there are more).

    Returns:
        ``{target: [[actual, predicted], ...]}``, rounded to 6 significant digits.
    """
    per = max(1, limit // max(1, len(targets)))
    out: dict[str, list[list[float]]] = {}
    for k, name in enumerate(targets):
        pairs = [
            [float(f"{float(actual):.6g}"), float(f"{float(pred[k]):.6g}")]
            for row, pred in zip(truth, predicted, strict=True)
            if (actual := row[k]) is not None
        ]
        step = max(1, math.ceil(len(pairs) / per))
        out[name] = pairs[::step][:per]
    return out


# ---------------------------------------------------------------------------
# Training (needs torch)


class ValueDataset:
    """Map-style dataset of ``{"path", "values"}`` items: the image, its normalised values and their mask."""

    def __init__(
        self, items: list[dict[str, Any]], transform: Any, means: list[float], stds: list[float]
    ) -> None:
        """Keep the items, the transform and the normalisation of the targets.

        Args:
            items: The images of one part (``path``, and ``values`` with one number or None per target).
            transform: The transforms applied to every image.
            means: Mean of each target (training part).
            stds: Standard deviation of each target (training part).
        """
        self.items, self.transform, self.means, self.stds = items, transform, means, stds

    def __len__(self) -> int:
        """Number of images.

        Returns:
            The number of items in the part.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        """The transformed image, its normalised values (0 where missing) and the mask of the known ones.

        Args:
            index: Position in the part.

        Returns:
            ``(image tensor, values tensor, mask tensor)``.
        """
        item = self.items[index]
        values = item["values"]
        normal = [
            0.0 if v is None else (float(v) - m) / s
            for v, m, s in zip(values, self.means, self.stds, strict=True)
        ]
        mask = [0.0 if v is None else 1.0 for v in values]
        with Image.open(item["path"]) as image:
            upright = ImageOps.exif_transpose(image) or image
            tensor = self.transform(upright.convert("RGB"))
        return tensor, torch.tensor(normal, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class MaskedLoss(torch.nn.Module):
    """MSE, MAE (L1) or Huber loss over the known values only (an image may lack a target)."""

    def __init__(self, name: str, delta: float = 1.0) -> None:
        """Choose the loss.

        Args:
            name: one of :data:`LOSSES`.
            delta: where the Huber loss turns from squared to linear (in standard deviations of the target).

        Raises:
            ValueError: an unknown loss.
        """
        super().__init__()
        if name not in LOSSES:
            raise ValueError(f"Unknown loss '{name}'.")
        self.name, self.delta = name, float(delta)

    def forward(self, output: Any, target: Any, mask: Any) -> Any:
        """The mean loss over the known values of the batch.

        Args:
            output: Predictions, ``(batch, targets)``.
            target: Normalised values, ``(batch, targets)``.
            mask: 1 where the value is known, 0 elsewhere.

        Returns:
            A scalar tensor (0 when no value is known).
        """
        if self.name == "mse":
            each = torch.nn.functional.mse_loss(output, target, reduction="none")
        elif self.name == "mae":
            each = torch.nn.functional.l1_loss(output, target, reduction="none")
        else:
            each = torch.nn.functional.huber_loss(output, target, reduction="none", delta=self.delta)
        return (each * mask).sum() / mask.sum().clamp(min=1.0)


def make_loader(
    items: list[dict[str, Any]],
    transform: Any,
    shuffle: bool,
    stats: tuple[list[float], list[float]],
    batch_size: int,
    workers: int,
    generator: Any,
) -> Any:
    """A data loader over ``items``.

    Args:
        items: the images of one part.
        transform: the transforms applied to every image.
        shuffle: shuffle with ``generator`` (the training part).
        stats: ``(means, stds)`` of the targets (:func:`target_stats`).
        batch_size: images per batch.
        workers: loader processes (0 loads in the main process).
        generator: the seeded random generator used when shuffling.

    Returns:
        A ``DataLoader`` of ``(images, values, mask)`` batches.
    """
    return DataLoader(
        ValueDataset(items, transform, *stats),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: Any,
    loader: Any,
    device: str,
    loss_fn: Any,
    stats: tuple[list[float], list[float]],
    optimizer: Any = None,
    on_step: Any = None,
    schedule: Any = None,
    precision: Any = None,
) -> tuple[float, list[list[float | None]], list[list[float]]]:
    """One pass over a loader. Trains when ``optimizer`` is given.

    Args:
        model: The network.
        loader: Data loader of ``(images, values, mask)`` batches.
        device: Torch device name.
        loss_fn: A :class:`MaskedLoss`.
        stats: ``(means, stds)`` of the targets, to give the values back in their own units.
        optimizer: The optimizer; ``None`` evaluates without gradients.
        on_step: Called with ``(step, steps)`` after each batch (optional).
        schedule: The learning rate schedule (``options.LrSchedule``), advanced before each training step.
        precision: Mixed precision of the training steps (``options.Precision``); float32 by default.

    Returns:
        ``(mean loss per image, real values per image (None where missing), predictions per image)``.
    """
    training = optimizer is not None
    amp = precision or options.Precision(False)
    means, stds = stats
    total, count = 0.0, 0
    truth: list[list[float | None]] = []
    predicted: list[list[float]] = []
    with torch.set_grad_enabled(training):
        for step, (images, values, mask) in enumerate(loader, start=1):
            images, values, mask = images.to(device), values.to(device), mask.to(device)
            with amp.autocast(training):
                output = model(images).float()
                loss = loss_fn(output, values, mask)
            if training:
                if schedule is not None:
                    schedule.advance()
                amp.step(loss, optimizer)
            total += float(loss.detach()) * len(images)
            count += len(images)
            for row, known, out in zip(values.tolist(), mask.tolist(), output.detach().tolist(), strict=True):
                truth.append(
                    [v * s + m if k else None for v, k, m, s in zip(row, known, means, stds, strict=True)]
                )
                predicted.append([o * s + m for o, m, s in zip(out, means, stds, strict=True)])
            if on_step:
                on_step(step, len(loader))
    return (total / count if count else 0.0), truth, predicted


def batch_loss(model: Any, batch: Any, device: str, loss_fn: Any) -> Any:
    """The loss of one batch of ``(images, values, mask)``, for the GPU memory probe.

    Args:
        model: The network.
        batch: One batch from the data loader.
        device: Torch device name.
        loss_fn: A :class:`MaskedLoss`.

    Returns:
        The loss tensor of the batch.
    """
    images, values, mask = (t.to(device) for t in batch)
    return loss_fn(model(images).float(), values, mask)


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


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """Train, keep the best epoch by the selection metric (RMSE and MAE: lowest), evaluate the test part.

    Writes ``best.pt``, ``last.pt`` and ``results.json`` to ``output_dir``.

    Args:
        spec: The checked job spec.

    Returns:
        The results: ``metric``, ``best_epoch``, ``epochs_run``, ``stopped_early``, ``val`` and ``test``
        (``None`` without a test part); each part has ``loss``, ``mae``, ``rmse``, ``r2``, ``per_target``,
        ``points`` and ``targets``.

    Raises:
        SpecError: A model or size the worker cannot use.
        options.DeviceError: The device the run asked for is not available.
        options.CodedFailure: Nothing to resume from, or the training changed since it stopped.
        augment.AugmentError: A technique the worker cannot apply.
        ImportError: A timm model was asked for and timm is not installed.
    """
    p, targets, metric = spec["params"], spec["classes"], spec["metric"]
    lower = bool(spec.get("lower_is_better", metric != "r2"))
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    classification.seed_everything(int(spec.get("seed", p["seed"])), torch)
    device = options.pick_device(torch, str(spec.get("device") or "auto"))
    emit("log", message=f"Device: {device}. {classification.split_folds_note(spec)}")
    use_amp, amp_note = options.amp_enabled(str(p["amp"]), device)
    if amp_note:
        emit("warning", message=amp_note, code="amp_needs_cuda", params={})
    precision = options.Precision(use_amp)
    emit("setup", device=device, amp=use_amp)
    size, batch, policy = int(p["image_size"]), int(p["batch_size"]), str(p["resize"])
    techniques = spec["augmentations"]
    emit("log", message=f"Augmentation: {augment.describe(techniques)}. Resize: {policy}.")
    try:
        model, head = classification.build_model(spec, pretrained=spec["start"] != "scratch")
    except classification.SpecError as exc:
        raise SpecError(str(exc)) from exc
    mean, std = classification.normalisation(model)
    train_tf = augment.classification_transform(techniques, size, mean, std, options.Fit(size, policy))
    eval_tf = classification.eval_transform(size, policy, mean, std)
    parts = spec["parts"]
    stats = target_stats(parts["train"], len(targets))
    emit(
        "log", message=f"Targets normalised with the training part: means {[round(v, 4) for v in stats[0]]}."
    )
    generator = torch.Generator().manual_seed(int(p["seed"]))
    workers = 4 if device.startswith("cuda") else 0
    resumed = (
        options.load_resume(out / "last.pt", str(spec.get("digest") or "")) if spec.get("resume") else None
    )
    if resumed is not None:
        batch = int(resumed["resume"]["batch"])
    loader = partial(make_loader, stats=stats, batch_size=batch, workers=workers, generator=generator)
    train_loader, val_loader = loader(parts["train"], train_tf, True), loader(parts["val"], eval_tf, False)
    model.to(device)
    optimizer = classification.make_optimizer(spec, model, head, torch)
    loss_fn = MaskedLoss(str(p.get("loss", "mse")), float(p.get("huber_delta", 1.0))).to(device)
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
        classification.seed_everything(int(spec.get("seed", p["seed"])), torch)  # the probe drew numbers
        loader = partial(loader, batch_size=batch)
        train_loader, val_loader = (
            loader(parts["train"], train_tf, True),
            loader(parts["val"], eval_tf, False),
        )
    emit("setup", batch_size=batch)

    epochs, patience = int(p["epochs"]), int(p["patience"])
    schedule = options.LrSchedule(
        optimizer, str(p["scheduler"]), epochs, len(train_loader), int(p["warmup_epochs"]), lower=lower
    )
    best: dict[str, Any] = {}
    since_best, ran = 0, 0
    if resumed is not None:
        model.load_state_dict(resumed["state_dict"])
        loop = options.restore(resumed["resume"], optimizer, schedule, precision, generator)
        best, since_best, ran = loop["best"], int(loop["since_best"]), int(loop["epoch"])
        emit("resumed", epoch=ran)
        emit("log", message=f"Resumed after epoch {ran} of {epochs}.")
    meta = {
        "task": "regression",
        "classes": targets,
        "model": spec["model"],
        "family": spec["family"],
        "image_size": size,
        "resize": policy,
        "mean": mean,
        "std": std,
        "target_mean": stats[0],
        "target_std": stats[1],
    }

    for epoch in range(ran + 1, epochs + 1):
        started = time.monotonic()
        model.train()
        if spec["start"] == "transfer":
            model.eval()  # frozen backbone: no batch-norm statistics or dropout changes
            head.train()
        on_step = partial(report_step, epoch, epochs)
        train_loss, _, _ = run_epoch(
            model, train_loader, device, loss_fn, stats, optimizer, on_step, schedule, precision
        )
        model.eval()
        val_loss, truth, predicted = run_epoch(model, val_loader, device, loss_fn, stats)
        scores = evaluation(truth, predicted, targets)
        schedule.end_epoch(scores[metric])
        ran = epoch
        improved = options.improves(scores[metric], best["scores"][metric] if best else None, lower)
        if improved:
            best = {"epoch": epoch, "scores": scores, "val_loss": val_loss}
            since_best = 0
            classification.save_checkpoint(out / "best.pt", model, meta, epoch)
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
        classification.save_checkpoint(out / "last.pt", model, {**meta, "resume": resume}, epoch)
        emit(
            "epoch",
            epoch=epoch,
            checkpoint=True,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            val_rmse=scores["rmse"],
            val_mae=scores["mae"],
            val_r2=scores["r2"],
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
        "val": {"loss": best["val_loss"], **best["scores"], "targets": targets},
        "test": None,
    }
    if parts.get("test"):
        emit("progress", phase="evaluating", epoch=ran, epochs=epochs)
        state = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        test_loss, truth, predicted = run_epoch(
            model, loader(parts["test"], eval_tf, False), device, loss_fn, stats
        )
        results["test"] = {"loss": test_loss, **evaluation(truth, predicted, targets), "targets": targets}
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
