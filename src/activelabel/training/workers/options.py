"""Training options shared by the workers and the prediction and export processes (docs/TRAINING.md).

The resize policy turns a photo of any shape into the square input of the network, the same way in training,
evaluation, pre-annotation and ONNX export:

- ``stretch``: both sides scaled to the square (the proportions change);
- ``letterbox``: the longer side fits, the rest is filled (the proportions stay, nothing is cut);
- ``center_crop``: the shorter side fits, the centre is kept (the proportions stay, the edges are cut).

Each policy is one scale per axis plus a shift (an affine matrix), so boxes, masks and keypoints go into the
network and come back to the photo with the same numbers.

The learning rate follows one schedule per step in every worker (:class:`LrSchedule`): an optional linear
warmup, then ``none``, ``cosine``, ``step``, ``onecycle`` (its own ramp up and down) or ``plateau``
(divided by ten when the selection metric stops improving).

The device (:func:`pick_device`) is the one the run asked for (``auto``, ``cpu``, ``cuda:N`` or ``mps``),
checked before anything is trained. Automatic mixed precision (:class:`Precision`) runs the training steps
in float16 on a CUDA GPU (``amp`` ``auto`` or ``on``); on the CPU and on Apple GPUs (MPS) training stays in
float32. On CUDA the batch size is tried first (:func:`fit_batch`): when the GPU runs out of memory the batch
is halved, down to the smallest the model accepts, with a warning.

Standalone like the workers: it runs in the training environment, which has no ActiveLabel.
"""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from activelabel.training.workers import augment
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling file
    import augment  # type: ignore[no-redef]

Array = Any
RESIZE_POLICIES = ("stretch", "letterbox", "center_crop")
#: Policy of a saved model that does not record one: classification and segmentation always stretched.
LEGACY_RESIZE = "stretch"
#: Colour of the bands a letterbox adds (the ImageNet mean, so it is about zero after normalisation).
FILL = (124, 116, 104)
SCHEDULERS = ("none", "cosine", "step", "onecycle", "plateau")
WARMUP_START = 0.001  # share of the learning rate at the first warmup step
ONECYCLE_PEAK = 0.3  # share of the steps that one-cycle spends going up
ONECYCLE_LOW = 1 / 25  # one-cycle starts at this share of the learning rate...
ONECYCLE_END = 1 / 25e4  # ...and ends at this one (the PyTorch defaults)
PLATEAU_PATIENCE = 2  # epochs without improvement before plateau lowers the rate
PLATEAU_FACTOR = 0.1
AMP_CHOICES = ("auto", "on", "off")


class DeviceError(ValueError):
    """The device asked for is not available to the training environment (the message is readable)."""


class OutOfMemory(RuntimeError):
    """Not even the smallest batch fits in the GPU memory (the message is readable)."""


def fit_layout(width: int, height: int, size: int, policy: str) -> tuple[int, int, int, int]:
    """Where a ``width`` x ``height`` photo goes in the ``size`` x ``size`` input.

    Args:
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.

    Returns:
        ``(scaled width, scaled height, left, top)``: the photo is scaled to that size and its top-left corner
        put at ``(left, top)`` (negative with ``center_crop``: that part is cut).

    Raises:
        ValueError: an unknown policy.
    """
    if policy == "stretch":
        return size, size, 0, 0
    if policy not in RESIZE_POLICIES:
        raise ValueError(f"Unknown resize policy '{policy}': use one of {', '.join(RESIZE_POLICIES)}.")
    side = max(width, height) if policy == "letterbox" else min(width, height)
    scale = size / side
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    return new_w, new_h, (size - new_w) // 2, (size - new_h) // 2


def fit_matrix(width: int, height: int, size: int, policy: str) -> Array:
    """Affine matrix from photo pixels to input pixels (see :func:`fit_layout`).

    Args:
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.

    Returns:
        The 3 x 3 matrix.
    """
    new_w, new_h, left, top = fit_layout(width, height, size, policy)
    return np.array([[new_w / width, 0.0, float(left)], [0.0, new_h / height, float(top)], [0.0, 0.0, 1.0]])


def fit_image(
    image: Any, size: int, policy: str, resample: Any = Image.Resampling.BILINEAR, fill: Any = FILL
) -> Any:
    """A PIL image turned into the ``size`` x ``size`` input by ``policy``.

    Args:
        image: PIL image (a photo, or a class mask with ``NEAREST`` resampling).
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.
        resample: PIL resampling filter.
        fill: value of the letterbox bands.

    Returns:
        The square image.
    """
    new_w, new_h, left, top = fit_layout(image.width, image.height, size, policy)
    scaled = image.resize((new_w, new_h), resample)
    if (new_w, new_h, left, top) == (size, size, 0, 0):
        return scaled
    canvas = Image.new(image.mode, (size, size), fill)
    canvas.paste(scaled, (left, top))
    return canvas


def fit_target(target: dict[str, Array], width: int, height: int, size: int, policy: str) -> dict[str, Array]:
    """A detection target in photo pixels moved into the input (objects cut away by a crop are dropped).

    Args:
        target: ``boxes`` (N, 4), ``labels`` (N,) and maybe ``masks`` (N, H, W) or ``keypoints`` (N, K, 3).
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.

    Returns:
        The target in input pixels, of the same keys.
    """
    return augment.move_target(target, fit_matrix(width, height, size, policy), size, size, [])


def unfit_boxes(boxes: Array, width: int, height: int, size: int, policy: str) -> Array:
    """Boxes ``(N, 4)`` xyxy in input pixels taken back to the photo (clipped to it).

    Args:
        boxes: boxes in input pixels.
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.

    Returns:
        The boxes in photo pixels (float32), one per input box.
    """
    inverse = np.linalg.inv(fit_matrix(width, height, size, policy))
    moved, _ = augment.move_boxes(boxes, inverse, width, height)
    return moved


def unfit_points(points: Array, width: int, height: int, size: int, policy: str) -> Array:
    """Keypoints ``(N, K, 2 or 3)`` in input pixels taken back to the photo (the visibility is kept).

    Args:
        points: points in input pixels; a third value per point is kept as it is.
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.

    Returns:
        The points in photo pixels (float64), of the same shape.
    """
    out = np.array(points, dtype=np.float64, copy=True)
    if not out.size:
        return out
    inverse = np.linalg.inv(fit_matrix(width, height, size, policy))
    flat = augment.move_points(out[..., :2].reshape(-1, 2), inverse)
    out[..., :2] = flat.reshape(*out.shape[:-1], 2)
    return out


def unfit_map(
    maps: Any, width: int, height: int, size: int, policy: str, out_size: tuple[int, int], fill: float = 0.0
) -> Any:
    """Per-pixel maps ``(N, C, size, size)`` of the input taken back to a grid over the photo.

    The part of the input that shows the photo is scaled (bilinear) onto the grid; grid cells the input did
    not show (the edges a centre crop cut) get ``fill``.

    Args:
        maps: tensor of the network's maps (logits or probabilities).
        width: photo width in pixels.
        height: photo height in pixels.
        size: side of the square input.
        policy: one of :data:`RESIZE_POLICIES`.
        out_size: ``(grid width, grid height)`` covering the whole photo.
        fill: value of the cells the input did not show.

    Returns:
        The tensor ``(N, C, grid height, grid width)``.
    """
    new_w, new_h, left, top = fit_layout(width, height, size, policy)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + new_w, size), min(top + new_h, size)
    grid_w, grid_h = out_size
    # the shown part, in grid cells (photo pixels scaled to the grid)
    gx0, gx1 = round((x0 - left) / new_w * grid_w), round((x1 - left) / new_w * grid_w)
    gy0, gy1 = round((y0 - top) / new_h * grid_h), round((y1 - top) / new_h * grid_h)
    gx1, gy1 = max(gx1, gx0 + 1), max(gy1, gy0 + 1)
    part = torch.nn.functional.interpolate(
        maps[:, :, y0:y1, x0:x1], size=(gy1 - gy0, gx1 - gx0), mode="bilinear", align_corners=False
    )
    if (gx0, gy0, gx1, gy1) == (0, 0, grid_w, grid_h):
        return part
    out = maps.new_full((maps.shape[0], maps.shape[1], grid_h, grid_w), fill)
    out[:, :, gy0:gy1, gx0:gx1] = part[:, :, : grid_h - gy0, : grid_w - gx0]
    return out


class Fit:
    """Transform that turns a PIL photo into the square input (picklable, for data loader processes)."""

    def __init__(self, size: int, policy: str) -> None:
        """Keep the input side and the policy.

        Args:
            size: side of the square input.
            policy: one of :data:`RESIZE_POLICIES`.

        Raises:
            ValueError: an unknown policy.
        """
        if policy not in RESIZE_POLICIES:
            raise ValueError(f"Unknown resize policy '{policy}': use one of {', '.join(RESIZE_POLICIES)}.")
        self.size = size
        self.policy = policy

    def __call__(self, image: Any) -> Any:
        """The square input of a photo.

        Args:
            image: RGB PIL image.

        Returns:
            The ``size`` x ``size`` PIL image.
        """
        return fit_image(image, self.size, self.policy)

    def __repr__(self) -> str:
        """Short description, shown by ``transforms.Compose``.

        Returns:
            For example ``Fit(size=224, policy=letterbox)``.
        """
        return f"Fit(size={self.size}, policy={self.policy})"


def _cosine(start: float, end: float, t: float) -> float:
    """A half cosine from ``start`` (t = 0) to ``end`` (t = 1).

    Args:
        start: value at the beginning.
        end: value at the end.
        t: position, 0..1 (clipped).

    Returns:
        The value at ``t``.
    """
    t = min(1.0, max(0.0, t))
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))


def lr_factor(
    name: str, step: int, steps_per_epoch: int, epochs: int, warmup_epochs: float, plateau: float = 1.0
) -> float:
    """Share of the learning rate at a training step (the pure schedule; see :class:`LrSchedule`).

    Args:
        name: one of :data:`SCHEDULERS`.
        step: the step about to run, from 0 over the whole training.
        steps_per_epoch: batches in an epoch.
        epochs: epochs of the training.
        warmup_epochs: epochs of the linear warmup (0 = none; one-cycle has its own ramp and ignores it).
        plateau: the share ``plateau`` has reached (it changes with the selection metric, epoch by epoch).

    Returns:
        The factor that multiplies the learning rate of every parameter group.

    Raises:
        ValueError: an unknown schedule.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown learning rate schedule '{name}': use one of {', '.join(SCHEDULERS)}.")
    steps_per_epoch = max(1, steps_per_epoch)
    total = max(1, steps_per_epoch * epochs)
    if name == "onecycle":
        peak = max(1, round(ONECYCLE_PEAK * total))
        if step < peak:
            return _cosine(ONECYCLE_LOW, 1.0, step / peak)
        return _cosine(1.0, ONECYCLE_END, (step - peak) / max(1, total - peak))
    warm_steps = round(warmup_epochs * steps_per_epoch)
    warm = WARMUP_START + (1.0 - WARMUP_START) * step / warm_steps if step < warm_steps else 1.0
    if name == "cosine":
        base = _cosine(1.0, 0.0, (step - warm_steps) / max(1, total - warm_steps))
    elif name == "step":
        base = 0.1 ** ((step // steps_per_epoch) // max(1, epochs // 3))
    elif name == "plateau":
        base = plateau
    else:
        base = 1.0
    return warm * base


class LrSchedule:
    """Sets the learning rate of an optimizer before every training step (warmup and schedule)."""

    def __init__(
        self, optimizer: Any, name: str, epochs: int, steps_per_epoch: int, warmup_epochs: float = 0
    ) -> None:
        """Remember the starting rates of the optimizer (the 100 % of the schedule).

        Args:
            optimizer: the torch optimizer (its parameter groups keep their own base rates).
            name: one of :data:`SCHEDULERS`.
            epochs: epochs of the training.
            steps_per_epoch: batches in an epoch.
            warmup_epochs: epochs of the linear warmup (0 = none).

        Raises:
            ValueError: an unknown schedule.
        """
        lr_factor(name, 0, steps_per_epoch, epochs, warmup_epochs)  # checks the name
        self.optimizer = optimizer
        self.name, self.epochs, self.steps_per_epoch, self.warmup = (
            name,
            epochs,
            steps_per_epoch,
            warmup_epochs,
        )
        self.base = [float(g["lr"]) for g in optimizer.param_groups]
        self.step_index = 0
        self.plateau = 1.0
        self.best: float | None = None
        self.waited = 0

    def advance(self) -> None:
        """Set the rate of the step about to run and count it."""
        factor = lr_factor(
            self.name, self.step_index, self.steps_per_epoch, self.epochs, self.warmup, self.plateau
        )
        for group, base in zip(self.optimizer.param_groups, self.base, strict=True):
            group["lr"] = base * factor
        self.step_index += 1

    def end_epoch(self, score: float) -> None:
        """Take the selection metric of an epoch (higher is better): ``plateau`` lowers the rate if it stalls.

        Args:
            score: the validation value of the selection metric.
        """
        if self.name != "plateau":
            return
        if self.best is None or score > self.best:
            self.best, self.waited = score, 0
            return
        self.waited += 1
        if self.waited >= PLATEAU_PATIENCE:
            self.plateau *= PLATEAU_FACTOR
            self.waited = 0

    def lr(self) -> float:
        """The rate of the last parameter group (the head), for the epoch log.

        Returns:
            The current learning rate.
        """
        return float(self.optimizer.param_groups[-1]["lr"])


def amp_enabled(choice: str, device: str) -> tuple[bool, str]:
    """Whether a training uses mixed precision, and a warning when ``on`` cannot be honoured.

    Args:
        choice: the ``amp`` parameter (one of :data:`AMP_CHOICES`).
        device: the torch device of the training (``cpu``, ``mps``, ``cuda`` or ``cuda:N``).

    Returns:
        ``(enabled, warning)``: mixed precision only on CUDA and not ``off``; the warning is empty unless
        ``on`` was asked for on a device without it.

    Raises:
        ValueError: an unknown choice.
    """
    if choice not in AMP_CHOICES:
        raise ValueError(f"Unknown amp choice '{choice}': use one of {', '.join(AMP_CHOICES)}.")
    if choice == "off":
        return False, ""
    if device.startswith("cuda"):
        return True, ""
    if choice == "on":
        return False, f"Mixed precision needs a CUDA GPU: this training runs in full precision on {device}."
    return False, ""


class Precision:
    """Forward passes in float16 (autocast) and a gradient scaler on CUDA, or plain float32 steps."""

    def __init__(self, enabled: bool) -> None:
        """Create the gradient scaler when mixed precision is on.

        Args:
            enabled: use mixed precision (only on a CUDA device: see :func:`amp_enabled`).
        """
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler("cuda") if enabled else None

    def autocast(self, active: bool = True) -> Any:
        """Context of a forward pass: float16 where safe when mixed precision is on.

        Args:
            active: False for evaluation passes, which stay in float32.

        Returns:
            A context manager.
        """
        if self.enabled and active:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def step(self, loss: Any, optimizer: Any) -> None:
        """Backward pass and optimizer step (scaled with mixed precision).

        Args:
            loss: the loss tensor of the batch.
            optimizer: the optimizer.

        Raises:
            RuntimeError: the loss is not finite (NaN or infinite), with a hint about mixed precision when on.
        """
        if not bool(torch.isfinite(loss)):
            if self.enabled:
                raise RuntimeError(
                    "The loss is not finite (NaN or infinite) with mixed precision: set amp to off or try a "
                    "lower learning rate."
                )
            raise RuntimeError("The loss is not finite: try a lower learning rate.")
        optimizer.zero_grad()
        if self.scaler is None:
            loss.backward()
            optimizer.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


def pick_device(torch: Any, choice: str = "auto") -> str:
    """The torch device of a choice: ``auto`` takes CUDA, then MPS (Apple), then the CPU.

    Args:
        torch: the torch module.
        choice: ``auto``, ``cpu``, ``cuda`` (the first GPU), ``cuda:N`` or ``mps``.

    Returns:
        The device name (``cpu``, ``mps``, ``cuda`` or ``cuda:N``).

    Raises:
        DeviceError: the device asked for does not exist here, or the choice is not a device.
    """
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    if choice in ("", "auto"):
        return "cuda" if cuda else "mps" if mps else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice == "mps":
        if not mps:
            raise DeviceError("The Apple GPU (MPS) is not available on this computer: choose auto or CPU.")
        return "mps"
    match = re.fullmatch(r"cuda(?::(\d+))?", choice)
    if match is None:
        raise DeviceError(f"Unknown device '{choice}': use auto, cpu, cuda:N (a GPU) or mps.")
    if not cuda:
        raise DeviceError(
            "No CUDA GPU is available to the training environment (no NVIDIA driver, or PyTorch without "
            "CUDA): choose auto or CPU."
        )
    index, count = int(match.group(1) or 0), int(torch.cuda.device_count())
    if index >= count:
        raise DeviceError(f"GPU {index} does not exist: this computer has {count} (numbered from 0).")
    return choice


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether an error is the GPU running out of memory.

    Args:
        exc: the error raised by a training step.

    Returns:
        True for ``torch.cuda.OutOfMemoryError`` or a runtime error that says "out of memory".
    """
    kind = getattr(torch.cuda, "OutOfMemoryError", None)
    if kind is not None and isinstance(exc, kind):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def probe(batch: int, model: Any, loader_of: Any, loss_of: Any, precision: Precision, optimizer: Any) -> None:
    """One forward and backward pass of a first batch, then all put back (weights, statistics, gradients).

    Args:
        batch: the batch size to try.
        model: the network, on its device.
        loader_of: function ``(batch_size=...)`` that gives a data loader of the training images.
        loss_of: function ``(model, batch)`` that gives the loss of one batch.
        precision: the mixed precision of the training.
        optimizer: the optimizer (its gradients are cleared).
    """
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    try:
        first = next(iter(loader_of(batch_size=batch)))
        with precision.autocast():
            loss = loss_of(model, first)
        loss.backward()
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.load_state_dict(saved)


def fit_batch(try_batch: Any, batch: int, emit: Any, smallest: int = 1, clear: Any = None) -> int:
    """The batch size to train with: the one asked for, halved while the GPU runs out of memory.

    Args:
        try_batch: function ``(batch)`` that runs one training step with that batch (see :func:`probe`).
        batch: the batch size asked for.
        emit: function ``(event, **fields)`` that tells the runner (a ``warning`` per halving).
        smallest: the smallest batch the model accepts.
        clear: function that frees the GPU memory after a failure (``torch.cuda.empty_cache``).

    Returns:
        The batch size that fits.

    Raises:
        OutOfMemory: not even ``smallest`` fits.
    """
    while True:
        try:
            try_batch(batch)
            return batch
        except Exception as exc:  # noqa: BLE001 - only running out of memory is handled here
            if not is_out_of_memory(exc):
                raise
            if clear is not None:
                clear()
            if batch <= smallest:
                raise OutOfMemory(
                    f"Not even a batch of {smallest} image(s) fits in the GPU memory: lower the image size "
                    "or train on the CPU."
                ) from None
            smaller = max(smallest, batch // 2)
            emit(
                "warning",
                message=f"The GPU ran out of memory with a batch of {batch} images: training goes on with "
                f"{smaller}.",
                code="batch_halved",
                batch_from=batch,
                batch_to=smaller,
            )
            batch = smaller
