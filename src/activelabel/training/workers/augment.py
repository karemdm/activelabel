"""Data augmentation of the training workers (docs/TRAINING.md): applies the techniques the run chose.

The runner resolves the ``augmentation`` level into a list ``[{"name", "params"}]`` (see
``activelabel/training/augmentations.py``, the catalogue) and puts it in ``spec["augmentations"]``; the
workers apply exactly that list, in its order, to the training images only (validation and test images are
only resized). Standalone like the workers: it runs in the training environment, which has no ActiveLabel.

Geometric techniques are combined into one affine matrix (continuous pixel coordinates: pixel ``i`` spans
``[i, i + 1)``) applied to the image and, with the same matrix, to boxes, masks and keypoints, so the labels
stay on the objects. A horizontal flip swaps the keypoint ``flip_pairs``.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

Array = Any
Entries = list[dict[str, Any]]
#: Share of a moved box that must stay inside the image for the object to be kept.
MIN_VISIBLE = 0.25
#: Techniques that only the classification worker applies (they do not move labels or mix batches).
CLASSIFICATION_ONLY = ("randaugment", "trivialaugment", "random_erasing", "mixup", "cutmix")
MIXES = ("mixup", "cutmix")
GEOMETRIC = ("random_resized_crop", "hflip", "vflip", "rotation", "scale_jitter", "translate")


class AugmentError(ValueError):
    """A technique this worker cannot apply (the message is readable)."""


def params_of(entries: Entries) -> dict[str, dict[str, Any]]:
    """The parameters of each chosen technique, by name.

    Args:
        entries: the resolved techniques ``[{"name", "params"}]``.

    Returns:
        Technique name to its parameters.
    """
    return {e["name"]: dict(e.get("params") or {}) for e in entries}


def describe(entries: Entries) -> str:
    """One line for the log: the techniques and their settings.

    Args:
        entries: the resolved techniques.

    Returns:
        For example ``hflip (p=0.5), color_jitter (brightness=0.2, ...)``, or ``none``.
    """
    if not entries:
        return "none"
    parts = []
    for entry in entries:
        settings = ", ".join(f"{k}={v}" for k, v in sorted((entry.get("params") or {}).items()))
        parts.append(f"{entry['name']} ({settings})" if settings else entry["name"])
    return ", ".join(parts)


def blur_kernel(sigma_max: float) -> int:
    """Odd Gaussian kernel size that holds three sigmas on each side.

    Args:
        sigma_max: the largest sigma, in pixels.

    Returns:
        The kernel size (at least 3).
    """
    return max(3, 2 * math.ceil(3 * sigma_max) + 1)


# ---------------------------------------------------------------------------
# Classification (torchvision transforms on PIL images)


def _classification_step(name: str, params: dict[str, Any], size: int) -> Any:
    """The torchvision transform of one technique, for classification.

    Args:
        name: technique name.
        params: its parameters.
        size: side of the square input, in pixels.

    Returns:
        The transform (``None`` for ``random_erasing``, ``mixup`` and ``cutmix``, applied elsewhere).

    Raises:
        AugmentError: a technique the classification worker does not know.
    """
    if name == "random_resized_crop":
        return transforms.RandomResizedCrop((size, size), scale=(params["scale_min"], 1.0))
    if name == "hflip":
        return transforms.RandomHorizontalFlip(params["p"])
    if name == "vflip":
        return transforms.RandomVerticalFlip(params["p"])
    if name == "rotation":
        return transforms.RandomRotation(params["degrees"])
    if name == "randaugment":
        return transforms.RandAugment(num_ops=params["num_ops"], magnitude=params["magnitude"])
    if name == "trivialaugment":
        return transforms.TrivialAugmentWide()
    if name == "color_jitter":
        return transforms.ColorJitter(
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            hue=params["hue"],
        )
    if name == "blur":
        sigma = params["sigma_max"]
        blur = transforms.GaussianBlur(blur_kernel(sigma), sigma=(0.1, sigma))
        return transforms.RandomApply([blur], p=params["p"])
    if name == "grayscale":
        return transforms.RandomGrayscale(params["p"])
    if name in ("random_erasing", *MIXES):
        return None
    raise AugmentError(f"The classification worker cannot apply the augmentation technique '{name}'.")


def classification_transform(
    entries: Entries, size: int, mean: tuple[float, ...], std: tuple[float, ...]
) -> Any:
    """Training transform of a classifier: the chosen techniques, tensor conversion and normalisation.

    Without ``random_resized_crop`` the image is resized to ``size`` x ``size`` first.

    Args:
        entries: the resolved techniques.
        size: side of the square input, in pixels.
        mean: normalisation mean per channel.
        std: normalisation standard deviation per channel.

    Returns:
        The composed transform (PIL image in, normalised tensor out).

    Raises:
        AugmentError: a technique the classification worker does not know.
    """
    by_name = params_of(entries)
    steps = [] if "random_resized_crop" in by_name else [transforms.Resize((size, size))]
    steps += [s for s in (_classification_step(e["name"], by_name[e["name"]], size) for e in entries) if s]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if "random_erasing" in by_name:
        steps.append(transforms.RandomErasing(p=by_name["random_erasing"]["p"]))
    return transforms.Compose(steps)


def mix_settings(entries: Entries) -> list[tuple[str, float]]:
    """The batch mixes chosen (mixup, cutmix) with their ``alpha``.

    Args:
        entries: the resolved techniques.

    Returns:
        ``(name, alpha)`` pairs; empty when neither is chosen.
    """
    return [(e["name"], float(e["params"]["alpha"])) for e in entries if e["name"] in MIXES]


def cut_box(height: int, width: int, lam: float, rng: Any = random) -> tuple[int, int, int, int]:
    """Random box of about ``1 - lam`` of the image area, for cutmix.

    Args:
        height: image height in pixels.
        width: image width in pixels.
        lam: share of the image that stays.
        rng: random source with ``uniform`` (the ``random`` module by default).

    Returns:
        ``(top, left, bottom, right)`` in pixels, clipped to the image.
    """
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = height * ratio, width * ratio
    cy, cx = rng.uniform(0, height), rng.uniform(0, width)
    top, bottom = int(max(0.0, cy - cut_h / 2)), int(min(height, cy + cut_h / 2))
    left, right = int(max(0.0, cx - cut_w / 2)), int(min(width, cx + cut_w / 2))
    return top, left, bottom, right


def mix_batch(images: Any, mixes: list[tuple[str, float]], rng: Any = random) -> tuple[Any, Any, float]:
    """Mixup or cutmix (one of the chosen, at random) of a batch with a shuffled copy of itself.

    The loss is then ``lam * loss(output, labels) + (1 - lam) * loss(output, labels[order])``.

    Args:
        images: batch tensor ``(N, C, H, W)``.
        mixes: the chosen ``(name, alpha)`` pairs (see ``mix_settings``).
        rng: random source with ``choice``, ``betavariate`` and ``uniform``.

    Returns:
        ``(mixed images, order of the partner images, lam)``; ``lam`` is the share of each image's own class.
    """
    name, alpha = rng.choice(mixes)
    lam = rng.betavariate(alpha, alpha)
    order = torch.randperm(images.shape[0], device=images.device)
    if name == "mixup":
        return lam * images + (1.0 - lam) * images[order], order, lam
    height, width = images.shape[-2:]
    top, left, bottom, right = cut_box(height, width, lam, rng)
    mixed = images.clone()
    mixed[:, :, top:bottom, left:right] = images[order][:, :, top:bottom, left:right]
    kept = 1.0 - (bottom - top) * (right - left) / float(height * width)
    return mixed, order, kept


# ---------------------------------------------------------------------------
# Objects and masks (detection, instance, keypoints, semantic)


def _translation(tx: float, ty: float) -> Array:
    """Affine matrix of a shift.

    Args:
        tx: shift along x, in pixels.
        ty: shift along y, in pixels.

    Returns:
        The 3 x 3 matrix.
    """
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _about_centre(matrix: Array, width: int, height: int) -> Array:
    """A linear map applied around the image centre.

    Args:
        matrix: 3 x 3 matrix of the map about the origin.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        The 3 x 3 matrix about the centre.
    """
    return _translation(width / 2, height / 2) @ matrix @ _translation(-width / 2, -height / 2)


def crop_box(width: int, height: int, min_scale: float, rng: Any = random) -> tuple[int, int, int, int]:
    """Random crop ``(left, top, right, bottom)`` covering ``min_scale..1`` of the area (aspect 3/4..4/3).

    Args:
        width: image width in pixels.
        height: image height in pixels.
        min_scale: smallest share of the area (1.0 or more keeps the whole image).
        rng: random source with ``uniform`` and ``randint``.

    Returns:
        The crop box in pixels.
    """
    if min_scale >= 1.0:
        return 0, 0, width, height
    area = rng.uniform(min_scale, 1.0) * width * height
    ratio = rng.uniform(3 / 4, 4 / 3)
    w = min(width, max(1, round((area * ratio) ** 0.5)))
    h = min(height, max(1, round((area / ratio) ** 0.5)))
    left, top = rng.randint(0, width - w), rng.randint(0, height - h)
    return left, top, left + w, top + h


def _step_matrix(name: str, params: dict[str, Any], width: int, height: int, rng: Any) -> Array | None:
    """The random affine matrix of one geometric technique.

    Args:
        name: technique name.
        params: its parameters.
        width: image width in pixels.
        height: image height in pixels.
        rng: random source with ``random``, ``uniform`` and ``randint``.

    Returns:
        The 3 x 3 matrix, or ``None`` when the technique is not applied this time (probability) or is not
        geometric.
    """
    if name == "random_resized_crop":
        left, top, right, bottom = crop_box(width, height, params["scale_min"], rng)
        scale = np.diag([width / (right - left), height / (bottom - top), 1.0])
        return scale @ _translation(-left, -top)
    if name in ("hflip", "vflip"):
        if rng.random() >= params["p"]:
            return None
        if name == "hflip":
            return np.array([[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height)], [0.0, 0.0, 1.0]])
    if name == "rotation":
        angle = math.radians(rng.uniform(-params["degrees"], params["degrees"]))
        c, s = math.cos(angle), math.sin(angle)
        return _about_centre(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), width, height)
    if name == "scale_jitter":
        zoom = rng.uniform(params["scale_min"], params["scale_max"])
        return _about_centre(np.diag([zoom, zoom, 1.0]), width, height)
    if name == "translate":
        f = params["fraction"]
        return _translation(rng.uniform(-f, f) * width, rng.uniform(-f, f) * height)
    return None


def geometry(entries: Entries, width: int, height: int, rng: Any = random) -> tuple[Array, bool]:
    """One random affine matrix for the geometric techniques chosen, applied in their order.

    Args:
        entries: the resolved techniques.
        width: image width in pixels.
        height: image height in pixels.
        rng: random source with ``random``, ``uniform`` and ``randint``.

    Returns:
        ``(3 x 3 matrix from input to output pixel coordinates, whether it mirrors left and right)``.
    """
    matrix = np.eye(3)
    for entry in entries:
        step = _step_matrix(entry["name"], entry.get("params") or {}, width, height, rng)
        if step is not None:
            matrix = step @ matrix
    return matrix, mirrors(matrix)


def mirrors(matrix: Array) -> bool:
    """Whether an affine matrix mirrors the image (negative determinant).

    Args:
        matrix: 3 x 3 affine matrix.

    Returns:
        True when it is a reflection.
    """
    return bool(np.linalg.det(matrix[:2, :2]) < 0)


def warp(image: Any, matrix: Array, resample: Any, fill: Any) -> Any:
    """A PIL image moved by an affine matrix (same size; uncovered pixels get ``fill``).

    Args:
        image: PIL image.
        matrix: 3 x 3 matrix from input to output pixel coordinates.
        resample: PIL resampling filter (bilinear for photos, nearest for masks).
        fill: value of the pixels the moved image does not cover.

    Returns:
        The moved image.
    """
    inverse = np.linalg.inv(matrix)
    data = tuple(float(v) for v in (*inverse[0], *inverse[1]))
    return image.transform(image.size, Image.Transform.AFFINE, data, resample=resample, fillcolor=fill)


def move_points(points: Array, matrix: Array) -> Array:
    """Points ``(N, 2)`` moved by an affine matrix.

    Args:
        points: ``x, y`` pixel coordinates.
        matrix: 3 x 3 affine matrix.

    Returns:
        The moved points ``(N, 2)``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def move_boxes(boxes: Array, matrix: Array, width: int, height: int) -> tuple[Array, Array]:
    """Boxes ``(N, 4)`` xyxy moved by an affine matrix: the extent of the moved corners, clipped.

    Args:
        boxes: boxes in pixels.
        matrix: 3 x 3 affine matrix.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        ``(boxes, keep)``: the moved boxes (float32) and which objects stay (at least one pixel a side and
        ``MIN_VISIBLE`` of the moved box inside the image).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
    moved = move_points(corners, matrix).reshape(-1, 4, 2)
    low, high = moved.min(axis=1), moved.max(axis=1)
    full = np.prod(high - low, axis=1)
    clipped_low = np.clip(low, 0, [width, height])
    clipped_high = np.clip(high, 0, [width, height])
    sides = clipped_high - clipped_low
    keep = (sides >= 1.0).all(axis=1) & (np.prod(sides, axis=1) >= MIN_VISIBLE * full)
    return np.concatenate([clipped_low, clipped_high], axis=1).astype(np.float32), keep


def mask_box(mask: Array) -> list[float] | None:
    """Box ``[x1, y1, x2, y2]`` around the pixels of a mask.

    Args:
        mask: ``(H, W)`` array, non-zero inside.

    Returns:
        The box in pixels, or ``None`` for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def move_keypoints(keypoints: Array, matrix: Array, width: int, height: int, swap: list[list[int]]) -> Array:
    """Keypoints ``(N, K, 3)`` moved by an affine matrix; points that leave the image become missing.

    Args:
        keypoints: absolute ``x, y, visibility`` (visibility 0 = missing).
        matrix: 3 x 3 affine matrix.
        width: image width in pixels.
        height: image height in pixels.
        swap: pairs of point indexes that trade places (a mirrored image), or empty.

    Returns:
        The moved keypoints (float32).
    """
    out = np.array(keypoints, dtype=np.float32, copy=True)
    if not out.size:
        return out
    visible = out[:, :, 2] > 0
    moved = move_points(out[:, :, :2].reshape(-1, 2), matrix).reshape(out.shape[0], out.shape[1], 2)
    inside = (moved[..., 0] >= 0) & (moved[..., 0] < width) & (moved[..., 1] >= 0) & (moved[..., 1] < height)
    out[:, :, :2] = moved
    out[~(visible & inside)] = 0.0
    for left, right in swap:
        out[:, [left, right]] = out[:, [right, left]]
    return out


def move_target(
    target: dict[str, Array], matrix: Array, width: int, height: int, swap: list[list[int]]
) -> dict[str, Array]:
    """A detection target moved by an affine matrix; objects that leave the image are dropped.

    With ``masks`` the boxes are taken from the moved masks (tight); otherwise from the moved corners.

    Args:
        target: ``boxes`` (N, 4), ``labels`` (N,) and maybe ``masks`` (N, H, W) or ``keypoints`` (N, K, 3).
        matrix: 3 x 3 affine matrix.
        width: image width in pixels.
        height: image height in pixels.
        swap: keypoint pairs that trade places (a mirrored image), or empty.

    Returns:
        The new target, of the same keys.
    """
    out = dict(target)
    if "masks" in target:
        masks = [
            np.asarray(warp(Image.fromarray(m), matrix, Image.Resampling.NEAREST, 0), dtype=np.uint8)
            for m in target["masks"]
        ]
        found = [mask_box(m) for m in masks]
        keep = np.array([b is not None for b in found], dtype=bool)
        boxes = np.array([b for b in found if b is not None], dtype=np.float32).reshape(-1, 4)
        out["masks"] = (
            np.stack([m for m, k in zip(masks, keep, strict=True) if k])
            if keep.any()
            else np.zeros((0, height, width), dtype=np.uint8)
        )
    else:
        boxes, keep = move_boxes(target["boxes"], matrix, width, height)
        boxes = boxes[keep]
    out["boxes"] = boxes
    out["labels"] = target["labels"][keep]
    if "keypoints" in target:
        out["keypoints"] = move_keypoints(target["keypoints"], matrix, width, height, swap)[keep]
    return out


def photometric(image: Any, entries: Entries, rng: Any = random) -> Any:
    """Colour, blur and grey techniques on a PIL image (labels do not change).

    Args:
        image: RGB PIL image.
        entries: the resolved techniques (the geometric ones are skipped).
        rng: random source with ``random`` and ``uniform``.

    Returns:
        The changed image.
    """
    for entry in entries:
        name, params = entry["name"], entry.get("params") or {}
        if name == "color_jitter":
            image = transforms.ColorJitter(
                brightness=params["brightness"],
                contrast=params["contrast"],
                saturation=params["saturation"],
                hue=params["hue"],
            )(image)
        elif name == "blur" and rng.random() < params["p"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, params["sigma_max"])))
        elif name == "grayscale" and rng.random() < params["p"]:
            image = image.convert("L").convert("RGB")
    return image


def check_objects(entries: Entries) -> None:
    """Refuse techniques the object workers cannot apply (they only make sense for classification).

    Args:
        entries: the resolved techniques.

    Raises:
        AugmentError: a classification-only or unknown technique.
    """
    known = (*GEOMETRIC, "color_jitter", "blur", "grayscale")
    for entry in entries:
        if entry["name"] not in known:
            raise AugmentError(f"This worker cannot apply the augmentation technique '{entry['name']}'.")


def flip_pairs(entries: Entries) -> list[list[int]]:
    """The keypoint pairs a horizontal flip swaps.

    Args:
        entries: the resolved techniques.

    Returns:
        The ``[left, right]`` pairs of the ``hflip`` entry, or an empty list.
    """
    return [list(p) for p in params_of(entries).get("hflip", {}).get("flip_pairs", [])]


def augment_objects(
    image: Any, target: dict[str, Array], entries: Entries, rng: Any = random
) -> tuple[Any, dict[str, Array]]:
    """Training augmentation of an image and its detection target (boxes, masks, keypoints move together).

    Args:
        image: RGB PIL image.
        target: target of numpy arrays in pixels (see ``move_target``).
        entries: the resolved techniques.
        rng: random source with ``random``, ``uniform`` and ``randint``.

    Returns:
        ``(image, target)`` augmented, the image keeping its size.

    Raises:
        AugmentError: a technique this worker cannot apply.
    """
    check_objects(entries)
    width, height = image.size
    matrix, mirrored = geometry(entries, width, height, rng)
    if not np.allclose(matrix, np.eye(3)):
        image = warp(image, matrix, Image.Resampling.BILINEAR, (0, 0, 0))
        target = move_target(target, matrix, width, height, flip_pairs(entries) if mirrored else [])
    return photometric(image, entries, rng), target


def augment_pair(image: Any, mask: Any, entries: Entries, rng: Any = random) -> tuple[Any, Any]:
    """Training augmentation of an image and its class mask (semantic segmentation), same size out.

    Args:
        image: RGB PIL image.
        mask: PIL class mask of the same size (0 is the background, which also fills uncovered pixels).
        entries: the resolved techniques.
        rng: random source with ``random``, ``uniform`` and ``randint``.

    Returns:
        ``(image, mask)`` augmented.

    Raises:
        AugmentError: a technique this worker cannot apply.
    """
    check_objects(entries)
    matrix, _ = geometry(entries, image.width, image.height, rng)
    if not np.allclose(matrix, np.eye(3)):
        image = warp(image, matrix, Image.Resampling.BILINEAR, (0, 0, 0))
        mask = warp(mask, matrix, Image.Resampling.NEAREST, 0)
    return photometric(image, entries, rng), mask
