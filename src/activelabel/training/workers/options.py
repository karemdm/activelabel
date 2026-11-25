"""Training options shared by the workers and the prediction and export processes (docs/TRAINING.md).

The resize policy turns a photo of any shape into the square input of the network, the same way in training,
evaluation, pre-annotation and ONNX export:

- ``stretch``: both sides scaled to the square (the proportions change);
- ``letterbox``: the longer side fits, the rest is filled (the proportions stay, nothing is cut);
- ``center_crop``: the shorter side fits, the centre is kept (the proportions stay, the edges are cut).

Each policy is one scale per axis plus a shift (an affine matrix), so boxes, masks and keypoints go into the
network and come back to the photo with the same numbers. Standalone like the workers: it runs in the training
environment, which has no ActiveLabel.
"""

from __future__ import annotations

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
