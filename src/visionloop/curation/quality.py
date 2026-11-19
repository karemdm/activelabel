"""Measures of image quality and a perceptual hash for near-duplicate detection.

Metrics stored in ``ImageRecord.metrics``:

* ``brightness``: mean of the gray image (0..255).
* ``contrast``: standard deviation of the gray image.
* ``sharpness``: variance of the Laplacian of the gray image reduced to at most 512 px on its
  longest side (so the value does not depend on the resolution).
* ``colorfulness``: mean absolute difference between color channels (0 for gray images).
* ``grayscale``: 1.0 when the channels are practically equal, else 0.0.

The 64-bit perceptual hash (DCT based, hex) goes to ``ImageRecord.phash``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from visionloop.images.manifest import ImageRecord, Manifest
from visionloop.jobs import Progress

MAX_SIDE = 512
GRAYSCALE_TOLERANCE = 2.0  # mean channel difference (0..255) below which an image counts as gray
BATCH = 64

_HASH_SIZE = 32
_DCT = np.array(
    [
        [np.cos(np.pi * (2 * x + 1) * u / (2 * _HASH_SIZE)) for x in range(_HASH_SIZE)]
        for u in range(_HASH_SIZE)
    ]
)


def perceptual_hash(gray: np.ndarray) -> str:
    """64-bit DCT perceptual hash of a gray image (2-D array), as 16 hex digits."""
    small = Image.fromarray(gray.astype(np.uint8)).resize((_HASH_SIZE, _HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:8, :8]
    median = np.median(low.flatten()[1:])  # the DC term would dominate the median
    bits = (low.flatten() > median).astype(np.uint8)
    return f"{int(''.join(map(str, bits)), 2):016x}"


def hamming(a: str, b: str) -> int:
    """Number of different bits between two hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def measure_file(path: Path) -> tuple[dict[str, float], str]:
    """Metrics and perceptual hash of one image file.

    Raises:
        OSError: the file cannot be read as an image.
    """
    with Image.open(path) as image:
        image.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float64)
    gray = array @ np.array([0.299, 0.587, 0.114])
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    colorfulness = float((np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3)
    laplacian = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        if min(gray.shape) >= 3
        else np.zeros((1, 1))
    )
    metrics = {
        "brightness": round(float(gray.mean()), 3),
        "contrast": round(float(gray.std()), 3),
        "sharpness": round(float(laplacian.var()), 3),
        "colorfulness": round(colorfulness, 3),
        "grayscale": 1.0 if colorfulness < GRAYSCALE_TOLERANCE else 0.0,
    }
    return metrics, perceptual_hash(gray)


def measure_images(
    manifest: Manifest, progress: Progress | None = None, recompute: bool = False, workers: int = 4
) -> int:
    """Measure active images that have no metrics yet (all of them with ``recompute``).

    Results are saved every few images, so a cancelled run keeps what it measured. Files that
    cannot be read are skipped (they stay unmeasured).

    Args:
        manifest: images of the project.
        progress: reports progress; its ``update`` raises ``Cancelled`` when the user cancels.
        recompute: measure again images that already have metrics.
        workers: threads.

    Returns:
        How many images were measured.

    Raises:
        Cancelled: the user cancelled (after saving what was measured).
    """
    todo = [r for r in manifest.active() if recompute or not r.metrics or not r.phash]
    if progress:
        progress.update(done=0, total=len(todo), message="Measuring images")
    measured = 0

    def work(record: ImageRecord) -> tuple[str, dict[str, Any] | None]:
        try:
            metrics, phash = measure_file(manifest.file_path(record))
        except (OSError, ValueError):
            return record.id, None
        return record.id, {"metrics": metrics, "phash": phash}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="visionloop-measure") as pool:
        for start in range(0, len(todo), BATCH):
            if progress:
                progress.check()
            chunk = todo[start : start + BATCH]
            results = list(pool.map(work, chunk))
            changes = {image_id: values for image_id, values in results if values is not None}
            manifest.update(changes)
            measured += len(changes)
            if progress:
                progress.update(done=min(start + BATCH, len(todo)))
    return measured
