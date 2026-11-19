"""Small synthetic images for tests (no downloads)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_image(path: Path, size=(64, 48), seed=0, brightness=128, blur=0.0, fmt=None) -> Path:
    """Write a textured image (random squares). ``blur`` > 0 makes it blurry, low ``brightness`` dark."""
    rng = np.random.default_rng(seed)
    width, height = size
    data = np.full((height, width, 3), brightness, dtype=np.int16)
    for _ in range(12):
        x, y = rng.integers(0, width - 8), rng.integers(0, height - 8)
        data[y : y + 8, x : x + 8] += rng.integers(-60, 60, size=3)
    image = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=fmt)
    return path


def make_corrupt(path: Path) -> Path:
    """Write a file named like a JPEG that is not a valid image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return path
