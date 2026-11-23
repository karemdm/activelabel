"""Small synthetic videos for tests."""

from pathlib import Path

import cv2
import numpy as np


def make_video(path: Path, frames: int = 60, fps: float = 30.0, same: bool = False) -> Path:
    """Write a 64x48 video; each frame has its own colour unless ``same``. Returns the path.

    Tries mp4v in .mp4 and falls back to MJPG in .avi when the platform lacks the first codec.
    """
    for suffix, codec in ((path.suffix or ".mp4", "mp4v"), (".avi", "MJPG")):
        target = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), fps, (64, 48))
        if not writer.isOpened():
            continue
        for k in range(frames):
            value = 100 if same else (k * 4) % 256
            image = np.full((48, 64, 3), (value, 255 - value, (k * 7) % 256), np.uint8)
            cv2.putText(image, str(k), (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(image)
        writer.release()
        return target
    raise RuntimeError("OpenCV cannot write a test video here")
