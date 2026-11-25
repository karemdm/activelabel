"""Smoke test of the Ultralytics worker with the real package (not run by pytest).

Trains a tiny detection model from scratch (no download) for two epochs on the CPU and checks the events
and the files the runner expects. Run it where ultralytics is installed (the training environment)::

    python tests/smoke/yolo_smoke.py [path to a site-packages folder that has ultralytics]

An optional argument adds that folder to ``sys.path`` (read only); the script says "skipped" and exits 0 when
ultralytics cannot be imported.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "src" / "activelabel" / "training" / "workers" / "yolo.py"
PARAMS = {
    "epochs": 2,
    "batch_size": 4,
    "lr": 0.01,
    "optimizer": "sgd",
    "weight_decay": 0.0005,
    "scheduler": "cosine",
    "warmup_epochs": 0,
    "image_size": 64,
    "resize": "letterbox",
    "augmentation": "light",
    "patience": 0,
    "amp": "off",
    "seed": 1,
    "start": "scratch",
    "backbone_lr_factor": 0.1,
    "mosaic": True,
    "close_mosaic": 1,
}


def make_items(folder: Path, count: int) -> list[dict]:
    """Images with one bright square each, and its box as the label.

    Args:
        folder: where the images go.
        count: how many images.

    Returns:
        Spec items ``{"path", "objects"}``.
    """
    rng = np.random.default_rng(0)
    items = []
    for k in range(count):
        data = rng.integers(0, 60, size=(96, 128, 3), dtype=np.uint8)
        x, y = int(rng.integers(5, 70)), int(rng.integers(5, 40))
        data[y : y + 40, x : x + 40] = 230
        path = folder / f"{k}.png"
        Image.fromarray(data).save(path)
        box = [x / 128, y / 96, 40 / 128, 40 / 96]
        items.append({"path": str(path), "objects": [{"label": 0, "bbox": box}]})
    return items


def main(argv: list[str]) -> int:
    """Run the worker on a synthetic dataset and check what it wrote.

    Args:
        argv: command line; ``argv[1]`` may be a site-packages folder to read ultralytics from.

    Returns:
        0 when it passes or is skipped, 1 when a check fails.
    """
    if len(argv) > 1:
        sys.path.append(argv[1])
    if importlib.util.find_spec("ultralytics") is None:
        print("skipped: ultralytics is not installed")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        items = make_items(base, 12)
        spec = {
            "task": "detection", "model": "yolov8n", "family": "ultralytics", "start": "scratch",
            "params": PARAMS, "seed": 1, "metric": "map50_95", "classes": ["square"],
            "parts": {"train": items[:8], "val": items[8:]}, "output_dir": str(base / "out"),
            "device": "cpu", "augmentations": [{"name": "hflip", "params": {"p": 0.5}}], "digest": "smoke",
        }  # fmt: skip
        (base / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        cmd = [sys.executable, str(WORKER), str(base / "spec.json")]
        run = subprocess.run(
            cmd,
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(argv[1:2])},
        )
        events = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
        names = [e["event"] for e in events]
        epochs = [e for e in events if e["event"] == "epoch"]
        out = base / "out"
        problems = []
        if run.returncode != 0 or "error" in names:
            problems.append(f"exit {run.returncode}: {run.stdout[-800:]} {run.stderr[-800:]}")
        if [e["epoch"] for e in epochs] != [1, 2]:
            problems.append(f"epochs: {[e['epoch'] for e in epochs]}")
        for name in ("best.pt", "last.pt", "results.json"):
            if not (out / name).exists():
                problems.append(f"missing {name}")
        if epochs and not {"val_map50", "val_map50_95", "train_loss", "lr"} <= set(epochs[-1]):
            problems.append(f"epoch event fields: {sorted(epochs[-1])}")
    print(
        "FAILED: " + "; ".join(problems)
        if problems
        else f"ok: {len(epochs)} epochs, events {sorted(set(names))}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
