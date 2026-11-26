"""Pretrained weights a job downloaded, measured and deleted when the job ends (docs/TRAINING.md, #643).

Every training run, pre-annotation and export works in its own temporary folder under
``<workspace>/training/tmp``, and the worker's download caches point inside it (``TORCH_HOME``,
``HF_HOME`` for timm and Hugging Face, ``XDG_CACHE_HOME``, ``YOLO_CONFIG_DIR`` and the working folder,
where Ultralytics saves its weights: see ``runner.download_env``). So whatever was downloaded goes away
with the folder, whatever the outcome; the trained weights (``best.pt``, ``last.pt``) are written to the
run's own folder, never to the temporary one, and caches outside ActiveLabel are never touched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

#: File endings of model weights (PyTorch, safetensors, Hugging Face binaries, checkpoints).
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin", ".ckpt")


def downloaded(tmp: Path) -> list[dict[str, Any]]:
    """The weight files inside a job's temporary folder.

    Args:
        tmp: the temporary folder of the job.

    Returns:
        ``{name, bytes}`` per file (``name`` relative to ``tmp``, with ``/``), sorted by name; empty when the
        folder does not exist.
    """
    if not tmp.is_dir():
        return []
    files = []
    for path in tmp.rglob("*"):
        try:
            if path.suffix.lower() in WEIGHT_SUFFIXES and path.is_file():
                files.append({"name": path.relative_to(tmp).as_posix(), "bytes": path.stat().st_size})
        except OSError:
            continue  # removed meanwhile or unreadable: not counted
    return sorted(files, key=lambda f: str(f["name"]))


def clean(tmp: Path, remove: Callable[[Path], None]) -> dict[str, Any]:
    """Measure the downloaded weights of a job, then delete its temporary folder.

    Args:
        tmp: the temporary folder of the job.
        remove: deletes a folder tree (``environment.remove_tree``: it logs instead of raising).

    Returns:
        ``{files, bytes, deleted}``: the weight files found, their total size and whether the folder is gone.
    """
    files = downloaded(tmp)
    remove(tmp)
    return {"files": files, "bytes": sum(int(f["bytes"]) for f in files), "deleted": not tmp.exists()}
