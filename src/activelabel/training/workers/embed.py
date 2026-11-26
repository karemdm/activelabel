"""Embeddings of the images for the similarity map of the curation (run by ActiveLabel in the training env).

Usage: ``python embed.py spec.json``. The spec has ``items`` (``[{"id", "path"}]``), ``backbone`` (a key of
``BACKBONES``), ``device`` (``auto``, ``cpu``, ``cuda:N`` or ``mps``), ``output`` (folder that receives
``vectors.npy``, one float32 row per embedded image, and ``ids.json``, the image id of each row) and
optionally ``batch_size``. torchvision downloads the ImageNet weights into the job's temporary folder
(``TORCH_HOME``), removed afterwards, like every download of a run.

The backbone is a small ImageNet network without its classifier: the image is resized to ``SIZE`` x ``SIZE``
(stretched, as only the content matters here), normalised with the ImageNet mean and deviation, and the pooled
features are the embedding, scaled to length 1 so that distances compare directions.

Prints JSON lines on stdout: ``progress`` (``step``/``steps``), ``skipped`` (``{"id", "message"}``) for a file
that cannot be read, then ``done`` (``{"images", "dims", "backbone", "unreadable"}``), or ``error``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms import functional as tf

try:
    from activelabel.training.workers import options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling file
    import options  # type: ignore[no-redef]

#: Backbone key -> (torchvision builder, weights enum name). Small networks: they run well on a CPU.
BACKBONES = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
}
DEFAULT_BACKBONE = "mobilenet_v3_small"
SIZE = 224
BATCH_SIZE = 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def emit(event: str, **fields: Any) -> None:
    """One JSON line for ActiveLabel.

    Args:
        event: Event name (``progress``, ``skipped``, ``done`` or ``error``).
        **fields: Extra fields of the JSON object.
    """
    print(json.dumps({"event": event, **fields}), flush=True)


def build_backbone(name: str) -> Any:
    """The ImageNet network ``name`` with its classifier replaced by an identity (it returns the features).

    Args:
        name: A key of ``BACKBONES``.

    Returns:
        The network, with its default ImageNet weights (downloaded by torchvision the first time).

    Raises:
        ValueError: ``name`` is not a known backbone.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'.")
    builder, weights = BACKBONES[name]
    model = getattr(torchvision.models, builder)(weights=getattr(torchvision.models, weights).DEFAULT)
    if hasattr(model, "fc"):  # ResNet
        model.fc = torch.nn.Identity()
    else:  # MobileNet: features -> pooling -> classifier
        model.classifier = torch.nn.Identity()
    return model


def to_tensor(image: Any) -> Any:
    """The network input of one RGB image.

    Args:
        image: A PIL image in RGB.

    Returns:
        A float tensor ``3 x SIZE x SIZE``, normalised with the ImageNet mean and deviation.
    """
    resized = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return tf.normalize(tf.to_tensor(resized), MEAN, STD)


def normalise(vectors: Any) -> Any:
    """Rows scaled to length 1 (a zero row stays zero).

    Args:
        vectors: A float array ``N x D``.

    Returns:
        The array with every non-zero row divided by its length.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def read_image(path: str) -> Any:
    """An image file as RGB, turned as it is shown (EXIF rotation applied).

    Args:
        path: The image file.

    Returns:
        The PIL image in RGB.
    """
    with Image.open(path) as opened:
        return (ImageOps.exif_transpose(opened) or opened).convert("RGB")


def embed(spec: dict[str, Any]) -> dict[str, Any]:
    """Embed every readable item and write ``vectors.npy`` and ``ids.json`` into ``spec["output"]``.

    Args:
        spec: The job spec (see the module docstring).

    Returns:
        Totals ``{"images", "dims", "backbone", "unreadable"}``.

    Raises:
        ValueError: No image could be read.
    """
    name = str(spec.get("backbone") or DEFAULT_BACKBONE)
    model = build_backbone(name)
    device = options.pick_device(torch, str(spec.get("device") or "auto"))
    model.to(device).eval()
    items = spec["items"]
    batch_size = int(spec.get("batch_size") or BATCH_SIZE)
    ids: list[str] = []
    rows: list[Any] = []
    batch: list[Any] = []
    batch_ids: list[str] = []
    unreadable = 0
    with torch.no_grad():
        for step, item in enumerate(items, start=1):
            try:
                batch.append(to_tensor(read_image(item["path"])))
                batch_ids.append(str(item["id"]))
            except (OSError, ValueError) as exc:  # a corrupt or missing file: the others go on
                unreadable += 1
                emit("skipped", id=item["id"], message=str(exc) or type(exc).__name__)
            if batch and (len(batch) == batch_size or step == len(items)):
                features = model(torch.stack(batch).to(device))
                rows.append(features.flatten(1).cpu().numpy().astype(np.float32))
                ids.extend(batch_ids)
                batch, batch_ids = [], []
            emit("progress", step=step, steps=len(items))
    if not ids:
        raise ValueError("None of the images could be read.")
    vectors = normalise(np.concatenate(rows))
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "vectors.npy", vectors)
    (output / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return {"images": len(ids), "dims": int(vectors.shape[1]), "backbone": name, "unreadable": unreadable}


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command line; ``argv[1]`` is the path of the spec JSON file.

    Returns:
        Process exit code: 0 on success, 1 after an ``error`` event.
    """
    try:
        spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        totals = embed(spec)
    except Exception as exc:  # noqa: BLE001 - reported to ActiveLabel as an event
        emit("error", message=str(exc) or type(exc).__name__)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=totals)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
