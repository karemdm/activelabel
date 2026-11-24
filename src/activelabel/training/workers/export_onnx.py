"""Export the weights of a saved model to ONNX (run by ActiveLabel in the training environment).

Usage: ``python export_onnx.py spec.json``. The spec has ``checkpoint`` (``best.pt``), ``output``
(``model.onnx``), ``worker`` (the training worker module that builds the network:
``classification``, ``detection`` or ``segmentation``), ``task`` (kind of task) and ``build`` (the
spec that worker's ``build_model`` reads: ``model``, ``family``, ``classes``, ``params``, ``start``
and, for keypoints, ``num_keypoints``).

The network is rebuilt without downloading anything and the weights are loaded into it. Inputs and
outputs of the ONNX graph:

- classification: ``images`` (N×3×S×S, normalised with the ImageNet mean and std) → ``logits``;
- semantic segmentation: ``images`` → ``logits`` (N×(classes+1)×S×S, class 0 is the background).

Detection models are not exported yet: the traced torchvision detectors break in ONNX Runtime on
images without candidate boxes.

Prints JSON lines on stdout: ``{"event": "log"|"done"|"error", ...}``.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

try:
    import onnx
except ImportError:  # the export installs onnx on demand; report it as an error event
    onnx = None

OPSET = 17
KINDS = ("classification", "semantic")  # detection graphs break in ONNX Runtime: later


def emit(event: str, **fields: Any) -> None:
    """One JSON line for ActiveLabel."""
    print(json.dumps({"event": event, **fields}), flush=True)


def check_keys(missing: list[str], unexpected: list[str]) -> None:
    """Accept only the differences between a pretrained and a plain build of the same network.

    A pretrained torchvision detector uses frozen batch norm (no ``num_batches_tracked``) and a
    pretrained segmentation model has an auxiliary classifier, which inference does not use.

    Raises:
        ValueError: any other missing or unexpected weight.
    """
    bad_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    bad_unexpected = [k for k in unexpected if not k.startswith("aux_classifier.")]
    if bad_missing or bad_unexpected:
        sample = ", ".join((bad_missing + bad_unexpected)[:5])
        raise ValueError(f"The weights do not match the network ({sample}).")


class Logits(torch.nn.Module):
    """A torchvision segmentation model that returns only its logits (ONNX wants a plain tensor)."""

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: Any) -> Any:
        """The ``out`` logits of the wrapped model for the images ``x``."""
        return self.inner(x)["out"]


def export(spec: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the network, load the weights and write the ONNX file. Returns ``{"size", "inputs"}``."""
    task = spec["task"]
    if task not in KINDS:
        raise ValueError(f"ONNX export of {task} models is not available yet.")
    worker = importlib.import_module(spec["worker"])
    # weights_only=False: best.pt is written by ActiveLabel's own training (never a downloaded file)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    build = {**spec["build"], "num_keypoints": checkpoint.get("num_keypoints") or 0}
    built = worker.build_model(build, False)
    model = built[0] if isinstance(built, tuple) else built
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    check_keys(list(result.missing_keys), list(result.unexpected_keys))
    model.eval()
    size = int(checkpoint.get("image_size") or spec["build"]["params"].get("image_size") or 224)
    out = Path(spec["output"])
    tmp = out.with_suffix(".part")
    emit("log", message=f"exporting {spec['build']['model']} ({task}) at {size}px")
    net = model
    if task == "semantic":
        net = Logits(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            net,
            (torch.zeros(1, 3, size, size),),
            str(tmp),
            dynamo=False,
            opset_version=OPSET,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )
    inputs = {"images": ["batch", "3", str(size), str(size)]}
    if onnx is None:
        raise ImportError("onnx is not installed in the training environment.")
    onnx.checker.check_model(str(tmp))
    os.replace(tmp, out)
    return {"size": out.stat().st_size, "inputs": inputs, "opset": OPSET, "image_size": size}


def main(argv: list[str]) -> int:
    """Entry point."""
    try:
        spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        info = export(spec)
    except Exception as exc:  # noqa: BLE001 - reported to ActiveLabel as an event
        emit("error", message=str(exc) or type(exc).__name__)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=info)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
