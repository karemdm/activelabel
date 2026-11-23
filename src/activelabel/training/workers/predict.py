"""Labels from a saved model for new images (run by ActiveLabel in the training environment).

Usage: ``python predict.py spec.json``. The spec has ``checkpoint`` (``best.pt``), ``worker`` (the
training worker module that builds the network: ``classification`` or ``detection``), ``task``
(``classification``, ``detection`` or ``keypoints``), ``build`` (what that worker's ``build_model``
reads), ``classes`` (in the order of the network's outputs), ``threshold`` (lowest confidence kept),
``model`` (the saved model's name, e.g. ``m1``) and ``items`` (``[{"id", "path"}]``).

Prints JSON lines on stdout: ``progress`` (``step``/``steps``), one ``prediction`` per image
(``{"id", "labels"}`` in ActiveLabel's label format, coordinates relative to the image), then
``done`` (``{"images", "objects"}``), or ``error``. Preprocessing is the one used to evaluate the
model in training: resize to the model's input size and ImageNet normalisation for classification,
the image as it is (values 0..1) for detection, which resizes inside the network.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

KINDS = ("classification", "detection", "keypoints")
DETECTION_KINDS = ("detection", "keypoints")


def emit(event: str, **fields: Any) -> None:
    """One JSON line for ActiveLabel."""
    print(json.dumps({"event": event, **fields}), flush=True)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def classification_labels(
    probabilities: list[float], classes: list[str], threshold: float, model: str
) -> list[dict[str, Any]]:
    """The most likely class when it reaches ``threshold`` (else no label)."""
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    score = float(probabilities[best])
    if score < threshold or best >= len(classes):
        return []
    return [{"label": classes[best], "score": round(score, 4), "model": model}]


def detection_labels(
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    width: int,
    height: int,
    classes: list[str],
    threshold: float,
    model: str,
    keypoints: list[list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Objects at or above ``threshold``: relative box ``[x, y, w, h]``, class name (0 is background)."""
    out = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=True)):
        if score < threshold or not 1 <= int(label) <= len(classes):
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x, y = _clamp(x1 / width), _clamp(y1 / height)
        w, h = _clamp(x2 / width) - x, _clamp(y2 / height) - y
        if w <= 0 or h <= 0:
            continue
        item: dict[str, Any] = {
            "label": classes[int(label) - 1],
            "bbox": [round(x, 6), round(y, 6), round(w, 6), round(h, 6)],
            "score": round(float(score), 4),
            "model": model,
        }
        if keypoints is not None:
            item["keypoints"] = [
                [
                    round(_clamp(p[0] / width), 6),
                    round(_clamp(p[1] / height), 6),
                    2 if (p[2] if len(p) > 2 else 1) > 0 else 0,
                ]
                for p in keypoints[i]
            ]
        out.append(item)
    return out


def predict(spec: dict[str, Any]) -> dict[str, int]:
    """Run the model over every item, emitting one ``prediction`` per image."""
    import torch
    from PIL import Image

    task = spec["task"]
    if task not in KINDS:
        raise ValueError(f"Pre-annotation with {task} models is not available yet.")
    worker = importlib.import_module(spec["worker"])
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    build = {**spec["build"], "num_keypoints": checkpoint.get("num_keypoints") or 0}
    built = worker.build_model(build, False)
    model = built[0] if isinstance(built, tuple) else built
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    bad = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    if bad or [k for k in result.unexpected_keys if not k.startswith("aux_classifier.")]:
        raise ValueError("The weights do not match the network.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    classes, threshold, name = list(spec["classes"]), float(spec["threshold"]), str(spec["model"])
    if task in DETECTION_KINDS:
        for part in ("roi_heads", "head"):  # keep candidates down to the threshold (the default is 0.05)
            module = getattr(model, part, None)
            if module is not None and hasattr(module, "score_thresh"):
                module.score_thresh = min(float(module.score_thresh), threshold)
        if hasattr(model, "score_thresh"):
            model.score_thresh = min(float(model.score_thresh), threshold)
    else:
        size = int(checkpoint.get("image_size") or build["params"]["image_size"])
        transform = worker.build_transforms(worker.eval_plan(size))
    items = spec["items"]
    totals = {"images": 0, "objects": 0, "unreadable": 0}
    with torch.no_grad():
        for step, item in enumerate(items, start=1):
            try:
                with Image.open(item["path"]) as opened:
                    image = opened.convert("RGB")
            except (OSError, ValueError) as exc:  # a corrupt or missing file: the others go on
                totals["unreadable"] += 1
                emit("skipped", id=item["id"], message=str(exc) or type(exc).__name__)
                emit("progress", step=step, steps=len(items))
                continue
            width, height = image.size
            if task in DETECTION_KINDS:
                from torchvision.transforms.functional import pil_to_tensor

                tensor = (pil_to_tensor(image).float() / 255.0).to(device)
                out = model([tensor])[0]
                labels = detection_labels(
                    out["boxes"].cpu().tolist(),
                    out["labels"].cpu().tolist(),
                    out["scores"].cpu().tolist(),
                    width,
                    height,
                    classes,
                    threshold,
                    name,
                    out["keypoints"].cpu().tolist() if task == "keypoints" and "keypoints" in out else None,
                )
            else:
                logits = model(transform(image).unsqueeze(0).to(device))
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
                labels = classification_labels(probabilities, classes, threshold, name)
            totals["images"] += 1 if labels else 0
            totals["objects"] += len(labels)
            emit("prediction", id=item["id"], labels=labels)
            emit("progress", step=step, steps=len(items))
    return totals


def main(argv: list[str]) -> int:
    """Entry point."""
    try:
        spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        totals = predict(spec)
    except Exception as exc:  # noqa: BLE001 - reported to ActiveLabel as an event
        emit("error", message=str(exc) or type(exc).__name__)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=totals)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
