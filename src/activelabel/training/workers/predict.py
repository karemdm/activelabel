"""Labels from a saved model for new images (run by ActiveLabel in the training environment).

Usage: ``python predict.py spec.json``. The spec has ``checkpoint`` (``best.pt``), ``worker`` (the
training worker module that builds the network: ``classification`` or ``detection``), ``task``
(``classification``, ``detection`` or ``keypoints``), ``build`` (what that worker's ``build_model``
reads), ``classes`` (in the order of the network's outputs), ``threshold`` (lowest confidence kept),
``model`` (the saved model's name, e.g. ``m1``), ``items`` (``[{"id", "path"}]``) and ``device`` (``auto``,
``cpu``, ``cuda:N`` or ``mps``; ``auto`` when absent). For a public
pretrained model the spec has ``public`` (``builder``, ``categories``, ``mapping``, ``task``) instead
of ``checkpoint``, ``worker`` and ``build``: torchvision downloads the weights, and categories missing
from the mapping are dropped.

Prints JSON lines on stdout: ``progress`` (``step``/``steps``), one ``prediction`` per image
(``{"id", "labels"}`` in ActiveLabel's label format, coordinates relative to the image), then
``done`` (``{"images", "objects"}``), or ``error``. Instance and semantic segmentation models give
every label a ``mask`` (``size`` and run-length ``counts`` of a binary mask of at most
``MAX_MASK_SIDE`` pixels on its longer side, relative to the whole image): this process has no
OpenCV, ActiveLabel turns the masks into polygons. Preprocessing is the one used to evaluate the
model in training: the resize policy saved with the weights (``options.py``) brings the photo to the model's
input size, with ImageNet normalisation for classification and segmentation and values 0..1 for detection;
boxes, keypoints and masks are taken back to the photo with the same policy. A model saved before the policy
existed is read as it was trained: stretched (classification, segmentation), or the photo as it is
(detection, which resized inside the network), like a public model.
"""

from __future__ import annotations

import importlib
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
from torchvision.transforms.functional import pil_to_tensor

try:
    from activelabel.training.workers import options
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling file
    import options  # type: ignore[no-redef]

KINDS = ("classification", "detection", "keypoints", "instance", "semantic")
DETECTION_KINDS = ("detection", "keypoints", "instance")
MAX_MASK_SIDE = 512
MASK_THRESHOLD = 0.5  # instance mask probability that counts as inside the object


def emit(event: str, **fields: Any) -> None:
    """One JSON line for ActiveLabel.

    Args:
        event: Event name (``progress``, ``prediction``, ``skipped``, ``done`` or ``error``).
        **fields: Extra fields of the JSON object.
    """
    print(json.dumps({"event": event, **fields}), flush=True)


def _clamp(value: float) -> float:
    """``value`` limited to the range 0..1.

    Args:
        value: Any number.

    Returns:
        The number clamped to 0..1.
    """
    return min(1.0, max(0.0, value))


def classification_labels(
    probabilities: list[float], classes: list[str], threshold: float, model: str
) -> list[dict[str, Any]]:
    """The most likely class when it reaches ``threshold`` (else no label).

    Args:
        probabilities: Probability of each class, in the order of ``classes``.
        classes: Class names, in the order of the network's outputs.
        threshold: Lowest confidence kept.
        model: Name of the saved model, stored in the label.

    Returns:
        A list with one label (``label``, ``score``, ``model``), or an empty list.
    """
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    score = float(probabilities[best])
    if score < threshold or best >= len(classes):
        return []
    return [{"label": classes[best], "score": round(score, 4), "model": model}]


def encode_mask(mask: Any) -> dict[str, Any]:
    """Run-length form of a 2-D boolean array: ``{"size": [h, w], "counts": [...]}``.

    ``counts`` are the lengths of the alternating runs of the flattened (row by row) mask, the first
    one of zeros (possibly 0 long).

    Args:
        mask: 2-D boolean array (anything ``numpy`` converts to one; needs a ``shape``).

    Returns:
        ``{"size": [h, w], "counts": [...]}``.
    """
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(flat)]))
    counts = np.diff(bounds).tolist()
    if len(flat) and flat[0]:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_step(width: int, height: int) -> int:
    """Subsampling step that brings the longer side of a mask to at most ``MAX_MASK_SIDE``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The step, at least 1.
    """
    return max(1, -(-max(width, height) // MAX_MASK_SIDE))


def semantic_labels(
    classes_map: Any, confidence: Any, classes: list[str | None], threshold: float, model: str
) -> list[dict[str, Any]]:
    """One label per class found in a per-pixel class map (0 is background, then the project's classes).

    The score of a class is the mean confidence of its pixels; classes below ``threshold`` are left
    out. The label has only a mask: ActiveLabel splits it into one polygon object per region.

    Args:
        classes_map: 2-D array with the class index of each pixel.
        confidence: 2-D array, same shape, with the confidence of each pixel.
        classes: Project class names (index 0 of the list is class 1 of the map); ``None`` marks a
            category the project does not use.
        threshold: Lowest mean confidence kept.
        model: Name of the saved model, stored in each label.

    Returns:
        A list of labels (``label``, ``score``, ``model``, ``mask``).
    """
    out = []
    for index in np.unique(classes_map):
        if index < 1 or index > len(classes) or classes[int(index) - 1] is None:
            continue
        inside = classes_map == index
        score = float(confidence[inside].mean())
        if score >= threshold:
            out.append(
                {
                    "label": classes[int(index) - 1],
                    "score": round(score, 4),
                    "model": model,
                    "mask": encode_mask(inside),
                }
            )
    return out


def detection_labels(
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    width: int,
    height: int,
    classes: list[str | None],
    threshold: float,
    model: str,
    keypoints: list[list[list[float]]] | None = None,
    masks: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Objects at or above ``threshold``: relative box ``[x, y, w, h]``, class name (0 is background).

    Args:
        boxes: Boxes ``[x1, y1, x2, y2]`` in pixels.
        labels: Class index of each box (0 is background).
        scores: Confidence of each box.
        width: Image width in pixels.
        height: Image height in pixels.
        classes: Project class names (index 0 of the list is class 1); ``None`` marks a category the
            project does not use.
        threshold: Lowest confidence kept.
        model: Name of the saved model, stored in each label.
        keypoints: Per detection, a list of ``[x, y, visibility]`` points in pixels (optional).
        masks: One boolean 2-D array per detection, any resolution; adds a ``mask`` to each object
            (optional).

    Returns:
        A list of labels with ``label``, ``bbox``, ``score``, ``model`` and, when given, ``keypoints``
        and ``mask``.
    """
    out = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=True)):
        if score < threshold or not 1 <= int(label) <= len(classes) or classes[int(label) - 1] is None:
            continue  # below the threshold, background, or a category the project does not use
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
        if masks is not None:
            item["mask"] = encode_mask(masks[i])
        out.append(item)
    return out


def instance_masks(probabilities: Any, width: int, height: int) -> list[Any]:
    """Boolean masks (reduced to at most ``MAX_MASK_SIDE`` pixels) of the ``(N, 1, H, W)`` mask output.

    Args:
        probabilities: Tensor of mask probabilities, shape ``(N, 1, H, W)``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One 2-D boolean array per detection.
    """
    step = mask_step(width, height)
    return [m[::step, ::step] for m in (probabilities[:, 0] > MASK_THRESHOLD).cpu().numpy()]


def semantic_map(
    model: Any,
    image: Any,
    size: int,
    mean: Any,
    std: Any,
    device: str,
    policy: str = options.LEGACY_RESIZE,
) -> tuple[Any, Any]:
    """Per-pixel ``(class map, confidence)`` of an image, at most ``MAX_MASK_SIDE`` pixels wide.

    The image is resized to the training size by the resize policy, like in evaluation, and the logits are
    taken back to the (reduced) size of the photo before the argmax (pixels a centre crop cut are background).

    Args:
        model: The segmentation network, already on ``device`` and in eval mode.
        image: RGB ``PIL`` image.
        size: Side of the square the image is resized to for the network.
        mean: Per-channel mean used to normalise the image.
        std: Per-channel standard deviation used to normalise the image.
        device: Torch device name the model is on.
        policy: The resize policy the model was trained with (one of ``options.RESIZE_POLICIES``).

    Returns:
        ``(class map, confidence)`` as 2-D ``numpy`` arrays.
    """
    width, height = image.size
    step = mask_step(width, height)
    small = options.fit_image(image, size, policy)
    array = (np.asarray(small, dtype="float32") / 255.0 - np.array(mean, dtype="float32")) / np.array(
        std, dtype="float32"
    )
    batch = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    logits = model(batch)["out"]
    logits = options.unfit_map(logits, width, height, size, policy, (-(-width // step), -(-height // step)))
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence, classes_map = probabilities.max(0)
    return classes_map.cpu().numpy(), confidence.cpu().numpy()


def detect(model: Any, image: Any, task: str, device: str, size: int = 0, policy: str = "") -> dict[str, Any]:
    """Run a detector on a photo; boxes, keypoints and masks come back in photo pixels.

    Args:
        model: The detection network, already on ``device`` and in eval mode.
        image: RGB ``PIL`` image.
        task: ``detection``, ``keypoints`` or ``instance``.
        device: Torch device name the model is on.
        size: Side of the square input; 0 gives the photo as it is (the network resizes it).
        policy: The resize policy the model was trained with, used when ``size`` is given.

    Returns:
        ``boxes`` (``[x1, y1, x2, y2]``), ``labels`` and ``scores`` lists, ``keypoints`` (per detection, or
        ``None``) and ``masks`` (boolean arrays of at most ``MAX_MASK_SIDE`` pixels, or ``None``).
    """
    width, height = image.size
    shown = options.fit_image(image, size, policy) if size else image
    out = model([(pil_to_tensor(shown).float() / 255.0).to(device)])[0]
    boxes = out["boxes"].cpu().numpy()
    points = out["keypoints"].cpu().numpy() if task == "keypoints" and "keypoints" in out else None
    masks = None
    if size:
        boxes = options.unfit_boxes(boxes, width, height, size, policy)
        points = None if points is None else options.unfit_points(points, width, height, size, policy)
    if task == "instance":
        if size:
            step = mask_step(width, height)
            grid = (-(-width // step), -(-height // step))
            back = options.unfit_map(out["masks"], width, height, size, policy, grid)
            masks = list((back[:, 0] > MASK_THRESHOLD).cpu().numpy())
        else:
            masks = instance_masks(out["masks"], width, height)
    return {
        "boxes": boxes.tolist(),
        "labels": out["labels"].cpu().tolist(),
        "scores": out["scores"].cpu().tolist(),
        "keypoints": None if points is None else points.tolist(),
        "masks": masks,
    }


def predict(spec: dict[str, Any]) -> dict[str, int]:
    """Run the model over every item, emitting one ``prediction`` per image.

    Args:
        spec: The job spec (see the module docstring).

    Returns:
        Totals ``{"images", "objects", "unreadable"}``: images with at least one label, labels, and
        files that could not be read.

    Raises:
        ValueError: The task is not supported, or the weights do not match the network.
    """
    task = spec["task"]
    if task not in KINDS:
        raise ValueError(f"Pre-annotation with {task} models is not available yet.")
    public = spec.get("public")
    checkpoint: dict[str, Any] = {}
    if public:
        # a public pretrained model: torchvision downloads its weights (into the run's temporary
        # folder, removed afterwards); its categories become project classes through the mapping

        model = getattr(torchvision.models.detection, public["builder"])(weights="DEFAULT")
        classes: list[str | None] = [public["mapping"].get(c) for c in public["categories"][1:]]
    else:
        worker = importlib.import_module(spec["worker"])
        # weights_only=False: best.pt is written by ActiveLabel's own training (never a downloaded file)
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        build = {**spec["build"], "num_keypoints": checkpoint.get("num_keypoints") or 0}
        built = worker.build_model(build, False)
        model = built[0] if isinstance(built, tuple) else built
        result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        bad = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
        if bad or [k for k in result.unexpected_keys if not k.startswith("aux_classifier.")]:
            raise ValueError("The weights do not match the network.")
        classes = list(spec["classes"])
    device = options.pick_device(torch, str(spec.get("device") or "auto"))
    model.to(device).eval()
    threshold, name = float(spec["threshold"]), str(spec["model"])
    if task in DETECTION_KINDS:
        for part in ("roi_heads", "head"):  # keep candidates down to the threshold (the default is 0.05)
            module = getattr(model, part, None)
            if module is not None and hasattr(module, "score_thresh"):
                module.score_thresh = min(float(module.score_thresh), threshold)
        if hasattr(model, "score_thresh"):
            model.score_thresh = min(float(model.score_thresh), threshold)
    # the resize policy saved with the weights; a model saved before it existed has none
    policy = str(checkpoint.get("resize") or "")
    size = int(checkpoint.get("image_size") or 0) if policy else 0
    if task not in DETECTION_KINDS:
        size = int(checkpoint.get("image_size") or spec["build"]["params"]["image_size"])
        policy = policy or options.LEGACY_RESIZE
        if task == "classification":
            transform = worker.eval_transform(size, policy)
    items = spec["items"]
    totals = {"images": 0, "objects": 0, "unreadable": 0}
    with torch.no_grad():
        for step, item in enumerate(items, start=1):
            try:
                with Image.open(item["path"]) as opened:
                    # boxes must fit the photo as shown (EXIF rotation applied), like the labels
                    image = (ImageOps.exif_transpose(opened) or opened).convert("RGB")
            except (OSError, ValueError) as exc:  # a corrupt or missing file: the others go on
                totals["unreadable"] += 1
                emit("skipped", id=item["id"], message=str(exc) or type(exc).__name__)
                emit("progress", step=step, steps=len(items))
                continue
            width, height = image.size
            if task == "semantic":
                classes_map, confidence = semantic_map(
                    model,
                    image,
                    size,
                    checkpoint.get("mean") or worker.MEAN,
                    checkpoint.get("std") or worker.STD,
                    device,
                    policy,
                )
                labels = semantic_labels(classes_map, confidence, classes, threshold, name)
            elif task in DETECTION_KINDS:
                found = detect(model, image, task, device, size, policy)
                labels = detection_labels(
                    found["boxes"],
                    found["labels"],
                    found["scores"],
                    width,
                    height,
                    classes,
                    threshold,
                    name,
                    found["keypoints"],
                    found["masks"],
                )
            else:
                logits = model(transform(image).unsqueeze(0).to(device))
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
                labels = classification_labels(probabilities, [c or "" for c in classes], threshold, name)
            totals["images"] += 1 if labels else 0
            totals["objects"] += len(labels)
            emit("prediction", id=item["id"], labels=labels)
            emit("progress", step=step, steps=len(items))
    return totals


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command line; ``argv[1]`` is the path of the spec JSON file.

    Returns:
        Process exit code: 0 on success, 1 after an ``error`` event.
    """
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
