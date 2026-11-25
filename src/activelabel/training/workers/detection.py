"""Detection, instance segmentation and keypoints worker (torchvision): ``python detection.py spec.json``.

Standalone, like ``classification.py``: it imports the standard library and numpy at the top and
torch, torchvision and PIL only inside the functions that need them, so it runs in the training
environment (which has no ActiveLabel) and its pure helpers (target conversion, IoU, COCO-style AP,
spec parsing) can be tested without torch.

Spec ``parts`` items are ``{"path", "objects": [{"label", "bbox", "polygons", "keypoints"}]}`` with
relative coordinates (ActiveLabel labels, ``labels.py``); ``label`` is the 0-based class index (the model
gets class + 1,
0 being the background). An image without objects is a background image (an empty target).

Output: one JSON object per line on stdout (``log``, ``progress``, ``epoch``, ``done``, ``error``).
Files written to ``output_dir``: ``best.pt``, ``last.pt``, ``results.json``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

try:
    from activelabel.training.workers import augment
except ImportError:  # run as a script in the training environment (no ActiveLabel): the sibling file
    import augment  # type: ignore[no-redef]

METRICS = ("map50_95", "map50")
TASKS = ("detection", "instance", "keypoints")
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
MAX_SIDE_FACTOR = 1.667  # torchvision's default max_size / min_size (1333 / 800)
WARMUP_STEPS = 100
EVAL_SCORE_THRESH = 0.0  # evaluation sees every detection (COCO), whatever the user's score_thresh
Array = Any


class SpecError(ValueError):
    """The spec cannot be used (the message is readable)."""


# ---------------------------------------------------------------------------
# Pure helpers (no torch)


def emit(event: str, **fields: Any) -> None:
    """Print one event line.

    Args:
        event: Event name (``log``, ``progress``, ``epoch``, ``done`` or ``error``).
        **fields: Extra fields of the JSON object.
    """
    print(json.dumps({"event": event, **fields}), flush=True)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Read and check ``spec.json``.

    Args:
        path: Path of the spec file.

    Returns:
        The spec as a dict.

    Raises:
        SpecError: missing keys, an unsupported task, family, start or metric, or no images.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task", "model", "family", "start", "params", "classes", "parts", "output_dir", "metric"):
        if key not in spec:
            raise SpecError(f"spec.json has no '{key}'.")
    if spec["task"] not in TASKS:
        raise SpecError(f"This worker trains {', '.join(TASKS)}, not {spec['task']}.")
    if spec.get("fold") == "all":
        raise SpecError("Training all folds comes later: choose one fold.")
    if spec["family"] != "torchvision":
        raise SpecError(f"This worker trains torchvision models, not {spec['family']}.")
    if spec["start"] not in ("transfer", "finetune", "scratch"):
        raise SpecError(f"Unknown start '{spec['start']}'.")
    if spec["metric"] not in METRICS:
        raise SpecError(f"The selection metric must be one of: {', '.join(METRICS)}.")
    parts = spec["parts"]
    if not parts.get("train") or not parts.get("val"):
        raise SpecError("Training needs images in the train and validation parts.")
    if spec["task"] == "keypoints" and int(spec.get("num_keypoints") or 0) < 1:
        raise SpecError("The labels have no keypoints.")
    return dict(spec)


def warmup_factor(step: int, steps: int) -> float:
    """Learning rate factor for step ``step`` (1-based) of the first epoch: a short linear ramp.

    Args:
        step: The batch about to run (from 1).
        steps: Batches in the epoch.

    Returns:
        The factor, from 0.001 up to 1.0 (1.0 once the ramp is over or the epoch is too short).
    """
    length = min(WARMUP_STEPS, steps - 1)
    if length <= 0 or step > length:
        return 1.0
    return 0.001 + (1.0 - 0.001) * (step - 1) / length


def polygon_mask(polygons: list[list[list[float]]] | None, width: int, height: int) -> Array:
    """``(height, width)`` uint8 mask, 1 inside the rings (relative points), drawn with PIL.

    Args:
        polygons: Rings of relative ``[x, y]`` points (``None`` or empty gives an empty mask).
        width: Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        The mask array.
    """
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for ring in polygons or []:
        points = [(min(max(p[0], 0.0), 1.0) * width, min(max(p[1], 0.0), 1.0) * height) for p in ring]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.array(image, dtype=np.uint8)


def keypoints_array(keypoints: list[list[float]] | None, count: int, width: int, height: int) -> Array:
    """``(count, 3)`` array of absolute ``[x, y, visibility]``; missing points are ``[0, 0, 0]``.

    Args:
        keypoints: Relative ``[x, y]`` or ``[x, y, visibility]`` points (visibility defaults to 2).
        count: Number of points per object.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The float32 array.
    """
    out = np.zeros((count, 3), dtype=np.float32)
    for i, point in enumerate((keypoints or [])[:count]):
        visibility = point[2] if len(point) > 2 else 2
        if visibility > 0:
            out[i] = (point[0] * width, point[1] * height, visibility)
    return out


def object_box(obj: dict[str, Any], width: int, height: int) -> list[float] | None:
    """Absolute ``[x1, y1, x2, y2]`` of an object (its box, else the extent of polygons or points).

    Args:
        obj: Spec object with a relative ``bbox``, ``polygons`` or ``keypoints``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The box clipped to the image, or None when it has no area (less than one pixel on a side).
    """
    box: list[float] | None = None
    if obj.get("bbox"):
        x, y, w, h = obj["bbox"]
        box = [x * width, y * height, (x + w) * width, (y + h) * height]
    elif obj.get("polygons"):
        points = [p for ring in obj["polygons"] for p in ring]
        if points:
            xs, ys = [p[0] * width for p in points], [p[1] * height for p in points]
            box = [min(xs), min(ys), max(xs), max(ys)]
    elif obj.get("keypoints"):
        points = [p for p in obj["keypoints"] if (p[2] if len(p) > 2 else 2) > 0]
        if points:
            xs, ys = [p[0] * width for p in points], [p[1] * height for p in points]
            box = [min(xs), min(ys), max(xs), max(ys)]
    if box is None:
        return None
    box = [min(max(box[0], 0.0), width), min(max(box[1], 0.0), height)] + [
        min(max(box[2], 0.0), width),
        min(max(box[3], 0.0), height),
    ]
    if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
        return None
    return box


def build_target(
    item: dict[str, Any], width: int, height: int, task: str, num_keypoints: int
) -> dict[str, Array]:
    """Numpy target of an image.

    ``boxes`` (N, 4) xyxy in pixels, ``labels`` (N,) = class + 1 and, by task,
    ``masks`` (N, H, W) uint8 or ``keypoints`` (N, K, 3).

    Args:
        item: Spec item with its ``objects``.
        width: Image width in pixels.
        height: Image height in pixels.
        task: ``detection``, ``instance`` or ``keypoints``.
        num_keypoints: Points per object (used by the ``keypoints`` task).

    Returns:
        The target dict of numpy arrays; objects without a usable box are left out.
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    masks: list[Array] = []
    points: list[Array] = []
    for obj in item.get("objects", []):
        box = object_box(obj, width, height)
        if box is None:
            continue
        boxes.append(box)
        labels.append(int(obj["label"]) + 1)
        if task == "instance":
            if obj.get("polygons"):
                mask = polygon_mask(obj["polygons"], width, height)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[int(box[1]) : int(np.ceil(box[3])), int(box[0]) : int(np.ceil(box[2]))] = 1
            masks.append(mask)
        if task == "keypoints":
            points.append(keypoints_array(obj.get("keypoints"), num_keypoints, width, height))
    target: dict[str, Array] = {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
    }
    if task == "instance":
        target["masks"] = np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
    if task == "keypoints":
        target["keypoints"] = (
            np.stack(points) if points else np.zeros((0, num_keypoints, 3), dtype=np.float32)
        )
    return target


def box_iou(a: Array, b: Array) -> Array:
    """``(len(a), len(b))`` IoU of xyxy boxes.

    Args:
        a: Boxes ``[x1, y1, x2, y2]``.
        b: Boxes ``[x1, y1, x2, y2]``.

    Returns:
        The IoU matrix.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    side = np.clip(bottom_right - top_left, 0.0, None)
    inter = side[..., 0] * side[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def mask_iou(a: Array, b: Array) -> Array:
    """``(len(a), len(b))`` IoU of masks of shape (N, H, W) (any value above 0 counts as inside).

    Args:
        a: Masks of shape (N, H, W).
        b: Masks of shape (M, H, W), same H and W as ``a``.

    Returns:
        The IoU matrix.
    """
    if not len(a) or not len(b):  # before reshaping: an empty (0, H, W) array cannot take -1
        return np.zeros((len(a), len(b)))
    fa = (np.asarray(a) > 0).reshape(len(a), -1).astype(np.float32)
    fb = (np.asarray(b) > 0).reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(scores: Array, hits: Array, n_gt: int) -> float:
    """COCO-style AP (101 recall points, precision made monotonic) of detections of one class.

    Args:
        scores: confidence of every detection.
        hits: whether each detection matched an object (true positive).
        n_gt: objects of the class in the ground truth.

    Returns:
        The AP, 0..1 (0.0 without ground truth or detections).
    """
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(hits, dtype=bool)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_POINTS, side="left")
    found = index < len(precision)
    sampled = np.zeros(len(RECALL_POINTS))
    sampled[found] = precision[index[found]]
    return float(sampled.mean())


class APAccumulator:
    """Collects detections image by image and computes per-class AP (COCO protocol).

    AP is averaged over IoU thresholds .50:.05:.95 (``ap50_95``) and also reported at .50 (``ap50``);
    the mean is taken over the classes that have ground truth. Ground-truth objects marked as ignored
    (as COCO does for a pose without labelled points) are not counted, and a detection that only
    matches an ignored object is dropped (neither a true nor a false positive).
    """

    def __init__(self, n_classes: int, thresholds: Array = IOU_THRESHOLDS) -> None:
        """Start empty.

        Args:
            n_classes: Number of classes (0-based indices below it are counted).
            thresholds: IoU thresholds to average over; 0.5 must be one of them (it gives ``ap50``).

        Raises:
            ValueError: 0.5 is not among the thresholds (``ap50`` would silently be another IoU).
        """
        self.n_classes = n_classes
        self.thresholds = [float(t) for t in thresholds]
        if 0.5 not in self.thresholds:
            raise ValueError("The IoU thresholds must include 0.5 (it gives ap50).")
        self._records: list[list[tuple[Array, Array, Array]]] = [[] for _ in range(n_classes)]
        self._n_gt = [0] * n_classes

    def add(self, scores: Array, labels: Array, gt_labels: Array, ious: Array, ignore: Array = None) -> None:
        """Add one image.

        Args:
            scores: Confidence of each detection.
            labels: 0-based class of each detection.
            gt_labels: 0-based class of each ground-truth object.
            ious: ``(detections, ground truth)`` IoU matrix (of boxes, masks or keypoints).
            ignore: Per ground-truth object, True when it must be ignored (None = none is).
        """
        scores, labels, gt_labels = np.asarray(scores), np.asarray(labels), np.asarray(gt_labels)
        ious = np.asarray(ious).reshape(len(labels), len(gt_labels))
        ignored = np.zeros(len(gt_labels), dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool)
        for c in {int(x) for x in labels} | {int(x) for x in gt_labels}:
            if not 0 <= c < self.n_classes:
                continue
            pred, truth = np.where(labels == c)[0], np.where(gt_labels == c)[0]
            self._n_gt[c] += int((~ignored[truth]).sum())
            if len(pred):
                self._records[c].append((scores[pred], ious[np.ix_(pred, truth)], ignored[truth]))

    def _hits(self, c: int, threshold: float) -> tuple[Array, Array]:
        """Scores and true-positive flags of class ``c`` at an IoU ``threshold`` (greedy matching).

        Args:
            c: 0-based class index.
            threshold: Lowest IoU for a detection to match a ground-truth object.

        Returns:
            ``(scores, hits)`` of the class's detections, each image's detections sorted by score;
            detections that only match an ignored object are left out.
        """
        all_scores: list[Array] = []
        all_hits: list[Array] = []
        for scores, ious, ignored in self._records[c]:
            order = np.argsort(-scores, kind="stable")
            used = np.zeros(ious.shape[1], dtype=bool)
            hits = np.zeros(len(scores), dtype=bool)
            keep = np.ones(len(scores), dtype=bool)
            for rank, i in enumerate(order):
                for pool in (~ignored, ignored):  # objects that count first, as in COCO
                    candidates = np.where(used | ~pool, -1.0, ious[i])
                    j = int(np.argmax(candidates)) if len(candidates) else -1
                    if j >= 0 and candidates[j] >= threshold:
                        used[j] = True
                        hits[rank], keep[rank] = not ignored[j], not ignored[j]
                        break
            all_scores.append(scores[order][keep])
            all_hits.append(hits[keep])
        if not all_scores:
            return np.zeros(0), np.zeros(0, dtype=bool)
        return np.concatenate(all_scores), np.concatenate(all_hits)

    def summary(self, names: list[str]) -> dict[str, Any]:
        """``{"map50", "map50_95", "per_class": {name: {"ap50", "ap50_95", "support"}}}``.

        Args:
            names: Class names, in the order of the class indices.

        Returns:
            The mean AP over the classes that have ground truth, and the AP of each of them.
        """
        per_class: dict[str, dict[str, float]] = {}
        for c, name in enumerate(names[: self.n_classes]):
            if self._n_gt[c] == 0:
                continue
            aps = [average_precision(*self._hits(c, t), self._n_gt[c]) for t in self.thresholds]
            at50 = aps[self.thresholds.index(0.5)]
            per_class[name] = {"ap50": at50, "ap50_95": float(np.mean(aps)), "support": self._n_gt[c]}
        values = list(per_class.values())
        return {
            "map50": float(np.mean([v["ap50"] for v in values])) if values else 0.0,
            "map50_95": float(np.mean([v["ap50_95"] for v in values])) if values else 0.0,
            "per_class": per_class,
        }


def match_pairs(
    pred_boxes: Array,
    pred_scores: Array,
    pred_labels: Array,
    gt_boxes: Array,
    gt_labels: Array,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Greedy ``(prediction, ground truth)`` pairs of the same class with IoU at least ``threshold``.

    Args:
        pred_boxes: Predicted boxes, xyxy.
        pred_scores: Confidence of each prediction (higher scores pick first).
        pred_labels: Class of each prediction.
        gt_boxes: Ground-truth boxes, xyxy.
        gt_labels: Class of each ground-truth object.
        threshold: Lowest IoU for a pair.

    Returns:
        The ``(prediction index, ground-truth index)`` pairs.
    """
    ious = box_iou(pred_boxes, gt_boxes)
    used: set[int] = set()
    pairs = []
    for i in np.argsort(-np.asarray(pred_scores), kind="stable"):
        best, best_iou = -1, threshold
        for j in range(len(gt_labels)):
            if j in used or pred_labels[i] != gt_labels[j] or ious[i, j] < best_iou:
                continue
            best, best_iou = j, ious[i, j]
        if best >= 0:
            used.add(best)
            pairs.append((int(i), best))
    return pairs


def oks_matrix(pred: Array, gt: Array, areas: Array, boxes: Array = None) -> Array:
    """``(detections, ground truth)`` object keypoint similarity (OKS), as in COCO.

    ``pred`` and ``gt`` are ``(N, K, 3)`` absolute ``[x, y, visibility]``; ``areas`` is the scale
    ``s**2`` of each ground-truth object (its box area). Every point has the same tolerance
    ``sigma = 1 / K`` (COCO's per-point values exist only for the 17 body points); only the ground
    truth's labelled points count. An object without labelled points has OKS 0, or, with ``boxes``,
    COCO's value: every predicted point is measured to the object's box enlarged to three times its
    size (0 inside it), so a detection on such an object can match it and be ignored with it.

    Args:
        pred: Predicted keypoints, ``(N, K, 3)``.
        gt: Ground-truth keypoints, ``(M, K, 3)``.
        areas: Box area of each ground-truth object.
        boxes: ``(M, 4)`` xyxy boxes of the ground truth (None = no fallback for objects without points).

    Returns:
        The ``(N, M)`` OKS matrix.
    """
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)))
    visible = gt[..., 2] > 0
    variance = (2.0 / gt.shape[1]) ** 2
    squared = ((pred[:, None, :, :2] - gt[None, :, :, :2]) ** 2).sum(-1)  # (D, G, K)
    scale = 2.0 * variance * (np.asarray(areas, dtype=np.float64) + np.spacing(1))
    similarity = np.exp(-squared / scale[None, :, None])
    counts = visible.sum(1)
    total = (similarity * visible[None]).sum(-1)
    out = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)
    if boxes is not None and not counts.all():
        box = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        size = box[:, 2:] - box[:, :2]
        low, high = box[:, :2] - size, box[:, 2:] + size  # COCO: x - w .. x + 2w
        points = pred[:, None, :, :2]  # (D, 1, K, 2)
        outside = np.maximum(0.0, low[None, :, None] - points) + np.maximum(0.0, points - high[None, :, None])
        fallback = np.exp(-(outside**2).sum(-1) / scale[None, :, None]).mean(-1)
        out = np.where(counts > 0, out, fallback)
    return out


def pck_counts(
    pred: dict[str, Array], gt: dict[str, Array], min_score: float = 0.5, alpha: float = 0.1
) -> tuple[int, int]:
    """``(correct, total)`` keypoints of an image for PCK@alpha.

    A labelled point is correct when the matching prediction (same class, box IoU of 0.5) has it
    within ``alpha`` times the longer side of the ground-truth box; unmatched objects count as wrong.

    Args:
        pred: Predictions of the image: ``boxes``, ``scores``, ``labels`` and ``keypoints``.
        gt: Ground truth of the image: ``boxes``, ``labels`` and ``keypoints``.
        min_score: Predictions below this score are ignored.
        alpha: Fraction of the longer box side allowed as distance.

    Returns:
        ``(correct, total)`` counts of labelled points.
    """
    gt_kps = gt["keypoints"]
    visible = gt_kps[..., 2] > 0
    total = int(visible.sum())
    keep = np.asarray(pred["scores"]) >= min_score
    if not total or not keep.any():
        return 0, total
    boxes, labels, kps = pred["boxes"][keep], pred["labels"][keep], pred["keypoints"][keep]
    pairs = match_pairs(boxes, pred["scores"][keep], labels, gt["boxes"], gt["labels"])
    correct = 0
    for i, j in pairs:
        side = max(gt["boxes"][j][2] - gt["boxes"][j][0], gt["boxes"][j][3] - gt["boxes"][j][1])
        distance = np.linalg.norm(kps[i][:, :2] - gt_kps[j][:, :2], axis=1)
        correct += int(((distance <= alpha * side) & visible[j]).sum())
    return correct, total


def split_note(spec: dict[str, Any]) -> str:
    """One-line description of the data, for the log.

    Args:
        spec: The job spec (``parts``, ``classes`` and optionally ``fold``).

    Returns:
        The sentence with image counts per part, the fold, the classes and the training objects.
    """
    parts = spec["parts"]
    sizes = ", ".join(f"{k} {len(v)}" for k, v in parts.items())
    objects = sum(len(i.get("objects", [])) for i in parts["train"])
    fold = f", fold {spec['fold']}" if spec.get("fold") is not None else ""
    return f"Images: {sizes}{fold}. Classes: {len(spec['classes'])}. Training objects: {objects}."


# ---------------------------------------------------------------------------
# Torch side


class DetectionDataset:
    """Map-style dataset of spec items: ``(image tensor 0..1, target dict of tensors)``."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        task: str,
        num_keypoints: int,
        augmentations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Keep the items and the augmentation techniques.

        Args:
            items: The images of one part (``path`` and ``objects`` each).
            task: ``detection``, ``instance`` or ``keypoints``.
            num_keypoints: Points per object (0 when the task has none).
            augmentations: The techniques applied to each image (training); ``None`` or empty reads as is.
        """
        self.items, self.task, self.num_keypoints = items, task, num_keypoints
        self.augmentations = augmentations or []

    def __len__(self) -> int:
        """Number of images.

        Returns:
            The number of items in the part.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """The image tensor and the target tensors at ``index``.

        Args:
            index: Position in the part.

        Returns:
            ``(image tensor 0..1, target dict of tensors)``, after the augmentation (boxes, masks and
            keypoints move with the image).
        """
        with Image.open(self.items[index]["path"]) as opened:
            # labels are drawn on the photo as shown (EXIF rotation applied)
            image = (ImageOps.exif_transpose(opened) or opened).convert("RGB")
        width, height = image.size
        target = build_target(self.items[index], width, height, self.task, self.num_keypoints)
        if self.augmentations:
            image, target = augment.augment_objects(image, target, self.augmentations)
        tensor = pil_to_tensor(image).float() / 255.0
        return tensor, {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in target.items()}


def collate(batch: list[Any]) -> tuple[Any, ...]:
    """Detection batches are lists (images differ in size).

    Args:
        batch: The ``(image, target)`` pairs of a batch.

    Returns:
        ``(images, targets)`` as tuples.
    """
    return tuple(zip(*batch, strict=True))


def pick_device(torch: Any) -> str:
    """``cuda`` when there is a GPU, ``mps`` on Apple silicon, else ``cpu``.

    Args:
        torch: The torch module.

    Returns:
        The device name.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, torch: Any) -> None:
    """Seed Python, numpy and torch.

    Args:
        seed: The seed.
        torch: The torch module.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_conv(old: Any, out_channels: int, torch: Any, kernel: int, padding: int, prior: float | None) -> Any:
    """A new convolution with the input channels of ``old``, small random weights and a prior-based bias.

    Args:
        old: The convolution it replaces (gives ``in_channels``).
        out_channels: Output channels of the new layer.
        torch: The torch module.
        kernel: Kernel size.
        padding: Padding.
        prior: Expected foreground probability for the bias initialisation; ``None`` or 0 gives a zero bias.

    Returns:
        The new ``Conv2d``.
    """
    conv = torch.nn.Conv2d(old.in_channels, out_channels, kernel, padding=padding)
    torch.nn.init.normal_(conv.weight, std=0.01)
    bias = -float(np.log((1 - prior) / prior)) if prior else 0.0
    torch.nn.init.constant_(conv.bias, bias)
    return conv


def replace_heads(
    model: Any, model_id: str, classes_with_background: int, num_keypoints: int, torch: Any
) -> None:
    """Replace the predictors of a COCO-pretrained model for this project's classes.

    Args:
        model: The torchvision detection model, changed in place.
        model_id: The torchvision model name (picks the mask or keypoint predictor).
        classes_with_background: Project classes plus the background.
        num_keypoints: Points per object (for ``keypointrcnn`` models).
        torch: The torch module.
    """
    n = classes_with_background
    if hasattr(model, "roi_heads"):
        predictor = model.roi_heads.box_predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(predictor.cls_score.in_features, n)
        if model_id.startswith("maskrcnn"):
            old = model.roi_heads.mask_predictor
            model.roi_heads.mask_predictor = MaskRCNNPredictor(old.conv5_mask.in_channels, 256, n)
        if model_id.startswith("keypointrcnn"):
            old = model.roi_heads.keypoint_predictor
            model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
                old.kps_score_lowres.in_channels, num_keypoints
            )
        return
    head = model.head.classification_head
    if model_id.startswith("ssd"):
        for block in head.module_list:
            old = block[-1]
            anchors = old.out_channels // head.num_columns
            new = torch.nn.Conv2d(old.in_channels, anchors * n, old.kernel_size, padding=old.padding)
            torch.nn.init.xavier_uniform_(new.weight)
            torch.nn.init.zeros_(new.bias)
            block[-1] = new
        head.num_columns = n
        return
    # RetinaNet and FCOS: a new final convolution keeps the pretrained tower
    head.cls_logits = _new_conv(head.cls_logits, head.num_anchors * n, torch, 3, 1, 0.01)
    head.num_classes = n


def get_score_thresh(model: Any) -> float | None:
    """Lowest score a detection needs to be returned (see :func:`set_score_thresh`).

    Args:
        model: The torchvision detection model.

    Returns:
        The model's current score threshold, or None when the model has none.
    """
    holder = model.roi_heads if hasattr(model, "roi_heads") else model
    value = getattr(holder, "score_thresh", None)
    return None if value is None else float(value)


def set_score_thresh(model: Any, value: float) -> None:
    """Lowest score a detection needs to be returned.

    Args:
        model: The torchvision detection model, changed in place.
        value: The score threshold.
    """
    if hasattr(model, "roi_heads"):
        model.roi_heads.score_thresh = value
    else:
        model.score_thresh = value


def build_model(spec: dict[str, Any], pretrained: bool) -> Any:
    """The torchvision model with predictors for the project's classes (+ background).

    Args:
        spec: The job spec (``model``, ``classes``, ``start``, ``params`` and ``num_keypoints``).
        pretrained: Start from COCO weights (downloads them when needed) instead of random ones.

    Returns:
        The model.

    Raises:
        SpecError: torchvision has no such detection model.
    """
    p, model_id = spec["params"], spec["model"]
    builder = getattr(torchvision.models.detection, model_id, None)
    if builder is None:
        raise SpecError(f"torchvision has no detection model '{model_id}'.")
    n = len(spec["classes"]) + 1
    keypoints = int(spec.get("num_keypoints") or 0)
    extra: dict[str, Any] = {}
    if not model_id.startswith("ssd"):  # SSDlite has a fixed 320 pixel input
        size = int(p["image_size"])
        extra = {"min_size": size, "max_size": int(size * MAX_SIDE_FACTOR)}
    if pretrained:
        if spec["start"] == "finetune":
            extra["trainable_backbone_layers"] = int(p.get("trainable_backbone_layers", 3))
        model = builder(weights="DEFAULT", **extra)
        replace_heads(model, model_id, n, keypoints, torch)
    else:
        if model_id.startswith("keypointrcnn"):
            extra["num_keypoints"] = keypoints
        model = builder(weights=None, weights_backbone=None, num_classes=n, **extra)
    set_score_thresh(model, float(p.get("score_thresh", 0.05)))
    return model


def make_optimizer(spec: dict[str, Any], model: Any, torch: Any) -> Any:
    """Optimizer for the start mode.

    Heads only (transfer), backbone at a lower rate (fine-tuning) or everything (scratch).
    The backbone includes the feature pyramid.

    Args:
        spec: The job spec (``start`` and ``params``).
        model: The detection model; in ``transfer`` mode its backbone is frozen.
        torch: The torch module.

    Returns:
        The optimizer (SGD, Adam or AdamW as in ``params.optimizer``).

    Raises:
        SpecError: Nothing in the model is trainable.
    """
    p = spec["params"]
    lr, start = float(p["lr"]), spec["start"]
    backbone_ids = {id(x) for x in model.backbone.parameters()}
    backbone = [x for x in model.parameters() if id(x) in backbone_ids]
    heads = [x for x in model.parameters() if id(x) not in backbone_ids]
    if start == "transfer":
        for param in backbone:
            param.requires_grad = False
        groups = [{"params": heads, "lr": lr}]
    elif start == "finetune":
        groups = [
            {"params": [x for x in backbone if x.requires_grad], "lr": lr * float(p["backbone_lr_factor"])},
            {"params": heads, "lr": lr},
        ]
    else:
        groups = [{"params": backbone + heads, "lr": lr}]
    groups = [g for g in groups if g["params"]]
    if not groups:
        raise SpecError("The model has nothing to train.")
    kwargs: dict[str, Any] = {"weight_decay": float(p["weight_decay"])}
    name = p["optimizer"]
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, **kwargs)
    if name == "adam":
        return torch.optim.Adam(groups, **kwargs)
    return torch.optim.AdamW(groups, **kwargs)


def make_scheduler(spec: dict[str, Any], optimizer: Any, torch: Any) -> Any:
    """Learning rate schedule, or ``None``.

    Args:
        spec: The job spec (``params.scheduler`` and ``params.epochs``).
        optimizer: The optimizer to schedule.
        torch: The torch module.

    Returns:
        A cosine or step scheduler, or ``None`` for a constant rate.
    """
    name, epochs = spec["params"]["scheduler"], int(spec["params"]["epochs"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


def train_epoch(
    model: Any, loader: Any, optimizer: Any, device: str, torch: Any, first: bool, on_step: Any = None
) -> float:
    """One training pass; returns the mean of the summed model losses per image batch.

    Args:
        model: The detection model in training mode.
        loader: Data loader of ``(images, targets)`` batches.
        optimizer: The optimizer.
        device: Torch device name.
        torch: The torch module.
        first: Whether this is the first epoch (the learning rate warms up).
        on_step: Called with ``(step, steps)`` after each batch (optional).

    Returns:
        The mean loss per batch.

    Raises:
        RuntimeError: The loss is not finite.
    """
    base = [g["lr"] for g in optimizer.param_groups]
    total, count = 0.0, 0
    steps = len(loader)
    for step, (images, targets) in enumerate(loader, start=1):
        if first:
            factor = warmup_factor(step, steps)
            for group, lr in zip(optimizer.param_groups, base, strict=True):
                group["lr"] = lr * factor
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            raise RuntimeError("The loss is not finite: try a lower learning rate.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss)
        count += 1
        if on_step:
            on_step(step, steps)
    for group, lr in zip(optimizer.param_groups, base, strict=True):
        group["lr"] = lr
    return total / count if count else 0.0


def evaluate(model: Any, loader: Any, spec: dict[str, Any], device: str, torch: Any) -> dict[str, Any]:
    """Box mAP (plus mask mAP for instance; keypoint OKS mAP and PCK@0.1 for keypoints) of a loader.

    The model returns every detection while it is evaluated (score threshold ``EVAL_SCORE_THRESH``, at
    most its ``detections_per_img``, 100 by default, per image), as in the COCO protocol: the user's
    ``score_thresh`` is put back afterwards and only limits what a saved model shows or pre-annotates.

    Args:
        model: The detection model (put in eval mode here).
        loader: Data loader of ``(images, targets)`` batches.
        spec: The job spec (``task`` and ``classes``).
        device: Torch device name.
        torch: The torch module.

    Returns:
        ``map50``, ``map50_95`` and ``per_class`` plus, by task, the ``mask_*`` or ``kp_*`` and
        ``pck`` fields.
    """
    task, classes = spec["task"], spec["classes"]
    model.eval()
    kept = get_score_thresh(model)
    if kept is None:  # not a torchvision detector (a stand-in): nothing to lower
        return _evaluate_batches(model, loader, task, classes, device, torch)
    set_score_thresh(model, EVAL_SCORE_THRESH)
    try:
        return _evaluate_batches(model, loader, task, classes, device, torch)
    finally:
        set_score_thresh(model, kept)


def _evaluate_batches(
    model: Any, loader: Any, task: str, classes: list[str], device: str, torch: Any
) -> dict[str, Any]:
    """The metrics of :func:`evaluate`, computed over the loader with the model as it is.

    Args:
        model: The detection model, in eval mode.
        loader: Data loader of ``(images, targets)`` batches.
        task: ``detection``, ``instance`` or ``keypoints``.
        classes: The project's class names.
        device: Torch device name.
        torch: The torch module.

    Returns:
        The same dict as :func:`evaluate`.
    """
    boxes_ap = APAccumulator(len(classes))
    masks_ap = APAccumulator(len(classes)) if task == "instance" else None
    points_ap = APAccumulator(len(classes)) if task == "keypoints" else None
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([i.to(device) for i in images])
            for out, tgt in zip(outputs, targets, strict=True):
                pred = {
                    "boxes": out["boxes"].cpu().numpy(),
                    "scores": out["scores"].cpu().numpy(),
                    "labels": out["labels"].cpu().numpy(),
                }
                gt = {k: v.numpy() for k, v in tgt.items()}
                ious = box_iou(pred["boxes"], gt["boxes"])
                boxes_ap.add(pred["scores"], pred["labels"] - 1, gt["labels"] - 1, ious)
                if masks_ap is not None:
                    masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
                    ious = mask_iou(masks, gt["masks"])
                    masks_ap.add(pred["scores"], pred["labels"] - 1, gt["labels"] - 1, ious)
                if points_ap is not None:
                    pred["keypoints"] = out["keypoints"].cpu().numpy()
                    areas = (gt["boxes"][:, 2] - gt["boxes"][:, 0]) * (gt["boxes"][:, 3] - gt["boxes"][:, 1])
                    ious = oks_matrix(pred["keypoints"], gt["keypoints"], areas, gt["boxes"])
                    unlabelled = ~(gt["keypoints"][..., 2] > 0).any(1)  # COCO ignores poses without points
                    points_ap.add(pred["scores"], pred["labels"] - 1, gt["labels"] - 1, ious, unlabelled)
                    hit, seen = pck_counts(pred, gt)
                    correct, total = correct + hit, total + seen
    result: dict[str, Any] = boxes_ap.summary(classes)
    if masks_ap is not None:
        masks_summary = masks_ap.summary(classes)
        result["mask_map50"] = masks_summary["map50"]
        result["mask_map50_95"] = masks_summary["map50_95"]
        result["mask_per_class"] = masks_summary["per_class"]
    if points_ap is not None:
        points_summary = points_ap.summary(classes)
        result["kp_map50"] = points_summary["map50"]
        result["kp_map50_95"] = points_summary["map50_95"]
        result["pck"] = correct / total if total else 0.0
    return result


def epoch_scores(result: dict[str, Any]) -> dict[str, float]:
    """Metrics of an evaluation as the ``val_*`` fields of an ``epoch`` event.

    Args:
        result: The dict returned by ``evaluate``.

    Returns:
        The ``val_`` prefixed scores.
    """
    fields = {"val_map50": result["map50"], "val_map50_95": result["map50_95"]}
    for key in ("mask_map50", "mask_map50_95", "kp_map50", "kp_map50_95", "pck"):
        if key in result:
            fields[f"val_{key}"] = result[key]
    return fields


def make_loader(
    items: list[dict[str, Any]],
    train_mode: bool,
    task: str,
    keypoints: int,
    augmentations: list[dict[str, Any]],
    batch_size: int,
    workers: int,
    generator: Any,
) -> Any:
    """A data loader over ``items``.

    Args:
        items: the images of one part.
        train_mode: shuffle and augment (the training part) instead of only reading.
        task: ``detection``, ``instance`` or ``keypoints``.
        keypoints: points per object (0 when the task has none).
        augmentations: the techniques applied in train mode.
        batch_size: images per batch.
        workers: loader processes (0 loads in the main process).
        generator: the seeded random generator used when shuffling.

    Returns:
        A ``DataLoader`` yielding ``(images, targets)`` batches.
    """
    return DataLoader(
        DetectionDataset(items, task, keypoints, augmentations if train_mode else None),
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=workers,
        collate_fn=collate,
        generator=generator if train_mode else None,
    )


def report_step(epoch: int, epochs: int, step: int, steps: int) -> None:
    """Tell the runner about the progress of a training epoch (about twenty times per epoch).

    Args:
        epoch: the epoch (from 1).
        epochs: how many epochs the run has.
        step: the batch just done (from 1).
        steps: batches in the epoch.
    """
    if step == steps or step % max(1, steps // 20) == 0:
        emit("progress", phase="training", epoch=epoch, epochs=epochs, step=step, steps=steps)


def save_checkpoint(path: Path, model: Any, meta: dict[str, Any], epoch: int) -> None:
    """Write the weights of ``model`` with what is needed to rebuild it.

    Args:
        path: the ``.pt`` file.
        model: the network.
        meta: classes, model name, family, image size... saved next to the weights.
        epoch: the epoch the weights come from.
    """
    torch.save({"state_dict": model.state_dict(), **meta, "epoch": epoch}, path)


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """Train, keep the best epoch by the selection metric, evaluate the test part.

    Writes ``best.pt``, ``last.pt`` and ``results.json`` to ``output_dir``.

    Args:
        spec: The checked job spec.

    Returns:
        The results: ``metric``, ``best_epoch``, ``epochs_run``, ``stopped_early``, ``val`` and
        ``test`` (``None`` without a test part).

    Raises:
        SpecError: A model the worker cannot use.
        augment.AugmentError: A technique the worker cannot apply.
        RuntimeError: The loss is not finite.
    """
    p, classes, metric, task = spec["params"], spec["classes"], spec["metric"], spec["task"]
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(int(spec.get("seed", p["seed"])), torch)
    device = pick_device(torch)
    emit("log", message=f"Device: {device}. {split_note(spec)}")
    batch, keypoints = int(p["batch_size"]), int(spec.get("num_keypoints") or 0)
    techniques = spec["augmentations"]
    augment.check_objects(techniques)
    emit("log", message=f"Augmentation: {augment.describe(techniques)}.")
    generator = torch.Generator().manual_seed(int(p["seed"]))
    workers = 4 if device == "cuda" else 0
    parts = spec["parts"]

    loader = partial(
        make_loader,
        task=task,
        keypoints=keypoints,
        augmentations=techniques,
        batch_size=batch,
        workers=workers,
        generator=generator,
    )
    train_loader, val_loader = loader(parts["train"], True), loader(parts["val"], False)
    model = build_model(spec, pretrained=spec["start"] != "scratch")
    model.to(device)
    optimizer = make_optimizer(spec, model, torch)
    scheduler = make_scheduler(spec, optimizer, torch)
    epochs, patience = int(p["epochs"]), int(p["patience"])
    best: dict[str, Any] = {}
    since_best, ran = 0, 0

    meta = {
        "classes": classes,
        "model": spec["model"],
        "family": spec["family"],
        "task": task,
        "num_keypoints": keypoints,
        "image_size": int(p["image_size"]),
    }

    for epoch in range(1, epochs + 1):
        started = time.monotonic()

        model.train()
        on_step = partial(report_step, epoch, epochs)
        train_loss = train_epoch(model, train_loader, optimizer, device, torch, epoch == 1, on_step)
        scores = evaluate(model, val_loader, spec, device, torch)
        if scheduler is not None:
            scheduler.step()
        ran = epoch
        improved = not best or scores[metric] > best["scores"][metric]
        if improved:
            best = {"epoch": epoch, "scores": scores}
            since_best = 0
            save_checkpoint(out / "best.pt", model, meta, epoch)
        else:
            since_best += 1
        save_checkpoint(out / "last.pt", model, meta, epoch)
        emit(
            "epoch",
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            **epoch_scores(scores),
            lr=optimizer.param_groups[-1]["lr"],
            seconds=round(time.monotonic() - started, 2),
            best=improved,
        )
        if patience and since_best >= patience:
            emit("log", message=f"Early stopping: no improvement for {patience} epoch(s).")
            break

    results: dict[str, Any] = {
        "metric": metric,
        "best_epoch": best["epoch"],
        "epochs_run": ran,
        "stopped_early": ran < epochs,
        "val": {**best["scores"], "classes": classes},
        "test": None,
    }
    if parts.get("test"):
        emit("progress", phase="evaluating", epoch=ran, epochs=epochs)
        state = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        results["test"] = {
            **evaluate(model, loader(parts["test"], False), spec, device, torch),
            "classes": classes,
        }
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return results


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command line; ``argv[1]`` is the path of the spec JSON file.

    Returns:
        Process exit code: 0 on success, 1 after an ``error`` event.
    """
    try:
        spec = load_spec(argv[1])
        results = train(spec)
    except Exception as exc:  # noqa: BLE001 - reported to the runner as an event
        emit("error", message=str(exc) or type(exc).__name__)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    emit("done", results=results)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main(sys.argv))
