"""Precision x recall and F1 x confidence curves of the objects a model finds (docs/TRAINING.md).

The detection worker (torchvision and transformers detectors) and the YOLO worker add these curves to each
part of ``results.json`` (validation and test), computed from the evaluation pass the run already does:

- **Curves** (:func:`class_curve`, :func:`curves_report`): for every class with ground truth, the precision,
  recall and F1 of the detections whose confidence is at least ``t``, for ``t`` on a fixed grid of
  :data:`GRID_POINTS` confidences from 0 to 1 (so at most that many points per curve and class). A detection
  is right when it matches an object of the same class with an IoU of at least 0.5 (the same greedy matching
  as ``ap50``). Where no detection is that confident, precision is 1 and recall 0 (as scikit-learn and
  Ultralytics do). ``all`` is the mean of the classes' curves.
- **Suggested threshold** (:func:`best_threshold`, :func:`suggestions`): the confidence where F1 is
  highest, for the mean of the classes (``suggested_threshold``, the default confidence of a pre-annotation
  with the saved model) and for each class (``suggested_thresholds``). Only the validation part suggests one;
  the test part is measured with the validation's.
- **Confusion matrix of objects** (:func:`image_pairs`, :class:`ConfusionAccumulator`): the detections at
  or above the suggested threshold are paired with the objects at IoU 0.5, first with an object of the same
  class, then with one of another class (a class swap); a detection left alone is a false positive (the
  background row), an object left alone a miss (the background column). The format is the one of the
  evaluation of a saved model in FiftyOne (``{"classes", "other", "missing", "matrix"}``, ground truth in
  rows), so the same table shows both.

Standalone like the workers: it runs in the training environment, which has no ActiveLabel, and needs only
numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Confidences at which every curve is sampled (0, 0.01 … 1).
GRID_POINTS = 101
CONFIDENCE = np.linspace(0.0, 1.0, GRID_POINTS)
#: Decimals kept in ``results.json``.
DIGITS = 4
#: IoU at which a detection matches an object for the curves and the confusion matrix.
CURVE_IOU = 0.5
#: Confidence of the confusion matrix when the validation suggests none (the default of a pre-annotation).
FALLBACK_THRESHOLD = 0.25
Array = Any


def _rounded(values: Array) -> list[float]:
    """Values as a JSON list, rounded to :data:`DIGITS` decimals.

    Args:
        values: Numbers.

    Returns:
        The rounded floats.
    """
    return [round(float(v), DIGITS) for v in np.asarray(values, dtype=np.float64)]


def class_curve(scores: Array, hits: Array, n_gt: int, grid: Array = CONFIDENCE) -> dict[str, list[float]]:
    """Precision, recall and F1 of one class's detections at each confidence of ``grid``.

    Args:
        scores: Confidence of each detection of the class.
        hits: Whether each detection matched an object of the class (true positive).
        n_gt: Objects of the class in the ground truth.
        grid: Confidences (the lowest a detection needs to count), in increasing order.

    Returns:
        ``{"precision", "recall", "f1"}``, one value per confidence of ``grid``; precision is 1 where no
        detection is that confident, recall and F1 are 0 without ground truth.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    hits = np.asarray(hits, dtype=bool).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ranked, right = scores[order], hits[order]
    true_pos = np.concatenate([[0], np.cumsum(right)])
    # detections with a score of at least each confidence
    kept = np.searchsorted(-ranked, -grid, side="right")
    found = true_pos[kept].astype(np.float64)
    precision = np.where(kept > 0, found / np.maximum(kept, 1), 1.0)
    recall = found / n_gt if n_gt else np.zeros(len(grid))
    total = precision + recall
    f1 = np.where(total > 0, 2 * precision * recall / np.where(total > 0, total, 1.0), 0.0)
    return {"precision": _rounded(precision), "recall": _rounded(recall), "f1": _rounded(f1)}


def resample(confidence: Array, values: Array, grid: Array = CONFIDENCE) -> list[float]:
    """A curve sampled at other confidences (linear interpolation), for curves computed on a finer grid.

    Args:
        confidence: Confidences at which ``values`` were computed, in increasing order.
        values: The curve.
        grid: Confidences to sample at.

    Returns:
        The values at each confidence of ``grid``, rounded.
    """
    return _rounded(np.interp(np.asarray(grid, dtype=np.float64), confidence, values))


def mean_curve(classes: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    """The mean of the classes' curves, point by point.

    Args:
        classes: ``{name: {"precision", "recall", "f1", ...}}`` sampled on the same grid.

    Returns:
        ``{"precision", "recall", "f1"}`` (empty lists without classes).
    """
    if not classes:
        return {"precision": [], "recall": [], "f1": []}
    return {
        key: _rounded(np.mean([np.asarray(c[key], dtype=np.float64) for c in classes.values()], axis=0))
        for key in ("precision", "recall", "f1")
    }


def best_threshold(confidence: Array, f1: Array) -> float | None:
    """The confidence where F1 is highest.

    F1 is flat between two detections' scores, so the highest values are a run of confidences: the middle of
    the first such run is taken, away from both scores.

    Args:
        confidence: Confidences, in increasing order.
        f1: F1 at each confidence.

    Returns:
        The confidence, rounded to 2 decimals, or ``None`` when F1 is 0 everywhere (nothing was found).
    """
    values = np.asarray(f1, dtype=np.float64)
    if not len(values) or values.max() <= 0:
        return None
    first = int(np.argmax(values))
    last = first
    while last + 1 < len(values) and values[last + 1] >= values[first] - 1e-9:
        last += 1
    return round((float(confidence[first]) + float(confidence[last])) / 2, 2)


def suggestions(report: dict[str, Any]) -> dict[str, Any]:
    """The suggested thresholds of a ``curves`` entry, as fields of its part of ``results.json``.

    Args:
        report: The ``curves`` entry (:func:`curves_report`).

    Returns:
        ``suggested_threshold`` (best F1 of the mean of the classes; left out when there is none) and
        ``suggested_thresholds`` (``{class: confidence}`` of the classes that have one).
    """
    confidence = report["confidence"]
    out: dict[str, Any] = {}
    overall = best_threshold(confidence, report["all"]["f1"]) if report["all"]["f1"] else None
    if overall is not None:
        out["suggested_threshold"] = overall
    each = {name: best_threshold(confidence, c["f1"]) for name, c in report["classes"].items()}
    out["suggested_thresholds"] = {name: value for name, value in each.items() if value is not None}
    return out


def _greedy(order: Array, allowed: Array, ious: Array, used: Array, iou: float) -> dict[int, int]:
    """Pairs of detections and objects, the most confident detection first, each with its best free object.

    Args:
        order: Detection indices, the most confident first.
        allowed: ``(detections, objects)`` booleans: the pairs that may be made.
        ious: ``(detections, objects)`` IoU.
        used: Per object, whether it is already paired (updated in place).
        iou: Lowest IoU of a pair.

    Returns:
        ``{detection: object}``.
    """
    pairs: dict[int, int] = {}
    for i in order:
        candidates = np.where(allowed[i] & ~used & (ious[i] >= iou), ious[i], -1.0)
        if len(candidates) and candidates.max() >= 0:
            j = int(np.argmax(candidates))
            used[j] = True
            pairs[int(i)] = j
    return pairs


def image_pairs(
    scores: Array,
    labels: Array,
    gt_labels: Array,
    ious: Array,
    threshold: float,
    iou: float = CURVE_IOU,
) -> list[tuple[int, int]]:
    """``(real class, predicted class)`` of every detection and object of one image; -1 is the background.

    Only detections with a confidence of at least ``threshold`` count. They are paired with objects at
    ``iou`` or more, the most confident first: with an object of the same class when one is free, then with an
    object of another class. An unpaired detection gives ``(-1, class)`` (found where there is nothing), an
    unpaired object ``(class, -1)`` (missed).

    Args:
        scores: Confidence of each detection.
        labels: 0-based class of each detection.
        gt_labels: 0-based class of each object.
        ious: ``(detections, objects)`` IoU matrix, every class against every class.
        threshold: Lowest confidence of a detection.
        iou: Lowest IoU of a pair.

    Returns:
        One ``(real, predicted)`` pair per paired detection, unpaired detection and unpaired object.
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)
    ious = np.asarray(ious, dtype=np.float64).reshape(len(labels), len(gt_labels))
    confident = np.where(scores >= threshold)[0]
    order = confident[np.argsort(-scores[confident], kind="stable")]
    used = np.zeros(len(gt_labels), dtype=bool)
    same = labels[:, None] == gt_labels[None, :]
    pairs = _greedy(order, same, ious, used, iou)
    left = np.array([i for i in order if int(i) not in pairs], dtype=np.int64)
    pairs.update(_greedy(left, np.ones_like(same), ious, used, iou))
    out = [(int(gt_labels[j]), int(labels[i])) for i, j in pairs.items()]
    out += [(-1, int(labels[i])) for i in order if int(i) not in pairs]
    out += [(int(gt_labels[j]), -1) for j in np.where(~used)[0]]
    return out


class ConfusionAccumulator:
    """Collects the detections of a part image by image and counts the confusion matrix of objects.

    The matrix needs the threshold the validation suggests, known only after the whole part is seen, so the
    images are kept (their detections and IoU) and counted at the end (:meth:`report`).
    """

    def __init__(self, n_classes: int, iou: float = CURVE_IOU) -> None:
        """Start empty.

        Args:
            n_classes: Number of classes (0-based indices below it are counted; others are skipped).
            iou: Lowest IoU of a pair.
        """
        self.n_classes = n_classes
        self.iou = iou
        self._images: list[tuple[Array, Array, Array, Array]] = []

    def add(self, scores: Array, labels: Array, gt_labels: Array, ious: Array) -> None:
        """Add one image.

        Args:
            scores: Confidence of each detection.
            labels: 0-based class of each detection.
            gt_labels: 0-based class of each object.
            ious: ``(detections, objects)`` IoU matrix, every class against every class.
        """
        self._images.append((np.asarray(scores), np.asarray(labels), np.asarray(gt_labels), np.asarray(ious)))

    def matrix(self, threshold: float) -> list[list[int]]:
        """``(classes + 1)`` square counts, ground truth in rows; the last row and column are the background.

        Args:
            threshold: Lowest confidence of a detection.

        Returns:
            The counts.
        """
        n = self.n_classes
        counts = [[0] * (n + 1) for _ in range(n + 1)]
        for scores, labels, gt_labels, ious in self._images:
            for real, predicted in image_pairs(scores, labels, gt_labels, ious, threshold, self.iou):
                if real >= n or predicted >= n:
                    continue
                counts[n if real < 0 else real][n if predicted < 0 else predicted] += 1
        return counts

    def report(self, names: list[str], threshold: float | None) -> dict[str, Any]:
        """The ``confusion`` entry of a part of ``results.json``.

        Args:
            names: Class names, in the order of the class indices.
            threshold: Lowest confidence of a detection (the validation's suggestion); ``None`` uses
                :data:`FALLBACK_THRESHOLD`.

        Returns:
            ``{"classes", "other", "missing", "matrix", "iou", "threshold"}``: ``missing`` (the background) is
            the last row and column, as in the evaluation of a saved model in FiftyOne.
        """
        used = FALLBACK_THRESHOLD if threshold is None else float(threshold)
        return {
            "classes": list(names[: self.n_classes]),
            "other": False,
            "missing": True,
            "matrix": self.matrix(used),
            "iou": self.iou,
            "threshold": used,
        }


def curves_report(classes: dict[str, dict[str, Any]], match: str, grid: Array = CONFIDENCE) -> dict[str, Any]:
    """The ``curves`` entry of a part of ``results.json``.

    Args:
        classes: ``{name: {"precision", "recall", "f1", "support"}}`` of the classes with ground truth.
        match: What the IoU compares: ``box`` or ``mask``.
        grid: The confidences of the curves.

    Returns:
        ``{"iou", "match", "confidence", "classes", "all"}``.
    """
    return {
        "iou": CURVE_IOU,
        "match": match,
        "confidence": _rounded(grid),
        "classes": classes,
        "all": mean_curve(classes),
    }
