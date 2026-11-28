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
#: IoU at which a detection matches an object for the curves.
CURVE_IOU = 0.5
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
