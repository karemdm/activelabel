"""Quality alerts of a finished training run (see docs/TRAINING.md, "Quality alerts").

The alerts never block anything: the run can still be saved as a model. Each alert is
``{"code": str, "params": dict}``; the interface writes the text from ``tr.alert.<code>`` (pt and en).
The logic is pure: it reads what the workers already write (``results.json`` and ``metrics.jsonl``).
For a grid or all-folds run the alerts are those of the best child, whose results and curves are the
run's own (see ``Runner._finish_children``).
"""

from __future__ import annotations

import math
from typing import Any

#: Alert codes, in the order they are reported.
CODES = (
    "only_background",
    "metric_zero",
    "single_class",
    "class_zero",
    "class_far_below",
    "val_worse_than_train",
    "few_val_images",
    "few_val_per_class",
)

#: A class takes at least this share of the predictions (classification): the model predicts one class.
SINGLE_CLASS_SHARE = 0.95
#: Semantic segmentation: mIoU below this, with the pixel accuracy at or above ``ONLY_BACKGROUND_ACCURACY``.
ONLY_BACKGROUND_MIOU = 0.01
#: Pixel accuracy from which a near-zero mIoU means "everything predicted as background".
ONLY_BACKGROUND_ACCURACY = 0.5
#: A class is "far below" when its score is under this fraction of the mean of the classes...
CLASS_GAP_FRACTION = 0.5
#: ...and the mean is above this (a model that is weak everywhere is not a problem of one class).
CLASS_GAP_MIN_MEAN = 0.2
#: At most this many classes are named (the weakest ones), so a long list does not flood the card.
MAX_CLASSES_NAMED = 5
#: The validation loss is at least this many times the training loss at the best epoch...
LOSS_GAP_RATIO = 2.0
#: ...and at least this much higher (so two tiny losses do not warn).
LOSS_GAP_MIN_DIFF = 0.1
#: Fewer validation images than this is too few for a trustworthy score.
MIN_VAL_IMAGES = 10
#: Classification: a class with fewer validation images than this has an unreliable score.
MIN_VAL_PER_CLASS = 2
#: Score of each class by the keys of ``results["val"]["per_class"]`` (the first one present is used).
CLASS_SCORE_KEYS = ("recall", "ap50", "iou")


def _number(value: Any) -> float | None:
    """A finite number, or ``None``.

    Args:
        value: anything read from the results.

    Returns:
        The value as a float; ``None`` for a missing value, a non-number (a bool too), NaN or infinity.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _entries(val: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The per-class entries of the validation results that are well formed.

    Args:
        val: the ``val`` part of ``results.json``.

    Returns:
        ``{class: entry}`` (empty when there is no ``per_class``).
    """
    per_class = val.get("per_class")
    if not isinstance(per_class, dict):
        return {}
    return {str(k): v for k, v in per_class.items() if isinstance(v, dict)}


def _class_scores(val: dict[str, Any]) -> dict[str, float]:
    """The score of each class that had validation samples (the background is left out).

    Args:
        val: the ``val`` part of ``results.json``.

    Returns:
        ``{class: score}`` using ``recall``, ``ap50`` or ``iou`` (the first one the workers wrote).
    """
    names = list(_entries(val).items())
    if "background_iou" in val:  # semantic segmentation: the first class is the background
        names = names[1:]
    out: dict[str, float] = {}
    for name, entry in names:
        key = next((k for k in CLASS_SCORE_KEYS if k in entry), None)
        score = _number(entry.get(key)) if key else None
        if score is not None and (_number(entry.get("support")) or 0) > 0:
            out[name] = score
    return out


def _single_class(val: dict[str, Any]) -> dict[str, Any] | None:
    """The class that takes (almost) all the predictions of a classifier.

    Args:
        val: the ``val`` part of ``results.json``.

    Returns:
        ``{"class", "share"}`` (share in percent) when one class has at least ``SINGLE_CLASS_SHARE``
        of the predicted images; ``None`` otherwise (also without a matrix, with one class, or for
        segmentation, where the background legitimately takes most pixels).
    """
    matrix, classes = val.get("confusion_matrix"), val.get("classes")
    if "background_iou" in val or not isinstance(matrix, list) or not isinstance(classes, list):
        return None
    if len(classes) < 2 or len(matrix) != len(classes):
        return None
    try:
        columns = [sum(float(row[i]) for row in matrix) for i in range(len(classes))]
    except (TypeError, ValueError, IndexError):
        return None
    total = sum(columns)
    if total <= 0:
        return None
    top = max(range(len(columns)), key=columns.__getitem__)
    share = columns[top] / total
    if share < SINGLE_CLASS_SHARE:
        return None
    return {"class": str(classes[top]), "share": round(share * 100)}


def _loss_gap(results: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, float] | None:
    """Training and validation loss of the best epoch, when the validation one is much higher.

    Args:
        results: ``results.json`` (``best_epoch``).
        metrics: the rows of ``metrics.jsonl``.

    Returns:
        ``{"train", "val"}`` (rounded) when the gap passes ``LOSS_GAP_RATIO`` and ``LOSS_GAP_MIN_DIFF``;
        ``None`` otherwise (also when the rows do not have both losses).
    """
    rows = [m for m in metrics if isinstance(m, dict)]
    if not rows:
        return None
    best = results.get("best_epoch")
    row = next((m for m in rows if m.get("epoch") == best), rows[-1])
    train, val = _number(row.get("train_loss")), _number(row.get("val_loss"))
    if train is None or val is None or train < 0:
        return None
    if val - train < LOSS_GAP_MIN_DIFF or val < LOSS_GAP_RATIO * train:
        return None
    return {"train": round(train, 3), "val": round(val, 3)}


def _class_alerts(scores: dict[str, float]) -> list[dict[str, Any]]:
    """Alerts about classes much weaker than the others.

    Args:
        scores: score of each class (``_class_scores``).

    Returns:
        ``class_zero`` and ``class_far_below`` alerts, the weakest classes first, at most
        ``MAX_CLASSES_NAMED``; empty with fewer than two classes.
    """
    if len(scores) < 2:
        return []
    mean = sum(scores.values()) / len(scores)
    out: list[dict[str, Any]] = []
    for name, score in sorted(scores.items(), key=lambda kv: kv[1]):
        if score == 0 and mean > 0:
            out.append({"code": "class_zero", "params": {"class": name}})
        elif mean > CLASS_GAP_MIN_MEAN and score < CLASS_GAP_FRACTION * mean:
            params = {"class": name, "value": round(score, 3), "mean": round(mean, 3)}
            out.append({"code": "class_far_below", "params": params})
    return out[:MAX_CLASSES_NAMED]


def _main_metric_alert(results: dict[str, Any], val: dict[str, Any]) -> dict[str, Any] | None:
    """The alert about the main metric (zero, missing or NaN), or the "only background" one.

    Args:
        results: ``results.json`` (``metric``).
        val: the ``val`` part of ``results.json``.

    Returns:
        ``only_background`` when mIoU is about zero but most pixels are right, else ``metric_zero``
        when the selection metric is 0, NaN or ``None`` (not when it is not in the results), else ``None``.
    """
    if "miou" in val and (_number(val["miou"]) or 0) < ONLY_BACKGROUND_MIOU:
        if (_number(val.get("pixel_accuracy")) or 0) >= ONLY_BACKGROUND_ACCURACY:
            return {"code": "only_background", "params": {}}
    metric = results.get("metric")
    if isinstance(metric, str) and metric in val and not _number(val[metric]):
        return {"code": "metric_zero", "params": {"metric": metric}}
    return None


def _size_alerts(val: dict[str, Any], val_images: int | None) -> list[dict[str, Any]]:
    """Alerts about a validation part too small to trust.

    Args:
        val: the ``val`` part of ``results.json``.
        val_images: images of the validation part, or ``None`` when unknown.

    Returns:
        ``few_val_images`` (fewer than ``MIN_VAL_IMAGES``) and, for a classifier, ``few_val_per_class``
        (classes with fewer than ``MIN_VAL_PER_CLASS`` validation images).
    """
    out: list[dict[str, Any]] = []
    if val_images is not None and val_images < MIN_VAL_IMAGES:
        out.append({"code": "few_val_images", "params": {"n": val_images}})
    few = [
        n
        for n, e in _entries(val).items()
        if "recall" in e and (_number(e.get("support")) or 0) < MIN_VAL_PER_CLASS
    ]
    if few:
        out.append(
            {
                "code": "few_val_per_class",
                "params": {"classes": ", ".join(few[:MAX_CLASSES_NAMED]), "n": len(few)},
            }
        )
    return out


def quality_alerts(
    results: dict[str, Any] | None,
    metrics: list[dict[str, Any]] | None = None,
    val_images: int | None = None,
) -> list[dict[str, Any]]:
    """The quality alerts of a finished run.

    Args:
        results: ``results.json`` of the run (``metric``, ``best_epoch`` and ``val`` with the scores);
            ``None`` gives no alert.
        metrics: the rows of ``metrics.jsonl`` (``epoch``, ``train_loss``, ``val_loss``), or ``None``.
        val_images: images of the validation part, when known (without it only a classifier's
            per-class support is checked).

    Returns:
        A list of ``{"code", "params"}`` (codes in ``CODES``); empty when the model looks fine.
    """
    val = results.get("val") if isinstance(results, dict) else None
    if not isinstance(results, dict) or not isinstance(val, dict):
        return []
    alerts = [_main_metric_alert(results, val)]
    single = _single_class(val)
    alerts.append({"code": "single_class", "params": single} if single else None)
    found = [a for a in alerts if a]
    if not single:  # one predicted class already explains why every other class scores 0 (#148 QA)
        found.extend(_class_alerts(_class_scores(val)))
    gap = _loss_gap(results, metrics or [])
    if gap:
        found.append({"code": "val_worse_than_train", "params": gap})
    found.extend(_size_alerts(val, val_images))
    return found
