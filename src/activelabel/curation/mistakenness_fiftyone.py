"""Possible label errors (``compute_mistakenness``, #612) and hard images (``compute_hardness``, #662).

A saved model's predictions are written to the field ``PRED_FIELD`` of the project's FiftyOne dataset, next
to the ground truth, and FiftyOne Brain's ``compute_mistakenness`` compares them: for a classification, how
likely the image's class is wrong; for objects, how likely each object's class is wrong (``mistakenness``)
and its box (``mistakenness_loc``), plus the objects the model found that have no label
(``possible_missing``) and the labels the model does not find (``possible_spurious``). For a classification
model that sent its logits, the comparison uses them, and ``compute_hardness`` adds how hard each image is
for the model (``hardness``). The per-image numbers are read back so ActiveLabel can list them; the App keeps
every field for a closer look.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from activelabel import fiftyone_bridge
from activelabel.labels import Label
from activelabel.optional import fo, fob
from activelabel.tasks import Task

PRED_FIELD = "activelabel_predictions"
MISTAKENNESS = "mistakenness"
MISSING = "possible_missing"
SPURIOUS = "possible_spurious"
HARDNESS = "hardness"


def classification_from_logits(logits: list[float], classes: list[str | None]) -> Any:
    """A FiftyOne classification made from a model's raw outputs (its best class, with the logits).

    Args:
        logits: the raw output of each class, in the order of ``classes``.
        classes: the model's classes (None for a class the project no longer has).

    Returns:
        The classification, or None when the best class is not one of the project's.
    """
    values = np.asarray(logits, dtype=float)
    best = int(values.argmax())
    if best >= len(classes) or classes[best] is None:
        return None
    exp = np.exp(values - values.max())
    return fo.Classification(label=classes[best], confidence=float(exp[best] / exp.sum()), logits=values)


def score(
    dataset: Any,
    task: Task,
    predictions: dict[str, list[Label]],
    logits: dict[str, list[float]] | None = None,
    classes: list[str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write the predictions to the dataset, compute the mistakenness (and hardness) and read them back.

    Args:
        dataset: the project's FiftyOne dataset (refreshed with the ground truth).
        task: the project's task.
        predictions: image id to the model's labels (each with a ``score``).
        logits: for a classification, image id to the model's raw outputs (then used instead of the labels).
        classes: the model's classes, in the order of the logits (needed with ``logits``).

    Returns:
        Image id to ``{"mistakenness", "missing", "spurious", "hardness"}`` for the images that have ground
        truth and a prediction (``missing``/``spurious`` are counts, 0 for a classification; ``hardness`` is
        None without logits).
    """
    for field in (PRED_FIELD, HARDNESS):
        if dataset.has_sample_field(field):
            dataset.delete_sample_field(field)
    with_logits = bool(logits) and task is Task.CLASSIFICATION and classes is not None
    if with_logits and logits is not None and classes is not None:
        values = {image_id: classification_from_logits(raw, classes) for image_id, raw in logits.items()}
    else:
        values = {
            image_id: fiftyone_bridge.fo_label(task, labels) for image_id, labels in predictions.items()
        }
    present = set(dataset.values("image_id"))  # an image deleted meanwhile is left out
    values = {
        image_id: label for image_id, label in values.items() if label is not None and image_id in present
    }
    if not values:  # nothing predicted (every image unreadable, or below the threshold)
        return {}
    dataset.set_values(PRED_FIELD, values, key_field="image_id")
    view = dataset.exists(PRED_FIELD).exists(fiftyone_bridge.GROUND_TRUTH)
    if len(view) == 0:
        return {}
    fob.compute_mistakenness(
        view, PRED_FIELD, label_field=fiftyone_bridge.GROUND_TRUTH, use_logits=with_logits, progress=False
    )
    if with_logits:
        fob.compute_hardness(view, PRED_FIELD, hardness_field=HARDNESS, progress=False)
    ids = view.values("image_id")
    mistakes = view.values(MISTAKENNESS) if view.has_sample_field(MISTAKENNESS) else [None] * len(ids)
    objects = task is not Task.CLASSIFICATION
    missing = view.values(MISSING) if objects and view.has_sample_field(MISSING) else [0] * len(ids)
    spurious = view.values(SPURIOUS) if objects and view.has_sample_field(SPURIOUS) else [0] * len(ids)
    hard = view.values(HARDNESS) if with_logits and view.has_sample_field(HARDNESS) else [None] * len(ids)
    return {
        str(image_id): {
            "mistakenness": round(float(m), 6) if m is not None else 0.0,
            "missing": int(miss or 0),
            "spurious": int(spur or 0),
            "hardness": round(float(h), 6) if h is not None else None,
        }
        for image_id, m, miss, spur, h in zip(ids, mistakes, missing, spurious, hard, strict=True)
    }
