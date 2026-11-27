"""Evaluating a saved model in FiftyOne (#611): its predictions, the evaluation and the numbers read back.

The model's predictions on one part of a dataset version go to the field ``eval_m<N>`` of the project's
FiftyOne dataset (``eval_m<N>_pred`` for a classification: FiftyOne's own ``eval_m<N>`` is then the sample
field that says whether the image was right), next to the ground truth, and FiftyOne runs
``evaluate_detections`` (with the mean average precision; a polygon is compared as a polyline by its IoU) or
``evaluate_classifications`` under the evaluation key ``eval_m<N>``. That registers the evaluation, so the
App's Model Evaluation panel lists it, and writes the ``eval_m<N>_tp``/``_fp``/``_fn`` fields the App can
filter. What ActiveLabel keeps from the results: the main metrics, the metrics of each class, the confusion
matrix and the worst images. A saved view ``evaluation m<N>`` holds the evaluated images.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from activelabel import fiftyone_bridge
from activelabel.errors import CodedError
from activelabel.labels import Label
from activelabel.optional import F, fo
from activelabel.tasks import Task

#: Tasks the evaluation handles (keypoints and regression have no evaluation in FiftyOne's App here).
TASKS = (Task.CLASSIFICATION, Task.DETECTION, Task.SEGMENTATION)
#: Classes shown in the confusion matrix (the most frequent ones; the rest are summed in one extra column).
MATRIX_CLASSES = 20
#: Classes listed with their own metrics.
REPORT_CLASSES = 100
#: Images listed as the worst ones.
WORST_IMAGES = 50
_AVERAGES = ("micro avg", "macro avg", "weighted avg", "accuracy")


def key_of(model: int) -> str:
    """The evaluation key of a model (also the field of its predictions for objects).

    Args:
        model: Number of the saved model.

    Returns:
        ``eval_m<N>``.
    """
    return f"eval_m{model}"


def view_name(model: int) -> str:
    """The saved view with the images a model was evaluated on.

    Args:
        model: Number of the saved model.

    Returns:
        ``evaluation m<N>``.
    """
    return f"evaluation m{model}"


def prediction_field(task: Task, model: int) -> str:
    """The sample field that holds a model's predictions.

    Args:
        task: Task of the project.
        model: Number of the saved model.

    Returns:
        ``eval_m<N>`` for objects; ``eval_m<N>_pred`` for a classification (the evaluation key is the
        name of the field FiftyOne writes there, so the two cannot share one).
    """
    key = key_of(model)
    return f"{key}_pred" if task is Task.CLASSIFICATION else key


def refuse(task: Task) -> None:
    """Stop for a project type that cannot be evaluated here.

    Args:
        task: Task of the project.

    Raises:
        CodedError: ``evaluation_task_unsupported`` for keypoints and regression projects;
            ``evaluation_needs_shapely`` for a segmentation project when ``shapely`` (which FiftyOne
            compares polygons with) is not installed.
    """
    if task not in TASKS:
        raise CodedError(
            "evaluation_task_unsupported",
            f"A {task.value} model cannot be evaluated in FiftyOne here yet.",
            task=task.value,
        )
    if task is Task.SEGMENTATION and find_spec("shapely") is None:
        raise CodedError(
            "evaluation_needs_shapely",
            "Evaluating polygons needs the shapely package. Add it with: pip install shapely",
            install="pip install shapely",
        )


def _prediction(task: Task, labels: list[Label]) -> Any:
    """The FiftyOne label of one image's predictions (an empty one when the model found nothing).

    Args:
        task: Task of the project.
        labels: The labels the model predicted (each with a ``score``).

    Returns:
        The label, or None for a classification the model did not make.
    """
    label = fiftyone_bridge.fo_label(task, labels)
    if label is None and task is Task.DETECTION:
        return fo.Detections()
    if label is None and task is Task.SEGMENTATION:
        return fo.Polylines()
    return label


def _clean(dataset: Any, key: str, field: str) -> None:
    """Remove the previous evaluation of the model: its run and its fields (so a new one starts clean).

    Args:
        dataset: The project's FiftyOne dataset.
        key: The evaluation key.
        field: The field of the predictions.
    """
    if key in dataset.list_evaluations():
        dataset.delete_evaluation(key)
    for name in (field, key, f"{key}_tp", f"{key}_fp", f"{key}_fn"):
        if dataset.has_sample_field(name):
            dataset.delete_sample_field(name)


def _rate(top: float, bottom: float) -> float:
    """A share, 0 when there is nothing to divide.

    Args:
        top: The numerator.
        bottom: The denominator.

    Returns:
        ``top / bottom`` rounded to 6 places, or 0.0.
    """
    return round(top / bottom, 6) if bottom else 0.0


def per_class(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The metrics of each class from FiftyOne's report, the most frequent first.

    Args:
        report: ``results.report()``.

    Returns:
        ``[{"class", "precision", "recall", "f1", "support"}]`` (at most :data:`REPORT_CLASSES`).
    """
    rows: list[dict[str, Any]] = [
        {
            "class": str(name),
            "precision": round(float(row["precision"]), 6),
            "recall": round(float(row["recall"]), 6),
            "f1": round(float(row["f1-score"]), 6),
            "support": int(row["support"]),
        }
        for name, row in report.items()
        if name not in _AVERAGES and isinstance(row, dict)
    ]
    return sorted(rows, key=lambda row: (-row["support"], row["class"]))[:REPORT_CLASSES]


def confusion(results: Any, rows: list[dict[str, Any]], objects: bool) -> dict[str, Any]:
    """The confusion matrix of the most frequent classes (ground truth in rows, predictions in columns).

    The classes that are left out are summed in one extra row and column (``other``); for objects one more
    (``missing``) counts the objects with no match: a prediction with no ground truth (row) and a ground
    truth no prediction found (column).

    Args:
        results: The FiftyOne evaluation results.
        rows: The classes with their metrics (:func:`per_class`), already in order of frequency.
        objects: Whether the evaluation is of objects (detections or polygons).

    Returns:
        ``{"classes": [...], "other": bool, "missing": bool, "matrix": [[int]]}``; the extra row and column
        come after the classes, ``other`` first.
    """
    names = [row["class"] for row in rows[:MATRIX_CLASSES]]
    other = len(rows) > len(names)
    matrix = results.confusion_matrix(classes=names, include_other=other, include_missing=objects)
    return {
        "classes": names,
        "other": other,
        "missing": objects,
        "matrix": [[int(value) for value in line] for line in matrix.tolist()],
    }


def worst_images(view: Any, key: str, field: str, objects: bool) -> list[str]:
    """The images the model got worst: most false positives and misses, or wrong with the most confidence.

    Args:
        view: The evaluated samples.
        key: The evaluation key.
        field: The field of the predictions.
        objects: Whether the evaluation is of objects.

    Returns:
        At most :data:`WORST_IMAGES` image ids, the worst first.
    """
    scored: list[tuple[float, str]]
    if objects:
        ids, false_pos, false_neg = view.values(["image_id", f"{key}_fp", f"{key}_fn"])
        counts = [
            (int(fp or 0) + int(fn or 0), i) for i, fp, fn in zip(ids, false_pos, false_neg, strict=True)
        ]
        scored = [item for item in counts if item[0] > 0]
    else:
        ids, right, confidence = view.values(["image_id", key, f"{field}.confidence"])
        scored = [(float(conf or 0), i) for i, ok, conf in zip(ids, right, confidence, strict=True) if not ok]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [str(image_id) for _, image_id in scored[:WORST_IMAGES]]


def evaluate(dataset: Any, task: Task, model: int, predictions: dict[str, list[Label]]) -> dict[str, Any]:
    """Write the predictions to the dataset, evaluate them in FiftyOne and read the results back.

    Args:
        dataset: The project's FiftyOne dataset (refreshed with the ground truth).
        task: Task of the project (one of :data:`TASKS`).
        model: Number of the saved model.
        predictions: Image id to the labels the model predicted (each with a ``score``), only the images
            of the evaluated part that a person reviewed (an image with no objects has no ground truth
            there, and every prediction on it is a false positive).

    Returns:
        ``{"kind", "view", "images", "metrics", "classes", "confusion", "worst", "fields"}``; ``metrics`` has
        ``precision``, ``recall``, ``f1`` and ``map`` for objects (with the counts ``tp``, ``fp`` and ``fn``),
        ``accuracy``, ``macro_f1`` for a classification.

    Raises:
        CodedError: ``no_images_in_part`` when none of the images is in the dataset.
    """
    refuse(task)
    key, field = key_of(model), prediction_field(task, model)
    present = set(dataset.values("image_id"))  # an image deleted meanwhile is left out
    values = {image_id: _prediction(task, labels) for image_id, labels in predictions.items()}
    values = {i: label for i, label in values.items() if i in present}
    if not values:
        raise CodedError("no_images_in_part", "None of the images of this part could be evaluated.")
    _clean(dataset, key, field)
    if task is Task.CLASSIFICATION:  # the type cannot be inferred when the model made no prediction at all
        dataset.add_sample_field(field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Classification)
    dataset.set_values(field, values, key_field="image_id")
    view = dataset.match(F("image_id").is_in(list(values)))
    objects = task is not Task.CLASSIFICATION
    if objects:
        results = view.evaluate_detections(
            field, gt_field=fiftyone_bridge.GROUND_TRUTH, eval_key=key, compute_mAP=True, progress=False
        )
    else:
        results = view.evaluate_classifications(
            field, gt_field=fiftyone_bridge.GROUND_TRUTH, eval_key=key, progress=False
        )
    report = results.report()
    rows = per_class(report)
    if objects:
        tp, fp, fn = (int(view.sum(f"{key}_{name}") or 0) for name in ("tp", "fp", "fn"))
        precision, recall = _rate(tp, tp + fp), _rate(tp, tp + fn)
        metrics: dict[str, Any] = {
            "precision": precision,
            "recall": recall,
            "f1": _rate(2 * precision * recall, precision + recall),
            "map": round(float(results.mAP()), 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    else:
        metrics = {
            "accuracy": round(float(results.metrics()["accuracy"]), 6),
            "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 6),
        }
    dataset.save_view(view_name(model), view, overwrite=True)
    return {
        "kind": "detection" if objects else "classification",
        "view": view_name(model),
        "images": len(view),
        "metrics": metrics,
        "classes": rows,
        "confusion": confusion(results, rows, objects),
        "worst": worst_images(view, key, field, objects),
        "fields": list(
            dict.fromkeys([field, key, *([f"{key}_tp", f"{key}_fp", f"{key}_fn"] if objects else [])])
        ),
    }
