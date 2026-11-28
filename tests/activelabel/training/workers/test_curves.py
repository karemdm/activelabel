"""Curves by confidence threshold (workers/curves.py): numpy only, so they run in the app environment."""

import numpy as np
import pytest

from activelabel.training.workers import curves

SCORES = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1]
HITS = [True, True, False, True, False, True, False, False]


def test_the_grid_has_at_most_about_a_hundred_points():
    assert curves.GRID_POINTS <= 101
    assert curves.CONFIDENCE[0] == 0.0 and curves.CONFIDENCE[-1] == 1.0


def test_precision_and_recall_count_the_detections_at_or_above_each_confidence():
    got = curves.class_curve(SCORES, HITS, n_gt=5, grid=[0.0, 0.6, 0.65, 0.95])
    # all 8 detections, 4 right; at 0.6: 4 detections, 3 right; at 0.65: 3, 2 right; at 0.95: none
    assert got["precision"] == pytest.approx([0.5, 0.75, 2 / 3, 1.0], abs=1e-4)
    assert got["recall"] == pytest.approx([0.8, 0.6, 0.4, 0.0], abs=1e-4)
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(got["precision"], got["recall"], strict=True)]
    assert got["f1"] == pytest.approx(f1, abs=1e-4)
    assert got["f1"][0] == pytest.approx(2 * 0.5 * 0.8 / 1.3, abs=1e-4)


def test_the_curve_matches_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    scores = rng.random(200)
    truth = rng.random(200) < scores  # confident detections are more often right
    precision, recall, thresholds = metrics.precision_recall_curve(truth, scores)
    got = curves.class_curve(scores, truth, int(truth.sum()), grid=thresholds)
    assert got["precision"] == pytest.approx(precision[:-1], abs=1e-4)
    assert got["recall"] == pytest.approx(recall[:-1], abs=1e-4)
    for i in (0, 50, 120):
        predicted = scores >= thresholds[i]
        assert got["f1"][i] == pytest.approx(metrics.f1_score(truth, predicted), abs=1e-4)


def test_objects_never_found_lower_the_recall():
    got = curves.class_curve([0.9], [True], n_gt=4, grid=[0.5])
    assert got["recall"] == [0.25]
    assert got["precision"] == [1.0]


def test_a_class_without_detections_or_without_objects():
    empty = curves.class_curve([], [], n_gt=3, grid=[0.0, 0.5])
    assert empty == {"precision": [1.0, 1.0], "recall": [0.0, 0.0], "f1": [0.0, 0.0]}
    nothing = curves.class_curve([0.7], [False], n_gt=0, grid=[0.5])
    assert nothing["recall"] == [0.0] and nothing["f1"] == [0.0]


def test_resample_interpolates_a_finer_curve_on_the_grid():
    fine = np.linspace(0, 1, 1000)
    assert curves.resample(fine, fine * 2) == pytest.approx(list(curves.CONFIDENCE * 2), abs=1e-4)
    assert len(curves.resample(fine, fine)) == curves.GRID_POINTS


def test_the_mean_curve_and_the_report():
    a = {"precision": [1.0, 0.5], "recall": [0.0, 1.0], "f1": [0.0, 2 / 3], "support": 2}
    b = {"precision": [1.0, 1.0], "recall": [0.0, 0.5], "f1": [0.0, 2 / 3], "support": 1}
    report = curves.curves_report({"a": a, "b": b}, "mask", grid=[0.0, 1.0])
    assert report["iou"] == 0.5 and report["match"] == "mask"
    assert report["confidence"] == [0.0, 1.0]
    assert report["all"]["precision"] == [1.0, 0.75]
    assert report["all"]["recall"] == [0.0, 0.75]
    assert curves.mean_curve({}) == {"precision": [], "recall": [], "f1": []}


def test_the_suggested_threshold_is_the_middle_of_the_best_f1():
    confidence = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert curves.best_threshold(confidence, [0.2, 0.5, 0.8, 0.8, 0.8, 0.1]) == 0.3
    assert curves.best_threshold(confidence, [0.2, 0.9, 0.8, 0.9, 0.9, 0.1]) == 0.1  # the first run
    assert curves.best_threshold(confidence, [0.0] * 6) is None
    assert curves.best_threshold([], []) is None


def test_suggestions_of_a_report():
    a = {"precision": [0.5, 1.0], "recall": [1.0, 0.5], "f1": [2 / 3, 2 / 3], "support": 2}
    b = {"precision": [1.0, 1.0], "recall": [0.0, 0.0], "f1": [0.0, 0.0], "support": 1}
    report = curves.curves_report({"a": a, "b": b}, "box", grid=[0.2, 0.8])
    assert curves.suggestions(report) == {"suggested_threshold": 0.5, "suggested_thresholds": {"a": 0.5}}
    nothing = curves.curves_report({"b": b}, "box", grid=[0.2, 0.8])
    assert curves.suggestions(nothing) == {"suggested_thresholds": {}}


# -- confusion matrix of objects (#324), pairs built by hand ----------------------------------------

# objects: cat, dog, cat; detections: a cat on the first cat, a dog on the second cat (a swap), a cat on
# nothing, and an unsure dog on the dog (below the threshold: the dog is missed)
GT = [0, 1, 0]
LABELS = [0, 1, 0, 1]
SCORES_324 = [0.9, 0.8, 0.7, 0.1]
IOUS = [
    [0.8, 0.9, 0.0],  # overlaps the dog more, but the cat of its own class comes first
    [0.0, 0.0, 0.7],
    [0.0, 0.0, 0.0],
    [0.0, 0.9, 0.0],
]


def test_image_pairs_by_hand():
    pairs = curves.image_pairs(SCORES_324, LABELS, GT, IOUS, threshold=0.5)
    assert sorted(pairs) == sorted([(0, 0), (0, 1), (-1, 0), (1, -1)])
    # with every detection counted, the unsure dog finds its dog
    assert sorted(curves.image_pairs(SCORES_324, LABELS, GT, IOUS, threshold=0.0)) == sorted(
        [(0, 0), (0, 1), (-1, 0), (1, 1)]
    )


def test_a_pair_needs_the_iou():
    assert sorted(curves.image_pairs([0.9], [0], [0], [[0.49]], threshold=0.5)) == [(-1, 0), (0, -1)]
    assert curves.image_pairs([0.9], [0], [0], [[0.5]], threshold=0.5) == [(0, 0)]


def test_the_more_confident_detection_takes_the_object():
    pairs = curves.image_pairs([0.6, 0.9], [0, 0], [0], [[0.95], [0.6]], threshold=0.5)
    assert sorted(pairs) == [(-1, 0), (0, 0)]  # the 0.9 one, though the other overlaps more


def test_images_without_objects_or_detections():
    assert curves.image_pairs([0.9], [1], [], np.zeros((1, 0)), threshold=0.5) == [(-1, 1)]
    assert curves.image_pairs([], [], [0, 0], np.zeros((0, 2)), threshold=0.5) == [(0, -1), (0, -1)]


def test_the_matrix_has_the_format_of_the_fiftyone_evaluation():
    acc = curves.ConfusionAccumulator(2)
    acc.add(SCORES_324, LABELS, GT, IOUS)
    acc.add([0.9], [5], [0], [[0.9]])  # a pair with a class the project does not have is skipped
    report = acc.report(["cat", "dog"], 0.5)
    assert report["classes"] == ["cat", "dog"] and report["other"] is False and report["missing"] is True
    assert report["matrix"] == [[1, 1, 0], [0, 0, 1], [1, 0, 0]]  # real in rows, the background last
    assert (report["iou"], report["threshold"]) == (0.5, 0.5)
    assert acc.report(["cat", "dog"], None)["threshold"] == curves.FALLBACK_THRESHOLD
