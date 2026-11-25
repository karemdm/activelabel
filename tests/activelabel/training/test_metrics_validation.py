"""Hand-computed and scikit-learn checks of the metric formulas of the training workers.

Covers the classification, segmentation and detection workers and the K-fold ranking of the runner. Every
worked example carries its derivation in a comment above the assertion. The workers need torch, so the
``cls``, ``seg`` and ``det`` fixtures skip where torch is absent; ``rank_children`` needs nothing.
"""

import math
import statistics
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from worker_loader import load_worker

from activelabel.training.runner import rank_children

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def cls():
    return load_worker("classification")


@pytest.fixture(scope="module")
def seg():
    return load_worker("segmentation")


@pytest.fixture(scope="module")
def det():
    return load_worker("detection")


class FakeModel:
    """A detector that returns the stored outputs whatever the images are."""

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        """Keep the list of per-image output dicts of torch tensors.

        Args:
            outputs: what ``__call__`` returns.
        """
        self.outputs = outputs

    def eval(self) -> "FakeModel":
        """Switch to evaluation mode (nothing to do).

        Returns:
            The model itself, like a torch module.
        """
        return self

    def __call__(self, images: list[Any]) -> list[dict[str, Any]]:
        """Run the detector.

        Args:
            images: the batch (ignored).

        Returns:
            The stored outputs.
        """
        return self.outputs


class ThresholdModel(FakeModel):
    """A detector that drops the detections below ``roi_heads.score_thresh`` like torchvision does."""

    def __init__(self, outputs: list[dict[str, Any]], score_thresh: float) -> None:
        """Keep the outputs and the model's own score threshold.

        Args:
            outputs: the detections before filtering.
            score_thresh: lowest score the model returns.
        """
        super().__init__(outputs)
        self.roi_heads = SimpleNamespace(score_thresh=score_thresh)

    def __call__(self, images: list[Any]) -> list[dict[str, Any]]:
        """Run the detector.

        Args:
            images: the batch (ignored).

        Returns:
            The stored outputs without the detections under the threshold.
        """
        kept = []
        for out in self.outputs:
            mask = out["scores"] >= self.roi_heads.score_thresh
            kept.append({key: value[mask] for key, value in out.items()})
        return kept


def random_labels(rng: np.random.Generator, n_classes: int, size: int) -> tuple[list[int], list[int]]:
    """Random truth and predicted labels of ``size`` samples over ``n_classes`` values.

    Args:
        rng: the random generator.
        n_classes: number of distinct values.
        size: number of samples.

    Returns:
        The truth list and the predicted list.
    """
    truth = rng.integers(0, n_classes, size).tolist()
    predicted = rng.integers(0, n_classes, size).tolist()
    return truth, predicted


def add_image(
    acc: Any,
    det: Any,
    detections: list[tuple[list[float], int, float]],
    ground_truth: list[tuple[list[float], int]],
) -> None:
    """Feed one image of boxes to an ``APAccumulator``.

    Args:
        acc: the accumulator.
        det: the detection worker module (for ``box_iou``).
        detections: ``(box, class, score)`` of every detection.
        ground_truth: ``(box, class)`` of every object.
    """
    boxes = np.array([d[0] for d in detections], dtype=float).reshape(-1, 4)
    scores = np.array([d[2] for d in detections], dtype=float)
    labels = np.array([d[1] for d in detections], dtype=int)
    gt_boxes = np.array([g[0] for g in ground_truth], dtype=float).reshape(-1, 4)
    gt_labels = np.array([g[1] for g in ground_truth], dtype=int)
    acc.add(scores, labels, gt_labels, det.box_iou(boxes, gt_boxes))


def keypoint_loader(torch: Any) -> list[Any]:
    """One image whose ground truth has two persons, the second without any labelled point.

    Args:
        torch: the torch module.

    Returns:
        A one-batch loader of ``(images, targets)``.
    """
    points = torch.zeros(2, 3, 3)
    points[0] = torch.tensor([[2.0, 2.0, 2.0], [5.0, 5.0, 2.0], [8.0, 8.0, 2.0]])
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "keypoints": points,
    }
    return [([torch.zeros(3, 32, 32)], [target])]


def single_box_loader(torch: Any) -> list[Any]:
    """One image with one ground-truth box ``[0, 0, 10, 10]`` of the first class.

    Args:
        torch: the torch module.

    Returns:
        A one-batch loader of ``(images, targets)``.
    """
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}
    return [([torch.zeros(3, 32, 32)], [target])]


def child(name: str, combo: int, fold: int) -> dict[str, Any]:
    """A ``children`` entry of ``run.json``.

    Args:
        name: child id.
        combo: grid combination index.
        fold: fold index.

    Returns:
        The entry dict.
    """
    return {"id": name, "combo": combo, "fold": fold, "values": {"lr": 0.1 * combo}}


# ---------------------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------------------


def test_classification_worked_example(cls):
    """Confusion matrix, accuracy, per-class precision/recall/F1 and macro F1 of a 6-sample case."""
    truth, predicted = [0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]
    matrix = cls.confusion_matrix(truth, predicted, 3)
    # rows = truth, columns = predicted: truth 0 -> predicted 0, 0, 1; truth 1 -> 1, 2; truth 2 -> 2
    assert matrix == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    # accuracy = diagonal / total = (2 + 1 + 1) / 6
    assert cls.accuracy(matrix) == pytest.approx(4 / 6)
    scores = cls.per_class(matrix, ["a", "b", "c"])
    # a: tp 2, predicted 2 (column sum), support 3 -> P 1, R 2/3, F1 = 2 * 1 * 2/3 / (1 + 2/3) = 0.8
    assert scores["a"] == pytest.approx({"precision": 1.0, "recall": 2 / 3, "f1": 0.8, "support": 3})
    # b: tp 1, predicted 2, support 2 -> P 1/2, R 1/2, F1 1/2
    assert scores["b"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    # c: tp 1, predicted 2, support 1 -> P 1/2, R 1, F1 = 2 * 1/2 * 1 / (3/2) = 2/3
    assert scores["c"] == pytest.approx({"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1})
    # macro F1 is the unweighted mean of the three F1 values
    assert cls.macro_f1(matrix) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    result = cls.evaluation(truth, predicted, ["a", "b", "c"])
    assert result["images"] == 6
    assert result["confusion_matrix"] == matrix


def test_classification_zero_division(cls):
    """Classes without predictions or without samples never raise and follow the documented macro rule."""
    # class 2 is never true but predicted once: support 0, predicted 1 -> precision 0, recall 0/0 = 0, F1 0;
    # it still counts in the macro mean: (1 + 2/3 + 0) / 3 with F1 of class 0 = 1 and of class 1 = 2/3
    assert cls.macro_f1([[2, 0, 0], [0, 1, 1], [0, 0, 0]]) == pytest.approx(5 / 9)
    # class 2 with an all-zero row and column is absent from truth and prediction -> left out: (1 + 1) / 2
    assert cls.macro_f1([[2, 0, 0], [0, 1, 0], [0, 0, 0]]) == pytest.approx(1.0)
    # class 1 has support 2 but is never predicted: precision 0/0 = 0, recall 0, F1 0
    scores = cls.per_class([[3, 0], [2, 0]], ["a", "b"])
    assert scores["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    # an empty matrix has no sample: every ratio is 0
    zeros = [[0, 0], [0, 0]]
    assert cls.accuracy(zeros) == 0.0
    assert cls.macro_f1(zeros) == 0.0


def test_classification_matches_scikit_learn(cls):
    """300 random cases agree with scikit-learn (accuracy, macro F1, per class P/R/F1/support, confusion)."""
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 8))
        truth, predicted = random_labels(rng, n, int(rng.integers(1, 61)))
        names = [f"c{i}" for i in range(n)]
        result = cls.evaluation(truth, predicted, names)
        assert result["accuracy"] == pytest.approx(metrics.accuracy_score(truth, predicted), abs=TOLERANCE)
        macro = metrics.f1_score(truth, predicted, average="macro", zero_division=0)
        assert result["macro_f1"] == pytest.approx(macro, abs=TOLERANCE)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            truth, predicted, labels=list(range(n)), zero_division=0
        )
        for i, name in enumerate(names):
            got = result["per_class"][name]
            assert got["precision"] == pytest.approx(precision[i], abs=TOLERANCE)
            assert got["recall"] == pytest.approx(recall[i], abs=TOLERANCE)
            assert got["f1"] == pytest.approx(f1[i], abs=TOLERANCE)
            assert got["support"] == support[i]
        expected = metrics.confusion_matrix(truth, predicted, labels=list(range(n)))
        assert np.array_equal(result["confusion_matrix"], expected)


# ---------------------------------------------------------------------------------------------------------
# Segmentation (index 0 is the background)
# ---------------------------------------------------------------------------------------------------------


def test_segmentation_worked_example(seg):
    """mIoU, Dice and pixel accuracy of a 3x3 pixel confusion matrix with one phantom class."""
    matrix = [[5, 1, 0], [2, 3, 1], [0, 0, 0]]
    # class a (index 1): tp 3, true pixels = 2 + 3 + 1 = 6, predicted pixels = 1 + 3 + 0 = 4
    # IoU = 3 / (6 + 4 - 3) = 3/7, Dice = 2 * 3 / (6 + 4) = 6/10
    iou, dice, support, predicted = seg.class_scores(matrix, 1)
    assert (iou, dice, support, predicted) == (pytest.approx(3 / 7), pytest.approx(0.6), 6, 4)
    # class b (index 2): true 0, predicted 1 (a pixel of class a taken for b), tp 0 -> IoU 0, Dice 0
    assert seg.class_scores(matrix, 2) == (0.0, 0.0, 0, 1)
    # both classes are present (a in truth, b in prediction): mIoU = (3/7 + 0) / 2, Dice = (0.6 + 0) / 2
    miou, mdice = seg.mean_scores(matrix)
    assert miou == pytest.approx((3 / 7) / 2)
    assert mdice == pytest.approx(0.3)
    # pixel accuracy = diagonal / all pixels = (5 + 3 + 0) / 12
    assert seg.pixel_accuracy(matrix) == pytest.approx(8 / 12)
    result = seg.evaluation(matrix, ["a", "b"])
    # background: tp 5, true 6, predicted 7 -> IoU 5 / (6 + 7 - 5) = 5/8; reported, not in the means
    assert result["background_iou"] == pytest.approx(5 / 8)
    assert result["per_class"]["background"]["iou"] == pytest.approx(5 / 8)
    assert result["miou"] == pytest.approx((3 / 7) / 2)
    assert result["dice"] == pytest.approx(0.3)
    assert result["pixel_accuracy"] == pytest.approx(8 / 12)


def test_segmentation_absent_class_and_empty_matrix(seg):
    """A class in neither truth nor prediction is left out of the means; an empty matrix scores 0."""
    # class b has an all-zero row and column: only class a counts
    # a: tp 3, true 2 + 3 = 5, predicted 1 + 3 = 4 -> IoU 3 / (5 + 4 - 3) = 1/2, Dice 2 * 3 / 9 = 2/3
    miou, dice = seg.mean_scores([[5, 1, 0], [2, 3, 0], [0, 0, 0]])
    assert miou == pytest.approx(1 / 2)
    assert dice == pytest.approx(2 / 3)
    zeros = [[0, 0, 0]] * 3
    assert seg.mean_scores(zeros) == (0.0, 0.0)
    assert seg.pixel_accuracy(zeros) == 0.0


def test_segmentation_matches_scikit_learn(seg):
    """Random pixel arrays: per-class IoU/Dice, the means and pixel accuracy agree with scikit-learn."""
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 6))
        truth, predicted = random_labels(rng, n + 1, int(rng.integers(1, 201)))
        matrix = seg.confusion_from_lists(truth, predicted, n + 1)
        result = seg.evaluation(matrix, [f"c{i}" for i in range(1, n + 1)])
        labels = list(range(n + 1))
        jaccard = metrics.jaccard_score(truth, predicted, average=None, labels=labels, zero_division=0)
        dice = metrics.f1_score(truth, predicted, average=None, labels=labels, zero_division=0)
        names = ["background", *[f"c{i}" for i in range(1, n + 1)]]
        for i, name in enumerate(names):
            assert result["per_class"][name]["iou"] == pytest.approx(jaccard[i], abs=TOLERANCE)
            assert result["per_class"][name]["dice"] == pytest.approx(dice[i], abs=TOLERANCE)
        present = [i for i in range(1, n + 1) if i in truth or i in predicted]
        mean_iou = float(np.mean([jaccard[i] for i in present])) if present else 0.0
        mean_dice = float(np.mean([dice[i] for i in present])) if present else 0.0
        assert result["miou"] == pytest.approx(mean_iou, abs=TOLERANCE)
        assert result["dice"] == pytest.approx(mean_dice, abs=TOLERANCE)
        assert result["pixel_accuracy"] == pytest.approx(
            metrics.accuracy_score(truth, predicted), abs=TOLERANCE
        )


# ---------------------------------------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------------------------------------


def test_average_precision_is_coco_101_points(det):
    """The AP interpolates 101 recall points (COCO), not the exact area under the curve (VOC 2010)."""
    # sorted by score: hit, miss, hit, miss with 3 objects
    # precision 1, 1/2, 2/3, 1/2 and recall 1/3, 1/3, 2/3, 2/3; monotone envelope 1, 2/3, 2/3, 1/2
    # recall points 0.00..0.33 (34 points) take 1; 0.34..0.66 (33 points) take 2/3;
    # 0.67..1.00 are never reached and take 0 -> (34 + 33 * 2/3) / 101 = 56 / 101
    ap = det.average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False], 3)
    assert ap == pytest.approx(56 / 101)
    # the all-points area would be 1/3 * 1 + 1/3 * 2/3 = 5/9 (0.5556): a different number
    assert ap != pytest.approx(5 / 9, abs=1e-4)
    assert 5 / 9 - ap == pytest.approx(1 / 909)  # 5/9 - 56/101 = (505 - 504) / 909


def test_two_image_case_matches_pycocotools(det):
    """Two images, two classes: the values pycocotools 2.x COCOeval gives (bbox, iscrowd 0)."""
    acc = det.APAccumulator(2)
    add_image(
        acc,
        det,
        [([0, 0, 10, 10], 0, 0.9), ([1, 1, 11, 11], 0, 0.8), ([20, 20, 40, 36], 1, 0.7)],
        [([0, 0, 10, 10], 0), ([20, 20, 40, 40], 1)],
    )
    add_image(
        acc,
        det,
        [([0, 0, 20, 16], 0, 0.85), ([50, 50, 60, 60], 1, 0.95), ([0, 0, 20, 11], 0, 0.6)],
        [([0, 0, 20, 20], 0)],
    )
    out = acc.summary(["a", "b"])
    # class a (2 objects): the 0.8 detection of image 1 repeats an already matched object -> false positive.
    # The 0.85 detection of image 2 has IoU 16/20 = 0.8: a hit for the 7 thresholds 0.50..0.80 -> AP 1
    # (hit, hit, miss, miss gives recall 1/2 then 1); for 0.85, 0.90, 0.95 only the 0.9 detection hits ->
    # recall stops at 0.5 -> AP 51/101. The 0.6 detection (IoU 0.55) never adds a hit.
    assert out["per_class"]["a"]["ap50"] == pytest.approx(1.0)
    assert out["per_class"]["a"]["ap50_95"] == pytest.approx((7 * 1 + 3 * 51 / 101) / 10)
    assert out["per_class"]["a"]["ap50_95"] == pytest.approx(0.8514851485148515)
    # class b (1 object): the 0.95 detection lies on image 2, which has no class b -> false positive ranked
    # first; the 0.7 detection has IoU 16/20 = 0.8 -> hit for 7 thresholds: precision 0, 1/2 -> envelope
    # 1/2, 1/2 at recall 1 -> AP 1/2 for 7 thresholds and 0 for the other 3: 3.5 / 10 = 0.35
    assert out["per_class"]["b"]["ap50"] == pytest.approx(0.5)
    assert out["per_class"]["b"]["ap50_95"] == pytest.approx(0.35)
    assert out["map50"] == pytest.approx(0.75)
    assert out["map50_95"] == pytest.approx(0.6007425742574257)
    assert out["per_class"]["a"]["support"] == 2
    assert out["per_class"]["b"]["support"] == 1


def test_matching_rules(det):
    """Threshold equality is a hit, images are matched separately and classes without objects are skipped."""
    # an IoU exactly equal to the threshold counts as a hit
    acc = det.APAccumulator(1, thresholds=[0.5])
    acc.add(np.array([0.9]), np.array([0]), np.array([0]), np.array([[0.5]]))
    assert acc.summary(["a"])["map50"] == 1.0
    below = det.APAccumulator(1, thresholds=[0.5])
    below.add(np.array([0.9]), np.array([0]), np.array([0]), np.array([[0.4999]]))
    assert below.summary(["a"])["map50"] == 0.0
    # detections of image 1 never match objects of image 2: the detection sits on an image without the
    # object (false positive) and the object of image 2 is missed -> AP 0 with support 1
    apart = det.APAccumulator(1, thresholds=[0.5])
    apart.add(np.array([0.9]), np.array([0]), np.array([], dtype=int), np.zeros((1, 0)))
    apart.add(np.array([], dtype=float), np.array([], dtype=int), np.array([0]), np.zeros((0, 1)))
    assert apart.summary(["a"])["map50"] == 0.0
    assert apart.summary(["a"])["per_class"]["a"]["support"] == 1
    # class 1 only has a false positive and no object: it is not in per_class nor in the mean
    mixed = det.APAccumulator(2, thresholds=[0.5])
    mixed.add(np.array([0.9, 0.8]), np.array([0, 1]), np.array([0]), np.array([[1.0], [0.0]]))
    out = mixed.summary(["a", "b"])
    assert list(out["per_class"]) == ["a"]
    assert out["map50"] == 1.0


def test_mask_iou_without_masks(det):
    """Empty mask sets give empty IoU matrices (regression for GitHub issue #344)."""
    assert det.mask_iou(np.zeros((0, 4, 4)), np.ones((1, 4, 4))).shape == (0, 1)
    assert det.mask_iou(np.ones((1, 4, 4)), np.zeros((0, 4, 4))).shape == (1, 0)
    assert det.mask_iou(np.zeros((0, 4, 4)), np.zeros((0, 4, 4))).shape == (0, 0)
    acc = det.APAccumulator(1)
    ious = det.mask_iou(np.zeros((0, 4, 4)), np.ones((1, 4, 4)))
    acc.add(np.zeros(0), np.zeros(0, dtype=int), np.array([0]), ious)
    # the object is missed: AP 0 but it counts in the support
    out = acc.summary(["a"])
    assert out["map50"] == 0.0
    assert out["per_class"]["a"]["support"] == 1


def test_oks_worked_example(det):
    """OKS of two keypoints with a known offset, with and without a visible second point."""
    gt = np.array([[[10.0, 10.0, 2.0], [30.0, 30.0, 2.0]]])
    pred = np.array([[[12.0, 14.0, 1.0], [30.0, 30.0, 1.0]]])
    areas = np.array([100.0])
    # K = 2 -> sigma = 1/K = 0.5, variance (2 * sigma)^2 = 1, denominator 2 * variance * area = 200
    # point 1: d^2 = 2^2 + 4^2 = 20 -> exp(-20 / 200) = exp(-0.1); point 2 is exact -> exp(0) = 1
    assert det.oks_matrix(pred, gt, areas)[0, 0] == pytest.approx((math.exp(-0.1) + 1) / 2)
    # point 2 not labelled (visibility 0): only point 1 counts -> OKS = exp(-0.1)
    hidden = gt.copy()
    hidden[0, 1, 2] = 0
    assert det.oks_matrix(pred, hidden, areas)[0, 0] == pytest.approx(math.exp(-0.1))
    # no visible point: the object matches nothing
    blind = np.zeros((1, 2, 3))
    assert det.oks_matrix(pred, blind, areas)[0, 0] == 0.0


@pytest.mark.xfail(strict=True, reason="#345: COCO ignores objects without labelled keypoints")
def test_keypoint_map_ignores_objects_without_points(det):
    """An object with no labelled keypoint should not count as a missed object in the keypoint mAP."""
    torch = pytest.importorskip("torch")
    model = FakeModel(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "keypoints": torch.tensor([[[2.0, 2.0, 1.0], [5.0, 5.0, 1.0], [8.0, 8.0, 1.0]]]),
            }
        ]
    )
    spec = {"task": "keypoints", "classes": ["person"]}
    result = det.evaluate(model, keypoint_loader(torch), spec, "cpu", torch)
    # the first person is found exactly (OKS 1); the second has no point and is counted as missed today:
    # recall stops at 1/2 -> 51/101 instead of 1.0
    assert result["kp_map50"] == 1.0


@pytest.mark.xfail(strict=True, reason="#346: evaluation uses the model's score_thresh")
def test_detection_map_does_not_depend_on_score_thresh(det):
    """A perfect low-score detection should count: evaluation must not inherit the model's score threshold."""
    torch = pytest.importorskip("torch")
    model = ThresholdModel(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.3]),
                "labels": torch.tensor([1]),
            }
        ],
        score_thresh=0.5,
    )
    spec = {"task": "detection", "classes": ["a"]}
    result = det.evaluate(model, single_box_loader(torch), spec, "cpu", torch)
    # the detection (score 0.3) is dropped by the model's own 0.5 cut-off today -> mAP 0 instead of 1
    assert result["map50"] == 1.0


# ---------------------------------------------------------------------------------------------------------
# Runner aggregation
# ---------------------------------------------------------------------------------------------------------


def test_rank_children_mean_and_population_std():
    """Fold scores are averaged without weights; the spread is the population std (ddof = 0)."""
    entries = [child("c1", 1, 0), child("c2", 1, 1), child("c3", 1, 2)]
    entries += [child("c4", 2, 0), child("c5", 2, 1), child("c6", 2, 2)]
    results: dict[str, Any] = {
        "c1": {"val": {"map50": 0.6, "images": 10}},
        "c2": {"val": {"map50": 0.7, "images": 100}},
        "c3": {"val": {"map50": 0.8, "images": 1}},
        "c4": {"val": {"map50": 0.9}},
        "c5": {"val": {"map50": 0.5}},
        "c6": None,  # failed child
    }
    out = rank_children(entries, results, "map50", 3)
    first, second = out["ranking"]
    # combo 1: mean (0.6 + 0.7 + 0.8) / 3 = 0.7 whatever the fold sizes (unweighted, not a pooled metric);
    # population std = sqrt(((0.1)^2 + 0 + (0.1)^2) / 3) = sqrt(0.02 / 3) = 0.0816497 (ddof 0, not 0.1)
    assert (first["combo"], first["rank"], first["complete"]) == (1, 1, True)
    assert first["mean"] == pytest.approx(0.7)
    assert first["std"] == pytest.approx(math.sqrt(0.02 / 3))
    assert first["std"] == pytest.approx(statistics.pstdev([0.6, 0.7, 0.8]))
    assert first["std"] != pytest.approx(statistics.stdev([0.6, 0.7, 0.8]))
    # combo 2 has the same mean 0.7 from its two finished folds but one failed -> incomplete, ranked second
    assert (second["combo"], second["rank"], second["complete"]) == (2, 2, False)
    assert second["mean"] == pytest.approx(0.7)
    assert second["scores"] == {"c4": 0.9, "c5": 0.5}
    # the best fold of the best combination is c3; best_value is the mean, best_fold_value the fold score
    assert out["best_combo"] == 1
    assert out["best_child"] == "c3"
    assert out["best_value"] == pytest.approx(0.7)
    assert out["best_fold_value"] == pytest.approx(0.8)
    assert out["folds"] == 3
