import numpy as np
import pytest
from PIL import Image
from worker_loader import load_worker

try:  # torch lives only in the worker environment; tests that need it are skipped without it
    import torch
except ImportError:
    torch = None


@pytest.fixture(scope="module")
def module():
    return load_worker("predict")


def test_worker_skips_unmapped_categories(module):
    classes = [None, "car", None]  # category 1 unmapped, 2 → car, 3 unmapped
    out = module.detection_labels(
        [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
        [1, 2, 3],
        [0.9, 0.9, 0.9],
        100,
        100,
        classes,
        0.5,
        "m1",
    )
    assert [o["label"] for o in out] == ["car"]


def test_classification_labels_need_the_threshold(module):
    assert module.classification_labels([0.1, 0.7, 0.2], ["a", "b", "c"], 0.5, "m1") == [
        {"label": "b", "score": 0.7, "model": "m1"}
    ]
    assert module.classification_labels([0.4, 0.3, 0.3], ["a", "b", "c"], 0.5, "m1") == []


def test_mask_step_and_clamp(module):
    assert module.mask_step(10, 10) == 1
    assert module.mask_step(module.MAX_MASK_SIDE * 3, 5) == 3
    assert module._clamp(-1) == 0.0 and module._clamp(2) == 1.0 and module._clamp(0.3) == 0.3


def test_encode_mask_run_lengths(module):
    mask = np.array([[True, True, False], [False, True, True]])
    assert module.encode_mask(mask) == {"size": [2, 3], "counts": [0, 2, 2, 2]}
    assert module.encode_mask(np.zeros((2, 2), dtype=bool)) == {"size": [2, 2], "counts": [4]}


class _FakeDetector:
    """Returns one box at fixed input pixels and remembers the size of the image it was given."""

    def __call__(self, images):
        self.shape = tuple(images[0].shape)
        return [
            {
                "boxes": torch.tensor([[10.0, 35.0, 30.0, 45.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
                "keypoints": torch.tensor([[[10.0, 35.0, 1.0]]]),
                "masks": torch.ones(1, 1, *self.shape[1:]),
            }
        ]


def test_detect_takes_a_letterbox_prediction_back_to_the_photo(module):
    model = _FakeDetector()
    found = module.detect(model, Image.new("RGB", (400, 200)), "keypoints", "cpu", 100, "letterbox")
    assert model.shape == (3, 100, 100)
    assert found["boxes"][0] == pytest.approx([40, 40, 120, 80])
    assert found["keypoints"][0][0][:2] == pytest.approx([40, 40])
    legacy = module.detect(model, Image.new("RGB", (400, 200)), "instance", "cpu")
    assert model.shape == (3, 200, 400) and legacy["boxes"][0] == pytest.approx([10, 35, 30, 45])
    assert legacy["masks"][0].shape == (200, 400)
    boxed = module.detect(model, Image.new("RGB", (400, 200)), "instance", "cpu", 100, "letterbox")
    assert boxed["masks"][0].shape == (200, 400) and boxed["masks"][0].all()


BOXES = [[0, 0, 50, 50], [2, 2, 52, 52], [60, 60, 70, 70], [0, 0, 49, 49], [80, 80, 82, 82]]
LABELS = [1, 1, 1, 2, 1]
SCORES = [0.9, 0.8, 0.7, 0.85, 0.6]


def test_select_detections_applies_nms_per_class_by_default(module):
    picked = module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", "b"], 0.5, {"nms_iou": 0.5})
    assert picked == [0, 3, 2, 4]  # box 1 overlaps box 0 (same class); box 3 is another class


def test_select_detections_can_apply_nms_across_classes(module):
    opts = {"nms_iou": 0.5, "class_agnostic_nms": True}
    assert module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", "b"], 0.5, opts) == [0, 2, 4]


def test_select_detections_threshold_size_most_and_classes(module):
    assert module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", "b"], 0.75) == [0, 3, 1]
    opts = {"min_size": 0.05}
    assert 4 not in module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", "b"], 0.5, opts)
    opts = {"max_detections": 2}
    assert module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", "b"], 0.5, opts) == [0, 3]
    assert module.select_detections(BOXES, LABELS, SCORES, 100, 100, ["a", None], 0.5) == [0, 1, 2, 4]


def test_classification_margin_and_classes_not_chosen(module):
    assert module.classification_labels([0.5, 0.45, 0.05], ["a", "b", "c"], 0.4, "m1", margin=0.1) == []
    assert module.classification_labels([0.6, 0.3, 0.1], ["a", "b", "c"], 0.4, "m1", margin=0.1)
    # the best class was not chosen: no label, never the second best in its place
    assert module.classification_labels([0.7, 0.2, 0.1], ["", "b", "c"], 0.1, "m1") == []


class Head:
    def __init__(self) -> None:
        self.score_thresh, self.nms_thresh, self.detections_per_img = 0.05, 0.5, 100


class Detector:
    def __init__(self) -> None:
        self.roi_heads = Head()


def test_tune_detector_sets_threshold_nms_and_objects(module):
    model = Detector()
    module.tune_detector(model, 0.01, {"nms_iou": 0.3, "max_detections": 500})
    assert (model.roi_heads.score_thresh, model.roi_heads.nms_thresh) == (0.01, 0.3)
    assert model.roi_heads.detections_per_img == 500
    module.tune_detector(model, 0.5, {"max_detections": 5})
    assert model.roi_heads.score_thresh == 0.01 and model.roi_heads.detections_per_img == 500


def test_instance_masks_follow_the_mask_threshold(module):
    if torch is None:
        pytest.skip("needs torch")
    probabilities = torch.full((1, 1, 4, 4), 0.6)
    assert module.instance_masks(probabilities, 4, 4)[0].all()
    assert not module.instance_masks(probabilities, 4, 4, 0.7)[0].any()
