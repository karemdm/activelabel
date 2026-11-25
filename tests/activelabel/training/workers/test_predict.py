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
