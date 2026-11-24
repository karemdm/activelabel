import json

import pytest
from sample_images import make_image
from worker_loader import load_worker


@pytest.fixture(scope="module")
def worker():
    return load_worker("classification")


def test_worker_metrics(worker):
    truth, pred = [0, 0, 1, 1, 2], [0, 1, 1, 1, 0]
    ev = worker.evaluation(truth, pred, ["a", "b", "c"])
    assert ev["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert ev["accuracy"] == pytest.approx(0.6)
    assert ev["per_class"]["a"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert ev["per_class"]["b"]["precision"] == pytest.approx(2 / 3) and ev["per_class"]["b"]["recall"] == 1
    assert ev["per_class"]["c"]["f1"] == 0
    assert ev["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert worker.macro_f1([[2, 0], [0, 0]]) == 1.0  # a class absent everywhere does not count
    assert worker.accuracy([[0, 0], [0, 0]]) == 0.0


def test_worker_spec_and_plans(worker, tmp_path):
    spec = {
        "task": "classification",
        "model": "resnet18",
        "family": "torchvision",
        "start": "transfer",
        "params": {},
        "classes": ["a"],
        "parts": {"train": [1], "val": [1]},
        "output_dir": "x",
        "metric": "accuracy",
    }
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert worker.load_spec(file)["model"] == "resnet18"
    for change, message in (
        ({"fold": "all"}, "comes later"),
        ({"task": "detection"}, "classification"),
        ({"metric": "map"}, "metric"),
        ({"parts": {"train": [1], "val": []}}, "validation"),
    ):
        file.write_text(json.dumps({**spec, **change}))
        with pytest.raises(worker.SpecError, match=message):
            worker.load_spec(file)
    assert worker.augmentation_plan("none", 64) == [("Resize", {"size": (64, 64)})]
    assert len(worker.augmentation_plan("strong", 64)) > len(worker.augmentation_plan("light", 64))
    with pytest.raises(worker.SpecError):
        worker.augmentation_plan("x", 64)
    assert worker.head_path("resnet18") == "fc" and worker.head_path("mobilenet_v3_large") == "classifier.3"


def test_worker_dataset_needs_torch(worker, tmp_path):
    pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")
    image = make_image(tmp_path / "i.png", fmt="PNG")
    data = worker.ImageDataset([{"path": str(image), "label": 1}], transforms.ToTensor())
    tensor, label = data[0]
    assert label == 1 and tensor.shape[0] == 3
