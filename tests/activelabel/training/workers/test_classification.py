import json

import pytest
from PIL import Image
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


def test_worker_spec_and_head_path(worker, tmp_path):
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
    assert worker.head_path("resnet18") == "fc" and worker.head_path("mobilenet_v3_large") == "classifier.3"


def test_worker_dataset_needs_torch(worker, tmp_path):
    pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")
    image = make_image(tmp_path / "i.png", fmt="PNG")
    data = worker.ImageDataset([{"path": str(image), "label": 1}], transforms.ToTensor())
    tensor, label = data[0]
    assert label == 1 and tensor.shape[0] == 3


def test_eval_transform_uses_the_resize_policy(worker):
    photo = Image.new("RGB", (64, 32), (255, 255, 255))
    boxed = worker.eval_transform(16, "letterbox")(photo)
    assert tuple(boxed.shape) == (3, 16, 16)
    assert float(boxed[0, 0, 0]) == pytest.approx(0.0, abs=0.05)  # the band is the mean colour
    assert float(boxed[0, 8, 8]) > 2.0  # white, normalised
    assert float(worker.eval_transform(16, "stretch")(photo)[0, 0, 0]) > 2.0


# -- resuming an interrupted training (#321) ------------------------------------------


class Interrupted(Exception):
    """Stands for the app closing in the middle of a training."""


class Events:
    """Collects the events of a training; raises :class:`Interrupted` after the epoch ``stop``."""

    def __init__(self, stop=0):
        self.stop, self.seen = stop, []

    def __call__(self, event, **fields):
        self.seen.append((event, fields))
        if event == "epoch" and fields["epoch"] == self.stop:
            raise Interrupted


def small_spec(tmp_path, folder):
    paths = [make_image(tmp_path / f"i{k}.png", seed=k, fmt="PNG") for k in range(8)]
    items = [{"path": str(p), "label": k % 2} for k, p in enumerate(paths)]
    params = {
        "epochs": 3,
        "batch_size": 4,
        "lr": 0.01,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "plateau",
        "warmup_epochs": 1,
        "image_size": 32,
        "resize": "letterbox",
        "augmentation": "light",
        "patience": 0,
        "seed": 3,
        "start": "scratch",
        "backbone_lr_factor": 0.1,
        "amp": "auto",
        "label_smoothing": 0.0,
    }
    return {
        "task": "classification",
        "model": "resnet18",
        "family": "torchvision",
        "start": "scratch",
        "params": params,
        "seed": 3,
        "metric": "accuracy",
        "classes": ["a", "b"],
        "fold": None,
        "parts": {"train": items[:6], "val": items[6:]},
        "output_dir": str(tmp_path / folder),
        "device": "cpu",
        "augmentations": [{"name": "hflip", "params": {"p": 0.5}}],
        "digest": "d1",
    }


def epochs_of(events):
    return [
        (f["epoch"], round(f["train_loss"], 6), round(f["val_loss"], 6))
        for e, f in events.seen
        if e == "epoch"
    ]


def test_a_resumed_training_ends_like_a_continuous_one(worker, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    whole = Events()
    monkeypatch.setattr(worker, "emit", whole)
    continuous = worker.train(small_spec(tmp_path, "whole"))

    first = Events(stop=2)
    monkeypatch.setattr(worker, "emit", first)
    spec = small_spec(tmp_path, "cut")
    with pytest.raises(Interrupted):
        worker.train(spec)
    second = Events()
    monkeypatch.setattr(worker, "emit", second)
    resumed = worker.train({**spec, "resume": True})

    assert ("resumed", {"epoch": 2}) in second.seen
    assert epochs_of(first) + epochs_of(second) == epochs_of(whole)
    assert resumed["val"] == continuous["val"] and resumed["best_epoch"] == continuous["best_epoch"]
    a = torch.load(tmp_path / "whole" / "last.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "cut" / "last.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resume_refuses_a_changed_training_or_no_checkpoint(worker, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "emit", Events(stop=1))
    spec = small_spec(tmp_path, "run")
    with pytest.raises(worker.options.CodedFailure, match="no checkpoint"):
        worker.train({**spec, "resume": True})
    with pytest.raises(Interrupted):
        worker.train(spec)
    with pytest.raises(worker.options.CodedFailure) as changed:
        worker.train({**spec, "resume": True, "digest": "other"})
    assert changed.value.code == "resume_config_changed"


# -- class imbalance (#147) ------------------------------------------------------------


def test_label_counts(worker):
    items = [{"label": 0}, {"label": 1}, {"label": 1}]
    assert worker.label_counts(items, 3) == [1, 2, 0]


def test_training_with_inverse_weights_and_focal_loss(worker, tmp_path, monkeypatch):
    events = Events()
    monkeypatch.setattr(worker, "emit", events)
    spec = small_spec(tmp_path, "imbalanced")
    spec["params"] = {**spec["params"], "class_weights": "inverse", "loss": "focal", "focal_gamma": 1.5}
    results = worker.train(spec)
    assert results["epochs_run"] == 3
    assert any(e == "log" and "Class weights" in f["message"] for e, f in events.seen)


@pytest.mark.parametrize(
    ("model_id", "layer"),
    [
        ("mobilenet_v2", "Linear"),
        ("squeezenet1_1", "Conv2d"),
        ("densenet121", "Linear"),
        ("shufflenet_v2_x0_5", "Linear"),
        ("mnasnet0_5", "Linear"),
        ("efficientnet_b0", "Linear"),
    ],
)
def test_build_model_replaces_the_head_of_each_architecture(worker, model_id, layer):
    """#618: every architecture's classifier layer is found and gets one output per class."""
    spec = {
        "classes": ["a", "b", "c"],
        "model": model_id,
        "family": "torchvision",
        "params": {"image_size": 96},
    }
    model, head = worker.build_model(spec, False)
    assert type(head).__name__ == layer
    model.eval()
    with worker.torch.no_grad():
        assert tuple(model(worker.torch.zeros(1, 3, 96, 96)).shape) == (1, 3)


def test_networks_with_auxiliary_heads_give_plain_logits_in_training(worker):
    spec = {
        "classes": ["a", "b"],
        "model": "googlenet",
        "family": "torchvision",
        "params": {"image_size": 96},
    }
    model, _ = worker.build_model(spec, False)
    model.train()
    assert isinstance(model(worker.torch.zeros(2, 3, 96, 96)), worker.torch.Tensor)  # no GoogLeNetOutputs


def test_networks_with_a_fixed_input_refuse_another_size(worker):
    for model_id, side in (("vit_b_32", 224), ("vit_h_14", 518), ("maxvit_t", 224)):
        spec = {
            "classes": ["a"],
            "model": model_id,
            "family": "torchvision",
            "params": {"image_size": side + 32},
        }
        with pytest.raises(worker.SpecError, match=f"only takes {side} pixel"):
            worker.build_model(spec, False)
