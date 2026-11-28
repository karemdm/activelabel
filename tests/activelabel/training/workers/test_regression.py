"""The regression worker (#619): its metrics, normalisation and loss, and a short training on CPU."""

import json
import math

import pytest
from sample_images import make_image
from worker_loader import load_worker


@pytest.fixture(scope="module")
def worker():
    return load_worker("regression")


def items(tmp_path, values):
    """Images with ``values`` (one list per image)."""
    paths = [make_image(tmp_path / f"i{k}.png", seed=k, fmt="PNG") for k in range(len(values))]
    return [{"path": str(p), "values": v} for p, v in zip(paths, values, strict=True)]


def spec_of(tmp_path, parts, **extra):
    params = {
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.01,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "plateau",
        "warmup_epochs": 0,
        "image_size": 32,
        "resize": "stretch",
        "augmentation": "light",
        "patience": 0,
        "seed": 3,
        "start": "scratch",
        "backbone_lr_factor": 0.1,
        "amp": "auto",
        "loss": "huber",
        "huber_delta": 1.0,
    }
    return {
        "task": "regression",
        "model": "resnet18",
        "family": "torchvision",
        "start": "scratch",
        "params": params,
        "seed": 3,
        "metric": "rmse",
        "lower_is_better": True,
        "classes": ["age", "height"],
        "fold": None,
        "parts": parts,
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "augmentations": [{"name": "hflip", "params": {"p": 0.5}}],
        "digest": "d",
        **extra,
    }


def test_metrics_per_target_and_over_the_targets(worker):
    truth = [[1.0, 10.0], [2.0, None], [3.0, 30.0]]
    predicted = [[1.0, 12.0], [3.0, 0.0], [3.0, 28.0]]
    ev = worker.evaluation(truth, predicted, ["a", "b"])
    a, b = ev["per_target"]["a"], ev["per_target"]["b"]
    assert (
        a["count"] == 3 and a["mae"] == pytest.approx(1 / 3) and a["rmse"] == pytest.approx(math.sqrt(1 / 3))
    )
    assert a["r2"] == pytest.approx(1 - 1 / 2)  # squared error 1, total 2
    assert b["count"] == 2 and b["mae"] == pytest.approx(2.0) and b["r2"] == pytest.approx(1 - 8 / 200)
    assert ev["rmse"] == pytest.approx((a["rmse"] + b["rmse"]) / 2)
    assert ev["points"] == {"a": [[1.0, 1.0], [2.0, 3.0], [3.0, 3.0]], "b": [[10.0, 12.0], [30.0, 28.0]]}


def test_a_target_without_spread_or_values_is_safe(worker):
    assert worker._one_target([], []) == {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "count": 0}
    assert worker._one_target([5.0, 5.0], [5.0, 6.0])["r2"] == 0.0  # R² is not defined: 0
    means, stds = worker.target_stats([{"values": [2.0, None]}, {"values": [4.0, None]}], 2)
    assert means == [3.0, 0.0] and stds == [1.0, 1.0]
    means, stds = worker.target_stats([{"values": [0.0]}, {"values": [10.0]}], 1)
    assert means == [5.0] and stds == [5.0]


def test_sample_points_keeps_at_most_the_limit(worker):
    truth = [[float(k)] for k in range(100)]
    points = worker.sample_points(truth, truth, ["a"], limit=10)["a"]
    assert len(points) == 10 and points[0] == [0.0, 0.0] and points[-1] == [90.0, 90.0]


def test_the_loss_counts_only_known_values(worker):
    torch = worker.torch
    output = torch.tensor([[1.0, 5.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert float(worker.MaskedLoss("mse")(output, target, mask)) == pytest.approx((1 + 4) / 2)
    assert float(worker.MaskedLoss("mae")(output, target, mask)) == pytest.approx((1 + 2) / 2)
    assert float(worker.MaskedLoss("huber", 1.0)(output, target, mask)) == pytest.approx((0.5 + 1.5) / 2)
    with pytest.raises(ValueError, match="Unknown loss"):
        worker.MaskedLoss("hinge")


def test_load_spec_checks_what_the_worker_trains(worker, tmp_path):
    base = spec_of(tmp_path, {"train": [1], "val": [1]})
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(base))
    assert worker.load_spec(file)["task"] == "regression"
    for change, message in (
        ({"task": "classification"}, "regression"),
        ({"metric": "accuracy"}, "metric"),
        ({"params": {**base["params"], "loss": "focal"}}, "loss"),
        ({"classes": []}, "target"),
        ({"fold": "all"}, "comes later"),
        ({"parts": {"train": [1], "val": []}}, "validation"),
    ):
        file.write_text(json.dumps({**base, **change}))
        with pytest.raises(worker.SpecError, match=message):
            worker.load_spec(file)


def test_a_short_training_keeps_the_best_epoch_and_the_normalisation(worker, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(worker, "emit", lambda event, **fields: seen.append((event, fields)))
    data = items(tmp_path, [[float(20 + k), 1.5 + k / 10] for k in range(6)] + [[30.0, None], [40.0, 1.9]])
    results = worker.train(spec_of(tmp_path, {"train": data[:4], "val": data[4:6], "test": data[6:]}))
    assert results["epochs_run"] == 2 and results["metric"] == "rmse"
    assert set(results["val"]) >= {"loss", "mae", "rmse", "r2", "per_target", "points", "targets"}
    assert results["test"]["per_target"]["height"]["count"] == 1  # the missing value is not scored
    epochs = [f for e, f in seen if e == "epoch"]
    assert {"val_rmse", "val_mae", "val_r2"} <= set(epochs[0])
    best = min(epochs, key=lambda f: f["val_rmse"])
    assert results["best_epoch"] == best["epoch"]  # the lowest error
    state = worker.torch.load(tmp_path / "out" / "best.pt", weights_only=False)
    assert state["task"] == "regression" and state["target_mean"] == pytest.approx([21.5, 1.65])
    assert json.loads((tmp_path / "out" / "results.json").read_text())["best_epoch"] == results["best_epoch"]


def test_the_loss_of_a_batch_for_the_memory_probe_uses_the_mask(worker):
    torch = worker.torch
    model = torch.nn.Linear(2, 2)
    batch = (torch.ones(3, 2), torch.zeros(3, 2), torch.tensor([[1.0, 0.0]] * 3))
    loss = worker.batch_loss(model, batch, "cpu", worker.MaskedLoss("mse"))
    expected = float(model(batch[0])[:, 0].pow(2).mean())
    assert float(loss) == pytest.approx(expected)
