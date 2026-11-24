import pytest
import yaml

from activelabel.training.config import MAX_GRID, ConfigError, RunConfig, validate


def _cfg(**kw):
    return RunConfig(**{"dataset": 1, "model": "resnet18", **kw})


@pytest.mark.parametrize(
    "kw, project, folds, text",
    [
        ({"model": "nope"}, "classification", 0, "Unknown model"),
        ({"model": "yolov8n"}, "classification", 0, "not for classification"),
        ({"params": {"epochs": 0}}, "classification", 0, "at least"),
        ({"params": {"epochs": 2.5}}, "classification", 0, "whole"),
        ({"params": {"optimizer": "x"}}, "classification", 0, "one of"),
        ({"params": {"foo": 1}}, "classification", 0, "not a hyperparameter"),
        ({"params": {"mosaic": True}}, "classification", 0, "not a hyperparameter"),
        ({"params": {"start": "scratch"}}, "classification", 0, "'start' option"),
        ({"start": "x"}, "classification", 0, "start"),
        ({"fold": 0}, "classification", 0, "no folds"),
        ({"fold": 3}, "classification", 3, "fold"),
        ({"metric": "miou"}, "classification", 0, "metric"),
        ({"mode": "grid"}, "classification", 0, "at least one list"),
        ({"grid": {"lr": [0.1]}}, "classification", 0, "single run"),
        ({"mode": "grid", "grid": {"lr": []}}, "classification", 0, "empty"),
        ({"mode": "grid", "grid": {"lr": [0.1, 0.1]}}, "classification", 0, "repeats"),
        ({"mode": "grid", "grid": {"epochs": list(range(1, 52))}}, "classification", 0, "at most"),
    ],
)
def test_validation_errors(kw, project, folds, text):
    with pytest.raises(ConfigError, match=text):
        validate(_cfg(**kw), project, folds)


def test_validation_normalises():
    ok = validate(_cfg(params={"epochs": 3.0}), "classification")
    assert ok.metric == "accuracy" and ok.fold is None and ok.params == {"epochs": 3}
    assert validate(_cfg(), "classification", 3).fold == "all"
    assert validate(_cfg(fold=2), "classification", 3).fold == 2
    assert validate(_cfg(model="lraspp_mobilenet_v3_large"), "segmentation").metric == "miou"
    assert validate(_cfg(model="maskrcnn_resnet50_fpn"), "segmentation").metric == "map50_95"


def test_yaml_round_trip_has_no_device_or_paths():
    cfg = validate(
        _cfg(mode="grid", grid={"lr": [0.01, 0.1], "start": ["transfer", "scratch"]}, params={"seed": 1}),
        "classification",
    )
    cfg.dataset_fingerprint = "abc"
    text = cfg.to_yaml()
    assert RunConfig.from_yaml(text) == cfg and text == RunConfig.from_yaml(text).to_yaml()
    keys = ["dataset", "dataset_fingerprint", "fold", "model", "start", "mode", "metric", "params", "grid"]
    assert list(yaml.safe_load(text)) == keys
    assert "device" not in text and "path" not in text
    for bad in ({"dataset": 1, "model": "resnet18", "device": "cuda"}, {"model": "x"}, []):
        with pytest.raises(ConfigError):
            RunConfig.from_dict(bad)
    with pytest.raises(ConfigError):
        RunConfig.from_yaml("a: [")


def test_grid_expansion_order_and_cap():
    grid = {"lr": [0.1, 0.01], "batch_size": [4, 8, 16], "start": ["scratch"]}
    cfg = validate(_cfg(mode="grid", params={"seed": 7}, grid=grid), "classification")
    assert cfg.grid_size() == 6
    runs = cfg.expand()
    pairs = [(r.params["batch_size"], r.params["lr"]) for r in runs]
    assert pairs == [(4, 0.1), (4, 0.01), (8, 0.1), (8, 0.01), (16, 0.1), (16, 0.01)]
    assert all(
        r.mode == "single" and not r.grid and r.start == "scratch" and r.params["seed"] == 7 for r in runs
    )
    assert [r.to_dict() for r in runs] == [r.to_dict() for r in cfg.expand()]
    assert _cfg().expand() == [_cfg()]
    big = {"epochs": list(range(1, 8)), "batch_size": list(range(1, 9))}  # 56
    with pytest.raises(ConfigError, match=str(MAX_GRID)):
        validate(_cfg(mode="grid", grid=big), "classification")
    fifty = {"epochs": list(range(1, 11)), "batch_size": [1, 2, 3, 4, 5]}
    assert validate(_cfg(mode="grid", grid=fifty), "classification").grid_size() == 50
    ten = {"epochs": list(range(1, 11))}
    folded = validate(_cfg(mode="grid", grid=ten, fold="all"), "classification", folds=5)
    assert folded.is_composite() and len(folded.children(5)) == 50
    with pytest.raises(ConfigError, match="grid and the folds make 60"):
        validate(_cfg(mode="grid", grid=ten, fold="all"), "classification", folds=6)
    kids = folded.children(5)
    assert [(k.params["epochs"], k.fold) for k in kids[:6]] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
    ]
    assert all(k.mode == "single" and not k.grid and not k.is_composite() for k in kids)
    assert not _cfg().is_composite() and _cfg().children(0) == [_cfg()]
