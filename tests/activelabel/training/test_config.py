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


AUGS = [{"name": "hflip", "params": {"p": 0.3}}, {"name": "blur"}]


def _checked_augs():
    return [
        {"name": "hflip", "params": {"p": 0.3}},
        {"name": "blur", "params": {"p": 0.2, "sigma_max": 1.5}},
    ]


def test_augmentations_round_trip():
    cfg = validate(_cfg(augmentations=AUGS), "classification")
    assert cfg.augmentations == _checked_augs()
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    assert RunConfig.from_yaml(cfg.to_yaml()) == cfg
    assert yaml.safe_load(cfg.to_yaml())["augmentations"] == _checked_augs()


def test_old_configs_have_no_augmentations_key():
    assert "augmentations" not in _cfg().to_dict()
    old = {"dataset": 1, "model": "resnet18", "params": {"epochs": 3}}
    loaded = RunConfig.from_dict(old)
    assert loaded.augmentations is None
    ok = validate(loaded, "classification")
    assert ok.augmentations is None and "augmentation" not in ok.params
    assert "augmentations" not in ok.to_dict()


def test_list_sets_the_custom_level_and_sorts():
    ok = validate(_cfg(augmentations=[{"name": "blur"}, {"name": "hflip"}]), "classification")
    assert ok.params["augmentation"] == "custom"
    assert [e["name"] for e in ok.augmentations] == ["hflip", "blur"]
    explicit = validate(_cfg(params={"augmentation": "custom"}, augmentations=[]), "classification")
    assert explicit.augmentations == []


@pytest.mark.parametrize(
    "kw, code",
    [
        ({"params": {"augmentation": "light"}, "augmentations": AUGS}, "augmentation_list_needs_custom"),
        (
            {"mode": "grid", "grid": {"augmentation": ["custom", "none"]}, "augmentations": AUGS},
            "augmentation_list_needs_custom",
        ),
        ({"params": {"augmentation": "custom"}}, "augmentation_custom_needs_list"),
        ({"mode": "grid", "grid": {"augmentation": ["light", "custom"]}}, "augmentation_custom_needs_list"),
        ({"augmentations": [{"name": "nope"}]}, "augmentation_unknown"),
        ({"augmentations": "hflip"}, "augmentations_not_list"),
    ],
)
def test_augmentation_errors(kw, code):
    with pytest.raises(ConfigError) as info:
        validate(_cfg(**kw), "classification")
    assert info.value.code == code


def test_grid_children_keep_the_augmentations():
    grid = {"lr": [0.1, 0.01]}
    cfg = validate(_cfg(mode="grid", grid=grid, augmentations=AUGS, fold="all"), "classification", folds=2)
    assert len(cfg.expand()) == 2 and len(cfg.children(2)) == 4
    for child in [*cfg.expand(), *cfg.children(2)]:
        assert child.augmentations == _checked_augs() and child.params["augmentation"] == "custom"


@pytest.mark.parametrize(
    "kw, code",
    [
        ({"params": {"epochs": 10, "warmup_epochs": 30}}, "warmup_exceeds_epochs"),
        ({"params": {"epochs": 2, "warmup_epochs": 3, "scheduler": "onecycle"}}, "warmup_exceeds_epochs"),
        ({"params": {"epochs": 3, "patience": 50}}, "patience_exceeds_epochs"),
        ({"params": {"epochs": 3, "patience": 3}}, "patience_exceeds_epochs"),
        (
            {"mode": "grid", "grid": {"epochs": [3, 20]}, "params": {"warmup_epochs": 5}},
            "warmup_exceeds_epochs",
        ),
    ],
)
def test_a_warmup_or_patience_longer_than_the_training_is_refused(kw, code):
    """#546: the combination makes no sense, so the server says why."""
    with pytest.raises(ConfigError) as error:
        validate(_cfg(**kw), "classification")
    assert error.value.code == code


def test_a_warmup_or_patience_that_fits_is_accepted():
    ok = validate(_cfg(params={"epochs": 10, "warmup_epochs": 10, "patience": 5}), "classification")
    assert ok.params["patience"] == 5


def test_the_range_message_is_not_in_scientific_notation():
    with pytest.raises(ConfigError, match=r"at most 2147483647\."):
        validate(_cfg(params={"seed": 2**31}), "classification")
