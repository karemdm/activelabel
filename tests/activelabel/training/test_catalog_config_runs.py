import pytest
import yaml

from activelabel.training.catalog import MODELS, PROJECT_TYPES, TASK_KINDS, get_model, models_for
from activelabel.training.config import MAX_GRID, ConfigError, RunConfig, validate
from activelabel.training.runs import RunBusy, RunNotFound, RunStore


def test_catalog_consistency():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))
    for m in MODELS:
        assert m.packages and m.task_kind in TASK_KINDS and m.types
        for spec in m.param_specs():
            assert spec.check(spec.default) == spec.default, (m.id, spec.name)
            if spec.kind == "choice":
                assert spec.default in spec.choices
        assert m.param("image_size").default == m.image_size
    for t in PROJECT_TYPES:
        assert models_for(t)
    assert {m.task_kind for m in models_for("segmentation")} == {"semantic", "instance"}
    for wanted in (
        "resnet18",
        "vit_b_16",
        "ssdlite320_mobilenet_v3_large",
        "yolov8n",
        "yolo11l-seg",
        "yolo11m-pose",
    ):
        assert get_model(wanted)
    assert "ultralytics" in get_model("yolov8s").packages


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


def test_run_store_numbers_and_files(tmp_path):
    store = RunStore(tmp_path)
    assert store.list() == []
    first = store.create(_cfg())
    assert first["number"] == 1 and first["status"] == "queued" and first["children"] == []
    assert (tmp_path / "runs" / "r1" / "config.yaml").exists()
    assert store.config(1) == _cfg()
    store.create(_cfg())
    store.delete(2)
    assert store.create(_cfg())["number"] == 3  # never reused
    assert [r["number"] for r in store.list()] == [3, 1]

    store.update(1, status="running", started="t")
    assert store.get(1)["status"] == "running" and store.get(1)["created"] == first["created"]
    with pytest.raises(RunBusy):
        store.delete(1)
    with pytest.raises(ValueError):
        store.update(1, status="weird")
    store.update(1, status="done")
    assert store.metrics(1) == [] and store.results(1) is None
    folder = tmp_path / "runs" / "r1"
    (folder / "metrics.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n{"epo', encoding="utf-8")
    (folder / "results.json").write_text('{"best": 2}', encoding="utf-8")
    assert store.metrics(1) == [{"epoch": 1}, {"epoch": 2}] and store.results(1) == {"best": 2}
    store.delete(1)
    with pytest.raises(RunNotFound):
        store.get(1)
