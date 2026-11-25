"""The catalogue declares, per family and task, the hyperparameters its worker really reads.

The workers run in the separate training environment, so they are read here as source (``ast``), not imported:
every key a worker takes from ``params`` must be declared by each model it trains, and every parameter a model
declares must be read by its worker (``start`` is read from the run spec itself).
"""

import ast
from pathlib import Path

import pytest

from activelabel.training.catalog import (
    COMMON_PARAMS,
    MODELS,
    PROJECT_TYPES,
    TASK_KINDS,
    get_model,
    models_for,
)
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.runner import RESOLVED_PARAMS, WORKER_FILES, WORKERS_DIR

#: Families listed in the catalogue that no worker trains yet (the API refuses them, the form hides them).
NOT_TRAINABLE = {"ultralytics"}


def _is_params(node: ast.expr) -> bool:
    """``p`` or ``spec["params"]``: where a worker keeps the hyperparameters."""
    if isinstance(node, ast.Name):
        return node.id == "p"
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "params"
    )


def _reads(script: Path) -> tuple[set[str], set[str]]:
    """Keys a worker reads from its hyperparameters, and keys it reads from the run spec itself."""
    params: set[str] = set()
    spec: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
            if isinstance(key, str) and _is_params(node.value):
                params.add(key)
            elif isinstance(key, str) and isinstance(node.value, ast.Name) and node.value.id == "spec":
                spec.add(key)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_params(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            params.add(node.args[0].value)
    return params, spec


TRAINABLE = [m for m in MODELS if m.family not in NOT_TRAINABLE]


@pytest.mark.parametrize("model", TRAINABLE, ids=lambda m: m.id)
def test_every_declared_parameter_is_read_by_the_worker(model):
    params, spec = _reads(WORKERS_DIR / WORKER_FILES[model.task_kind])
    declared = {p.name for p in model.param_specs()}
    assert "start" in spec  # the start is a field of the run, not of params
    assert "augmentations" in spec  # the augmentation level reaches the worker as a list of techniques
    unread = declared - params - {"start"} - set(RESOLVED_PARAMS)
    assert not unread, f"{model.id}: declared but not read: {unread}"
    assert params <= declared, f"{model.id}: read but not declared: {params - declared}"


def test_untrainable_families_have_no_worker_reading_their_own_parameters():
    every_read = set().union(*(_reads(WORKERS_DIR / name)[0] for name in set(WORKER_FILES.values())))
    own = {p.name for m in MODELS if m.family in NOT_TRAINABLE for p in m.params}
    assert own and not own & every_read  # when a YOLO worker comes, move the family out of NOT_TRAINABLE


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.id)
def test_defaults_are_valid_for_each_model(model):
    for spec in model.param_specs():
        assert spec.check(spec.default) == spec.default, f"{model.id}.{spec.name}"
        if spec.min is not None:
            assert spec.min <= spec.max


@pytest.mark.parametrize(
    ("model_id", "size", "low", "high"),
    [
        ("resnet18", 224, 32, 2048),
        ("efficientnet_b0", 224, 32, 2048),
        ("efficientnet_b3", 288, 32, 2048),
        ("vit_b_16", 224, 224, 224),  # fixed position embeddings
        ("fasterrcnn_resnet50_fpn", 640, 32, 2048),
        ("ssdlite320_mobilenet_v3_large", 320, 320, 320),  # fixed input
        ("deeplabv3_resnet50", 512, 32, 2048),
        ("yolov8n", 640, 32, 2048),
        ("yolo11s-seg", 640, 32, 2048),
    ],
)
def test_image_size_per_family(model_id, size, low, high):
    spec = get_model(model_id).param("image_size")
    assert (spec.default, spec.min, spec.max) == (size, low, high)


def test_family_defaults():
    yolo = {p.name: p.default for p in get_model("yolov8n").param_specs()}
    assert (yolo["optimizer"], yolo["lr"], yolo["weight_decay"]) == ("sgd", 0.01, 0.0005)
    assert {p.name for p in get_model("yolov8n").params} == {"mosaic", "close_mosaic"}
    detection = {p.name: p.default for p in get_model("retinanet_resnet50_fpn").param_specs()}
    assert (detection["optimizer"], detection["lr"], detection["batch_size"]) == ("sgd", 0.005, 4)
    assert get_model("vit_b_16").param("lr").default == 0.0001
    assert get_model("resnet18").param("lr").default == 0.001
    semantic = {p.name for p in get_model("deeplabv3_resnet50").params}
    assert semantic == set()  # no score threshold nor backbone layers: the worker reads neither


def test_limits_do_not_change_the_common_parameters():
    assert get_model("vit_b_16").param("image_size").max == 224
    assert next(p for p in COMMON_PARAMS if p.name == "image_size").max == 2048


def test_validation_uses_the_model_limits():
    base = {"dataset": 1, "model": "vit_b_16", "start": "transfer", "mode": "single"}
    with pytest.raises(ConfigError):
        validate(RunConfig.from_dict({**base, "params": {"image_size": 256}}), "classification")
    assert validate(RunConfig.from_dict({**base, "params": {"image_size": 224}}), "classification")


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
