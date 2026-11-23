import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.datasets import DatasetStore
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training import runner as runner_module
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig
from activelabel.training.runner import RunnerError, build_spec, worker_script
from activelabel.training.runs import RunStore

WORKER = Path(__file__).parents[3] / "src" / "activelabel" / "training" / "workers" / "detection.py"


@pytest.fixture(scope="module")
def w():
    spec = importlib.util.spec_from_file_location("detection_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["detection_worker"] = module
    spec.loader.exec_module(module)
    return module


def test_iou(w):
    a = [[0, 0, 10, 10]]
    got = w.box_iou(a, [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])[0]
    assert got == pytest.approx([1.0, 1 / 3, 0.0])
    assert w.box_iou([], a).shape == (0, 1)
    m1 = np.zeros((1, 4, 4), np.uint8)
    m1[0, :2] = 1
    m2 = np.zeros((1, 4, 4), np.uint8)
    m2[0, :, :1] = 1
    assert w.mask_iou(m1, m2)[0, 0] == pytest.approx(2 / 10)


def test_average_precision(w):
    assert w.average_precision([0.9], [True], 1) == 1.0
    assert w.average_precision([0.9], [True], 2) == pytest.approx(51 / 101)  # recall stops at 0.5
    assert w.average_precision([], [], 2) == 0.0
    # hit, miss, hit: precision 1, 1/2, 2/3 -> envelope 1, 2/3, 2/3; recall .5, .5, 1
    ap = w.average_precision([0.9, 0.8, 0.7], [True, False, True], 2)
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_accumulator(w):
    acc = w.APAccumulator(2)
    gt = np.array([0, 1])
    ious = np.array([[1.0, 0.0], [0.0, 0.6]])  # class 1 matches up to IoU 0.60
    acc.add(np.array([0.9, 0.8]), np.array([0, 1]), gt, ious)
    out = acc.summary(["a", "b"])
    assert out["per_class"]["a"]["ap50_95"] == 1.0
    assert out["per_class"]["b"]["ap50"] == 1.0
    assert out["per_class"]["b"]["ap50_95"] == pytest.approx(0.3)  # thresholds .50 .55 .60 of ten
    assert out["map50"] == 1.0 and out["map50_95"] == pytest.approx(0.65)
    empty = w.APAccumulator(2)
    empty.add(np.array([0.5]), np.array([0]), np.array([], int), np.zeros((1, 0)))
    assert empty.summary(["a", "b"]) == {"map50": 0.0, "map50_95": 0.0, "per_class": {}}
    miss = w.APAccumulator(1)
    miss.add(np.array([]), np.array([], int), np.array([0]), np.zeros((0, 1)))
    assert miss.summary(["a"])["map50"] == 0.0


def test_targets(w):
    item = {
        "path": "x",
        "objects": [
            {"label": 1, "bbox": [0.1, 0.2, 0.5, 0.5]},
            {"label": 0, "polygons": [[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]]},
            {"label": 0, "bbox": [0.1, 0.1, 0.0, 0.0]},  # no area: dropped
        ],
    }
    t = w.build_target(item, 100, 50, "detection", 0)
    assert t["boxes"].tolist() == [[10.0, 10.0, 60.0, 35.0], [0.0, 0.0, 50.0, 25.0]]
    assert t["labels"].tolist() == [2, 1] and "masks" not in t
    t = w.build_target(item, 100, 50, "instance", 0)
    assert t["masks"].shape == (2, 50, 100) and t["masks"][1].sum() > 0 and t["masks"][1, 40, 90] == 0
    flipped = w.flip_target(t, 100)
    assert flipped["boxes"][0].tolist() == [40.0, 10.0, 90.0, 35.0]
    assert flipped["masks"].shape == t["masks"].shape
    empty = w.build_target({"path": "x", "objects": []}, 10, 8, "instance", 0)
    assert empty["boxes"].shape == (0, 4) and empty["masks"].shape == (0, 8, 10)


def test_polygon_mask_size(w):
    mask = w.polygon_mask([[[0, 0], [1, 0], [1, 1], [0, 1]]], 7, 5)
    assert mask.shape == (5, 7) and mask.sum() >= 30
    assert w.polygon_mask(None, 7, 5).sum() == 0


def test_keypoints(w):
    arr = w.keypoints_array([[0.5, 0.5, 2], [0.1, 0.1, 0]], 3, 10, 20)
    assert arr.tolist() == [[5.0, 10.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    item = {"path": "x", "objects": [{"label": 0, "keypoints": [[0.2, 0.2, 2], [0.4, 0.6, 1]]}]}
    t = w.build_target(item, 10, 10, "keypoints", 3)
    assert t["keypoints"].shape == (1, 3, 3) and t["boxes"].tolist() == [[2.0, 2.0, 4.0, 6.0]]
    gt = {"boxes": t["boxes"], "labels": t["labels"], "keypoints": t["keypoints"]}
    good = {**gt, "scores": np.array([0.9]), "keypoints": t["keypoints"] + 0.1}
    assert w.pck_counts(good, gt) == (2, 2)
    far = {**good, "keypoints": t["keypoints"] + 3.0}
    assert w.pck_counts(far, gt) == (0, 2)
    assert w.pck_counts({**good, "scores": np.array([0.1])}, gt) == (0, 2)


def test_plan_and_spec(w, tmp_path):
    assert w.augmentation_plan("light", "detection")["flip"] is True
    assert w.augmentation_plan("strong", "keypoints")["flip"] is False
    assert w.augmentation_plan("none", "instance")["flip"] is False
    with pytest.raises(w.SpecError):
        w.augmentation_plan("x", "detection")
    assert w.warmup_factor(1, 1000) == pytest.approx(0.001) and w.warmup_factor(500, 1000) == 1.0
    assert w.warmup_factor(1, 1) == 1.0
    spec = {
        "task": "detection",
        "model": "fasterrcnn_resnet50_fpn",
        "family": "torchvision",
        "start": "transfer",
        "params": {},
        "classes": ["a"],
        "parts": {"train": [1], "val": [1]},
        "output_dir": "x",
        "metric": "map50_95",
    }
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert w.load_spec(file)["model"] == "fasterrcnn_resnet50_fpn"
    for change, message in (
        ({"task": "classification"}, "not classification"),
        ({"family": "ultralytics"}, "torchvision"),
        ({"metric": "accuracy"}, "metric"),
        ({"task": "keypoints"}, "keypoints"),
        ({"parts": {"train": [1], "val": []}}, "validation"),
    ):
        file.write_text(json.dumps({**spec, **change}))
        with pytest.raises(w.SpecError, match=message):
            w.load_spec(file)


def test_torch_model_and_dataset(w, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image = make_image(tmp_path / "i.png", fmt="PNG")
    item = {"path": str(image), "objects": [{"label": 0, "bbox": [0.1, 0.1, 0.5, 0.5]}]}
    tensor, target = w.DetectionDataset([item], "detection", 0, True, 0.2)[0]
    assert tensor.shape[0] == 3 and target["boxes"].shape == (1, 4) and target["labels"].tolist() == [1]
    spec = {
        "model": "fasterrcnn_mobilenet_v3_large_fpn",
        "start": "scratch",
        "classes": ["a", "b"],
        "params": {"image_size": 64, "lr": 0.01, "weight_decay": 0.0, "optimizer": "sgd"},
    }
    model = w.build_model(spec, pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    assert w.make_optimizer(spec, model, torch).param_groups[0]["lr"] == 0.01


# -- runner ------------------------------------------------------------------------------------


def test_worker_mapping(tmp_path, monkeypatch):
    for kind in ("detection", "instance", "keypoints"):
        assert worker_script(kind).name == "detection.py"
    assert worker_script("classification").name == "classification.py"
    monkeypatch.setattr(runner_module, "WORKERS_DIR", tmp_path)
    with pytest.raises(RunnerError, match="comes later"):
        worker_script("semantic")  # no segmentation.py there
    (tmp_path / "segmentation.py").write_text("")
    assert worker_script("semantic").name == "segmentation.py"
    with pytest.raises(RunnerError, match="comes later"):
        worker_script("unknown")


@pytest.fixture
def det_project(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a", "b"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(10)]
    assert upload(client, 1, paths).status_code == 200
    ids = [r.id for r in state.manifest(1).all()]
    poly = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]]
    kps = [[0.2, 0.2, 2], [0.3, 0.3, 1], [0.4, 0.4, 0]]
    images = {
        i: [
            Label("a", bbox=[0.1, 0.1, 0.3, 0.3], polygons=poly, keypoints=kps[: 2 + k % 2]),
            Label("b"),
        ]
        for k, i in enumerate(ids[:9])
    }
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["a", "b"], images, {}))
    assert client.post("/api/projects/1/datasets", json={}).status_code == 200
    return state


def test_build_spec_objects(det_project):
    project_dir = det_project.store.dir(1)
    version = DatasetStore(project_dir).read(1)
    run_dir = project_dir / "runs" / "r1"
    for model_id, kind in (
        ("fasterrcnn_resnet50_fpn", "detection"),
        ("maskrcnn_resnet50_fpn", "instance"),
        ("keypointrcnn_resnet50_fpn", "keypoints"),
        ("deeplabv3_resnet50", "semantic"),
    ):
        model = get_model(model_id)
        config = RunConfig(dataset=1, model=model_id)
        spec = build_spec(project_dir, config, model, run_dir, version)
        assert spec["task"] == kind and spec["metric"] in ("map50_95", "miou")
        items = [i for p in spec["parts"].values() for i in p]
        assert items and all("label" not in i and Path(i["path"]).is_absolute() for i in items)
        objects = [o for i in items for o in i["objects"]]
        assert objects and all(o["label"] == 0 and o["bbox"] and o["polygons"] for o in objects)
        assert all("keypoints" in o for o in objects)
        assert ("num_keypoints" in spec) == (kind == "keypoints")
        if kind == "keypoints":
            assert spec["num_keypoints"] == 3


def test_warning_event(det_project, tmp_path):
    store = RunStore(det_project.store.dir(1))
    n = store.create(RunConfig(dataset=1, model="fasterrcnn_resnet50_fpn"))["number"]
    runner = runner_module.TrainingRunner(tmp_path / "ws", det_project.store.dir)
    state = {"results": None, "error": "", "last": 0.0, "warnings": []}
    line = json.dumps({"event": "warning", "message": "boxes only"})
    with (tmp_path / "log.txt").open("w", encoding="utf-8") as log:
        for _ in range(2):
            target = runner_module._Target(
                store.path(n), tmp_path / "log.txt", lambda **f: store.update(n, **f)
            )
            runner._handle(line, target, None, log, "val_map50_95", state)
    assert store.get(n)["warnings"] == ["boxes only"]
    assert (tmp_path / "log.txt").read_text(encoding="utf-8").count("boxes only") == 2
