"""The Ultralytics worker, with a fake ``YOLO`` (no ultralytics is installed: see scripts/yolo_smoke.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sample_images import make_image
from worker_loader import load_worker

from activelabel.training.augmentations import preset
from activelabel.training.catalog import get_model

torch = pytest.importorskip("torch", reason="the YOLO worker needs torch (CI job 'training workers')")

CLASSES = ["a", "b"]


@pytest.fixture(scope="module")
def w():
    return load_worker("yolo")


# -- a fake of the parts of ultralytics the worker uses ----------------------------------------


def fake_metric(map50, map_, indexes=(0, 1)):
    return SimpleNamespace(
        map50=map50,
        map=map_,
        ap_class_index=list(indexes),
        ap50=[map50 for _ in indexes],
        ap=[map_ for _ in indexes],
    )


def fake_metrics(value, task="detection"):
    metrics = SimpleNamespace(box=fake_metric(value + 0.1, value), nt_per_class=[5, 7])
    if task == "instance":
        metrics.seg = fake_metric(value + 0.2, value + 0.05)
    if task == "keypoints":
        metrics.pose = fake_metric(value + 0.3, value + 0.06)
    return metrics


class FakeYOLO:
    """Stands in for ``ultralytics.YOLO``: trains ``len(curve)`` epochs, one metric value per epoch."""

    curve = [0.2, 0.5, 0.4, 0.3, 0.2]
    task = "detection"
    saved_first = False  # whether the model save comes before the validation callback
    failure: Exception | None = None
    instances: list = []

    def __init__(self, weights):
        self.weights = weights
        self.callbacks = {}
        self.train_kwargs = None
        self.val_kwargs = None
        self.data_seen = None
        FakeYOLO.instances.append(self)

    def add_callback(self, name, function):
        self.callbacks[name] = function

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        if FakeYOLO.failure is not None:
            raise FakeYOLO.failure
        if kwargs.get("resume"):
            args = torch.load(Path(self.weights), weights_only=False)["train_args"]
            start, project, epochs = int(args["epoch"]) + 1, args["project"], int(args["epochs"])
            self.data_seen = args["data"]
        else:
            start, project, epochs = 0, kwargs["project"], kwargs["epochs"]
        wdir = Path(project) / "yolo" / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        trainer = SimpleNamespace(
            epochs=epochs,
            train_loader=[0, 1, 2],
            label_loss_items=lambda tloss: {"box": 1.0, "cls": 0.5},
            tloss=None,
            lr={"lr/pg0": 0.01},
            last=wdir / "last.pt",
            stop=False,
            epoch=0,
            validator=SimpleNamespace(metrics=None),
        )
        order = (
            ("on_model_save", "on_fit_epoch_end")
            if self.saved_first
            else ("on_fit_epoch_end", "on_model_save")
        )
        for epoch in range(start, min(epochs, len(self.curve))):
            trainer.epoch = epoch
            self.callbacks["on_train_epoch_start"](trainer)
            for _ in range(3):
                self.callbacks["on_train_batch_end"](trainer)
            trainer.validator.metrics = fake_metrics(self.curve[epoch], self.task)
            checkpoint = {"epoch": epoch, "epochs": epochs, "train_args": {**kwargs, "epoch": epoch}}
            torch.save(checkpoint, trainer.last)
            for name in order:
                self.callbacks[name](trainer)
            if trainer.stop:
                break

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return fake_metrics(0.77, FakeYOLO.task)


@pytest.fixture
def fake(w, monkeypatch, tmp_path):
    FakeYOLO.instances = []
    FakeYOLO.failure = None
    FakeYOLO.saved_first = False
    FakeYOLO.curve = [0.2, 0.5, 0.4, 0.3, 0.2]
    FakeYOLO.task = "detection"
    settings = SimpleNamespace(updated=[], update=lambda data: settings.updated.append(data))
    monkeypatch.setattr(w, "YOLO", FakeYOLO)
    monkeypatch.setattr(w, "settings", settings)
    monkeypatch.chdir(tmp_path)
    return settings


def make_spec(tmp_path, model="yolov8n", task="detection", **params):
    """A spec with real images, the catalogue defaults and the given parameter changes."""
    defaults = {p.name: p.default for p in get_model(model).param_specs()}
    defaults.update({"epochs": 5, "image_size": 64, "batch_size": 2, **params})
    items = []
    for k in range(4):
        image = make_image(tmp_path / "src" / f"{k}.png", seed=k, size=(80, 40), fmt="PNG")
        objects = [{"label": k % 2, "bbox": [0.1, 0.2, 0.5, 0.5]}]
        if task == "instance":
            objects = [{"label": k % 2, "polygons": [[[0.1, 0.2], [0.6, 0.2], [0.6, 0.7]]]}]
        if task == "keypoints":
            points = [[0.2, 0.3, 2], [0.4, 0.5, 1]]
            objects = [{"label": 0, "bbox": [0.1, 0.2, 0.5, 0.5], "keypoints": points}]
        items.append({"path": str(image), "objects": objects})
    return {
        "task": task,
        "model": model,
        "family": "ultralytics",
        "start": "transfer",
        "params": defaults,
        "seed": defaults["seed"],
        "metric": "map50_95",
        "classes": CLASSES,
        "parts": {"train": items[:2], "val": items[2:]},
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "augmentations": preset("light", task),
        "digest": "d1",
        "num_keypoints": 2 if task == "keypoints" else 0,
    }


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


# -- pure helpers ----------------------------------------


def test_load_spec_checks_the_worker_can_train_it(w, tmp_path):
    spec = make_spec(tmp_path)
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert w.load_spec(file)["model"] == "yolov8n"
    for change, message in (
        ({"task": "classification"}, "accuracy, macro_f1"),  # its metrics are not those of boxes
        ({"task": "semantic"}, "not semantic"),
        ({"family": "torchvision"}, "not torchvision"),
        ({"start": "x"}, "Unknown start"),
        ({"metric": "accuracy"}, "metric"),
        ({"fold": "all"}, "all folds"),
        ({"parts": {"train": [1], "val": []}}, "validation"),
        ({"task": "keypoints", "num_keypoints": 0}, "no keypoints"),
    ):
        file.write_text(json.dumps({**spec, **change}))
        with pytest.raises(w.SpecError, match=message):
            w.load_spec(file)
    file.write_text(json.dumps({k: v for k, v in spec.items() if k != "classes"}))
    with pytest.raises(w.SpecError, match="no 'classes'"):
        w.load_spec(file)


def test_device_weights_and_schedule(w):
    assert [w.device_arg(d) for d in ("cpu", "mps", "cuda", "cuda:1")] == ["cpu", "mps", "0", "1"]
    assert w.weights_name("yolov8n", "transfer") == "yolov8n.pt" == w.weights_name("yolov8n", "finetune")
    assert w.weights_name("yolo11s-seg", "scratch") == "yolo11s-seg.yaml"
    assert w.schedule_args("cosine") == ({"cos_lr": True, "lrf": 0.01}, None)
    assert w.schedule_args("none") == ({"cos_lr": False, "lrf": 1.0}, None)
    assert w.schedule_args("onecycle") == ({"cos_lr": False, "lrf": 0.01}, "onecycle")


def test_augmentation_args_follow_the_techniques(w):
    off, flip, unsupported = w.augmentation_args([], False, "detection", 0)
    assert off["fliplr"] == off["hsv_v"] == off["degrees"] == off["scale"] == off["mosaic"] == 0.0
    assert flip is None and unsupported == []
    techniques = [
        {"name": "hflip", "params": {"p": 0.3}},
        {"name": "vflip", "params": {"p": 0.2}},
        {"name": "rotation", "params": {"degrees": 12.0}},
        {"name": "scale_jitter", "params": {"scale_min": 0.75, "scale_max": 1.5}},
        {"name": "translate", "params": {"fraction": 0.2}},
        {
            "name": "color_jitter",
            "params": {"brightness": 0.4, "saturation": 0.3, "hue": 0.1, "contrast": 0.2},
        },
        {"name": "blur", "params": {}},
        {"name": "grayscale", "params": {}},
    ]
    args, _, unsupported = w.augmentation_args(techniques, True, "detection", 0)
    assert (args["fliplr"], args["flipud"], args["degrees"], args["translate"]) == (0.3, 0.2, 12.0, 0.2)
    assert args["scale"] == 0.5 and (args["hsv_h"], args["hsv_s"], args["hsv_v"]) == (0.1, 0.3, 0.4)
    assert args["mosaic"] == 1.0 and unsupported == ["blur", "grayscale"]


def test_pose_flip_pairs_become_the_flip_index(w):
    flip = [{"name": "hflip", "params": {"p": 0.5, "flip_pairs": [[0, 2]]}}]
    _, index, _ = w.augmentation_args(flip, False, "keypoints", 4)
    assert index == [2, 1, 0, 3]
    assert w.augmentation_args([], False, "keypoints", 3)[1] == [0, 1, 2]


def test_object_extent_and_label_lines(w):
    assert w.object_extent({"bbox": [0.1, 0.2, 0.3, 0.4]}) == pytest.approx([0.1, 0.2, 0.4, 0.6])
    assert w.object_extent({"polygons": [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.9]]]}) == [0.1, 0.2, 0.5, 0.9]
    assert w.object_extent({"keypoints": [[0.2, 0.2, 2], [0.8, 0.9, 0], [0.4, 0.6, 1]]}) == [
        0.2,
        0.2,
        0.4,
        0.6,
    ]
    assert w.object_extent({}) is None and w.object_extent({"keypoints": [[0.5, 0.5, 0]]}) is None
    box = {"label": 1, "bbox": [0.1, 0.2, 0.4, 0.6]}
    assert w.label_lines([box], "detection", 0) == ["1 0.300000 0.500000 0.400000 0.600000"]
    poly = {"label": 0, "polygons": [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], [[0, 0], [1, 1]]]}
    assert w.label_lines([poly], "instance", 0) == ["0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000"]
    pose = {"label": 0, "bbox": [0.1, 0.1, 0.4, 0.4], "keypoints": [[0.2, 0.2, 2], [0.3, 0.4, 1]]}
    assert w.label_lines([pose], "keypoints", 2) == [
        "0 0.300000 0.300000 0.400000 0.400000 0.200000 0.200000 2 0.300000 0.400000 1"
    ]
    assert w.label_lines([pose], "keypoints", 3) == []  # a pose with another number of points is left out
    assert w.label_lines([{"label": 0, "bbox": [0.1, 0.1, 0, 0.4]}, {"label": 0}], "detection", 0) == []


def test_move_object_follows_the_fit(w):
    layout = (100, 50, 0, 25)  # letterbox of a 2:1 photo in a 100 input
    moved = w.move_object({"label": 0, "bbox": [0.0, 0.0, 1.0, 1.0]}, layout, 100)
    assert moved["bbox"] == pytest.approx([0.0, 0.25, 1.0, 0.5])
    assert w.move_object({"label": 0}, None, 100) == {"label": 0}
    crop = (200, 100, -50, 0)  # a center crop of a 2:1 photo cuts a quarter on each side
    out = w.move_object(
        {
            "label": 0,
            "bbox": [0.0, 0.0, 0.5, 1.0],
            "keypoints": [[0.05, 0.5, 2], [0.5, 0.5, 1]],
            "polygons": [[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]],
        },
        crop,
        100,
    )
    assert out["bbox"][0] == 0.0 and out["bbox"][2] == pytest.approx(0.5)  # cut at the left edge
    assert out["keypoints"][0] == [0.0, 0.0, 0]  # outside the crop: not labelled
    assert out["keypoints"][1] == pytest.approx([0.5, 0.5, 1])
    assert all(0.0 <= v <= 1.0 for ring in out["polygons"] for p in ring for v in p)


def test_data_yaml(w, tmp_path):
    text = w.data_yaml(tmp_path, ["a", "b c"], ["train", "val"], 0, None)
    assert "train: images/train" in text and "val: images/val" in text and "\ntest:" not in text
    assert '  0: "a"' in text and '  1: "b c"' in text and "kpt_shape" not in text
    pose = w.data_yaml(tmp_path, ["a"], ["train", "val", "test"], 2, [1, 0])
    assert "test: images/test" in pose and "kpt_shape: [2, 3]" in pose and "flip_idx: [1, 0]" in pose


def test_scores_have_the_shape_of_the_other_workers(w):
    scores = w.scores_of(fake_metrics(0.4, "instance"), CLASSES, "instance")
    assert scores["map50_95"] == 0.4 and scores["map50"] == pytest.approx(0.5)
    assert scores["mask_map50"] == pytest.approx(0.6) and scores["mask_map50_95"] == pytest.approx(0.45)
    assert scores["per_class"]["b"] == {"ap50": pytest.approx(0.5), "ap50_95": 0.4, "support": 7}
    assert w.epoch_scores(scores) == {
        "val_map50": scores["map50"],
        "val_map50_95": 0.4,
        "val_mask_map50": scores["mask_map50"],
        "val_mask_map50_95": scores["mask_map50_95"],
    }
    pose = w.scores_of(fake_metrics(0.4, "keypoints"), CLASSES, "keypoints")
    assert "kp_map50" in pose and "kp_map50_95" in pose and "mask_map50" not in pose
    short = w.per_class_scores(fake_metric(0.5, 0.4, indexes=(1, 5)), CLASSES)
    assert (
        list(short) == ["b"] and short["b"]["support"] == 0
    )  # an index the project does not have is skipped


def curved_metric(indexes=(0, 1)):
    """A fake ``Metric`` with Ultralytics' curves on 1000 confidences (precision 1, recall falling)."""
    px = np.linspace(0, 1, 1000)
    rows = len(indexes)
    return SimpleNamespace(
        **vars(fake_metric(0.5, 0.4, indexes)),
        px=px,
        p_curve=np.ones((rows, 1000)),
        r_curve=np.tile(1 - px, (rows, 1)),
        f1_curve=np.tile(2 * (1 - px) / (2 - px), (rows, 1)),
    )


def test_curves_come_from_ultralytics_on_the_grid_of_the_other_workers(w):
    metrics = SimpleNamespace(box=curved_metric(), nt_per_class=[5, 7])
    scores = w.scores_of(metrics, CLASSES, "detection")
    curves = scores["curves"]
    assert curves["match"] == "box" and len(curves["confidence"]) == 101
    assert curves["classes"]["b"]["support"] == 7
    assert curves["classes"]["a"]["recall"][50] == pytest.approx(0.5, abs=1e-3)
    assert "val_curves" not in w.epoch_scores(scores)  # epoch rows keep only the metrics
    assert scores["suggested_threshold"] == 0.0  # F1 falls as the confidence grows: best with everything
    assert "val_suggested_threshold" not in w.epoch_scores(scores)
    assert "suggested_threshold" not in w.scores_of(metrics, CLASSES, "detection", suggest=False)
    instance = SimpleNamespace(box=fake_metric(0.5, 0.4), seg=curved_metric(), nt_per_class=[5, 7])
    assert w.scores_of(instance, CLASSES, "instance")["curves"]["match"] == "mask"
    assert "curves" not in w.scores_of(fake_metrics(0.4), CLASSES, "detection")  # Ultralytics gave none


# -- the dataset ----------------------------------------


def test_letterbox_links_the_photos_and_writes_one_label_file_each(w, tmp_path):
    spec = make_spec(tmp_path, resize="letterbox")
    spec["parts"]["test"] = spec["parts"]["val"][:1]
    root = tmp_path / "ds"
    yaml = w.prepare_dataset(spec, root, None)
    assert yaml == root / "data.yaml" and "test: images/test" in yaml.read_text()
    assert len(list((root / "images" / "train").iterdir())) == 2
    assert len(list((root / "images" / "test").iterdir())) == 1
    assert (root / "labels" / "train" / "000000.txt").read_text() == "0 0.350000 0.450000 0.500000 0.500000\n"
    with Image.open(next((root / "images" / "train").iterdir())) as image:
        assert image.size == (80, 40)  # not resized: Ultralytics letterboxes itself


@pytest.mark.parametrize("policy", ["stretch", "center_crop"])
def test_stretch_and_center_crop_fit_the_photo_and_its_labels(w, tmp_path, policy):
    spec = make_spec(tmp_path, resize=policy)
    root = tmp_path / "ds"
    w.prepare_dataset(spec, root, None)
    with Image.open(root / "images" / "train" / "000000.jpg") as image:
        assert image.size == (64, 64)
    _, cx, cy, bw, bh = (float(v) for v in (root / "labels" / "train" / "000000.txt").read_text().split())
    if policy == "stretch":
        assert (cx, cy, bw, bh) == pytest.approx((0.35, 0.45, 0.5, 0.5), abs=1e-5)
    else:  # a 2:1 photo cropped to its middle square: the box keeps its height and is cut at the edges
        assert 0.0 <= cx - bw / 2 and cx + bw / 2 <= 1.0 and bh == pytest.approx(0.5, abs=1e-5)


def test_images_without_objects_get_an_empty_label_file(w, tmp_path):
    spec = make_spec(tmp_path, resize="letterbox")
    spec["parts"]["train"][0]["objects"] = []
    root = tmp_path / "ds"
    w.prepare_dataset(spec, root, None)
    assert (root / "labels" / "train" / "000000.txt").read_text() == ""


def test_link_or_copy_falls_back_to_a_copy(w, tmp_path, monkeypatch):
    source = tmp_path / "a.txt"
    source.write_text("x")

    def refuse(*args):
        raise OSError("another disk")

    monkeypatch.setattr(w.os, "link", refuse)
    w.link_or_copy(source, tmp_path / "b.txt")
    assert (tmp_path / "b.txt").read_text() == "x"


# -- training with the fake ----------------------------------------


def test_training_reports_the_same_events_and_files_as_the_other_workers(w, fake, tmp_path, capsys):
    spec = make_spec(tmp_path, epochs=3, mosaic=True, close_mosaic=2, warmup_epochs=1)
    results = w.train(spec)
    out = Path(spec["output_dir"])
    seen = events(capsys)
    epochs = [e for e in seen if e["event"] == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2, 3] and all(e["checkpoint"] for e in epochs)
    assert [e["best"] for e in epochs] == [True, True, False]
    assert epochs[1]["val_map50_95"] == 0.5 and epochs[1]["train_loss"] == 1.5 and epochs[1]["lr"] == 0.01
    assert {"val_map50", "seconds", "epochs"} <= set(epochs[0])
    assert {"setup", "log", "progress"} <= {e["event"] for e in seen}
    assert any(e["event"] == "progress" and e["step"] == 3 and e["steps"] == 3 for e in seen)
    assert results["best_epoch"] == 2 and results["epochs_run"] == 3 and results["stopped_early"] is False
    assert (
        results["val"]["map50_95"] == 0.5 and results["val"]["classes"] == CLASSES and results["test"] is None
    )
    assert json.loads((out / "results.json").read_text()) == results
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    state = json.loads((out / "yolo_state.json").read_text())
    assert state["epoch"] == 3 and state["digest"] == "d1" and state["best"]["epoch"] == 2
    assert not (out / "yolo").exists() and not list(tmp_path.glob("yolo-data-*"))  # temporary files are gone
    assert fake.updated == [{"sync": False}]


def test_the_best_checkpoint_is_the_best_epoch_by_the_selection_metric(w, fake, tmp_path):
    torch = pytest.importorskip("torch")
    w.train(make_spec(tmp_path, epochs=3))
    best = torch.load(tmp_path / "out" / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "out" / "last.pt", weights_only=False)
    assert best["epoch"] == 1 and last["epoch"] == 2  # zero-based: epoch 2 was the best, epoch 3 the last
    spec = make_spec(tmp_path, epochs=3)
    spec["metric"] = "map50"
    spec["output_dir"] = str(tmp_path / "out50")
    assert w.train(spec)["metric"] == "map50"


def test_the_train_arguments_carry_the_options(w, fake, tmp_path):
    spec = make_spec(
        tmp_path,
        epochs=2,
        lr=0.02,
        optimizer="adamw",
        weight_decay=0.001,
        scheduler="cosine",
        warmup_epochs=2,
        mosaic=False,
        close_mosaic=1,
        seed=7,
        batch_size=4,
        patience=3,
    )
    w.train(spec)
    kwargs = FakeYOLO.instances[0].train_kwargs
    assert FakeYOLO.instances[0].weights == "yolov8n.pt"
    assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"], kwargs["device"]) == (2, 64, 4, "cpu")
    assert kwargs["optimizer"] == "AdamW" and kwargs["lr0"] == 0.02 and kwargs["weight_decay"] == 0.001
    assert kwargs["warmup_epochs"] == 2.0 and kwargs["cos_lr"] is True and kwargs["seed"] == 7
    assert kwargs["workers"] == 0 and kwargs["mosaic"] == 0.0 and kwargs["close_mosaic"] == 1
    assert kwargs["amp"] is False and kwargs["fliplr"] == 0.5
    assert kwargs["patience"] == 0  # the worker stops by the selection metric itself
    assert kwargs["freeze"] == w.FROZEN_LAYERS and kwargs["pretrained"] is True
    assert kwargs["project"] == str(tmp_path / "out") and kwargs["data"].endswith("data.yaml")


@pytest.mark.parametrize(
    ("start", "weights", "pretrained", "frozen"),
    [
        ("transfer", "yolov8n.pt", True, True),
        ("finetune", "yolov8n.pt", True, False),
        ("scratch", "yolov8n.yaml", False, False),
    ],
)
def test_start_picks_the_weights_and_what_is_frozen(w, fake, tmp_path, start, weights, pretrained, frozen):
    spec = make_spec(tmp_path, epochs=1)
    spec["start"] = start
    w.train(spec)
    model = FakeYOLO.instances[0]
    assert model.weights == weights and model.train_kwargs["pretrained"] is pretrained
    assert ("freeze" in model.train_kwargs) is frozen


def test_patience_stops_by_the_selection_metric(w, fake, tmp_path, capsys):
    results = w.train(make_spec(tmp_path, epochs=5, patience=2))
    seen = events(capsys)
    assert [e["epoch"] for e in seen if e["event"] == "epoch"] == [
        1,
        2,
        3,
        4,
    ]  # best at 2, stops after 2 more
    assert results["epochs_run"] == 4 and results["stopped_early"] is True and results["best_epoch"] == 2
    assert any(e["event"] == "log" and "Early stopping" in e["message"] for e in seen)


def test_epochs_close_whichever_callback_comes_second(w, fake, tmp_path, capsys):
    FakeYOLO.saved_first = True
    results = w.train(make_spec(tmp_path, epochs=2))
    assert [e["epoch"] for e in events(capsys) if e["event"] == "epoch"] == [1, 2]
    assert results["best_epoch"] == 2


@pytest.mark.parametrize("task", ["instance", "keypoints"])
def test_instance_and_pose_report_their_own_scores(w, fake, tmp_path, capsys, task):
    FakeYOLO.task = task
    model = {"instance": "yolov8n-seg", "keypoints": "yolov8n-pose"}[task]
    results = w.train(make_spec(tmp_path, model=model, task=task, epochs=2))
    prefix = "mask" if task == "instance" else "kp"
    epoch = next(e for e in events(capsys) if e["event"] == "epoch")
    assert f"val_{prefix}_map50_95" in epoch and f"{prefix}_map50" in results["val"]


def test_the_test_part_is_evaluated_with_the_best_weights(w, fake, tmp_path, capsys):
    spec = make_spec(tmp_path, epochs=2)
    spec["parts"]["test"] = spec["parts"]["val"][:1]
    results = w.train(spec)
    tester = FakeYOLO.instances[-1]
    assert tester.weights == str(tmp_path / "out" / "best.pt") and tester.val_kwargs["split"] == "test"
    assert results["test"]["map50_95"] == 0.77 and results["test"]["classes"] == CLASSES
    assert any(e["event"] == "progress" and e["phase"] == "evaluating" for e in events(capsys))


def test_unsupported_options_are_reported_as_warnings(w, fake, tmp_path, capsys):
    spec = make_spec(tmp_path, epochs=1, scheduler="onecycle")
    spec["start"] = "finetune"
    spec["augmentations"] = [{"name": "blur", "params": {}}, {"name": "hflip", "params": {"p": 0.5}}]
    w.train(spec)
    codes = [(e["code"], e["params"]) for e in events(capsys) if e["event"] == "warning"]
    assert ("yolo_technique_unsupported", {"technique": "blur"}) in codes
    assert ("yolo_scheduler_unsupported", {"scheduler": "onecycle"}) in codes
    assert ("yolo_backbone_lr_ignored", {}) in codes


def test_mixed_precision_follows_the_device(w, fake, tmp_path, capsys):
    w.train(make_spec(tmp_path, epochs=1, amp="on"))
    assert any(e["event"] == "warning" and e["code"] == "amp_needs_cuda" for e in events(capsys))
    assert FakeYOLO.instances[0].train_kwargs["amp"] is False  # on the CPU


def test_a_device_that_is_not_here_is_a_coded_failure(w, fake, tmp_path):
    spec = make_spec(tmp_path, epochs=1)
    spec["device"] = "cuda:5"
    with pytest.raises(w.options.DeviceError):
        w.train(spec)


def test_out_of_memory_becomes_the_coded_failure(w, fake, tmp_path):
    FakeYOLO.failure = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    with pytest.raises(w.options.OutOfMemory):
        w.train(make_spec(tmp_path, epochs=1))
    FakeYOLO.failure = ValueError("other")
    with pytest.raises(ValueError, match="other"):
        w.train(make_spec(tmp_path, epochs=1))
    assert not list(tmp_path.glob("yolo-data-*"))


def test_training_without_ultralytics_says_so(w, monkeypatch, tmp_path):
    monkeypatch.setattr(w, "YOLO", None)
    with pytest.raises(w.SpecError, match="not installed"):
        w.train(make_spec(tmp_path))


def test_a_training_that_ends_without_an_epoch_fails(w, fake, tmp_path):
    FakeYOLO.curve = []
    with pytest.raises(RuntimeError, match="without completing"):
        w.train(make_spec(tmp_path, epochs=2))


# -- resume ----------------------------------------


def test_resume_continues_after_the_checkpoint_and_keeps_the_best(w, fake, tmp_path, capsys):
    spec = make_spec(tmp_path, epochs=4)
    FakeYOLO.curve = [0.2, 0.5]
    w.train(spec)  # the first process ran two epochs, then stopped
    capsys.readouterr()
    FakeYOLO.curve = [0.2, 0.5, 0.4, 0.45]
    FakeYOLO.instances = []
    spec["resume"] = True
    results = w.train(spec)
    seen = events(capsys)
    assert [e for e in seen if e["event"] == "resumed"] == [{"event": "resumed", "epoch": 2}]
    assert [e["epoch"] for e in seen if e["event"] == "epoch"] == [3, 4]
    model = FakeYOLO.instances[0]
    assert model.train_kwargs == {"resume": True} and model.data_seen.endswith("data.yaml")
    assert Path(model.weights) == tmp_path / "out" / "yolo" / "weights" / "last.pt"
    assert results["best_epoch"] == 2 and results["epochs_run"] == 4  # the earlier best survives the restart


def test_resume_needs_a_checkpoint_of_the_same_run(w, fake, tmp_path):
    spec = make_spec(tmp_path, epochs=4)
    spec["resume"] = True
    with pytest.raises(w.options.CodedFailure) as none:
        w.train(spec)
    assert none.value.code == "resume_no_checkpoint"
    FakeYOLO.curve = [0.2]
    spec["resume"] = False
    w.train(spec)
    spec["resume"], spec["digest"] = True, "another"
    with pytest.raises(w.options.CodedFailure) as changed:
        w.train(spec)
    assert changed.value.code == "resume_config_changed"


# -- main ----------------------------------------


def test_main_reports_done_or_a_coded_error(w, fake, tmp_path, capsys):
    spec = make_spec(tmp_path, epochs=1)
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert w.main(["yolo.py", str(file)]) == 0
    assert events(capsys)[-1]["event"] == "done"
    file.write_text(json.dumps({**spec, "device": "cuda:9"}))
    assert w.main(["yolo.py", str(file)]) == 1
    error = next(e for e in events(capsys) if e["event"] == "error")
    assert error["code"] == "device_unavailable"


def test_classification_scores_come_from_ultralytics_confusion_matrix(w):
    """Ultralytics counts ``matrix[predicted][true]``; the scores use ``[true][predicted]``."""
    matrix = [[3, 1], [0, 2]]  # predicted a: 3 true a, 1 true b; predicted b: 2 true b
    scores = w.classification_scores(matrix, ["a", "b"])
    assert scores["confusion_matrix"] == [[3, 0], [1, 2]] and scores["images"] == 6
    assert scores["accuracy"] == pytest.approx(5 / 6) and 0 < scores["macro_f1"] < 1
    assert set(scores["per_class"]) == {"a", "b"}
    assert w.scores_of(None, ["a", "b"], "classification", matrix)["accuracy"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="confusion matrix"):
        w.scores_of(object(), ["a", "b"], "classification")


def test_a_classification_dataset_has_one_folder_per_class_in_each_part(w, tmp_path):
    spec = make_spec(tmp_path, model="yolo11n-cls", task="classification", resize="stretch")
    for part in spec["parts"].values():
        for k, item in enumerate(part):
            item["label"] = k % 2
    root = tmp_path / "data"
    assert w.prepare_classification(spec, root, ["train", "val"], 64, "stretch") == root
    assert sorted(p.name for p in (root / "train").iterdir()) == ["0000", "0001"]
    assert len(list((root / "val" / "0001").iterdir())) == 1
