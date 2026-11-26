import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from activelabel.training import augmentations

WORKER = Path(__file__).parents[4] / "src" / "activelabel" / "training" / "workers" / "segmentation.py"


@pytest.fixture(scope="module")
def worker():
    pytest.importorskip("torch", reason="the training workers import torch, absent from the app environment")
    spec = importlib.util.spec_from_file_location("segmentation_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["segmentation_worker"] = module
    spec.loader.exec_module(module)
    return module


CLASSES = ["cat", "dog"]
SQUARE = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]


def test_rasterize_polygons_over_each_other(worker):
    big = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    objs = [{"label": 0, "polygons": [big]}, {"label": "dog", "polygons": [SQUARE]}]
    mask = worker.rasterize_mask(objs, 20, 20, CLASSES)
    assert mask.size == (20, 20)
    assert mask.getpixel((2, 2)) == 2  # the later object (dog) is on top
    assert mask.getpixel((15, 15)) == 1  # cat elsewhere


def test_rasterize_background_box_only_and_degenerate(worker):
    objs = [
        {"label": 1, "bbox": [0.5, 0.5, 0.5, 0.5]},  # box only -> filled rectangle
        {"label": 0, "polygons": [[[0.0, 0.0], [0.1, 0.1]]]},  # 2 points: no polygon, no box -> skipped
        {"label": 0},
    ]
    mask = worker.rasterize_mask(objs, 10, 10, CLASSES)
    assert mask.getpixel((0, 0)) == 0  # background
    assert mask.getpixel((7, 7)) == 2
    assert dict((v, c) for c, v in mask.getcolors())[2] == 25
    assert {v for _, v in mask.getcolors()} == {0, 2}
    assert worker.count_box_only({"train": [{"objects": objs}], "val": [{"objects": [objs[0]]}]}) == 2
    with pytest.raises(worker.SpecError):
        worker.rasterize_mask([{"label": "bird", "bbox": [0, 0, 1, 1]}], 4, 4, CLASSES)
    with pytest.raises(worker.SpecError):
        worker.class_id(5, CLASSES)


def test_dataset_applies_the_techniques_to_image_and_mask(worker, tmp_path):
    torch = pytest.importorskip("torch")
    Image.new("RGB", (40, 30), (120, 120, 120)).save(tmp_path / "a.png")
    items = [{"path": str(tmp_path / "a.png"), "objects": [{"label": 0, "polygons": [SQUARE]}]}]
    plain = worker.SegmentationDataset(items, CLASSES, 16, [])[0][1]
    assert plain[2, 2] == 1 and plain[14, 14] == 0
    for level in ("none", "light", "standard", "strong"):
        techniques = augmentations.preset(level, "semantic")
        image, mask = worker.SegmentationDataset(items, CLASSES, 16, techniques)[0]
        assert image.shape == (3, 16, 16) and mask.shape == (16, 16)
        assert set(mask.unique().tolist()) <= {0, 1}  # nearest: no invented class values
    flip = [{"name": "hflip", "params": {"p": 1.0}}]
    flipped = worker.SegmentationDataset(items, CLASSES, 16, flip)[0][1]
    assert torch.equal(flipped, plain.flip(-1))
    with pytest.raises(worker.augment.AugmentError):
        worker.SegmentationDataset(items, CLASSES, 16, [{"name": "mixup", "params": {"alpha": 0.2}}])[0]


# classes: background, cat, dog. matrix[true][predicted]
MATRIX = [[50, 5, 5], [10, 20, 0], [0, 10, 0]]


def test_confusion_and_metrics(worker):
    assert worker.confusion_from_lists([0, 0, 1, 2], [0, 1, 1, 1], 3) == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    cat = worker.class_scores(MATRIX, 1)  # tp 20, support 30, predicted 35
    assert cat[0] == pytest.approx(20 / 45) and cat[1] == pytest.approx(40 / 65)
    assert cat[2:] == (30, 35)
    dog = worker.class_scores(MATRIX, 2)  # tp 0, support 10, predicted 5
    assert dog[0] == 0 and dog[1] == 0
    miou, dice = worker.mean_scores(MATRIX)  # background excluded
    assert miou == pytest.approx((20 / 45 + 0) / 2)
    assert dice == pytest.approx((40 / 65 + 0) / 2)
    assert worker.pixel_accuracy(MATRIX) == pytest.approx(70 / 100)
    ev = worker.evaluation(MATRIX, CLASSES)
    assert ev["classes"] == ["background", "cat", "dog"]
    assert ev["background_iou"] == pytest.approx(50 / 70)
    assert ev["per_class"]["background"]["support"] == 60
    assert ev["per_class"]["cat"]["iou"] == pytest.approx(20 / 45)
    assert ev["confusion_matrix"] == MATRIX


def test_absent_classes_do_not_count(worker):
    only_bg = [[10, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert worker.mean_scores(only_bg) == (0.0, 0.0)
    assert worker.pixel_accuracy(only_bg) == 1.0
    # dog never appears in truth or prediction: the mean is over cat only
    m = [[8, 2, 0], [0, 4, 0], [0, 0, 0]]
    assert worker.mean_scores(m)[0] == pytest.approx(4 / 6)
    assert worker.pixel_accuracy([[0, 0], [0, 0]]) == 0.0


def test_load_spec(worker, tmp_path):
    spec = {
        "task": "semantic",
        "model": "fcn_resnet50",
        "family": "torchvision",
        "start": "transfer",
        "params": {},
        "classes": CLASSES,
        "parts": {"train": [{"path": "a"}, {"path": "b"}], "val": [{"path": "c"}]},
        "output_dir": str(tmp_path),
        "metric": "miou",
    }
    path = tmp_path / "spec.json"

    def check(**changes):
        path.write_text(json.dumps({**spec, **changes}))
        return worker.load_spec(path)

    assert check()["model"] == "fcn_resnet50"
    for bad in (
        {"task": "classification"},
        {"model": "resnet18"},
        {"start": "x"},
        {"metric": "accuracy"},
        {"fold": "all"},
        {"parts": {"train": [{"path": "a"}, {"path": "b"}], "val": []}},
        {"parts": {"train": [{"path": "a"}], "val": [{"path": "c"}]}},
    ):
        with pytest.raises(worker.SpecError):
            check(**bad)


def test_dataset_and_tiny_model(worker, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    Image.new("RGB", (32, 24), (200, 10, 10)).save(tmp_path / "a.png")
    items = [{"path": str(tmp_path / "a.png"), "objects": [{"label": 1, "polygons": [SQUARE]}]}]
    image, mask = worker.SegmentationDataset(items, CLASSES, 16, None)[0]
    assert image.shape == (3, 16, 16) and mask.shape == (16, 16) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) == {0, 2}
    params = {"lr": 0.01, "weight_decay": 0.0, "optimizer": "sgd", "backbone_lr_factor": 0.1}
    spec = {"model": "lraspp_mobilenet_v3_large", "classes": CLASSES, "start": "scratch", "params": params}
    model = worker.build_model(spec, pretrained=False)
    out = model(image[None].repeat(2, 1, 1, 1))
    assert out["out"].shape == (2, 3, 16, 16)
    loss = worker.segmentation_loss(out, mask[None].repeat(2, 1, 1), torch.nn.CrossEntropyLoss())
    assert float(loss) > 0
    assert worker.make_optimizer(spec, model, torch) is not None
    assert worker.options.pick_device(torch) in ("cuda", "mps", "cpu")


def test_letterbox_bands_are_background_in_the_mask(worker, tmp_path):
    Image.new("RGB", (32, 16), (200, 10, 10)).save(tmp_path / "wide.png")
    full = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    items = [{"path": str(tmp_path / "wide.png"), "objects": [{"label": 0, "polygons": [full]}]}]
    _, mask = worker.SegmentationDataset(items, CLASSES, 16, None, "letterbox")[0]
    assert mask[0, 8] == 0 and mask[8, 8] == 1 and mask[15, 8] == 0
    _, stretched = worker.SegmentationDataset(items, CLASSES, 16, None)[0]
    assert stretched[0, 8] == 1


# -- class imbalance (#147) ----------------------------------------------------


def test_pixel_counts_include_the_background_and_the_bands(worker, tmp_path):
    Image.new("RGB", (32, 16), (200, 10, 10)).save(tmp_path / "wide.png")
    full = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    items = [{"path": str(tmp_path / "wide.png"), "objects": [{"label": 1, "polygons": [full]}]}]
    stretched = worker.pixel_counts(items, CLASSES, 16, "stretch")
    assert stretched == [0, 0, 256]
    boxed = worker.pixel_counts(items, CLASSES, 16, "letterbox")
    assert boxed[0] > 0 and boxed[2] > 0 and sum(boxed) == 256 and boxed[1] == 0


def test_training_with_weights_and_dice_loss(worker, tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    Image.new("RGB", (32, 32), (200, 10, 10)).save(tmp_path / "a.png")
    obj = {"label": 0, "polygons": [SQUARE]}
    items = [{"path": str(tmp_path / "a.png"), "objects": [obj]} for _ in range(4)]
    params = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.001,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "none",
        "warmup_epochs": 0,
        "image_size": 32,
        "resize": "stretch",
        "augmentation": "none",
        "patience": 0,
        "seed": 1,
        "start": "scratch",
        "backbone_lr_factor": 0.1,
        "amp": "auto",
        "class_weights": "inverse",
        "loss": "cross_entropy_dice",
    }
    seen = []
    monkeypatch.setattr(worker, "emit", lambda event, **fields: seen.append((event, fields)))
    results = worker.train(
        {
            "task": "semantic",
            "model": "lraspp_mobilenet_v3_large",
            "family": "torchvision",
            "classes": CLASSES,
            "start": "scratch",
            "params": params,
            "seed": 1,
            "metric": "miou",
            "fold": None,
            "parts": {"train": items, "val": items[:2]},
            "output_dir": str(tmp_path / "out"),
            "device": "cpu",
            "augmentations": [],
            "digest": "d",
        }
    )
    assert results["epochs_run"] == 1
    assert any(e == "log" and "Class weights" in f["message"] for e, f in seen)


class FakeSmpNet:
    """Builds a tiny network shaped like a segmentation_models_pytorch model (encoder, decoder, head)."""

    @staticmethod
    def build(torch, classes):
        net = torch.nn.Module()
        net.encoder = torch.nn.Conv2d(3, 4, 3, padding=1)
        net.decoder = torch.nn.Conv2d(4, 4, 3, padding=1)
        net.segmentation_head = torch.nn.Conv2d(4, classes, 1)
        net.forward = lambda x: net.segmentation_head(net.decoder(net.encoder(x)))
        return net


class FakeSmp:
    """Stands for the segmentation_models_pytorch package: records what it builds."""

    def __init__(self, torch):
        self.torch, self.calls = torch, []

    def create_model(self, arch, encoder_name, encoder_weights, classes):
        self.calls.append((arch, encoder_name, encoder_weights, classes))
        return FakeSmpNet.build(self.torch, classes)


def smp_spec(tmp_path, image_size=32, start="transfer"):
    Image.new("RGB", (32, 32), (200, 10, 10)).save(tmp_path / "a.png")
    items = [
        {"path": str(tmp_path / "a.png"), "objects": [{"label": 0, "polygons": [SQUARE]}]} for _ in range(4)
    ]
    params = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.001,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "none",
        "warmup_epochs": 0,
        "image_size": image_size,
        "resize": "stretch",
        "augmentation": "none",
        "patience": 0,
        "seed": 1,
        "start": start,
        "backbone_lr_factor": 0.1,
        "amp": "auto",
        "class_weights": "none",
        "loss": "cross_entropy",
    }
    return {
        "task": "semantic",
        "model": "Unet:resnet34",
        "family": "smp",
        "classes": CLASSES,
        "start": start,
        "params": params,
        "seed": 1,
        "metric": "miou",
        "fold": None,
        "parts": {"train": items, "val": items[:2]},
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "augmentations": [],
        "digest": "d",
    }


def test_an_smp_model_is_built_by_decoder_and_encoder_and_answers_like_torchvision(worker, monkeypatch):
    """#606: ``Unet:resnet34`` becomes smp's Unet on resnet34; the output is ``{"out": logits}``."""
    fake = FakeSmp(worker.torch)
    monkeypatch.setattr(worker, "smp", fake)
    model = worker.build_model({"model": "Unet:resnet34", "family": "smp", "classes": CLASSES}, True)
    assert fake.calls == [("Unet", "resnet34", "imagenet", 3)]
    assert tuple(model(worker.torch.zeros(1, 3, 32, 32))["out"].shape) == (1, 3, 32, 32)
    assert worker.head_modules(model) == [model.net.decoder, model.net.segmentation_head]
    worker.build_model({"model": "FPN:mobilenet_v2", "family": "smp", "classes": CLASSES}, False)
    assert fake.calls[-1] == ("FPN", "mobilenet_v2", None, 3)


def test_smp_needs_its_package_and_sides_that_are_a_multiple_of_32(worker, monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "smp", None)
    with pytest.raises(worker.SpecError, match="not installed"):
        worker.build_model({"model": "Unet:resnet34", "family": "smp", "classes": CLASSES}, True)
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(smp_spec(tmp_path, image_size=100)))
    with pytest.raises(worker.SpecError, match="multiple of 32"):
        worker.load_spec(file)
    file.write_text(json.dumps({**smp_spec(tmp_path), "model": "Unet"}))
    with pytest.raises(worker.SpecError, match="Unknown segmentation_models_pytorch"):
        worker.load_spec(file)


def test_an_smp_model_trains_with_the_segmentation_worker(worker, tmp_path, monkeypatch):
    """A short real training (CPU) of an smp-shaped network: transfer trains the decoder and head only."""
    monkeypatch.setattr(worker, "smp", FakeSmp(worker.torch))
    monkeypatch.setattr(worker, "emit", lambda event, **fields: None)
    results = worker.train(smp_spec(tmp_path))
    assert results["epochs_run"] == 1 and "miou" in results["val"]
    assert (tmp_path / "out" / "best.pt").exists()
