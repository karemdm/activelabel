"""Hugging Face transformers models shaped like torchvision's (#606): tiny networks built from configurations.

Nothing is downloaded: every network is built from a small configuration. The tests need torch and
transformers (they are skipped without them).
"""

import json

import pytest
from PIL import Image
from worker_loader import load_worker

SQUARE = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]]


@pytest.fixture(scope="module")
def hf():
    module = load_worker("hf_models")
    if module.transformers is None:
        pytest.skip("transformers is not installed")
    return module


def tiny_detr(hf, labels=2):
    tf = hf.transformers
    backbone = tf.ResNetConfig(
        embedding_size=8, hidden_sizes=[8, 16, 16, 16], depths=[1, 1, 1, 1], out_features=["stage4"]
    )
    config = tf.DetrConfig(
        backbone_config=backbone,
        num_queries=5,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        num_labels=labels,
    )
    return tf.DetrForObjectDetection(config)


def tiny_rtdetr(hf, labels=2):
    tf = hf.transformers
    backbone = tf.RTDetrResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 16, 16],
        depths=[1, 1, 1, 1],
        out_features=["stage2", "stage3", "stage4"],
    )
    config = tf.RTDetrConfig(
        num_labels=labels,
        num_queries=5,
        d_model=16,
        encoder_hidden_dim=16,
        decoder_layers=1,
        backbone_config=backbone,
        encoder_in_channels=[16, 16, 16],
        decoder_in_channels=[16, 16, 16],
        num_feature_levels=3,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
    )
    return tf.RTDetrForObjectDetection(config)


def tiny_mask2former(hf, labels=2):
    tf = hf.transformers
    swin = tf.SwinConfig(
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[1, 1, 1, 1],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        image_size=64,
        window_size=2,
    )
    config = tf.Mask2FormerConfig(
        num_labels=labels,
        num_queries=5,
        hidden_dim=32,
        mask_feature_size=32,
        feature_size=32,
        encoder_layers=1,
        decoder_layers=2,
        dim_feedforward=16,
        num_attention_heads=2,
        encoder_feedforward_dim=16,
        backbone_config=swin,
    )
    return tf.Mask2FormerForUniversalSegmentation(config)


def tiny_segformer(hf, labels=3):
    tf = hf.transformers
    config = tf.SegformerConfig(
        num_labels=labels,
        hidden_sizes=[8, 16, 16, 16],
        depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=16,
    )
    return tf.SegformerForSemanticSegmentation(config)


def targets(hf, masks=False):
    torch = hf.torch
    out = []
    for _ in range(2):
        target = {"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]), "labels": torch.tensor([1])}
        if masks:
            mask = torch.zeros(1, 64, 60)
            mask[0, 8:40, 8:40] = 1
            target["masks"] = mask
        out.append(target)
    return out


def test_ids_name_an_architecture_and_a_checkpoint(hf):
    assert hf.split_id("detr:facebook/detr-resnet-50") == ("detr", "facebook/detr-resnet-50")
    for bad in ("detr", "yolo:x", "segformer:"):
        with pytest.raises(ValueError, match="Unknown transformers model"):
            hf.split_id(bad)


def test_images_are_padded_to_a_multiple_of_32_and_masked(hf):
    torch = hf.torch
    pixels, mask, size = hf.batch_images([torch.ones(3, 40, 50), torch.ones(3, 64, 33)])
    assert size == (64, 64) and pixels.shape == (2, 3, 64, 64)
    assert int(mask[0].sum()) == 40 * 50 and int(mask[1].sum()) == 64 * 33
    assert float(pixels[0, 0, 0, 0]) == pytest.approx((1 - hf.MEAN[0]) / hf.STD[0])
    assert float(pixels[0, 0, 63, 63]) == 0.0


def test_boxes_go_to_detr_form_and_back(hf):
    torch = hf.torch
    boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    relative = hf.to_hf_boxes(boxes, 64, 128)
    assert relative.tolist() == [
        [pytest.approx(20 / 128), pytest.approx(40 / 64), pytest.approx(20 / 128), 40 / 64]
    ]
    assert hf.from_hf_boxes(relative, 64, 128).tolist() == [pytest.approx(v) for v in boxes.tolist()]
    masks = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks[0, 2:4, 3:6] = True
    assert hf.mask_boxes(masks).tolist() == [[3.0, 2.0, 6.0, 4.0], [0.0, 0.0, 0.0, 0.0]]


@pytest.mark.parametrize("arch", ["detr", "rtdetr", "mask2former"])
def test_a_detector_trains_and_detects_like_torchvision(hf, arch):
    torch = hf.torch
    build = {"detr": tiny_detr, "rtdetr": tiny_rtdetr, "mask2former": tiny_mask2former}[arch]
    model = hf.HfDetector(build(hf), arch, score_thresh=0.0, detections_per_img=3)
    images = [torch.rand(3, 64, 60), torch.rand(3, 64, 60)]
    model.train()
    loss = model(images, targets(hf, masks=arch == "mask2former"))
    assert set(loss) == {"loss"} and torch.isfinite(loss["loss"])
    loss["loss"].backward()
    model.eval()
    with torch.no_grad():
        found = model(images)
    assert len(found) == 2
    for out in found:
        assert len(out["boxes"]) <= 3 and len(out["scores"]) == len(out["labels"])
        assert bool((out["labels"] >= 1).all()) and bool((out["boxes"][:, 2] <= 60).all())
        assert list(out["scores"]) == sorted(out["scores"].tolist(), reverse=True)
        if arch == "mask2former":
            assert out["masks"].shape[1:] == (1, 64, 60)
    model.score_thresh = 1.1
    with torch.no_grad():
        assert all(len(out["boxes"]) == 0 for out in model(images))
    assert model.backbone is hf.backbone_of(model.net, arch)


def test_segformer_answers_at_the_input_size(hf):
    torch = hf.torch
    model = hf.HfSemantic(tiny_segformer(hf))
    out = model(torch.rand(2, 3, 64, 64))["out"]
    assert out.shape == (2, 3, 64, 64) and model.heads() == [model.net.decode_head]


def test_build_network_needs_transformers(hf, monkeypatch):
    monkeypatch.setattr(hf, "transformers", None)
    with pytest.raises(ImportError, match="not installed"):
        hf.build_network("detr:facebook/detr-resnet-50", 2, False)


def write_images(tmp_path, count=4):
    items = []
    for k in range(count):
        path = tmp_path / f"{k}.png"
        Image.new("RGB", (64, 64), (200, 10 * k, 10)).save(path)
        items.append(
            {"path": str(path), "objects": [{"label": 0, "bbox": [0.1, 0.1, 0.5, 0.5], "polygons": [SQUARE]}]}
        )
    return items


def base_params(**extra):
    return {
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.0001,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "none",
        "warmup_epochs": 0,
        "image_size": 64,
        "resize": "stretch",
        "augmentation": "none",
        "patience": 0,
        "seed": 1,
        "backbone_lr_factor": 0.1,
        "amp": "auto",
        **extra,
    }


@pytest.mark.parametrize(("arch", "task"), [("detr", "detection"), ("mask2former", "instance")])
def test_a_short_real_training_with_the_detection_worker(hf, tmp_path, monkeypatch, arch, task):
    """The detection worker trains a transformers detector end to end (a tiny network, on the CPU)."""
    worker = load_worker("detection")
    build = {"detr": tiny_detr, "mask2former": tiny_mask2former}[arch]
    monkeypatch.setattr(worker.hf_models, "build_network", lambda model_id, n, pretrained: build(hf, n))
    monkeypatch.setattr(worker, "emit", lambda event, **fields: None)
    items = write_images(tmp_path)
    spec = {
        "task": task,
        "model": f"{arch}:facebook/x",
        "family": "transformers",
        "classes": ["cat", "dog"],
        "start": "finetune",
        "params": base_params(trainable_backbone_layers=5, score_thresh=0.05),
        "seed": 1,
        "metric": "map50_95",
        "fold": None,
        "parts": {"train": items, "val": items[:2]},
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "augmentations": [],
        "digest": "d",
    }
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert worker.load_spec(file)["family"] == "transformers"
    results = worker.train(spec)
    assert results["epochs_run"] == 1 and "map50_95" in results["val"]
    assert (tmp_path / "out" / "best.pt").exists()


def test_a_short_real_training_with_the_segmentation_worker(hf, tmp_path, monkeypatch):
    worker = load_worker("segmentation")
    monkeypatch.setattr(
        worker.hf_models, "build_network", lambda model_id, n, pretrained: tiny_segformer(hf, n)
    )
    monkeypatch.setattr(worker, "emit", lambda event, **fields: None)
    items = write_images(tmp_path)
    spec = {
        "task": "semantic",
        "model": "segformer:nvidia/mit-b0",
        "family": "transformers",
        "classes": ["cat", "dog"],
        "start": "transfer",
        "params": base_params(class_weights="none", loss="cross_entropy"),
        "seed": 1,
        "metric": "miou",
        "fold": None,
        "parts": {"train": items, "val": items[:2]},
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "augmentations": [],
        "digest": "d",
    }
    results = worker.train(spec)
    assert results["epochs_run"] == 1 and "miou" in results["val"]


def test_the_workers_refuse_transformers_models_they_cannot_train(tmp_path):
    detection, segmentation = load_worker("detection"), load_worker("segmentation")
    items = write_images(tmp_path, 2)
    spec = {
        "task": "keypoints",
        "model": "detr:facebook/detr-resnet-50",
        "family": "transformers",
        "classes": ["a"],
        "start": "transfer",
        "params": base_params(),
        "metric": "map50_95",
        "parts": {"train": items, "val": items},
        "output_dir": str(tmp_path / "out"),
        "num_keypoints": 2,
    }
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    with pytest.raises(detection.SpecError, match="no keypoint"):
        detection.load_spec(file)
    file.write_text(json.dumps({**spec, "task": "semantic", "metric": "miou"}))
    with pytest.raises(segmentation.SpecError, match="not a semantic"):
        segmentation.load_spec(file)
