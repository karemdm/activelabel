"""Hugging Face ``transformers`` models in the shape the segmentation and detection workers use (#606).

The ``transformers`` family is trained by the existing workers: ``segmentation.py`` for SegFormer and
``detection.py`` for DETR, RT-DETR and Mask2Former. This module wraps each network so that it answers like
torchvision's models:

- :class:`HfSemantic`: ``model(images)`` gives ``{"out": logits}`` at the size of the input (SegFormer
  predicts at a quarter of it); its ``decode_head`` is the head, its encoder the backbone.
- :class:`HfDetector`: in training ``model(images, targets)`` gives ``{"loss": ...}`` from torchvision-style
  targets (``boxes`` in pixels ``[x1, y1, x2, y2]``, ``labels`` from 1, ``masks``); in evaluation
  ``model(images)`` gives one ``{"boxes", "labels", "scores"[, "masks"]}`` per image above ``score_thresh``,
  at most ``detections_per_img``. ``backbone`` is the network's image encoder.

A ``model`` id is ``<architecture>:<checkpoint>``, for example ``detr:facebook/detr-resnet-50``: the
checkpoint on the Hugging Face Hub gives the pretrained weights, or only the configuration for a start from
scratch. Images are normalised with the ImageNet mean and standard deviation, as the processors of these
checkpoints do, and padded (bottom and right) to a common side that is a multiple of ``PAD_TO``;
``pixel_mask`` marks the padding.

Standalone like the workers: it runs in the training environment, which has no ActiveLabel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

# the Hub's downloads go to the run's temporary folder (removed afterwards), and its telemetry is off
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("TMPDIR") or Path.cwd()) / "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import transformers
except ImportError:  # optional: only the transformers family needs it (installed with its first run)
    transformers = None

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD_TO = 32
#: The Hugging Face class of each architecture.
CLASSES = {
    "segformer": "AutoModelForSemanticSegmentation",
    "detr": "DetrForObjectDetection",
    "rtdetr": "RTDetrForObjectDetection",
    "mask2former": "Mask2FormerForUniversalSegmentation",
}
SEMANTIC = ("segformer",)
DETECTORS = ("detr", "rtdetr", "mask2former")


def split_id(model_id: str) -> tuple[str, str]:
    """The architecture and the Hub checkpoint of a model id.

    Args:
        model_id: ``<architecture>:<checkpoint>``.

    Returns:
        ``(architecture, checkpoint)``.

    Raises:
        ValueError: an id without both parts, or an unknown architecture.
    """
    arch, _, checkpoint = model_id.partition(":")
    if arch not in CLASSES or not checkpoint:
        raise ValueError(f"Unknown transformers model '{model_id}'.")
    return arch, checkpoint


def build_network(model_id: str, num_labels: int, pretrained: bool) -> Any:
    """The Hugging Face network of a model id, with ``num_labels`` outputs.

    Args:
        model_id: ``<architecture>:<checkpoint>``.
        num_labels: classes the network predicts (SegFormer counts the background, the detectors do not).
        pretrained: load the checkpoint's weights (downloaded when needed); else only its configuration.

    Returns:
        The network (its head new when the checkpoint had another number of classes).

    Raises:
        ImportError: transformers is not installed.
    """
    if transformers is None:
        raise ImportError("transformers is not installed in the training environment.")
    arch, checkpoint = split_id(model_id)
    cls = getattr(transformers, CLASSES[arch])
    names = {i: str(i) for i in range(num_labels)}
    labels = {"num_labels": num_labels, "id2label": names, "label2id": {v: k for k, v in names.items()}}
    if pretrained:
        return cls.from_pretrained(checkpoint, ignore_mismatched_sizes=True, **labels)
    config = transformers.AutoConfig.from_pretrained(checkpoint, **labels)
    if hasattr(config, "use_pretrained_backbone"):
        config.use_pretrained_backbone = False  # random weights everywhere
    return cls.from_config(config) if hasattr(cls, "from_config") else cls(config)


def normalise(images: Any) -> Any:
    """Images 0..1 normalised with the ImageNet mean and standard deviation.

    Args:
        images: ``(N, 3, H, W)`` tensor.

    Returns:
        The normalised tensor.
    """
    mean = torch.tensor(MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def batch_images(images: list[Any]) -> tuple[Any, Any, tuple[int, int]]:
    """Pad a list of images to one batch.

    Args:
        images: ``(3, h, w)`` tensors with values 0..1.

    Returns:
        ``(pixel_values, pixel_mask, (H, W))``: the normalised batch, 1 where an image is and 0 in the
        padding, and the padded size (a multiple of ``PAD_TO``).
    """
    height = -(-max(int(i.shape[-2]) for i in images) // PAD_TO) * PAD_TO
    width = -(-max(int(i.shape[-1]) for i in images) // PAD_TO) * PAD_TO
    first = images[0]
    pixels = torch.zeros(len(images), 3, height, width, device=first.device, dtype=first.dtype)
    mask = torch.zeros(len(images), height, width, device=first.device, dtype=torch.long)
    for k, image in enumerate(images):
        h, w = int(image.shape[-2]), int(image.shape[-1])
        pixels[k, :, :h, :w] = normalise(image.unsqueeze(0))[0]
        mask[k, :h, :w] = 1
    return pixels, mask, (height, width)


def to_hf_boxes(boxes: Any, height: int, width: int) -> Any:
    """Boxes ``[x1, y1, x2, y2]`` in pixels as DETR's ``[cx, cy, w, h]`` relative to the padded batch.

    Args:
        boxes: ``(N, 4)`` tensor.
        height: padded height.
        width: padded width.

    Returns:
        ``(N, 4)`` tensor of values 0..1.
    """
    scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=torch.float32)
    x1, y1, x2, y2 = (boxes.float() / scale).unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def from_hf_boxes(boxes: Any, height: int, width: int) -> Any:
    """DETR's relative ``[cx, cy, w, h]`` as ``[x1, y1, x2, y2]`` in pixels of the padded batch.

    Args:
        boxes: ``(N, 4)`` tensor.
        height: padded height.
        width: padded width.

    Returns:
        ``(N, 4)`` tensor in pixels.
    """
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    return xyxy * torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)


def mask_boxes(masks: Any) -> Any:
    """Bounding boxes ``[x1, y1, x2, y2]`` (pixels) of boolean masks; an empty mask gives zeros.

    Args:
        masks: ``(N, H, W)`` boolean tensor.

    Returns:
        ``(N, 4)`` float tensor.
    """
    out = torch.zeros(len(masks), 4, device=masks.device)
    for k, mask in enumerate(masks):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(xs):
            out[k] = torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()
    return out


class HfSemantic(torch.nn.Module):
    """A Hugging Face semantic segmentation network that answers like torchvision's: ``{"out": logits}``."""

    def __init__(self, net: Any) -> None:
        """Wrap the network.

        Args:
            net: a ``*ForSemanticSegmentation`` network (``decode_head`` and its encoder).
        """
        super().__init__()
        self.net = net

    def forward(self, x: Any) -> dict[str, Any]:
        """Logits at the size of the input.

        Args:
            x: normalised images ``(N, 3, H, W)`` (the worker normalises them).

        Returns:
            ``{"out": logits}``, ``(N, classes + 1, H, W)``.
        """
        logits = self.net(pixel_values=x).logits
        out = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": out}

    def heads(self) -> list[Any]:
        """The parts that are not the encoder.

        Returns:
            The decode head.
        """
        return [self.net.decode_head]


def backbone_of(net: Any, arch: str) -> Any:
    """The image encoder of a detection network.

    Args:
        net: the Hugging Face network.
        arch: its architecture (``detr``, ``rtdetr``, ``mask2former``).

    Returns:
        The encoder module.
    """
    if arch == "mask2former":
        return net.model.pixel_level_module.encoder
    return net.model.backbone


class HfDetector(torch.nn.Module):
    """A Hugging Face detector (DETR, RT-DETR) or instance segmenter (Mask2Former) shaped like torchvision's.

    Attributes:
        net: the Hugging Face network.
        arch: ``detr``, ``rtdetr`` or ``mask2former``.
        backbone: the image encoder (frozen in ``transfer``).
        score_thresh: lowest score an evaluated detection needs.
        detections_per_img: most detections per image.
    """

    def __init__(
        self, net: Any, arch: str, score_thresh: float = 0.05, detections_per_img: int = 100
    ) -> None:
        """Wrap the network.

        Args:
            net: the Hugging Face network.
            arch: its architecture.
            score_thresh: lowest score an evaluated detection needs.
            detections_per_img: most detections per image.
        """
        super().__init__()
        self.net = net
        self.arch = arch
        self.backbone = backbone_of(net, arch)
        self.score_thresh = score_thresh
        self.detections_per_img = detections_per_img

    def forward(self, images: list[Any], targets: list[dict[str, Any]] | None = None) -> Any:
        """Losses in training (with targets), detections otherwise.

        Args:
            images: ``(3, h, w)`` tensors with values 0..1.
            targets: per image ``boxes`` (pixels, ``[x1, y1, x2, y2]``), ``labels`` (from 1) and, for
                Mask2Former, ``masks`` (``(N, h, w)``).

        Returns:
            ``{"loss": tensor}`` in training; else one dict per image (see :meth:`detections`).
        """
        pixels, pixel_mask, (height, width) = batch_images(images)
        if self.training and targets is not None:
            if self.arch == "mask2former":
                masks = []
                for t in targets:
                    padded = torch.zeros(len(t["masks"]), height, width, device=pixels.device)
                    padded[:, : t["masks"].shape[-2], : t["masks"].shape[-1]] = t["masks"].float()
                    masks.append(padded)
                out = self.net(
                    pixel_values=pixels,
                    pixel_mask=pixel_mask,
                    mask_labels=masks,
                    class_labels=[t["labels"].long() - 1 for t in targets],
                )
            else:
                labels = [
                    {"class_labels": t["labels"].long() - 1, "boxes": to_hf_boxes(t["boxes"], height, width)}
                    for t in targets
                ]
                extra = {"pixel_mask": pixel_mask} if self.arch == "detr" else {}
                out = self.net(pixel_values=pixels, labels=labels, **extra)
            return {"loss": out.loss}
        extra = {"pixel_mask": pixel_mask} if self.arch in ("detr", "mask2former") else {}
        out = self.net(pixel_values=pixels, **extra)
        return [
            self.detections(out, k, (int(i.shape[-2]), int(i.shape[-1])), (height, width))
            for k, i in enumerate(images)
        ]

    def detections(self, out: Any, k: int, size: tuple[int, int], padded: tuple[int, int]) -> dict[str, Any]:
        """The detections of one image of the batch, best first.

        Args:
            out: the network's output for the batch.
            k: the image's index in the batch.
            size: ``(h, w)`` of the image.
            padded: ``(H, W)`` of the padded batch.

        Returns:
            ``boxes`` (pixels of the image), ``labels`` (from 1), ``scores`` and, for Mask2Former, ``masks``
            (``(N, 1, h, w)`` probabilities).
        """
        (h, w), (height, width) = size, padded
        masks = None
        if self.arch == "mask2former":
            probs = out.class_queries_logits[k].softmax(-1)[:, :-1]
            scores, labels = probs.max(-1)
            logits = out.masks_queries_logits[k].unsqueeze(0)
            logits = torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            masks = logits[0, :, :h, :w].sigmoid()
            inside = masks > 0.5
            # the score of a query is its class score times the mean probability inside its mask (as HF does)
            area = inside.flatten(1).sum(-1)
            scores = scores * (masks * inside).flatten(1).sum(-1) / area.clamp(min=1)
            keep = area > 0
            boxes = mask_boxes(inside)
        else:
            logits = out.logits[k]
            probs = logits.sigmoid() if self.arch == "rtdetr" else logits.softmax(-1)[:, :-1]
            scores, labels = probs.max(-1)
            boxes = from_hf_boxes(out.pred_boxes[k], height, width)
            limit = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
            boxes = torch.minimum(boxes.clamp(min=0), limit)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep = keep & (scores >= self.score_thresh)
        order = scores.masked_fill(~keep, -1.0).argsort(descending=True)[: self.detections_per_img]
        order = order[keep[order]]
        found = {"boxes": boxes[order], "labels": labels[order] + 1, "scores": scores[order]}
        if masks is not None:
            found["masks"] = masks[order].unsqueeze(1)
        return found
