"""Public pretrained models that can pre-annotate without any local training (docs/PREANNOTATE.md).

A public model is added to a project as a saved model (``m<N>``, source ``public``) with a mapping
from the model's categories to the project's classes; categories that are not mapped are ignored.
Its weights are downloaded into the temporary folder of each pre-annotation and removed at the end,
like every other download.

The catalogue is data: each entry names the torchvision builder, the kind of task, the categories in
the order of the network's outputs (index 0 is the background) and where the weights come from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: COCO categories in the order of torchvision's detection outputs ("N/A" are unused indexes).
COCO_CATEGORIES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip
SOURCE = "torchvision (BSD-3-Clause code); weights trained on COCO 2017"


@dataclass(frozen=True)
class PublicModel:
    """One public model.

    Attributes:
        id: catalogue id.
        label: name shown on screen.
        task_kind: ``detection`` or ``keypoints``.
        builder: torchvision.models.detection builder name.
        categories: output categories in index order (0 = background).
        box_map: COCO box mAP reported by torchvision (for comparing the entries).
        size_mb: download size of the weights.
        note: one line on when to choose it.
        keypoints: keypoint names (keypoint models only).
    """

    id: str
    label: str
    task_kind: str
    builder: str
    categories: tuple[str, ...]
    box_map: float
    size_mb: int
    note: str
    keypoints: tuple[str, ...] = field(default=())

    def classes(self) -> list[str]:
        """Categories a project class can be mapped from (no background, no unused indexes)."""
        return [c for c in self.categories[1:] if c != "N/A"]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (with the mappable classes)."""
        data = asdict(self)
        data["categories"] = list(self.categories)
        data["keypoints"] = list(self.keypoints)
        data["classes"] = self.classes()
        data["source"] = SOURCE
        return data


PUBLIC_MODELS: tuple[PublicModel, ...] = (
    PublicModel(
        "coco_fasterrcnn_mobilenet_v3",
        "Faster R-CNN MobileNetV3 (COCO)",
        "detection",
        "fasterrcnn_mobilenet_v3_large_fpn",
        COCO_CATEGORIES,
        32.8,
        74,
        "Fastest; good to start, also on a computer without a GPU.",
    ),
    PublicModel(
        "coco_retinanet_resnet50_v2",
        "RetinaNet ResNet-50 v2 (COCO)",
        "detection",
        "retinanet_resnet50_fpn_v2",
        COCO_CATEGORIES,
        41.5,
        146,
        "More accurate, slower.",
    ),
    PublicModel(
        "coco_fasterrcnn_resnet50_v2",
        "Faster R-CNN ResNet-50 v2 (COCO)",
        "detection",
        "fasterrcnn_resnet50_fpn_v2",
        COCO_CATEGORIES,
        46.7,
        167,
        "The most accurate here; slowest.",
    ),
    PublicModel(
        "coco_keypointrcnn_resnet50",
        "Keypoint R-CNN ResNet-50 (COCO people)",
        "keypoints",
        "keypointrcnn_resnet50_fpn",
        ("__background__", "person"),
        54.6,
        226,
        "People with 17 body keypoints.",
        COCO_KEYPOINTS,
    ),
)

#: Kind of task → the project type that uses its labels.
PROJECT_TYPE = {"detection": "detection", "keypoints": "keypoints"}


def get_public(model_id: str) -> PublicModel | None:
    """A public model by id, or None."""
    return next((m for m in PUBLIC_MODELS if m.id == model_id), None)


def public_for(project_type: str | None) -> list[PublicModel]:
    """Public models that make the labels a project of this type uses (all without a type)."""
    return [m for m in PUBLIC_MODELS if project_type is None or PROJECT_TYPE.get(m.task_kind) == project_type]


def suggest_mapping(model: PublicModel, project_classes: list[str]) -> dict[str, str]:
    """Public category → project class for names that match (ignoring case, spaces and underscores)."""

    def key(name: str) -> str:
        return name.casefold().replace("_", " ").replace("-", " ").strip()

    by_key = {key(c): c for c in project_classes}
    return {c: by_key[key(c)] for c in model.classes() if key(c) in by_key}
