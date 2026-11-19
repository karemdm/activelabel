"""Computer vision tasks a project can have."""

from __future__ import annotations

from enum import Enum


class Task(str, Enum):
    """Task of a project. The value is what the API and the files use."""

    CLASSIFICATION = "classification"  # one class per image
    DETECTION = "detection"  # boxes
    SEGMENTATION = "segmentation"  # instance masks / polygons
    KEYPOINTS = "keypoints"  # pose: points per object

    @property
    def label_type(self) -> str:
        """FiftyOne label type of the ground truth of this task."""
        return {
            Task.CLASSIFICATION: "Classification",
            Task.DETECTION: "Detections",
            Task.SEGMENTATION: "Detections",  # Detections with masks (or Polylines from polygons)
            Task.KEYPOINTS: "Keypoints",
        }[self]
