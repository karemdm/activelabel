"""Computer vision tasks a project can have."""

from __future__ import annotations

from enum import Enum

from activelabel.errors import CodedError


class Task(str, Enum):
    """Task of a project. The value is what the API and the files use."""

    CLASSIFICATION = "classification"  # one class per image
    DETECTION = "detection"  # boxes
    SEGMENTATION = "segmentation"  # instance masks / polygons
    KEYPOINTS = "keypoints"  # pose: points per object
    REGRESSION = "regression"  # one or more numbers per image (#619): the "classes" are the targets

    @property
    def label_type(self) -> str:
        """FiftyOne label type of the ground truth of this task."""
        return {
            Task.CLASSIFICATION: "Classification",
            Task.DETECTION: "Detections",
            Task.SEGMENTATION: "Detections",  # Detections with masks (or Polylines from polygons)
            Task.KEYPOINTS: "Keypoints",
            Task.REGRESSION: "Regression",
        }[self]


def refuse_regression(task: Task) -> None:
    """Refuse a feature that regression projects do not have yet (sync, FiftyOne, training, pre-annotation).

    Args:
        task: the task of the project.

    Raises:
        CodedError: the task is regression (``regression_not_supported_yet``).
    """
    if task is Task.REGRESSION:
        raise CodedError("regression_not_supported_yet", "This is not available for regression projects yet.")
