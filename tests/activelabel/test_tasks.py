import pytest

from activelabel.errors import CodedError
from activelabel.tasks import Task, refuse_regression


def test_values_are_what_the_api_and_the_files_use():
    assert [t.value for t in Task] == [
        "classification",
        "detection",
        "segmentation",
        "keypoints",
        "regression",
    ]
    assert Task("detection") is Task.DETECTION


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError):
        Task("tracking")


@pytest.mark.parametrize(
    ("task", "label_type"),
    [
        (Task.CLASSIFICATION, "Classification"),
        (Task.DETECTION, "Detections"),
        (Task.SEGMENTATION, "Detections"),
        (Task.KEYPOINTS, "Keypoints"),
        (Task.REGRESSION, "Regression"),
    ],
)
def test_label_type_is_the_fiftyone_type_of_the_ground_truth(task, label_type):
    assert task.label_type == label_type


def test_a_task_compares_equal_to_its_text():
    assert Task.KEYPOINTS == "keypoints"


def test_regression_is_refused_where_it_is_not_built_yet():
    with pytest.raises(CodedError) as caught:
        refuse_regression(Task.REGRESSION)
    assert caught.value.code == "regression_not_supported_yet"
    refuse_regression(Task.DETECTION)  # the other tasks pass
