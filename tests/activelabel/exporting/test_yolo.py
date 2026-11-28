"""The text of a YOLO export: label lines and data.yaml."""

import yaml

from activelabel.exporting.yolo import data_yaml, label_lines
from activelabel.labels import Label
from activelabel.tasks import Task

CLASSES = ["cat", "dog"]


def test_detection_line_is_class_centre_and_size():
    lines = label_lines(Task.DETECTION, [Label("dog", bbox=[0.1, 0.2, 0.4, 0.2])], CLASSES)
    assert lines == ["1 0.300000 0.300000 0.400000 0.200000"]


def test_a_background_image_has_no_lines():
    assert label_lines(Task.DETECTION, [], CLASSES) == []


def test_segmentation_writes_a_line_per_ring_and_skips_small_rings():
    ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    label = Label("cat", polygons=[ring, ring[:2], ring])
    assert (
        label_lines(Task.SEGMENTATION, [label], CLASSES)
        == ["0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000"] * 2
    )


def test_keypoints_follow_the_box_with_their_visibility():
    label = Label("cat", bbox=[0.0, 0.0, 0.5, 0.5], keypoints=[[0.1, 0.2, 2], [0.3, 0.4, 1]])
    assert label_lines(Task.KEYPOINTS, [label], CLASSES) == [
        "0 0.250000 0.250000 0.500000 0.500000 0.100000 0.200000 2 0.300000 0.400000 1"
    ]
    assert label_lines(Task.KEYPOINTS, [Label("cat", bbox=[0, 0, 1, 1])], CLASSES) == []


def test_an_object_without_a_box_is_left_out_and_a_polygon_gives_its_box():
    assert label_lines(Task.DETECTION, [Label("cat")], CLASSES) == []
    poly = Label("cat", polygons=[[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]]])
    assert label_lines(Task.DETECTION, [poly], CLASSES) == ["0 0.400000 0.400000 0.400000 0.400000"]


def test_coordinates_stay_inside_the_image():
    line = label_lines(
        Task.SEGMENTATION, [Label("cat", polygons=[[[-0.1, 0.0], [1.2, 0.0], [1.0, 1.0]]])], CLASSES
    )
    assert line == ["0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000"]


def test_data_yaml_keeps_names_with_accents_and_spaces():
    text = data_yaml(
        {"train": "images/train", "val": ["images/a", "images/b"]}, ["gato preto", "cão"], keypoints=3
    )
    data = yaml.safe_load(text)
    assert data["path"] == "." and data["train"] == "images/train" and data["val"] == ["images/a", "images/b"]
    assert data["names"] == {0: "gato preto", 1: "cão"} and data["kpt_shape"] == [3, 3]
    assert "kpt_shape" not in yaml.safe_load(data_yaml({}, CLASSES))
