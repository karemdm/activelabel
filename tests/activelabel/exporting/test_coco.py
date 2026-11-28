"""The COCO document of an export."""

import pytest

from activelabel.exporting.coco import coco_document
from activelabel.labels import Label
from activelabel.tasks import Task

CLASSES = ["cat", "dog"]


def test_boxes_are_in_pixels_with_categories_from_one():
    doc = coco_document(
        Task.DETECTION, [("a.jpg", 200, 100, [Label("dog", bbox=[0.1, 0.2, 0.5, 0.5])])], CLASSES
    )
    assert doc["categories"] == [
        {"id": 1, "name": "cat", "supercategory": ""},
        {"id": 2, "name": "dog", "supercategory": ""},
    ]
    assert doc["images"] == [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}]
    (ann,) = doc["annotations"]
    assert ann["image_id"] == 1 and ann["category_id"] == 2 and ann["iscrowd"] == 0 and ann["id"] == 1
    assert ann["bbox"] == pytest.approx([20, 20, 100, 50]) and ann["area"] == pytest.approx(5000)


def test_an_image_without_labels_is_listed_without_annotations():
    doc = coco_document(Task.DETECTION, [("a.jpg", 10, 10, []), ("b.jpg", 10, 10, [])], CLASSES)
    assert [i["id"] for i in doc["images"]] == [1, 2] and doc["annotations"] == []


def test_polygons_give_segmentation_area_and_box():
    square = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    doc = coco_document(Task.SEGMENTATION, [("a.jpg", 100, 100, [Label("cat", polygons=[square])])], CLASSES)
    (ann,) = doc["annotations"]
    assert ann["segmentation"] == [[0, 0, 50, 0, 50, 50, 0, 50]]
    assert ann["area"] == pytest.approx(2500) and ann["bbox"] == pytest.approx([0, 0, 50, 50])


def test_keypoints_are_flat_triples_and_the_categories_name_them():
    label = Label("cat", bbox=[0.1, 0.1, 0.5, 0.5], keypoints=[[0.2, 0.2, 2], [0.0, 0.0, 0]])
    doc = coco_document(Task.KEYPOINTS, [("a.jpg", 100, 50, [label])], CLASSES, keypoints=2)
    (ann,) = doc["annotations"]
    assert ann["keypoints"] == pytest.approx([20, 10, 2, 0, 0, 0]) and ann["num_keypoints"] == 1
    assert doc["categories"][0]["keypoints"] == ["kp1", "kp2"]


def test_an_object_without_a_box_or_polygon_is_left_out():
    doc = coco_document(Task.DETECTION, [("a.jpg", 10, 10, [Label("cat")])], CLASSES)
    assert doc["annotations"] == []
