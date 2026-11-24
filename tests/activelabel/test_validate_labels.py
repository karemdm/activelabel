from activelabel.labels import Label, validate_labels
from activelabel.tasks import Task

BOX = [0.1, 0.1, 0.3, 0.3]


def test_every_category_is_reported_with_examples():
    detection = {
        "a": [Label("cat", bbox=BOX), Label("cat")],  # the second has no box
        "b": [
            Label("cat", bbox=[0.2, 0.2, 0, 0.1]),
            Label("cat", bbox=[1.5, 0.1, 0.2, 0.2]),
        ],  # flat, outside
        "c": [Label("bird", bbox=BOX)],  # a class the project does not have
    }
    check = validate_labels(Task.DETECTION, detection, ["cat", "Dog"])
    assert check.counts["no_geometry"] == 1 and check.counts["box_invalid"] == 2
    assert check.counts["unknown_class"] == 1 and check.labels == 5 and not check.ok
    assert check.examples["box_invalid"] == ["b"]
    assert check.problems() == {"no_geometry": 1, "box_invalid": 2, "unknown_class": 1}
    assert validate_labels(
        Task.DETECTION, {"a": [Label("cat", bbox=BOX)]}, ["CAT"]
    ).ok  # case does not matter
    assert validate_labels(Task.DETECTION, {"a": [Label("x", bbox=BOX)]}, []).ok  # no class list: not checked
    assert validate_labels(Task.DETECTION, {}).ok


def test_polygons_keypoints_and_classification():
    ring3, ring2 = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], [[0.1, 0.1], [0.5, 0.1]]
    seg = validate_labels(
        Task.SEGMENTATION, {"a": [Label("c", polygons=[ring3]), Label("c", polygons=[ring2])]}
    )
    assert seg.counts["polygon_too_few_points"] == 1 and seg.counts["no_geometry"] == 0
    three = [[0.1, 0.1, 2], [0.2, 0.2, 2], [0.3, 0.3, 2]]
    pose = {
        "a": [Label("p", keypoints=three)],
        "b": [Label("p", keypoints=three)],
        "c": [Label("p", keypoints=three[:1])],
    }
    assert validate_labels(Task.KEYPOINTS, pose).counts["keypoints_count_mismatch"] == 1  # usual size: 3
    assert validate_labels(Task.KEYPOINTS, pose, keypoints=1).counts["keypoints_count_mismatch"] == 2
    assert validate_labels(Task.KEYPOINTS, {"a": [Label("p", bbox=BOX)]}).counts["no_geometry"] == 1
    cls = validate_labels(Task.CLASSIFICATION, {"a": [Label("x"), Label("y")], "b": [Label("x"), Label("X")]})
    assert cls.counts["multiple_classes"] == 1 and cls.examples["multiple_classes"] == ["a"]
