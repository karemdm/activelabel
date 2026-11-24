from activelabel.annotations import conversion_possible, convert_annotations
from activelabel.annotations.models import ParseResult
from activelabel.annotations.validation import validate
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.tasks import Task

POLY = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4]]


def segmentation():
    images = {
        "a": [Label("cat", bbox=[0.1, 0.1, 0.4, 0.3], polygons=[POLY]), Label("dog", polygons=[POLY])],
        "b": [Label("cat", polygons=[POLY]), Label("cat", polygons=[POLY])],
        "c": [],
    }
    return AnnotationSet(Task.SEGMENTATION, ["cat", "dog"], images, {"format": "coco"})


def test_possible_pairs():
    t = Task
    yes = [
        (t.SEGMENTATION, t.DETECTION),
        (t.KEYPOINTS, t.DETECTION),
        (t.DETECTION, t.CLASSIFICATION),
        (t.SEGMENTATION, t.CLASSIFICATION),
        (t.KEYPOINTS, t.CLASSIFICATION),
    ]
    assert all(conversion_possible(a, b) for a, b in yes)
    no = [(t.CLASSIFICATION, t.DETECTION), (t.DETECTION, t.SEGMENTATION), (t.DETECTION, t.KEYPOINTS)]
    assert not any(conversion_possible(a, b) for a, b in no)
    assert convert_annotations(AnnotationSet(t.DETECTION, [], {}, {}), t.KEYPOINTS) is None


def test_segmentation_to_detection_keeps_boxes():
    source = segmentation()
    out = convert_annotations(source, Task.DETECTION)
    assert out is not None and out.task is Task.DETECTION
    assert out.source["converted_from"] == "segmentation" and out.source["format"] == "coco"
    assert all(lb.bbox and lb.polygons is None for v in out.images.values() for lb in v)
    assert len(out.images["a"]) == 2 and out.images["c"] == []
    assert source.images["a"][0].polygons  # the original is untouched


def test_keypoints_to_detection():
    points = [[0.2, 0.2, 2], [0.6, 0.5, 2]]
    kp = AnnotationSet(Task.KEYPOINTS, ["p"], {"a": [Label("p", keypoints=points)]}, {})
    out = convert_annotations(kp, Task.DETECTION)
    assert out is not None and out.images["a"][0].bbox is not None and out.images["a"][0].keypoints is None


def test_objects_to_classification_uses_the_most_frequent_class():
    out = convert_annotations(segmentation(), Task.CLASSIFICATION)
    assert out is not None
    assert [lb.label for lb in out.images["a"]] == ["cat"]  # tie: first to appear
    assert [lb.label for lb in out.images["b"]] == ["cat"] and out.images["c"] == []
    assert all(lb.bbox is None for v in out.images.values() for lb in v)


def test_most_frequent_class_wins():
    images = {
        "a": [
            Label("dog", bbox=[0, 0, 1, 1]),
            Label("cat", bbox=[0, 0, 1, 1]),
            Label("cat", bbox=[0, 0, 1, 1]),
        ]
    }
    out = convert_annotations(AnnotationSet(Task.DETECTION, ["cat", "dog"], images, {}), Task.CLASSIFICATION)
    assert out is not None and [lb.label for lb in out.images["a"]] == ["cat"]


def test_several_classes_per_image_give_a_warning_for_classification():
    labels = {"a": [Label("cat"), Label("dog")], "b": [Label("cat")]}
    result = ParseResult("coco", labels, ["cat", "dog"], ["x.json"], 2)
    report = validate(result, Project(1, "t", "", Task.CLASSIFICATION, []))
    assert report.task_matches and report.can_import
    assert [(p.level, p.code, p.count, p.examples) for p in report.problems] == [
        ("warning", "multiple_classes", 1, ["a"])
    ]
    assert report.problems[0].message == "1 images have more than one class; only the first one is kept"
