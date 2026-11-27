from curation_images import build

from activelabel.curation.distributions import distributions
from activelabel.images.manifest import EXCLUDED
from activelabel.labels import AnnotationSet, Label
from activelabel.tasks import Task


def box(name, w, h):
    return Label(name, bbox=[0.0, 0.0, w, h])


def detection(manifest):
    ids = [r.id for r in manifest.all()]
    images = {
        ids[0]: [box("a", 0.05, 0.05), box("a", 0.1, 0.1), box("b", 0.9, 0.9)],  # tiny, small, large
        ids[1]: [box("b", 0.4, 0.4)],  # medium
        ids[2]: [box("a", 0.05, 0.05)] * 12,
    }
    return ids, AnnotationSet(Task.DETECTION, ["a", "b", "unused"], images, {})


def test_detection_counts_classes_objects_and_box_sizes(tmp_path):
    manifest = build(tmp_path)
    ids, annotations = detection(manifest)
    data = distributions(annotations, manifest, annotations.classes, Task.DETECTION)
    assert data["images"] == 7 and data["annotated"] == 3 and data["without_annotations"] == 4
    rows = {r["name"]: r for r in data["classes"]}
    assert rows["a"] == {"name": "a", "objects": 14, "images": 2}
    assert rows["b"] == {"name": "b", "objects": 2, "images": 2}
    assert rows["unused"]["objects"] == 0  # a class with no object is still listed
    assert [r["name"] for r in data["classes"]] == ["a", "b", "unused"]  # most objects first
    per_image = {b["key"]: b["images"] for b in data["objects_per_image"]}
    assert per_image == {"0": 4, "1": 1, "2-3": 1, "4-10": 0, "11+": 1}
    area = {b["key"]: b["objects"] for b in data["box_area"]}
    assert area == {"tiny": 13, "small": 1, "medium": 1, "large": 1} and data["boxes"] == 16


def test_objects_per_image_buckets(tmp_path):
    manifest = build(tmp_path)
    ids, annotations = detection(manifest)
    data = distributions(annotations, manifest, [], Task.DETECTION)
    assert {b["key"]: b["images"] for b in data["objects_per_image"]} == {
        "0": 4,
        "1": 1,
        "2-3": 1,
        "4-10": 0,
        "11+": 1,
    }
    assert [r["name"] for r in data["classes"]] == ["a", "b"]  # classes of the labels join the project's


def test_excluded_images_and_a_chosen_part_are_left_out(tmp_path):
    manifest = build(tmp_path)
    ids, annotations = detection(manifest)
    manifest.set_status([ids[0]], EXCLUDED, "x")
    data = distributions(annotations, manifest, ["a", "b"], Task.DETECTION)
    assert data["images"] == 6 and data["boxes"] == 13
    part = distributions(annotations, manifest, ["a", "b"], Task.DETECTION, [ids[0], ids[1]])
    assert part["images"] == 1 and part["boxes"] == 1  # the excluded image is not in the part either


def test_no_annotations_yet(tmp_path):
    manifest = build(tmp_path)
    data = distributions(None, manifest, ["a"], Task.DETECTION)
    assert data["annotated"] == 0 and data["without_annotations"] == 7 and data["boxes"] == 0
    assert data["classes"] == [{"name": "a", "objects": 0, "images": 0}]


def test_classification_counts_images_per_class_only(tmp_path):
    manifest = build(tmp_path)
    ids = [r.id for r in manifest.all()]
    images = {ids[0]: [Label("cat"), Label("dog")], ids[1]: [Label("cat")]}  # only the first class counts
    data = distributions(
        AnnotationSet(Task.CLASSIFICATION, ["cat", "dog"], images, {}),
        manifest,
        ["cat", "dog"],
        Task.CLASSIFICATION,
    )
    assert data["classes"] == [
        {"name": "cat", "objects": 2, "images": 2},
        {"name": "dog", "objects": 0, "images": 0},
    ]
    assert data["objects_per_image"] is None and data["box_area"] is None and data["without_annotations"] == 5


def test_regression_has_no_classes(tmp_path):
    manifest = build(tmp_path)
    ids = [r.id for r in manifest.all()]
    annotations = AnnotationSet(Task.REGRESSION, ["age"], {ids[0]: [Label("age", value=3.0)]}, {})
    data = distributions(annotations, manifest, ["age"], Task.REGRESSION)
    assert data["classes"] == [] and data["annotated"] == 0 and data["objects_per_image"] is None
    assert data["box_area"] is None and data["images"] == 7


def test_segmentation_without_boxes_has_empty_size_buckets(tmp_path):
    manifest = build(tmp_path)
    ids = [r.id for r in manifest.all()]
    polygon = Label("a", polygons=[[[0, 0], [1, 0], [1, 1]]])
    annotations = AnnotationSet(Task.SEGMENTATION, ["a"], {ids[0]: [polygon]}, {})
    data = distributions(annotations, manifest, ["a"], Task.SEGMENTATION)
    assert data["boxes"] == 0 and all(b["objects"] == 0 for b in data["box_area"])
