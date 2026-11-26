import math

import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, parse
from activelabel.labels import task_of
from activelabel.tasks import Task


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


def test_coco_detection_polygon_rle(env):
    folder, images, recs = env
    data = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 50},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
            {
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            },
            {
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 50, 25],
                "segmentation": {"counts": "x", "size": [50, 100]},
            },
            {"image_id": 2, "category_id": 9, "bbox": [0, 0, 5, 5]},
        ],
    }
    path = write(folder, "inst.json", data)
    assert detect_format([path]) == "coco"
    res = parse([path], "coco", images)
    assert res.classes == ["cat", "dog"]
    first = res.labels[recs["a.png"].id]
    assert first[0].bbox == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert first[1].polygons is not None and first[1].bbox == pytest.approx([0, 0, 0.1, 0.2])
    second = res.labels[recs["b.png"].id]
    assert len(second) == 1 and second[0].polygons is None
    codes = {p.code for p in res.problems}
    assert {"rle_bbox_only", "unknown_class_id"} <= codes
    assert task_of(res.all_labels()) is Task.SEGMENTATION


def test_coco_keypoints_and_size_mismatch(env):
    folder, images, recs = env
    data = {
        "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose", "eye", "ear"]}],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 10, 100, 50],
                "keypoints": [100, 50, 2, 0, 0, 0, 40, 20, 1],
            }
        ],
    }
    res = parse([write(folder, "k.json", data)], "coco", images)
    lb = res.labels[recs["a.png"].id][0]
    assert lb.keypoints == [[0.5, 0.5, 2.0], [0.0, 0.0, 0.0], [0.2, 0.2, 1.0]]  # scaled by the file size
    assert lb.bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert task_of(res.all_labels()) is Task.KEYPOINTS
    assert "size_mismatch" in {p.code for p in res.problems}
    data["categories"][0]["keypoints"] = ["nose", "eye"]
    res = parse([write(folder, "k2.json", data)], "coco", images)
    assert "keypoint_count_mismatch" in {p.code for p in res.problems}


def test_coco_invalid_file(env):
    folder, images, _ = env
    res = parse([write(folder, "bad.json", "{not json")], "coco", images)
    assert res.problems[0].code == "invalid_file" and res.problems[0].level == "error"


def test_coco_damaged_geometry_is_reported_never_imported_as_is(env):
    """#620: NaN or infinite numbers, boxes outside the image or without area, polygons with too few points
    and unknown classes are each counted in the report; what is kept is inside the image and finite."""
    folder, images, recs = env
    text = """{"images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 50},
                {"id": 2, "file_name": "b.png", "width": 100, "height": 50}],
     "categories": [{"id": 1, "name": "cat"}],
     "annotations": [
      {"image_id": 1, "category_id": 1, "bbox": [NaN, 5, 20, 10]},
      {"image_id": 1, "category_id": 1, "bbox": [80, 40, 100, 100]},
      {"image_id": 1, "category_id": 1, "bbox": [5, 5, 0, 10]},
      {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": [[0, 0, 10, 0]]},
      {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": [[0, 0, NaN, 0, 10, 10]]},
      {"image_id": 2, "category_id": 1, "bbox": [-Infinity, 0, 5, 5]},
      {"image_id": 2, "category_id": 7, "bbox": [0, 0, 5, 5]}]}"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "damaged.json"
    path.write_text(text, encoding="utf-8")
    res = parse([path], "coco", images)
    counts = {p.code: p.count for p in res.problems}
    assert counts["invalid_annotation"] == 3  # NaN box, NaN polygon point, infinite box
    assert counts["box_clipped"] == 1 and counts["box_degenerate"] == 1
    assert counts["polygon_too_few_points"] == 1 and counts["unknown_class_id"] == 1
    kept = [lb for items in res.labels.values() for lb in items]
    for lb in kept:
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in lb.bbox)
        assert all(math.isfinite(v) for ring in lb.polygons or [] for point in ring for v in point)
    assert len(res.labels[recs["a.png"].id]) == 1  # only the clipped box
