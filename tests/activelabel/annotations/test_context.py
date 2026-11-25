import pytest
from ann_helpers import make_manifest

from activelabel.annotations.context import ParseContext, bbox_of_points, norm_name, stem_parts
from activelabel.labels import Label


def test_names_are_normalised():
    assert norm_name("./a" + chr(92) + "b/c.png") == "a/b/c.png"
    assert norm_name("/x/y.png") == "x/y.png"
    assert stem_parts("cat/img.001.png") == ["cat", "img.001"]
    assert stem_parts("") == []


def test_bbox_of_points():
    assert bbox_of_points([[0.1, 0.2], [0.5, 0.6]]) == pytest.approx([0.1, 0.2, 0.4, 0.4])
    assert bbox_of_points([]) is None
    assert bbox_of_points([[0.3, 0.3], [0.3, 0.3]]) is None  # an empty box is no box


def test_context_matches_images_and_collects_classes(tmp_path):
    manifest = make_manifest(tmp_path, ["Ação/a.png", "b.png"])
    ctx = ParseContext("coco", manifest.all())
    rec = ctx.match("ação/A.PNG")  # same name ignoring case, accents kept
    assert rec is not None and rec.name == "Ação/a.png"
    assert ctx.match("missing.png", example="missing.png") is None
    ctx.add_class("Ação")
    ctx.add_class("Ação")
    ctx.add_label(rec, Label("Ação"))
    result = ctx.result()
    assert result.classes == ["Ação"] and result.unmatched_count == 1
    assert result.labels[rec.id][0].label == "Ação"


def test_problems_are_aggregated_by_code(tmp_path):
    ctx = ParseContext("coco", make_manifest(tmp_path, ["a.png"]).all())
    ctx.problem("invalid_annotation", "x")
    ctx.problem("invalid_annotation", "y")
    (problem,) = ctx.result().problems
    assert problem.code == "invalid_annotation" and problem.count == 2 and problem.examples == ["x", "y"]


def test_polygon_and_keypoints_outside_the_image_are_clipped_and_reported(tmp_path):
    ctx = ParseContext("coco", make_manifest(tmp_path, ["a.png"]).all())
    assert ctx.ring([(10, 5), (80, 8), (110, 30)], 100, 50, "a.png") == [[0.1, 0.1], [0.8, 0.16], [1.0, 0.6]]
    assert ctx.point(-5, 20, 2, 100, 50, "a.png") == [0.0, 0.4, 2.0]
    (problem,) = ctx.result().problems
    assert problem.code == "geometry_clipped" and problem.count == 2


def test_polygon_and_keypoints_inside_the_image_report_nothing(tmp_path):
    ctx = ParseContext("coco", make_manifest(tmp_path, ["a.png"]).all())
    ctx.ring([(0, 0), (100, 0), (100, 50)], 100, 50, "a.png")  # on the border is inside
    ctx.point(100, 50, 2, 100, 50, "a.png")
    assert ctx.result().problems == []
