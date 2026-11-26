import io
import json

import pytest
from ann_helpers import make_project

from activelabel.annotations import load_staged, parse, stage, validate
from activelabel.annotations.models import ParseResult
from activelabel.labels import Label


def coco_file(task="seg"):
    ann = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25]}
    if task == "seg":
        ann["segmentation"] = [[0, 0, 50, 0, 50, 25, 0, 25]]
    if task == "pose":
        ann["keypoints"] = [10, 10, 2, 20, 20, 2]
    data = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 50},
        ],
        "categories": [{"id": 1, "name": "Cat"}],
        "annotations": [ann, {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    return ("labels.json", io.BytesIO(json.dumps(data).encode()))


@pytest.fixture
def proj(tmp_path):
    return make_project(tmp_path, ["a.png", "b.png", "c.png"], task="detection", classes=["cat"])


def test_validate_matching_task_and_report(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("det")])
    files = load_staged(store.dir(project.id), token)
    report = validate(parse(files, "coco", manifest.all()), project)
    assert report.task_matches and report.options == ["import", "discard"] and report.can_import
    assert report.counts["images_matched"] == 2 and report.counts["images_without_labels"] == 1
    assert report.counts["labels_per_class"] == {"Cat": 2} and report.classes_new == []
    json.dumps(report.as_dict())


def test_validate_mismatch_options(proj):
    store, project, manifest = proj
    token = stage(store.dir(project.id), [coco_file("seg")])
    report = validate(parse(load_staged(store.dir(project.id), token), "coco", manifest.all()), project)
    assert report.inferred_task == "segmentation" and not report.task_matches
    assert report.options == ["convert_and_import", "change_task_and_import", "discard"]
    store.update(project.id, task="classification")
    report = validate(
        parse(load_staged(store.dir(project.id), token), "coco", manifest.all()), store.get(project.id)
    )
    assert report.options == ["change_task_and_import", "discard"]  # no detection to classification


def test_validate_problems_block_import(proj):
    store, project, manifest = proj
    bad = ("x.json", io.BytesIO(b"{broken"))
    token = stage(store.dir(project.id), [bad])
    report = validate(parse(load_staged(store.dir(project.id), token), "coco", manifest.all()), project)
    assert not report.can_import and report.options == ["discard"]
    assert {p.code for p in report.problems} >= {"invalid_file", "no_images_matched"}
    assert all(len(p.examples) <= 5 for p in report.problems)


def test_nothing_matched_is_one_problem_not_two(proj):
    store, project, manifest = proj
    data = {
        "images": [{"id": 1, "file_name": "nope.png", "width": 100, "height": 50}],
        "categories": [{"id": 1, "name": "Cat"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    token = stage(store.dir(project.id), [("labels.json", io.BytesIO(json.dumps(data).encode()))])
    report = validate(parse(load_staged(store.dir(project.id), token), "coco", manifest.all()), project)
    codes = [p.code for p in report.problems]
    assert codes.count("no_images_matched") == 1 and "unmatched_entries" not in codes
    assert report.unmatched["count"] == 1


def test_regression_values_are_matched_and_bad_values_reported_once(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png", "b.png"], task="regression", classes=["age"])
    csv_file = ("v.csv", io.BytesIO(b"image,age\na.png,3\nb.png,oops\n"))
    files = load_staged(store.dir(project.id), stage(store.dir(project.id), [csv_file]))
    report = validate(parse(files, "regression", manifest.all()), project)
    assert report.inferred_task == "regression" and report.task_matches and report.can_import
    assert [p.code for p in report.problems] == ["value_invalid"]
    assert report.counts["labels_per_class"] == {"age": 1} and report.classes_new == []


def test_regression_repeated_target_is_a_warning(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png"], task="regression", classes=["age"])
    labels = {manifest.all()[0].id: [Label("age", value=1.0), Label("age", value=2.0)]}
    result = ParseResult("regression", labels, ["age"], ["v.json"], 1)
    report = validate(result, project)
    found = next(p for p in report.problems if p.code == "repeated_target")
    assert found.level == "warning" and found.count == 1
