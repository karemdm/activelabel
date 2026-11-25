import json

import pytest
from ann_helpers import make_project

from activelabel.annotate import diff as d
from activelabel.annotate.base import Pulled
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.sync.links import label_hash
from activelabel.tasks import Task


def box(name, x=0.1, y=0.1, w=0.4, h=0.4):
    return Label(name, bbox=[x, y, w, h])


@pytest.fixture
def proj(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png", "b.png"], classes=["cat", "dog"])
    ids = [r.id for r in manifest.all()]
    folder = store.dir(project.id)
    old = {ids[0]: [box("cat"), box("dog", 0.6, 0.6, 0.2, 0.2)]}
    save_annotations(folder, AnnotationSet(Task.DETECTION, ["cat", "dog"], old, {}))
    return store, project, manifest, ids, folder


def test_added_removed(proj):
    _, project, _, ids, folder = proj
    new = {
        ids[0]: [box("cat", 0.1, 0.1, 0.4, 0.4), box("cat", 0.0, 0.8, 0.1, 0.1)],  # dog removed, one added
        ids[1]: [],
    }
    r = d.diff(folder, ids, new, project)
    assert (r["images_changed"], r["added"], r["removed"], r["edited"]) == (1, 1, 1, 0)
    assert r["per_class"] == {
        "cat": {"added": 1, "removed": 0, "edited": 0},
        "dog": {"added": 0, "removed": 1, "edited": 0},
    }
    assert r["examples"][0]["image_id"] == ids[0] and r["problems"] == []


def test_moved_box_is_edited_small_tolerance_is_not(proj):
    _, project, _, ids, folder = proj
    keep_dog = box("dog", 0.6, 0.6, 0.2, 0.2)
    same = d.diff(folder, ids, {ids[0]: [box("cat", 0.102), keep_dog], ids[1]: []}, project)
    assert same["images_changed"] == 0
    moved = d.diff(folder, ids, {ids[0]: [box("cat", 0.15, 0.1, 0.4, 0.4), keep_dog], ids[1]: []}, project)
    assert moved["edited"] == 1 and moved["added"] == 0 and moved["removed"] == 0


def test_class_change_and_unknown_class(proj):
    _, project, _, ids, folder = proj
    new = {ids[0]: [box("dog"), box("Bird", 0.6, 0.6, 0.2, 0.2)], ids[1]: []}
    r = d.diff(folder, ids, new, project)
    assert r["edited"] == 2  # cat -> dog, and dog -> Bird, each at the same place
    assert r["added"] == 0 and r["removed"] == 0
    assert r["unknown_classes"] == ["Bird"]
    assert [p["code"] for p in r["problems"]] == ["unknown_class"]


def test_case_insensitive_class_matches_project(proj):
    _, project, _, ids, folder = proj
    new = {ids[0]: [box("CAT"), box("dog", 0.6, 0.6, 0.2, 0.2)], ids[1]: []}
    r = d.diff(folder, ids, new, project)
    assert r["images_changed"] == 0 and r["unknown_classes"] == []


def test_missing_image_and_no_geometry(proj):
    _, project, _, ids, folder = proj
    r = d.diff(folder, ids, {ids[0]: [Label("cat")]}, project)
    assert {p["code"] for p in r["problems"]} == {"missing_images", "no_geometry"}
    assert r["removed"] == 2


def test_classification(tmp_path):
    names = ["a", "b", "c", "d"]
    store, project, manifest = make_project(tmp_path, names, task="classification", classes=["x", "y"])
    ids = [r.id for r in manifest.all()]
    folder = store.dir(project.id)
    old = {ids[0]: [Label("x")], ids[1]: [Label("x")]}
    save_annotations(folder, AnnotationSet(Task.CLASSIFICATION, ["x", "y"], old, {}))
    new = {ids[0]: [Label("y")], ids[1]: [], ids[2]: [Label("x")], ids[3]: [Label("x"), Label("y")]}
    r = d.diff(folder, ids, new, project)
    assert (r["added"], r["removed"], r["edited"]) == (2, 1, 1)
    assert [p["code"] for p in r["problems"]] == ["multiple_classes"]


def test_keypoints_and_polygons(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a"], task="keypoints", classes=["p"])
    (image,) = [r.id for r in manifest.all()]
    folder = store.dir(project.id)
    old = Label("p", keypoints=[[0.1, 0.1, 2], [0.5, 0.5, 2]])
    save_annotations(folder, AnnotationSet(Task.KEYPOINTS, ["p"], {image: [old]}, {}))
    same = Label("p", keypoints=[[0.1, 0.1, 1], [0.5, 0.5, 2]])  # hidden vs visible: same
    assert d.diff(folder, [image], {image: [same]}, project)["images_changed"] == 0
    moved = Label("p", keypoints=[[0.15, 0.1, 2], [0.5, 0.5, 2]])
    assert d.diff(folder, [image], {image: [moved]}, project)["edited"] == 1
    hidden = Label("p", keypoints=[[0.1, 0.1, 2], [0.0, 0.0, 0]])
    assert d.diff(folder, [image], {image: [hidden]}, project)["edited"] == 1

    ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    a = Label("p", polygons=[ring])
    b = Label("p", polygons=[[*ring, [0.1, 0.5]]])
    changes = d.compare_image(Task.SEGMENTATION, [a], [b])
    assert len(changes.edited) == 1


def test_apply(proj):
    store, project, manifest, ids, folder = proj
    new = {ids[0]: [box("cat"), box("Bird", 0.6, 0.6, 0.2, 0.2), Label("dog")], ids[1]: [box("dog")]}
    counts = d.apply(store, project.id, manifest, ids, new)
    assert counts["new_classes"] == ["Bird"] and counts["skipped"] == 1 and counts["images_updated"] == 2
    backups = list(folder.glob("annotations.*.bak.json"))
    assert len(backups) == 1 and counts["backup"] == backups[0].name
    assert len(json.loads(backups[0].read_text())["images"][ids[0]]) == 2  # the old version
    saved = load_annotations(folder)
    assert [x.label for x in saved.images[ids[0]]] == ["cat", "Bird"]
    assert [x.label for x in saved.images[ids[1]]] == ["dog"]
    assert saved.classes == ["cat", "dog", "Bird"]
    assert store.get(project.id).classes == ["cat", "dog", "Bird"]


def test_apply_creates_annotations_when_none(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a"], classes=["cat"])
    (image,) = [r.id for r in manifest.all()]
    counts = d.apply(store, project.id, manifest, [image], {image: [box("cat")]})
    assert counts["backup"] == "" and load_annotations(store.dir(project.id)).task is Task.DETECTION


def test_settle_takes_everything_from_a_tool_that_cannot_tell_what_was_edited():
    pulled = {"a": [box("cat")], "b": []}
    assert d.settle(pulled, {"b": [box("dog")]}) == (pulled, [])


def test_settle_keeps_the_project_labels_of_images_not_edited_in_the_tool():
    sent, now = [box("cat")], [box("dog")]
    hashes = {"a": label_hash(sent), "b": label_hash(sent)}
    pulled = Pulled({"a": [box("cat", 0.3)], "b": sent, "c": []}, {"a"}, hashes)
    labels, conflicts = d.settle(pulled, {"a": sent, "b": now})
    assert labels == {"a": [box("cat", 0.3)], "b": now, "c": []}  # b changed only in the project: kept
    assert conflicts == []


def test_settle_lists_images_edited_on_both_sides(proj):
    _, project, _, ids, folder = proj
    sent = [box("cat", 0.5)]
    pulled = Pulled({ids[0]: [box("cat", 0.7)]}, {ids[0]}, {ids[0]: label_hash(sent)})
    labels, conflicts = d.settle(pulled, load_annotations(folder).images)
    assert labels == {ids[0]: [box("cat", 0.7)]} and conflicts == [ids[0]]
    problems = d.diff(folder, [ids[0]], labels, project, conflicts)["problems"]
    assert [(p["code"], p["count"]) for p in problems] == [("edited_on_both_sides", 1)]
