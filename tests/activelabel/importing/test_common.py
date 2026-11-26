import pytest

from activelabel.importing import common
from activelabel.importing.common import Origin, plan_task, unique_name
from activelabel.labels import Label
from activelabel.projects import ProjectStore
from activelabel.tasks import Task

BOX = [0.1, 0.1, 0.4, 0.4]
RING = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]


def box(name="cat"):
    return Label(name, bbox=BOX)


def poly(name="cat"):
    return Label(name, bbox=[0.1, 0.1, 0.4, 0.5], polygons=[RING])


def codes(plan):
    return {p.code: p for p in plan.problems}


def test_no_labels_uses_hint():
    plan = plan_task({"a": [], "b": []}, Task.SEGMENTATION)
    assert plan.task is Task.SEGMENTATION and plan.problems == []
    assert set(plan.labels) == {"a", "b"}
    assert plan_task({}, Task.DETECTION).task is Task.DETECTION


def test_majority_detection_converts_polygons_and_drops_tags():
    plan = plan_task({"a": [box(), box(), poly()], "b": [Label("dog")]})
    assert plan.task is Task.DETECTION
    assert [lb.bbox is not None and lb.polygons is None for lb in plan.labels["a"]] == [True] * 3
    assert "b" not in plan.labels  # the tag does not fit detection: the image stays unannotated
    found = codes(plan)
    assert found["converted_to_box"].count == 1
    assert found["dropped_other_kind"].count == 1
    assert "classification: 1" in found["dropped_other_kind"].examples


def test_majority_segmentation_drops_boxes():
    plan = plan_task({"a": [poly(), poly()], "b": [box()]})
    assert plan.task is Task.SEGMENTATION
    assert plan.labels["a"][0].polygons and "b" not in plan.labels
    assert codes(plan)["dropped_other_kind"].count == 1


def test_tie_prefers_richer_kind():
    assert plan_task({"a": [poly()], "b": [box()]}).task is Task.SEGMENTATION


def test_keypoints_converted_to_box_when_detection_wins():
    kp = Label("hand", bbox=[0.2, 0.2, 0.2, 0.2], keypoints=[[0.2, 0.2, 2], [0.4, 0.4, 2]])
    plan = plan_task({"a": [box(), box(), kp]})
    assert plan.task is Task.DETECTION and plan.labels["a"][2].keypoints is None
    assert codes(plan)["converted_to_box"].examples == ["keypoints: 1"]


def test_classification_keeps_first_class():
    plan = plan_task({"a": [Label("x"), Label("y")], "b": [Label("x")]})
    assert plan.task is Task.CLASSIFICATION
    assert plan.labels["a"] == [Label("x")]
    assert codes(plan)["multiple_classes"].count == 1


def test_unique_name(tmp_path):
    store = ProjectStore(tmp_path)
    assert unique_name(store, "  Pets  ") == "Pets"
    store.create("Pets", "", "detection")
    assert unique_name(store, "pets") == "pets (2)"
    store.create("pets (2)", "", "detection")
    assert unique_name(store, "Pets") == "Pets (3)"
    long = unique_name(store, "x" * 150)
    assert len(long) == 100


def test_new_project_description_and_rollback(tmp_path):
    store = ProjectStore(tmp_path)
    origin = Origin("cvat", "http://c", "3", "project", "Street")
    with common.new_project(store, origin, "", "", Task.DETECTION) as project:
        assert project.name == "Street"
        assert project.description == ""  # the origin badge says where it came from
    with common.new_project(store, origin, "Mine", "my text", Task.DETECTION) as other:
        assert (other.name, other.description) == ("Mine", "my text")
    with pytest.raises(RuntimeError), common.new_project(store, origin, "Gone", "", Task.DETECTION) as bad:
        assert store.dir(bad.id).exists()
        raise RuntimeError("boom")
    assert not store.dir(bad.id).exists()
    assert [p.name for p in store.list()] == ["Mine", "Street"]


def test_origin_roundtrip(tmp_path):
    assert common.read_origin(tmp_path) is None
    (tmp_path / "origin.json").write_text('{"tool": "cvat"}', encoding="utf-8")
    assert common.read_origin(tmp_path) == {"tool": "cvat"}


def test_kind_of_a_value_is_regression() -> None:
    assert common.kind_of(Label("age", value=31.0)) is Task.REGRESSION
    assert common.kind_of(Label("age", value=0.0)) is Task.REGRESSION


def test_plan_task_picks_regression_for_value_labels() -> None:
    labels = {"a": [Label("age", value=31.0), Label("height", value=170.0)], "b": [Label("age", value=5.0)]}
    plan = plan_task(labels)
    assert plan.task is Task.REGRESSION
    assert [lb.value for lb in plan.labels["a"]] == [31.0, 170.0]
    assert plan.problems == []
