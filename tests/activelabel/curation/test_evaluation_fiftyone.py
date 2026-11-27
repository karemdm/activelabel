import uuid
from types import SimpleNamespace

import fiftyone as fo
import numpy as np
import pytest

from activelabel import fiftyone_bridge
from activelabel.curation import evaluation_fiftyone as ef
from activelabel.errors import CodedError
from activelabel.labels import Label
from activelabel.tasks import Task

BOX = [0.1, 0.1, 0.4, 0.4]
FAR = [0.6, 0.6, 0.3, 0.3]
SQUARE = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]]
FAR_SQUARE = [[[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]]


def make_dataset(tmp_path, task, truth):
    """A dataset whose samples ``i0``... have the given ground truth (a list of labels each)."""
    dataset = fo.Dataset(f"test-evaluation-{uuid.uuid4().hex[:8]}")
    samples = []
    for k, labels in enumerate(truth):
        sample = fo.Sample(filepath=str(tmp_path / f"{k}.png"), image_id=f"i{k}")
        sample[fiftyone_bridge.GROUND_TRUTH] = fiftyone_bridge.fo_label(task, labels)
        samples.append(sample)
    dataset.add_samples(samples)
    return dataset


def test_the_names_of_the_evaluation_of_a_model():
    assert ef.key_of(3) == "eval_m3" and ef.view_name(3) == "evaluation m3"
    assert ef.prediction_field(Task.DETECTION, 3) == "eval_m3"
    assert ef.prediction_field(Task.SEGMENTATION, 3) == "eval_m3"
    assert ef.prediction_field(Task.CLASSIFICATION, 3) == "eval_m3_pred"  # eval_m3 is FiftyOne's own field


@pytest.mark.parametrize("task", [Task.KEYPOINTS, Task.REGRESSION])
def test_keypoints_and_regression_are_refused(task):
    with pytest.raises(CodedError) as caught:
        ef.refuse(task)
    assert caught.value.code == "evaluation_task_unsupported" and caught.value.params == {"task": task.value}
    with pytest.raises(CodedError):
        ef.evaluate(None, task, 1, {})


def test_polygons_need_shapely(monkeypatch):
    monkeypatch.setattr(ef, "find_spec", lambda name: None)
    with pytest.raises(CodedError) as caught:
        ef.refuse(Task.SEGMENTATION)
    assert caught.value.code == "evaluation_needs_shapely" and caught.value.params == {
        "install": "pip install shapely"
    }
    ef.refuse(Task.DETECTION)  # boxes do not


def test_the_classes_are_listed_with_the_most_frequent_first():
    report = {
        "b": {"precision": 0.5, "recall": 1.0, "f1-score": 2 / 3, "support": 2},
        "a": {"precision": 1.0, "recall": 0.5, "f1-score": 2 / 3, "support": 4},
        "c": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        "micro avg": {"precision": 0.7, "recall": 0.7, "f1-score": 0.7, "support": 6},
        "macro avg": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 6},
        "accuracy": 0.7,
    }
    rows = ef.per_class(report)
    assert [r["class"] for r in rows] == ["a", "b", "c"]
    assert rows[0] == {"class": "a", "precision": 1.0, "recall": 0.5, "f1": 0.666667, "support": 4}


def test_the_confusion_matrix_keeps_the_frequent_classes_and_sums_the_rest(monkeypatch):
    monkeypatch.setattr(ef, "MATRIX_CLASSES", 2)
    asked = {}

    def confusion_matrix(classes, include_other, include_missing):
        asked.update(classes=classes, other=include_other, missing=include_missing)
        return np.arange(16).reshape(4, 4)

    rows = [{"class": name} for name in ("a", "b", "c")]
    got = ef.confusion(SimpleNamespace(confusion_matrix=confusion_matrix), rows, True)
    assert asked == {"classes": ["a", "b"], "other": True, "missing": True}
    assert got["classes"] == ["a", "b"] and got["other"] is True and got["missing"] is True
    assert got["matrix"][1] == [4, 5, 6, 7]


def test_the_worst_images_of_objects_have_the_most_false_positives_and_misses():
    view = SimpleNamespace(
        values=lambda names: [["i0", "i1", "i2", "i3"], [0, 2, 1, None], [0, 1, 0, 0]],
    )
    assert ef.worst_images(view, "eval_m1", "eval_m1", True) == ["i1", "i2"]


def test_the_worst_images_of_a_classification_are_the_wrong_ones_most_sure(monkeypatch):
    monkeypatch.setattr(ef, "WORST_IMAGES", 2)
    view = SimpleNamespace(
        values=lambda names: [["i0", "i1", "i2", "i3"], [True, False, False, False], [0.9, 0.6, 0.8, None]],
    )
    assert ef.worst_images(view, "eval_m1", "eval_m1_pred", False) == ["i2", "i1"]


@pytest.fixture
def classification(tmp_path):
    dataset = make_dataset(
        tmp_path, Task.CLASSIFICATION, [[Label("a")], [Label("a")], [Label("b")], [Label("b")], [Label("b")]]
    )
    yield dataset
    dataset.delete()


@pytest.mark.integration
def test_a_classification_is_evaluated_with_its_confusion_matrix_and_worst_images(classification):
    predictions = {
        "i0": [Label("a", score=0.9)],
        "i1": [Label("b", score=0.95)],  # wrong, and sure
        "i2": [Label("b", score=0.9)],
        "i3": [Label("a", score=0.6)],  # wrong, less sure
        "i4": [Label("b", score=0.8)],
    }
    result = ef.evaluate(classification, Task.CLASSIFICATION, 1, predictions)
    assert result["kind"] == "classification" and result["images"] == 5 and result["view"] == "evaluation m1"
    assert result["metrics"]["accuracy"] == 0.6 and 0 < result["metrics"]["macro_f1"] < 1
    assert result["worst"] == ["i1", "i3"]
    assert result["confusion"]["classes"] == ["b", "a"] and result["confusion"]["missing"] is False
    assert result["confusion"]["matrix"] == [[2, 1], [1, 1]]  # ground truth in rows, predictions in columns
    assert [r["class"] for r in result["classes"]] == ["b", "a"] and result["classes"][0]["support"] == 3
    assert result["fields"] == ["eval_m1_pred", "eval_m1"]
    assert "eval_m1" in classification.list_evaluations()  # the App's Model Evaluation panel lists it
    assert classification.has_sample_field("eval_m1_pred") and classification.has_sample_field("eval_m1")
    assert len(classification.load_saved_view("evaluation m1")) == 5


@pytest.mark.integration
def test_only_the_predicted_images_are_evaluated_and_again_starts_clean(classification):
    first = ef.evaluate(classification, Task.CLASSIFICATION, 1, {"i0": [Label("a", score=0.9)]})
    assert first["images"] == 1
    again = ef.evaluate(
        classification, Task.CLASSIFICATION, 1, {"i0": [Label("b", score=0.9)], "i2": [Label("b", score=0.9)]}
    )
    assert again["images"] == 2 and again["metrics"]["accuracy"] == 0.5
    assert classification.exists("eval_m1_pred").count() == 2
    assert len(classification.load_saved_view("evaluation m1")) == 2


@pytest.mark.integration
def test_a_classification_the_model_did_not_make_counts_as_wrong(classification):
    result = ef.evaluate(classification, Task.CLASSIFICATION, 2, {"i0": [Label("a", score=0.9)], "i1": []})
    assert result["images"] == 2 and result["metrics"]["accuracy"] == 0.5
    assert result["worst"] == ["i1"]


@pytest.mark.integration
def test_a_model_that_predicted_nothing_at_all_is_evaluated_as_wrong_everywhere(classification):
    result = ef.evaluate(classification, Task.CLASSIFICATION, 3, {"i0": [], "i2": []})
    assert result["images"] == 2 and result["metrics"]["accuracy"] == 0.0


@pytest.mark.integration
def test_nothing_to_evaluate_is_an_error(classification):
    with pytest.raises(CodedError) as caught:
        ef.evaluate(classification, Task.CLASSIFICATION, 1, {"gone": [Label("a", score=0.9)]})
    assert caught.value.code == "no_images_in_part"


@pytest.fixture
def detection(tmp_path):
    dataset = make_dataset(
        tmp_path,
        Task.DETECTION,
        [
            [Label("car", bbox=BOX)],
            [Label("car", bbox=BOX), Label("bus", bbox=FAR)],
            [],
            [Label("bus", bbox=BOX)],
        ],
    )
    yield dataset
    dataset.delete()


@pytest.mark.integration
def test_detections_are_evaluated_with_the_map_and_their_misses(detection):
    predictions = {
        "i0": [Label("car", bbox=BOX, score=0.9)],  # right
        "i1": [Label("car", bbox=BOX, score=0.8)],  # the bus is missed
        "i2": [Label("car", bbox=FAR, score=0.7)],  # nothing there: a false positive
        "i3": [Label("car", bbox=BOX, score=0.6)],  # wrong class
    }
    result = ef.evaluate(detection, Task.DETECTION, 4, predictions)
    metrics = result["metrics"]
    assert result["kind"] == "detection" and result["images"] == 4
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 2, 2)
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5 and metrics["f1"] == 0.5
    assert 0 < metrics["map"] <= 1
    assert result["worst"] == ["i3", "i1", "i2"]  # two mistakes, then one each; i0 was right
    assert result["confusion"]["missing"] is True and result["confusion"]["other"] is False
    assert len(result["confusion"]["matrix"]) == len(result["confusion"]["classes"]) + 1
    assert result["fields"] == ["eval_m4", "eval_m4_tp", "eval_m4_fp", "eval_m4_fn"]
    assert "eval_m4" in detection.list_evaluations() and detection.has_sample_field("eval_m4_fp")


@pytest.mark.integration
def test_an_image_where_the_model_found_nothing_counts_its_misses(detection):
    result = ef.evaluate(detection, Task.DETECTION, 5, {"i0": [], "i1": [Label("car", bbox=BOX, score=0.9)]})
    assert result["images"] == 2 and result["metrics"]["fn"] == 2 and result["metrics"]["tp"] == 1
    assert result["worst"] == ["i0", "i1"]  # i0: two misses; i1: the bus


@pytest.fixture
def segmentation(tmp_path):
    pytest.importorskip("shapely")  # FiftyOne compares polygons with it (an extra of activelabel[fiftyone])
    dataset = make_dataset(
        tmp_path,
        Task.SEGMENTATION,
        [
            [Label("leaf", polygons=SQUARE)],
            [Label("leaf", polygons=SQUARE)],
            [Label("leaf", polygons=SQUARE)],
        ],
    )
    yield dataset
    dataset.delete()


@pytest.mark.integration
def test_polygons_are_evaluated_as_polylines_with_their_iou(segmentation):
    predictions = {
        "i0": [Label("leaf", polygons=SQUARE, score=0.9)],
        "i1": [Label("leaf", polygons=FAR_SQUARE, score=0.9)],  # does not overlap
        "i2": [],
    }
    result = ef.evaluate(segmentation, Task.SEGMENTATION, 1, predictions)
    metrics = result["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 2)
    assert result["worst"] == ["i1", "i2"]
    assert segmentation.has_sample_field("eval_m1") and "eval_m1" in segmentation.list_evaluations()
