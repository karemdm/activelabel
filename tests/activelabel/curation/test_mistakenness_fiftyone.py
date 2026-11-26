import uuid

import fiftyone as fo
import pytest

from activelabel import fiftyone_bridge
from activelabel.curation import mistakenness_fiftyone as mf
from activelabel.labels import Label
from activelabel.tasks import Task

pytestmark = pytest.mark.integration


def make_dataset(tmp_path, task, truth):
    """A dataset whose samples ``i0``... have the given ground truth (a list of labels each)."""
    dataset = fo.Dataset(f"test-mistakenness-{uuid.uuid4().hex[:8]}")
    samples = []
    for k, labels in enumerate(truth):
        sample = fo.Sample(filepath=str(tmp_path / f"{k}.png"), image_id=f"i{k}")
        sample[fiftyone_bridge.GROUND_TRUTH] = fiftyone_bridge.fo_label(task, labels)
        samples.append(sample)
    dataset.add_samples(samples)
    return dataset


@pytest.fixture
def classification(tmp_path):
    dataset = make_dataset(
        tmp_path, Task.CLASSIFICATION, [[Label("a")], [Label("a")], [Label("b")], [Label("b")]]
    )
    yield dataset
    dataset.delete()


def test_classification_with_confidence_flags_the_label_the_model_disagrees_with(classification):
    predictions = {
        "i0": [Label("a", score=0.95)],
        "i1": [Label("b", score=0.95)],  # labelled a, the model is sure it is b
        "i2": [Label("b", score=0.9)],
        "i3": [Label("b", score=0.9)],
    }
    result = mf.score(classification, Task.CLASSIFICATION, predictions)
    assert set(result) == {"i0", "i1", "i2", "i3"}
    assert result["i1"]["mistakenness"] > result["i0"]["mistakenness"]
    assert result["i1"]["mistakenness"] > result["i2"]["mistakenness"]
    assert all(r["hardness"] is None and r["missing"] == 0 and r["spurious"] == 0 for r in result.values())
    assert classification.has_sample_field(mf.PRED_FIELD) and not classification.has_sample_field(mf.HARDNESS)


def test_images_without_a_prediction_are_left_out(classification):
    result = mf.score(classification, Task.CLASSIFICATION, {"i0": [Label("a", score=0.9)], "i1": []})
    assert set(result) == {"i0"}


def test_classification_with_logits_gives_the_hardness(classification):
    logits = {"i0": [4.0, 0.0], "i1": [0.0, 4.0], "i2": [0.1, 0.0], "i3": [0.0, 3.0]}
    result = mf.score(classification, Task.CLASSIFICATION, {}, logits, ["a", "b"])
    assert set(result) == {"i0", "i1", "i2", "i3"}
    assert all(r["hardness"] is not None for r in result.values())
    assert result["i2"]["hardness"] > result["i1"]["hardness"]  # unsure is harder than sure
    assert result["i1"]["mistakenness"] > result["i0"]["mistakenness"]
    assert classification.has_sample_field(mf.HARDNESS)


def test_scoring_again_replaces_the_previous_fields(classification):
    mf.score(classification, Task.CLASSIFICATION, {}, {"i0": [4.0, 0.0], "i1": [0.0, 4.0]}, ["a", "b"])
    mf.score(classification, Task.CLASSIFICATION, {"i0": [Label("a", score=0.9)]})
    assert not classification.has_sample_field(mf.HARDNESS)
    assert classification.exists(mf.PRED_FIELD).count() == 1


def test_classification_from_logits_takes_the_best_class_with_its_probability():
    value = mf.classification_from_logits([0.0, 2.0, 0.0], ["a", "b", "c"])
    assert value.label == "b" and list(value.logits) == [0.0, 2.0, 0.0]
    assert value.confidence == pytest.approx(0.78698, abs=1e-4)


def test_a_best_class_the_project_no_longer_has_gives_none():
    assert mf.classification_from_logits([0.0, 2.0], ["a", None]) is None
    assert mf.classification_from_logits([0.0, 2.0, 5.0], ["a", "b"]) is None  # more outputs than classes


def box(label, x, score=None):
    return Label(label, bbox=[x, 0.1, 0.3, 0.3], score=score)


@pytest.fixture
def detection(tmp_path):
    dataset = make_dataset(
        tmp_path,
        Task.DETECTION,
        [[box("car", 0.1)], [box("car", 0.1)], [box("car", 0.1), box("dog", 0.6)]],
    )
    yield dataset
    dataset.delete()


def test_detection_counts_the_missing_and_spurious_objects(detection):
    predictions = {
        "i0": [box("car", 0.1, 0.9), box("car", 0.6, 0.99)],  # a car the labels lack
        "i1": [box("car", 0.1, 0.9)],  # agrees
        "i2": [box("car", 0.1, 0.9)],  # the model does not see the dog label
    }
    result = mf.score(detection, Task.DETECTION, predictions)
    assert set(result) == {"i0", "i1", "i2"}
    assert result["i0"]["missing"] == 1 and result["i1"]["missing"] == 0
    assert result["i2"]["spurious"] == 1 and result["i1"]["spurious"] == 0
    assert all(r["hardness"] is None for r in result.values())


def test_nothing_predicted_or_unknown_images_give_no_scores(classification):
    assert mf.score(classification, Task.CLASSIFICATION, {}) == {}
    assert (
        mf.score(classification, Task.CLASSIFICATION, {"not-in-the-dataset": [Label("a", score=0.9)]}) == {}
    )
