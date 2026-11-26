import uuid

import fiftyone as fo
import pytest

from activelabel.curation import fiftyone_runs, similarity_fiftyone

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset(tmp_path):
    ds = fo.Dataset(f"test-runs-{uuid.uuid4().hex[:8]}")
    ds.add_samples([fo.Sample(filepath=str(tmp_path / f"{k}.png"), image_id=f"i{k}") for k in range(4)])
    yield ds
    ds.delete()


def publish_map(dataset):
    ids = ["i0", "i1", "i2", "i3"]
    points = {"ids": ids, "xy": [[k / 10, 1 - k / 10] for k in range(4)], "group": [0, 0, 1, 1]}
    similarity_fiftyone.publish(dataset, {"points": points})


def test_record_creates_a_custom_run_with_the_parameters(dataset):
    key = fiftyone_runs.record(dataset, "uniqueness", field="uniqueness", images=4)
    assert key == fiftyone_runs.RUN_KEYS["uniqueness"]
    assert dataset.list_runs() == [key]
    config = dataset.get_run_info(key).config.serialize()
    assert config["analysis"] == "uniqueness" and config["images"] == 4 and config["field"] == "uniqueness"
    assert config["recorded"]


def test_recording_again_replaces_the_run(dataset):
    fiftyone_runs.record(dataset, "clustering", groups=2)
    fiftyone_runs.record(dataset, "clustering", groups=5)
    key = fiftyone_runs.RUN_KEYS["clustering"]
    assert dataset.list_runs() == [key]
    assert dataset.get_run_info(key).config.serialize()["groups"] == 5


def test_listing_shows_custom_and_brain_runs(dataset):
    publish_map(dataset)
    fiftyone_runs.record(dataset, "mistakenness", model=1, images=4)
    runs = fiftyone_runs.listing(dataset)
    by_key = {run["key"]: run for run in runs}
    ours = by_key[fiftyone_runs.RUN_KEYS["mistakenness"]]
    assert ours["type"] == "run" and ours["ours"] is True and ours["timestamp"]
    assert ours["config"]["model"] == 1
    brain = by_key[similarity_fiftyone.BRAIN_KEY]
    assert brain["type"] == "brain" and brain["ours"] is False and brain["timestamp"]
    assert brain["config"]
    assert [run["type"] for run in runs] == ["brain", "run"]


def test_listing_an_empty_dataset(dataset):
    assert fiftyone_runs.listing(dataset) == []


def test_delete_removes_only_our_runs(dataset):
    publish_map(dataset)
    key = fiftyone_runs.record(dataset, "uniqueness", images=4)
    assert fiftyone_runs.delete(dataset, key) is True
    assert dataset.list_runs() == []
    assert fiftyone_runs.delete(dataset, key) is False  # already gone
    assert fiftyone_runs.delete(dataset, similarity_fiftyone.BRAIN_KEY) is False
    assert similarity_fiftyone.BRAIN_KEY in dataset.list_brain_runs()
    assert fiftyone_runs.delete(dataset, "unknown") is False
