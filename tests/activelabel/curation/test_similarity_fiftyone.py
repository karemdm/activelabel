import uuid

import fiftyone as fo
import pytest

from activelabel.curation import similarity_fiftyone as sf

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset(tmp_path):
    ds = fo.Dataset(f"test-similarity-{uuid.uuid4().hex[:8]}")
    ds.add_samples([fo.Sample(filepath=str(tmp_path / f"{k}.png"), image_id=f"i{k}") for k in range(4)])
    yield ds
    ds.delete()


def result(ids, groups):
    return {"points": {"ids": ids, "xy": [[k / 10, 1 - k / 10] for k in range(len(ids))], "group": groups}}


def test_publish_writes_the_map_and_the_groups(dataset):
    placed = sf.publish(dataset, result(["i0", "i1", "i2", "gone"], [0, 1, 1, 0]))
    assert placed == 3  # an image no longer in the dataset is left out
    assert sf.BRAIN_KEY in dataset.list_brain_runs()
    by_image = dict(zip(dataset.values("image_id"), dataset.values(sf.GROUP_FIELD), strict=True))
    assert by_image == {"i0": 1, "i1": 2, "i2": 2, "i3": None}  # groups are numbered from 1 in the App
    results = dataset.load_brain_results(sf.BRAIN_KEY)
    placed_at = dict(zip(results.current_sample_ids, results.current_points.tolist(), strict=True))
    ids = dict(zip(dataset.values("image_id"), dataset.values("id"), strict=True))
    assert placed_at[ids["i2"]] == pytest.approx([0.2, 0.8])  # each point stays with its image


def test_publishing_again_replaces_the_previous_map(dataset):
    sf.publish(dataset, result(["i0", "i1"], [0, 0]))
    sf.publish(dataset, result(["i2", "i3"], [1, 0]))
    assert dataset.list_brain_runs() == [sf.BRAIN_KEY]
    by_image = dict(zip(dataset.values("image_id"), dataset.values(sf.GROUP_FIELD), strict=True))
    assert by_image == {"i0": None, "i1": None, "i2": 2, "i3": 1}


def test_nothing_to_place_leaves_no_map(dataset):
    assert sf.publish(dataset, result(["gone"], [0])) == 0
    assert dataset.list_brain_runs() == []


def test_noise_has_no_group(dataset):
    sf.publish(dataset, result(["i0", "i1", "i2", "i3"], [0, -1, 1, -1]))
    by_image = dict(zip(dataset.values("image_id"), dataset.values(sf.GROUP_FIELD), strict=True))
    assert by_image == {"i0": 1, "i1": None, "i2": 2, "i3": None}
