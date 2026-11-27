import uuid

import fiftyone as fo
import pytest

from activelabel.curation import fiftyone_workspace as ws
from activelabel.curation import similarity_fiftyone as sf

pytestmark = pytest.mark.integration


def refuse(*args, **kwargs):
    raise RuntimeError("no database")


@pytest.fixture
def dataset(tmp_path):
    ds = fo.Dataset(f"test-workspace-{uuid.uuid4().hex[:8]}")
    ds.add_samples([fo.Sample(filepath=str(tmp_path / f"{k}.png"), image_id=f"i{k}") for k in range(3)])
    yield ds
    ds.delete()


def test_saves_the_grid_and_the_histograms(dataset):
    assert ws.save_workspace(dataset) is True
    assert dataset.list_workspaces() == [ws.WORKSPACE]
    assert ws._flat(dataset) == ["Samples", "Histograms"]


def test_saving_again_changes_nothing(dataset):
    ws.save_workspace(dataset)
    assert ws.save_workspace(dataset) is False
    assert dataset.list_workspaces() == [ws.WORKSPACE]


def test_the_embeddings_panel_comes_with_the_similarity_map(dataset):
    ws.save_workspace(dataset)
    ids = ["i0", "i1", "i2"]
    sf.publish(dataset, {"points": {"ids": ids, "xy": [[0, 0], [1, 1], [2, 0]], "group": [0, 0, 1]}})
    assert ws.save_workspace(dataset) is True
    assert ws._flat(dataset) == ["Samples", "Histograms", "Embeddings"]
    embeddings = dataset.load_workspace(ws.WORKSPACE).children[1].children[1]
    assert embeddings.state == {"brainResult": sf.BRAIN_KEY}
    assert ws.save_workspace(dataset) is False


def test_a_failure_is_only_logged(dataset, monkeypatch):
    monkeypatch.setattr(type(dataset), "save_workspace", refuse)
    assert ws.save_workspace(dataset) is False
