import uuid

import fiftyone as fo
import pytest
from api_client import make_client

from activelabel import fiftyone_bridge, optional
from activelabel.curation import fiftyone_runs, similarity_fiftyone

pytestmark = pytest.mark.integration

URL = "/api/projects/1/curation/runs"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A project with no FiftyOne dataset yet; the dataset gets an isolated name."""
    test_client, state = make_client(tmp_path)
    test_client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    name = f"test-runs-api-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    yield test_client, name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


def make_dataset(name):
    dataset = fo.Dataset(name)
    dataset.add_sample(fo.Sample(filepath="a.png", image_id="i0"))
    return dataset


def test_no_runs_before_fiftyone_was_used(client):
    test_client, _ = client
    assert test_client.get(URL).json() == {"runs": []}


def test_the_recorded_runs_are_listed(client):
    test_client, name = client
    fiftyone_runs.record(make_dataset(name), "uniqueness", images=1)
    runs = test_client.get(URL).json()["runs"]
    assert [(r["type"], r["key"], r["ours"]) for r in runs] == [("run", "activelabel_uniqueness", True)]
    assert runs[0]["config"]["images"] == 1 and runs[0]["timestamp"]


def test_one_of_our_runs_can_be_deleted(client):
    test_client, name = client
    dataset = make_dataset(name)
    fiftyone_runs.record(dataset, "uniqueness", images=1)
    fiftyone_runs.record(dataset, "clustering", groups=2)
    response = test_client.delete(f"{URL}/activelabel_uniqueness")
    assert response.status_code == 200
    assert [r["key"] for r in response.json()["runs"]] == ["activelabel_clustering"]
    assert [r["key"] for r in test_client.get(URL).json()["runs"]] == ["activelabel_clustering"]


def test_a_brain_or_unknown_run_is_not_found(client):
    test_client, name = client
    dataset = make_dataset(name)
    points = {"ids": ["i0"], "xy": [[0.1, 0.2]], "group": [0]}
    similarity_fiftyone.publish(dataset, {"points": points})
    for key in (similarity_fiftyone.BRAIN_KEY, "nothing"):
        response = test_client.delete(f"{URL}/{key}")
        assert response.status_code == 404 and response.json()["code"] == "run_not_found"
    assert similarity_fiftyone.BRAIN_KEY in dataset.list_brain_runs()


def test_without_a_dataset_nothing_is_found(client):
    test_client, _ = client
    assert test_client.delete(f"{URL}/activelabel_uniqueness").json()["code"] == "run_not_found"


def test_an_unknown_project(client):
    test_client, _ = client
    assert test_client.get("/api/projects/9/curation/runs").status_code == 404
    assert test_client.delete("/api/projects/9/curation/runs/activelabel_uniqueness").status_code == 404


def test_without_fiftyone_there_are_no_runs(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(optional, "FIFTYONE", False)
    assert test_client.get("/api/projects/1/curation/runs").json() == {"runs": []}
    response = test_client.delete("/api/projects/1/curation/runs/activelabel_uniqueness")
    assert response.status_code == 501 and response.json()["code"] == "fiftyone_not_installed"
