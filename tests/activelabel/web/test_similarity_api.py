import uuid

import fiftyone as fo
import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FAKE_EMBED, FakeEnv, wait_for

from activelabel import fiftyone_bridge
from activelabel.curation import similarity_fiftyone, similarity_job
from activelabel.training.runner import TrainingRunner

URL = "/api/projects/1/curation/similarity"
FINISHED = ("done", "failed", "cancelled")


@pytest.fixture
def client(tmp_path):
    """A project with 6 images, served with a runner whose embeddings worker is the fake one (no torch)."""
    test_client, state = make_client(tmp_path)
    test_client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    (tmp_path / "embed.py").write_text(FAKE_EMBED)
    state.runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: tmp_path / "train.py",
        poll=0.05,
    )
    yield test_client, tmp_path
    state.runner.shutdown()


def add_images(test_client, tmp_path, count=6):
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(count)]
    assert upload(test_client, 1, paths).status_code == 200


def finished(test_client):
    return lambda: (s := test_client.get(URL).json()).get("status") in FINISHED and s


def test_no_map_yet_and_no_images(client):
    test_client, _ = client
    assert test_client.get(URL).json() == {}
    response = test_client.post(URL, json={})
    assert response.status_code == 400 and response.json()["code"] == "no_images"
    assert test_client.get("/api/projects/9/curation/similarity").status_code == 404


def test_one_computation_at_a_time_and_cancel(client):
    test_client, tmp_path = client
    add_images(test_client, tmp_path)
    assert test_client.post(URL, json={"device": "slow"}).json()["status"] == "queued"
    second = test_client.post(URL, json={})
    assert second.status_code == 409 and second.json()["code"] == "similarity_running"
    wait_for(lambda: test_client.get(URL).json().get("phase") == "embedding")
    test_client.post(f"{URL}/cancel")
    assert wait_for(finished(test_client))["status"] == "cancelled"


@pytest.mark.integration
def test_the_map_reaches_the_interface_and_fiftyone(client, monkeypatch):
    test_client, tmp_path = client
    name = f"test-similarity-api-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    add_images(test_client, tmp_path)
    test_client.post(URL)
    try:
        done = wait_for(finished(test_client), timeout=120)
        assert done["status"] == "done" and done["fiftyone_error"] == "", done
        assert sum(g["size"] for g in done["groups"]) == 6 and done["outdated"] is False
        dataset = fo.load_dataset(name)
        assert similarity_fiftyone.BRAIN_KEY in dataset.list_brain_runs()
        assert set(dataset.values(similarity_fiftyone.GROUP_FIELD)) <= {1, 2, 3}
        extra = make_image(tmp_path / "more" / "new.png", seed=99, fmt="PNG")
        assert upload(test_client, 1, [extra]).status_code == 200  # a new image: the map is outdated
        assert test_client.get(URL).json()["outdated"] is True
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


CLUSTERING = f"{URL}/clustering"


def wait_map(test_client):
    """Compute the map with the fake worker and wait for it."""
    assert test_client.post(URL, json={}).status_code == 200
    assert wait_for(finished(test_client), timeout=60)["status"] == "done"


def regrouped(test_client):
    return lambda: (s := test_client.get(CLUSTERING).json()).get("status") in FINISHED and s


@pytest.fixture
def mapped(client, monkeypatch):
    """The client with 9 images and a finished map (FiftyOne is not touched)."""
    test_client, tmp_path = client
    monkeypatch.setattr(similarity_job, "_publish", lambda *args: "")
    add_images(test_client, tmp_path, 9)
    wait_map(test_client)
    return test_client, tmp_path


def test_clustering_without_a_map(client):
    test_client, _ = client
    assert test_client.get(CLUSTERING).json() == {}
    response = test_client.post(CLUSTERING, json={"method": "kmeans", "groups": 3})
    assert response.status_code == 409 and response.json()["code"] == "similarity_map_needed"
    assert test_client.get("/api/projects/9/curation/similarity/clustering").status_code == 404
    assert test_client.post("/api/projects/9/curation/similarity/clustering", json={}).status_code == 404
    assert test_client.post("/api/projects/9/curation/similarity/clustering/cancel").status_code == 404


def test_the_map_is_regrouped(mapped):
    test_client, _ = mapped
    queued = test_client.post(CLUSTERING, json={"method": "kmeans", "groups": 2})
    assert queued.status_code == 200 and queued.json()["status"] == "queued"
    done = wait_for(regrouped(test_client))
    assert done["status"] == "done" and done["method"] == "kmeans", done.get("error")
    shown = test_client.get(URL).json()
    assert len(shown["groups"]) == 2 and sum(g["size"] for g in shown["groups"]) == 9
    assert shown["clustering"]["status"] == "done"


@pytest.mark.parametrize(
    "body",
    [{"method": "dbscan"}, {"method": "kmeans", "groups": 1}, {"method": "hdbscan", "min_size": 1}],
)
def test_an_invalid_regrouping_is_a_bad_request(mapped, body):
    test_client, _ = mapped
    response = test_client.post(CLUSTERING, json=body)
    assert response.status_code == 400 and response.json()["code"] == "cluster_invalid"
    assert response.json()["params"]["reason"]
    assert test_client.get(CLUSTERING).json() == {}


def test_one_regrouping_at_a_time_and_cancel(mapped):
    test_client, _ = mapped
    runner = test_client.app.state.activelabel.runner
    gate: list[bool] = []
    runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    assert test_client.post(CLUSTERING, json={"method": "kmeans", "groups": 2}).json()["status"] == "queued"
    second = test_client.post(CLUSTERING, json={"method": "kmeans", "groups": 3})
    assert second.status_code == 409 and second.json()["code"] == "clustering_running"
    new_map = test_client.post(URL, json={})  # a new map waits for the regrouping
    assert new_map.status_code == 409 and new_map.json()["code"] == "clustering_running"
    assert test_client.post(f"{CLUSTERING}/cancel").json()["status"] == "cancelled"
    gate.append(True)
    assert test_client.get(CLUSTERING).json()["status"] == "cancelled"
    assert test_client.post(URL, json={}).status_code == 200  # nothing regroups any more


def test_no_regrouping_while_the_map_is_computed(client):
    test_client, tmp_path = client
    add_images(test_client, tmp_path)
    assert test_client.post(URL, json={"device": "slow"}).json()["status"] == "queued"
    response = test_client.post(CLUSTERING, json={"method": "kmeans", "groups": 2})
    assert response.status_code == 409 and response.json()["code"] == "similarity_running"
    test_client.post(f"{URL}/cancel")
    assert wait_for(finished(test_client))["status"] == "cancelled"
