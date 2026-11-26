import numpy as np
import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FakeEnv, wait_for

from activelabel.curation import similarity, uniqueness_job
from activelabel.training.runner import TrainingRunner

URL = "/api/projects/1/curation/uniqueness"
FINISHED = ("done", "failed", "cancelled")


def slow_scores(vectors, check):
    """Scores that never end by themselves: only a cancel stops them."""
    while True:
        check(0, len(vectors))
        wait_for(lambda: True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A project with 6 images served with a runner that has no worker (the job runs in its thread)."""
    test_client, state = make_client(tmp_path)
    test_client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(6)]
    assert upload(test_client, 1, paths).status_code == 200
    state.runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: tmp_path / "train.py",
        poll=0.05,
    )
    monkeypatch.setattr(uniqueness_job, "_publish", lambda *args: "")
    yield test_client, state, tmp_path
    state.runner.shutdown()


def save_vectors(state):
    ids = [r.id for r in state.manifest(1).active()]
    vectors = np.random.default_rng(1).normal(size=(len(ids), 5))
    similarity.SimilarityStore(state.store.dir(1)).save_vectors(ids, vectors, "fake")


def finished(test_client):
    return lambda: (s := test_client.get(URL).json()).get("status") in FINISHED and s


def test_nothing_computed_yet(client):
    test_client, _, _ = client
    assert test_client.get(URL).json() == {}
    assert test_client.get("/api/projects/9/curation/uniqueness").status_code == 404
    assert test_client.post("/api/projects/9/curation/uniqueness").status_code == 404


def test_without_the_similarity_map_it_is_a_conflict(client):
    test_client, _, _ = client
    response = test_client.post(URL)
    assert response.status_code == 409 and response.json()["code"] == "similarity_map_needed"


def test_the_scores_are_computed_and_shown(client):
    test_client, state, _ = client
    save_vectors(state)
    assert test_client.post(URL).json()["status"] == "queued"
    done = wait_for(finished(test_client))
    assert done["status"] == "done" and len(done["scores"]) == 6 and done["outdated"] is False


def test_one_computation_at_a_time_and_cancel(client, monkeypatch):
    test_client, state, _ = client
    save_vectors(state)
    monkeypatch.setattr(uniqueness_job.uniqueness, "scores", slow_scores)
    assert test_client.post(URL).json()["status"] == "queued"
    second = test_client.post(URL)
    assert second.status_code == 409 and second.json()["code"] == "uniqueness_running"
    wait_for(lambda: test_client.get(URL).json().get("status") == "running")
    assert test_client.post(f"{URL}/cancel").status_code == 200
    assert wait_for(finished(test_client))["status"] == "cancelled"
    assert test_client.post("/api/projects/9/curation/uniqueness/cancel").status_code == 404


def test_new_images_make_the_scores_outdated(client):
    test_client, state, tmp_path = client
    save_vectors(state)
    test_client.post(URL)
    assert wait_for(finished(test_client))["outdated"] is False
    extra = make_image(tmp_path / "more" / "new.png", seed=99, fmt="PNG")
    assert upload(test_client, 1, [extra]).status_code == 200
    assert test_client.get(URL).json()["outdated"] is True
