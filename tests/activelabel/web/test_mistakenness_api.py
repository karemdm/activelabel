import pytest
from training_fakes import saved_model_project, wait_for

from activelabel import optional
from activelabel.curation import mistakenness_job
from activelabel.labels import load_annotations, save_annotations

URL = "/api/projects/1/curation/mistakenness"
FINISHED = ("done", "failed", "cancelled")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A project with 8 labelled images and a saved model m1 (fake workers; FiftyOne replaced)."""
    test_client, state, runner = saved_model_project(tmp_path)
    scores = lambda dataset, task, predictions, logits, classes: {  # noqa: E731
        i: {"mistakenness": 0.5, "missing": 0, "spurious": 0, "hardness": 0.25} for i in predictions
    }
    monkeypatch.setattr(mistakenness_job.mistakenness_fiftyone, "score", scores)
    monkeypatch.setattr(mistakenness_job.fiftyone_bridge, "sync_dataset", lambda *args: "dataset")
    yield test_client, state
    runner.shutdown()


def finished(test_client):
    return lambda: (s := test_client.get(URL).json()).get("status") in FINISHED and s


def code(response, status):
    assert response.status_code == status, response.text
    return response.json()["code"]


def test_nothing_checked_yet(client):
    test_client, _ = client
    assert test_client.get(URL).json() == {}
    assert test_client.get("/api/projects/9/curation/mistakenness").status_code == 404
    assert test_client.post("/api/projects/9/curation/mistakenness", json={"model": 1}).status_code == 404
    assert test_client.post("/api/projects/9/curation/mistakenness/cancel").status_code == 404


def test_the_check_runs_and_its_scores_are_shown(client):
    test_client, _ = client
    queued = test_client.post(URL, json={"model": 1, "device": "cpu"}).json()
    assert queued["status"] == "queued" and queued["device"] == "cpu" and queued["model"] == 1
    done = wait_for(finished(test_client))
    assert done["status"] == "done" and len(done["scores"]) == 8 and done["checked"] == 8


def test_the_body_needs_a_model(client):
    test_client, _ = client
    assert test_client.post(URL, json={}).status_code == 422


def test_a_device_this_computer_does_not_have_is_refused(client):
    test_client, _ = client
    response = test_client.post(URL, json={"model": 1, "device": "tpu"})
    assert response.status_code == 400, response.text
    assert test_client.get(URL).json() == {}


def test_without_fiftyone_the_check_is_not_offered(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(optional, "FIFTYONE", False)
    assert code(test_client.post(URL, json={"model": 1}), 501) == "fiftyone_not_installed"


def test_a_model_that_does_not_exist_is_a_404(client):
    test_client, _ = client
    assert code(test_client.post(URL, json={"model": 9}), 404) == "model_not_found"


def test_a_model_that_does_not_fit_the_project_is_a_400(client):
    test_client, state = client
    state.store.update(1, task="detection")
    response = test_client.post(URL, json={"model": 1})
    assert code(response, 400) == "mistakenness_model_unusable"
    assert "the project is now" in response.json()["detail"]


def test_without_labelled_images_it_is_a_conflict(client):
    test_client, state = client
    annotations = load_annotations(state.store.dir(1))
    annotations.images.clear()
    save_annotations(state.store.dir(1), annotations)
    assert code(test_client.post(URL, json={"model": 1}), 409) == "no_labelled_images"


def test_one_check_at_a_time_and_cancel(client):
    test_client, state = client
    gate: list[bool] = []  # keeps the lane busy: the check waits in the queue
    state.runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    assert test_client.post(URL, json={"model": 1}).json()["status"] == "queued"
    assert code(test_client.post(URL, json={"model": 1}), 409) == "mistakenness_running"
    cancelled = test_client.post(f"{URL}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    gate.append(True)
    assert test_client.get(URL).json()["status"] == "cancelled"
    assert test_client.post(URL, json={"model": 1}).status_code == 200  # a new one may start
    assert wait_for(finished(test_client))["status"] == "done"
