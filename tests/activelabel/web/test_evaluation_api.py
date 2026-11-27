import pytest
from training_fakes import saved_model_project, wait_for

from activelabel import optional
from activelabel.curation import evaluation_job
from activelabel.labels import load_annotations, save_annotations

URL = "/api/projects/1/models/1/evaluation"
FINISHED = ("done", "failed", "cancelled")
RESULT = {
    "kind": "classification",
    "images": 2,
    "metrics": {"accuracy": 0.5, "macro_f1": 0.4},
    "classes": [],
    "confusion": {"classes": [], "other": False, "missing": False, "matrix": []},
    "worst": ["x"],
    "fields": [],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A project with 8 labelled images and a saved model m1 (fake workers; FiftyOne replaced)."""
    test_client, state, runner = saved_model_project(tmp_path)
    monkeypatch.setattr(evaluation_job.evaluation_fiftyone, "evaluate", lambda *args: dict(RESULT))
    monkeypatch.setattr(evaluation_job.fiftyone_bridge, "sync_dataset", lambda *args: "dataset")
    monkeypatch.setattr(evaluation_job.fiftyone_runs, "record", lambda *args, **kwargs: "key")
    yield test_client, state
    runner.shutdown()


def finished(test_client):
    return lambda: (s := test_client.get(URL).json()).get("status") in FINISHED and s


def code(response, status):
    assert response.status_code == status, response.text
    return response.json()["code"]


def test_nothing_evaluated_yet(client):
    test_client, _ = client
    assert test_client.get(URL).json() == {"active": None}
    assert code(test_client.get("/api/projects/1/models/9/evaluation"), 404) == "model_not_found"
    assert test_client.get("/api/projects/9/models/1/evaluation").status_code == 404
    assert test_client.post("/api/projects/9/models/1/evaluation").status_code == 404
    assert code(test_client.post("/api/projects/1/models/9/evaluation"), 404) == "model_not_found"
    assert code(test_client.post("/api/projects/1/models/9/evaluation/cancel"), 404) == "model_not_found"


def test_the_evaluation_runs_and_its_result_is_shown(client):
    test_client, _ = client
    queued = test_client.post(URL, json={"device": "cpu"}).json()
    assert queued["status"] == "queued" and queued["device"] == "cpu"
    assert (queued["version"], queued["part"], queued["images"]) == (1, "test", 2)
    done = wait_for(finished(test_client))
    assert done["status"] == "done" and done["result"]["worst"] == ["x"] and done["result"]["part"] == "test"
    assert test_client.get("/api/projects/1/models/2/evaluation").status_code == 404


def test_the_body_is_optional_and_can_choose_the_part(client):
    test_client, _ = client
    assert test_client.post(URL, json={"part": "val", "version": 1}).json()["part"] == "val"
    assert wait_for(finished(test_client))["status"] == "done"
    assert test_client.post(URL).json()["part"] == "test"  # no body: the defaults
    assert wait_for(lambda: (s := test_client.get(URL).json())["status"] == "done" and s["part"] == "test")


def test_a_part_or_version_that_does_not_fit_is_refused(client):
    test_client, _ = client
    assert code(test_client.post(URL, json={"part": "all"}), 400) == "part_unknown"
    assert code(test_client.post(URL, json={"fold": 1}), 400) == "version_has_no_folds"
    assert code(test_client.post(URL, json={"version": 7}), 404) == "dataset_not_found"
    assert test_client.get(URL).json() == {"active": None}


def test_a_device_this_computer_does_not_have_is_refused(client):
    test_client, _ = client
    assert test_client.post(URL, json={"device": "tpu"}).status_code == 400
    assert test_client.get(URL).json() == {"active": None}


def test_without_fiftyone_the_evaluation_is_not_offered(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(optional, "FIFTYONE", False)
    assert code(test_client.post(URL, json={}), 501) == "fiftyone_not_installed"


@pytest.mark.parametrize("task", ["keypoints", "regression"])
def test_a_project_that_fiftyone_cannot_evaluate_here_is_refused(client, task):
    test_client, state = client
    state.store.update(1, task=task)
    response = test_client.post(URL, json={})
    assert code(response, 400) == "evaluation_task_unsupported" and response.json()["params"] == {
        "task": task
    }


def test_a_model_that_does_not_fit_the_project_is_a_400(client):
    test_client, state = client
    state.store.update(1, task="detection")
    response = test_client.post(URL, json={})
    assert code(response, 400) == "evaluation_model_unusable"
    assert "the project is now" in response.json()["detail"]


def test_a_part_without_reviewed_images_is_a_conflict(client):
    test_client, state = client
    annotations = load_annotations(state.store.dir(1))
    annotations.images.clear()
    save_annotations(state.store.dir(1), annotations)
    response = test_client.post(URL, json={})
    assert code(response, 409) == "no_images_in_part" and response.json()["params"] == {
        "version": 1,
        "part": "test",
    }


def test_one_evaluation_at_a_time_and_cancel(client):
    test_client, state = client
    gate: list[bool] = []  # keeps the lane busy: the evaluation waits in the queue
    state.runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    assert test_client.post(URL, json={}).json()["status"] == "queued"
    again = test_client.post(URL, json={})
    assert code(again, 409) == "evaluation_running" and again.json()["params"] == {"model": 1}
    cancelled = test_client.post(f"{URL}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    gate.append(True)
    assert test_client.get(URL).json()["status"] == "cancelled"
    assert test_client.post(URL, json={}).status_code == 200  # a new one may start
    assert wait_for(finished(test_client))["status"] == "done"
