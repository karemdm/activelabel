"""What a regression project cannot do yet is refused with one coded error (#619)."""

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.sync.engine import SyncError, make_plan

CODE = "regression_not_supported_yet"


@pytest.fixture
def regression(tmp_path):
    client, state = make_client(tmp_path)
    made = client.post("/api/projects", json={"name": "R", "task": "regression", "classes": ["age"]})
    assert made.status_code == 200, made.text
    assert upload(client, 1, [make_image(tmp_path / "src" / "a.png", seed=1, fmt="PNG")]).status_code == 200
    return client, state


def _refused(response, code=CODE):
    assert response.status_code == 400, response.text
    assert response.json()["code"] == code


def test_a_regression_project_is_created_with_its_targets(regression):
    client, _ = regression
    body = client.get("/api/projects/1").json()
    assert body["task"] == "regression" and body["classes"] == ["age"]


def test_data_split_needs_values(regression):
    client, _ = regression
    answer = client.post("/api/projects/1/datasets", json={})
    assert answer.status_code == 400 and answer.json()["code"] == "nothing_to_split"  # no value yet


def test_training_is_refused(regression):
    client, _ = regression
    _refused(client.post("/api/projects/1/runs", json={"dataset": 1, "model": "resnet18"}))


def test_pre_annotation_is_refused(regression):
    client, _ = regression
    _refused(client.post("/api/projects/1/preannotations", json={"task": 1, "model": 1}))


def test_the_catalogues_answer_without_models(regression):
    client, _ = regression
    assert client.get("/api/training/catalog", params={"type": "regression"}).json()["models"] == []
    assert client.get("/api/training/augmentations", params={"type": "regression"}).json()["kinds"] == {}


def test_the_cvat_synchronisation_is_refused(regression):
    _, state = regression
    with pytest.raises(SyncError) as caught:
        make_plan(state.store, 1, "cvat", None, None, None)  # type: ignore[arg-type]
    assert getattr(caught.value, "code", "") == "cvat_not_for_regression"


def test_importing_from_cvat_as_a_regression_project_is_refused(regression):
    client, _ = regression
    body = {"tool": "cvat", "kind": "project", "remote_id": 1, "name": "x", "task": "regression"}
    _refused(client.post("/api/import", json=body), "cvat_not_for_regression")
