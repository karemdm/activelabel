"""The pre-annotation API on a project with no run yet (running them is tested with the training code)."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_client import make_client
from fastapi.testclient import TestClient

from activelabel.preannotate import PreannotationStore
from activelabel.training.environment import TrainingEnvironment
from activelabel.web import preannotate_api


@pytest.fixture
def env(tmp_path: Path) -> tuple[TestClient, int]:
    client, state = make_client(tmp_path)
    return client, state.store.create("Cats", "", "classification", ["cat"]).id


def test_overview_of_a_new_project_is_empty(env: tuple[TestClient, int]) -> None:
    client, pid = env
    answer = client.get(f"/api/projects/{pid}/preannotations").json()
    assert answer["models"] == [] and answer["runs"] == []
    assert not answer["tasks"] or isinstance(answer["tasks"], (list, dict))


def test_cancelling_a_missing_run_is_a_404(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.post(f"/api/projects/{pid}/preannotations/5/cancel")
    assert (response.status_code, response.json()["code"]) == (404, "preannotation_not_found")


def test_starting_with_a_model_that_does_not_exist_is_refused(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.post(f"/api/projects/{pid}/preannotations", json={"task": 1, "model": 9})
    assert response.status_code in (400, 404, 422), response.text


def test_accept_on_a_project_without_pre_annotations_accepts_nothing(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.post(f"/api/projects/{pid}/preannotations/accept", json={})
    assert response.status_code == 200, response.text
    assert response.json()["accepted"] == 0


class FakeModels:
    """A model store that has model 1."""

    def __init__(self, project_dir: Path) -> None:
        pass

    def get(self, number: int) -> dict:
        return {"number": number, "model": "resnet18", "task_kind": "classification"}


@pytest.fixture
def ready(env: tuple[TestClient, int], monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, int]:
    """A project that can start a pre-annotation with model 1 (the model and the images are faked)."""
    client, pid = env
    model = {"number": 1, "missing": False, "supported": True, "label": "ResNet"}
    monkeypatch.setattr(preannotate_api, "ModelStore", FakeModels)
    monkeypatch.setattr(preannotate_api, "_models", lambda state, project_id: [model])
    monkeypatch.setattr(preannotate_api.preannotate, "targets", lambda project_dir, task, scope: ["x"])
    monkeypatch.setattr(TrainingEnvironment, "gpu_count", lambda self: 1)
    monkeypatch.setattr(TrainingEnvironment, "mps_hint", staticmethod(lambda: False))
    return client, pid


def _start(client: TestClient, pid: int, **extra):
    return client.post(f"/api/projects/{pid}/preannotations", json={"task": 1, "model": 1, **extra})


@pytest.mark.parametrize(
    ("sent", "stored"), [({}, "auto"), ({"device": "cpu"}, "cpu"), ({"device": "cuda:0"}, "cuda:0")]
)
def test_start_stores_the_device(ready: tuple[TestClient, int], sent: dict, stored: str) -> None:
    client, pid = ready
    response = _start(client, pid, **sent)
    assert response.status_code == 200, response.text
    assert response.json()["device"] == stored
    assert client.get(f"/api/projects/{pid}/preannotations").json()["runs"][0]["device"] == stored


def test_start_refuses_a_device_this_computer_lacks(ready: tuple[TestClient, int], tmp_path: Path) -> None:
    client, pid = ready
    response = _start(client, pid, device="cuda:7")
    assert (response.status_code, response.json()["code"]) == (400, "device_unavailable"), response.text
    response = _start(client, pid, device="mps")
    assert (response.status_code, response.json()["code"]) == (400, "device_unavailable"), response.text
    assert client.get(f"/api/projects/{pid}/preannotations").json()["runs"] == []


def test_start_refuses_a_device_that_is_not_a_device(ready: tuple[TestClient, int]) -> None:
    client, pid = ready
    response = _start(client, pid, device="bogus")
    assert (response.status_code, response.json()["code"]) == (400, "device_invalid"), response.text
    assert client.get(f"/api/projects/{pid}/preannotations").json()["runs"] == []


def test_the_run_store_keeps_a_device_given_at_creation(env: tuple[TestClient, int], tmp_path: Path) -> None:
    _, pid = env
    run = PreannotationStore(tmp_path / "x").create(model=1, task=1, device="cpu")
    assert run["device"] == "cpu"


class SuggestingModels:
    """A model store with one detector whose validation suggested a confidence (#323)."""

    def __init__(self, project_dir: Path) -> None:
        pass

    def list(self) -> list[dict]:
        card = {"number": 1, "model": "fasterrcnn_resnet50_fpn", "task_kind": "detection", "classes": ["cat"]}
        return [{**card, "suggested_threshold": 0.4234}]


def test_a_model_offers_its_suggested_confidence_as_the_default(
    env: tuple[TestClient, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, pid = env
    monkeypatch.setattr(preannotate_api, "ModelStore", SuggestingModels)
    model = preannotate_api._models(client.app.state.activelabel, pid)[0]
    confidence = next(o for o in model["options"] if o["name"] == "confidence")
    assert model["suggested_threshold"] == 0.423
    assert (confidence["default"], confidence["suggested"]) == (0.423, True)


def test_start_uses_the_suggested_confidence_unless_one_is_sent(
    ready: tuple[TestClient, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, pid = ready
    model = {"number": 1, "missing": False, "supported": True, "label": "R", "suggested_threshold": 0.42}
    monkeypatch.setattr(preannotate_api, "_models", lambda state, project_id: [model])
    assert _start(client, pid).json()["options"]["confidence"] == 0.42
    sent = _start(client, pid, scope="unreviewed", options={"confidence": 0.7})
    assert sent.json()["options"]["confidence"] == 0.7, sent.text
