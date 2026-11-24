"""The saved-models API on a project with no model yet (the full flow is tested with the training runs)."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_client import make_client
from fastapi.testclient import TestClient


@pytest.fixture
def env(tmp_path: Path) -> tuple[TestClient, int]:
    client, state = make_client(tmp_path)
    return client, state.store.create("Cats", "", "classification", ["cat"]).id


def test_a_new_project_has_no_models(env: tuple[TestClient, int]) -> None:
    client, pid = env
    assert client.get(f"/api/projects/{pid}/models").json() == {"models": [], "current": None}


def test_unknown_model_is_a_404_with_a_code(env: tuple[TestClient, int]) -> None:
    client, pid = env
    for response in (
        client.get(f"/api/projects/{pid}/models/7"),
        client.patch(f"/api/projects/{pid}/models/7", json={"note": "x"}),
        client.post(f"/api/projects/{pid}/models/7/export", json={}),
        client.get(f"/api/projects/{pid}/models/7/download/pytorch"),
    ):
        assert response.status_code == 404, response.text
    assert client.get(f"/api/projects/{pid}/models/7").json()["code"] == "model_not_found"


def test_delete_needs_confirmation(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.request("DELETE", f"/api/projects/{pid}/models/1", json={})
    assert (response.status_code, response.json()["code"]) == (400, "confirm_required")


def test_saving_the_weights_of_a_missing_run_is_a_404(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.post(f"/api/projects/{pid}/models", json={"run": 3, "note": ""})
    assert (response.status_code, response.json()["code"]) == (404, "run_not_found")


def test_public_catalog_lists_models_with_the_project_classes(env: tuple[TestClient, int]) -> None:
    client, pid = env
    answer = client.get(f"/api/projects/{pid}/models/public-catalog").json()
    assert answer["project_classes"] == ["cat"]
    assert isinstance(answer["models"], list)


def test_unknown_public_model_is_a_404(env: tuple[TestClient, int]) -> None:
    client, pid = env
    response = client.post(f"/api/projects/{pid}/models/public", json={"public": "nope", "mapping": {}})
    assert (response.status_code, response.json()["code"]) == (404, "public_model_not_found")
