"""The pre-annotation API on a project with no run yet (running them is tested with the training code)."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_client import make_client
from fastapi.testclient import TestClient


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
