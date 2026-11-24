"""The annotation-tool API: settings of the tools and the helpers of the session routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_client import make_client
from fastapi.testclient import TestClient

from activelabel.labels import AnnotationSet, Label, annotations_lock, save_annotations
from activelabel.web import annotate_api
from activelabel.web.errors import ApiError


@pytest.fixture(autouse=True)
def _no_real_keyring(fake_keyring) -> None:
    """No test of this file may reach the real system keyring."""


def _client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)[0]


def test_unknown_tool_is_a_404_with_a_code(tmp_path: Path) -> None:
    response = _client(tmp_path).delete("/api/annotation-tools/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "tool_unknown"


def test_list_tools_has_every_tool_without_secrets(tmp_path: Path) -> None:
    tools = _client(tmp_path).get("/api/annotation-tools").json()
    assert {t["tool"] for t in tools} == set(annotate_api.TOOLS)
    assert all("secret" not in t for t in tools)


def test_saving_a_tool_needs_a_url(tmp_path: Path) -> None:
    response = _client(tmp_path).put("/api/annotation-tools/cvat", json={"url": "", "username": "me"})
    assert response.status_code == 400
    assert response.json()["code"] == "url_required"


def test_forget_tool_answers_removed(tmp_path: Path) -> None:
    assert _client(tmp_path).delete("/api/annotation-tools/fiftyone").json() == {"removed": True}


def test_unconfigured_tool_is_refused(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    try:
        annotate_api._settings(state, "cvat")
    except ApiError as error:
        assert (error.status_code, error.code) == (400, "tool_not_configured")
    else:
        raise AssertionError("an unconfigured tool must be refused")


def test_local_hashes_follow_the_labels(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    project = state.store.create("Cats", "", "classification", ["cat"])
    project_dir = state.store.dir(project.id)
    with annotations_lock(project_dir):
        save_annotations(project_dir, AnnotationSet(project.task, ["cat"], {"a": [Label("cat")]}, {}))
    first = annotate_api._local_hashes(project_dir, ["a", "b"])
    assert set(first) == {"a", "b"} and first["a"] != first["b"]
