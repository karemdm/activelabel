"""The shared state: lookups that answer with readable HTTP errors."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from activelabel.config import Settings
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState


@pytest.fixture
def state(tmp_path: Path) -> AppState:
    return AppState(Settings(workspace=tmp_path / "ws"))


def test_missing_project_is_a_404_with_a_code(state: AppState) -> None:
    with pytest.raises(ApiError) as caught:
        state.project(99)
    assert (caught.value.status_code, caught.value.code) == (404, "project_not_found")


def test_manifest_with_images_refuses_an_empty_project(state: AppState) -> None:
    project = state.store.create("Cats", "", "classification", ["cat"])
    with pytest.raises(ApiError) as caught:
        state.manifest_with_images(project.id)
    assert (caught.value.status_code, caught.value.code) == (400, "no_images")


def test_busy_is_quiet_for_an_idle_project(state: AppState) -> None:
    project = state.store.create("Cats", "", "classification", ["cat"])
    state.busy(project.id)
    state.busy(project.id, training=True)


def test_busy_refuses_while_a_job_runs(state: AppState) -> None:
    project = state.store.create("Cats", "", "classification", ["cat"])
    release = threading.Event()
    state.jobs.submit(project.id, "analyze", lambda progress: release.wait(5) and {})
    try:
        with pytest.raises(ApiError) as caught:
            state.busy(project.id)
        assert (caught.value.status_code, caught.value.code) == (409, "busy")
    finally:
        release.set()
        state.jobs.shutdown()
