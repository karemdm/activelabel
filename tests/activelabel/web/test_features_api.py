"""GET /api/features and DELETE /api/training/environment."""

from __future__ import annotations

from pathlib import Path

from api_client import make_client
from training_fakes import wait_for

from activelabel import optional
from activelabel.training.environment import TrainingEnvironment
from activelabel.web.features_api import folder_size


def _make_environment(workspace: Path) -> Path:
    path = TrainingEnvironment(workspace).path
    (path / "lib").mkdir(parents=True)
    (path / "lib" / "a.bin").write_bytes(b"x" * 10)
    (path / "b.bin").write_bytes(b"y" * 5)
    return path


def test_folder_size_adds_the_files_and_is_zero_for_a_missing_folder(tmp_path: Path) -> None:
    path = _make_environment(tmp_path)
    assert folder_size(path) == 15
    assert folder_size(tmp_path / "missing") == 0


def test_features_shape(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    data = client.get("/api/features").json()
    assert data["fiftyone"]["installed"] is optional.FIFTYONE
    assert data["fiftyone"]["install"] == optional.FIFTYONE_INSTALL
    assert data["fiftyone"]["provides"] == list(optional.FIFTYONE_PROVIDES)
    env = data["training_environment"]
    assert env["exists"] is False and env["bytes"] == 0 and env["packages"] == {}
    assert env["path"] == str(TrainingEnvironment(state.settings.workspace).path)


def test_features_without_fiftyone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(optional, "FIFTYONE", False)
    client, _ = make_client(tmp_path)
    assert client.get("/api/features").json()["fiftyone"]["installed"] is False


def test_removing_the_environment_needs_confirmation(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    path = _make_environment(state.settings.workspace)
    for body in ({}, {"confirm": False}):
        answer = client.request("DELETE", "/api/training/environment", json=body)
        assert answer.status_code == 400 and answer.json()["code"] == "confirm_required"
    assert path.exists()


def test_removing_the_environment_deletes_the_folder(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    path = _make_environment(state.settings.workspace)
    answer = client.request("DELETE", "/api/training/environment", json={"confirm": True})
    assert answer.status_code == 200
    assert not path.exists()
    assert answer.json()["training_environment"]["exists"] is False


def test_removing_the_environment_is_refused_while_a_job_uses_it(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    path = _make_environment(state.settings.workspace)
    gate: list[bool] = []
    job_id = state.runner.enqueue_tool(
        1, "training", lambda progress: wait_for(lambda: progress.check() or gate, 30)
    )
    try:
        answer = client.request("DELETE", "/api/training/environment", json={"confirm": True})
        assert answer.status_code == 409 and answer.json()["code"] == "busy_training"
        assert path.exists()
    finally:
        gate.append(True)
        wait_for(lambda: state.runner.tool_job(job_id).status == "done")
