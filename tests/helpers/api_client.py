"""A test client for the API alone (no pages), on a temporary workspace."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from visionloop.config import Settings
from visionloop.web import api
from visionloop.web.state import AppState


def make_client(tmp_path: Path, import_root: Path | None = None) -> tuple[TestClient, AppState]:
    state = AppState(Settings(workspace=tmp_path / "ws", import_root=import_root))
    app = FastAPI()
    app.state.visionloop = state
    app.include_router(api.router)
    return TestClient(app), state


def upload(client: TestClient, project_id: int, paths: list[Path], names: list[str] | None = None):
    files = [
        ("files", (name, path.read_bytes()))
        for name, path in zip(names or [p.name for p in paths], paths, strict=True)
    ]
    return client.post(f"/api/projects/{project_id}/images", files=files)
