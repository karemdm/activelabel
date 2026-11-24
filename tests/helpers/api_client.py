"""A test client for the API alone (no pages), on a temporary workspace."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from activelabel.config import Settings
from activelabel.web import (
    annotate_api,
    api,
    datasets_api,
    import_api,
    models_api,
    preannotate_api,
    sync_api,
    tasks_api,
    training_api,
    video_api,
)
from activelabel.web.errors import install_error_handlers
from activelabel.web.state import AppState


def make_client(tmp_path: Path, import_root: Path | None = None) -> tuple[TestClient, AppState]:
    state = AppState(Settings(workspace=tmp_path / "ws", import_root=import_root))
    state.runner.enqueue = lambda project_id, number: None  # type: ignore[method-assign]  # runs stay queued
    app = FastAPI()
    app.state.activelabel = state
    install_error_handlers(app)
    app.include_router(api.router)
    app.include_router(annotate_api.router)
    app.include_router(datasets_api.router)
    app.include_router(import_api.router)
    app.include_router(sync_api.router)
    app.include_router(tasks_api.router)
    app.include_router(training_api.router)
    app.include_router(models_api.router)
    app.include_router(preannotate_api.router)
    app.include_router(video_api.router)
    return TestClient(app), state


def upload(client: TestClient, project_id: int, paths: list[Path], names: list[str] | None = None):
    files = [
        ("files", (name, path.read_bytes()))
        for name, path in zip(names or [p.name for p in paths], paths, strict=True)
    ]
    return client.post(f"/api/projects/{project_id}/images", files=files)
