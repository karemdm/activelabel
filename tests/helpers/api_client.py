"""A test client for the API alone (no pages), on a temporary workspace."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from activelabel.config import Settings
from activelabel.web import (
    annotate_api,
    api,
    backups_api,
    datasets_api,
    features_api,
    fiftyone_runs_api,
    import_api,
    mistakenness_api,
    models_api,
    preannotate_api,
    queue_api,
    similarity_api,
    sync_api,
    tasks_api,
    training_api,
    uniqueness_api,
    values_api,
    video_api,
)
from activelabel.web.errors import install_error_handlers
from activelabel.web.state import AppState


def make_client(tmp_path: Path, import_root: Path | None = None) -> tuple[TestClient, AppState]:
    state = AppState(Settings(workspace=tmp_path / "ws", import_root=import_root))
    # runs and pre-annotations stay queued: a test that posts one must never start a real environment
    # install or worker (a test that wants the real thing builds its own runner, as the training tests do)
    state.runner.enqueue = lambda project_id, number: None  # type: ignore[method-assign]
    state.runner.enqueue_preannotation = lambda project_id, number: None  # type: ignore[method-assign]
    app = FastAPI()
    app.state.activelabel = state
    install_error_handlers(app)
    app.include_router(api.router)
    app.include_router(annotate_api.router)
    app.include_router(backups_api.router)
    app.include_router(datasets_api.router)
    app.include_router(import_api.router)
    app.include_router(sync_api.router)
    app.include_router(tasks_api.router)
    app.include_router(training_api.router)
    app.include_router(models_api.router)
    app.include_router(preannotate_api.router)
    app.include_router(similarity_api.router)
    app.include_router(mistakenness_api.router)
    app.include_router(fiftyone_runs_api.router)
    app.include_router(uniqueness_api.router)
    app.include_router(values_api.router)
    app.include_router(video_api.router)
    app.include_router(queue_api.router)
    app.include_router(features_api.router)
    return TestClient(app), state


def upload(client: TestClient, project_id: int, paths: list[Path], names: list[str] | None = None):
    files = [
        ("files", (name, path.read_bytes()))
        for name, path in zip(names or [p.name for p in paths], paths, strict=True)
    ]
    return client.post(f"/api/projects/{project_id}/images", files=files)
