"""The application factory: routers, static files and the server life cycle."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi.testclient import TestClient

from activelabel.config import Settings
from activelabel.web.app import create_app
from activelabel.web.state import AppState


def test_create_app_serves_pages_api_and_static_files(tmp_path: Path) -> None:
    app = create_app(Settings(workspace=tmp_path / "ws"))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/projects").json() == []
        assert client.get("/static/does-not-exist.js").status_code == 404
    assert app.state.settings.workspace == tmp_path / "ws"


def test_lifespan_recovers_cleans_and_shuts_down(tmp_path: Path) -> None:
    state = AppState(Settings(workspace=tmp_path / "ws"))
    project = state.store.create("Cats", "", "classification", ["cat"])
    staging = state.store.dir(project.id) / "staging"
    calls: list[str] = []
    state.runner.recover = lambda: calls.append("recover")  # type: ignore[method-assign]
    state.runner.shutdown = lambda: calls.append("runner")  # type: ignore[method-assign]
    state.jobs.shutdown = lambda: calls.append("jobs")  # type: ignore[method-assign]
    with TestClient(create_app(state=state)):
        assert calls == ["recover"]
    assert calls == ["recover", "jobs", "runner"]
    assert not staging.exists() or not any(staging.iterdir())


def test_scripts_are_served_as_javascript_on_every_system(tmp_path: Path) -> None:
    """A module script served as ``text/plain`` (a Windows registry default) is refused by the browser."""
    assert mimetypes.guess_type("main.js")[0] == "text/javascript"
    with TestClient(create_app(Settings(workspace=tmp_path / "ws"))) as client:
        response = client.get("/static/js/main.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_static_files_are_revalidated_by_the_browser(tmp_path: Path) -> None:
    """Imported files carry no version in their address: ``no-cache`` plus the ETag keeps them current."""
    with TestClient(create_app(Settings(workspace=tmp_path / "ws"))) as client:
        first = client.get("/static/app.css")
        again = client.get("/static/app.css", headers={"If-None-Match": first.headers["etag"]})
    assert first.headers["cache-control"] == "no-cache"
    assert again.status_code == 304
    assert again.headers["cache-control"] == "no-cache"
