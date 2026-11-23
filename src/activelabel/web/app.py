"""The web application: pages, API and static files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from activelabel import __version__
from activelabel.config import Settings
from activelabel.web import annotate_api, api, import_api, pages, sync_api, tasks_api, video_api
from activelabel.web.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application (``state`` is injectable for tests)."""
    state = state or AppState(settings or Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        state.jobs.shutdown()

    app = FastAPI(title="activelabel", version=__version__, lifespan=lifespan)
    app.state.activelabel = state
    app.state.settings = state.settings
    app.include_router(api.router)
    app.include_router(annotate_api.router)
    app.include_router(import_api.router)
    app.include_router(sync_api.router)
    app.include_router(tasks_api.router)
    app.include_router(video_api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
