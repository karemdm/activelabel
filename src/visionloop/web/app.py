"""The web application: pages, API and static files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from visionloop import __version__
from visionloop.config import Settings
from visionloop.web import api, pages
from visionloop.web.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application (``state`` is injectable for tests)."""
    state = state or AppState(settings or Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        state.jobs.shutdown()

    app = FastAPI(title="visionloop", version=__version__, lifespan=lifespan)
    app.state.visionloop = state
    app.state.settings = state.settings
    app.include_router(api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
