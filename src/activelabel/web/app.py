"""The web application: pages, API and static files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from activelabel import __version__
from activelabel.annotations import cleanup_staging
from activelabel.config import Settings
from activelabel.web import (
    annotate_api,
    api,
    datasets_api,
    import_api,
    models_api,
    pages,
    preannotate_api,
    sync_api,
    tasks_api,
    training_api,
    video_api,
)
from activelabel.web.errors import install_error_handlers
from activelabel.web.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(state: AppState, app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shutdown of the server: recover interrupted runs, clean staging, stop the workers.

    Args:
        state: The shared state of the application.
        app: The application (unused; FastAPI passes it).

    Yields:
        Once, while the server is running.
    """
    state.runner.recover()
    for project in state.store.list():  # staged annotation files nobody applied or discarded
        cleanup_staging(state.store.dir(project.id))
    yield
    state.jobs.shutdown()
    state.runner.shutdown()


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application (``state`` is injectable for tests).

    Args:
        settings: The settings to build the state from (read from the environment when omitted).
        state: A ready state, used as is (``settings`` is then ignored).

    Returns:
        The FastAPI application with every router, the error handlers and the static files.
    """
    state = state or AppState(settings or Settings.from_env())
    app = FastAPI(title="activelabel", version=__version__, lifespan=partial(lifespan, state))
    app.state.activelabel = state
    app.state.settings = state.settings
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
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
