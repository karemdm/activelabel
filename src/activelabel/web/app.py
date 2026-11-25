"""The web application: pages, API and static files."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from activelabel import __version__
from activelabel.annotations import cleanup_staging
from activelabel.config import Settings
from activelabel.web import (
    annotate_api,
    api,
    backups_api,
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
from activelabel.web.guard import LocalOnly
from activelabel.web.state import AppState

STATIC_DIR = Path(__file__).parent / "static"

# Windows may map ".js" to "text/plain" in its registry: browsers refuse a module script served that way.
mimetypes.add_type("text/javascript", ".js")


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate before using its copy (``Cache-Control: no-cache``).

    The page links its entry files with a version in the address, but the files they import have none:
    revalidating (a 304 when unchanged) keeps an old copy from being used after an update.
    """

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        """Answer with the file, or with "not modified", and the no-cache header.

        Args:
            full_path: Path of the file on disk.
            stat_result: The file's ``os.stat`` result.
            scope: The ASGI scope of the request.
            status_code: Status of a full answer.

        Returns:
            The response of :class:`StaticFiles` with ``Cache-Control: no-cache``.
        """
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


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
        The FastAPI application with every router, the error handlers, the static files and the check that
        only this server's own pages (or a local script) call it (:class:`~activelabel.web.guard.LocalOnly`).
    """
    state = state or AppState(settings or Settings.from_env())
    app = FastAPI(title="activelabel", version=__version__, lifespan=partial(lifespan, state))
    app.state.activelabel = state
    app.state.settings = state.settings
    install_error_handlers(app)
    app.add_middleware(LocalOnly, allowed_hosts=state.settings.allowed_hosts)
    app.add_exception_handler(StarletteHTTPException, pages.unknown_address)
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
    app.include_router(video_api.router)
    app.include_router(pages.router)
    app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")
    return app
