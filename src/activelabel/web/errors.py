"""Errors the whole API answers the same way."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from activelabel.storage import DamagedFile


def install_error_handlers(app: FastAPI) -> None:
    """Answer a damaged data file with HTTP 500 and a readable ``detail`` (it names the file)."""

    async def damaged_file(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    app.add_exception_handler(DamagedFile, damaged_file)
