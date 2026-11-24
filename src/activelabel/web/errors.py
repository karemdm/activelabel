"""Errors the whole API answers the same way.

Every error body has ``detail`` (a readable English text, as before). Errors the interface should
translate also have ``code`` (stable) and ``params`` (values for the translated text)::

    {"detail": "A project named 'Cats' already exists.", "code": "project_name_taken",
     "params": {"name": "Cats"}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from activelabel.errors import CodedError
from activelabel.storage import DamagedFile
from activelabel.textcheck import message_of


class ApiError(HTTPException):
    """An HTTP error with a stable ``code`` and ``params`` next to its English ``detail``."""

    def __init__(self, status_code: int, code: str, message: str, **params: Any) -> None:
        super().__init__(status_code, message)
        self.code = code
        self.params = params


def bad_request(exc: Exception) -> HTTPException:
    """HTTP 400 for a rejected input (keeps the code of a :class:`~activelabel.errors.CodedError`)."""
    if isinstance(exc, CodedError):
        return ApiError(400, exc.code, message_of(exc), **exc.params)
    return HTTPException(400, message_of(exc))


async def api_error(request: Request, exc: Exception) -> JSONResponse:
    """Exception handler: the JSON body of an :class:`ApiError` (``detail``, ``code`` and ``params``).

    Args:
        request: The request that failed (unused).
        exc: The error raised; an :class:`ApiError` (the handler is registered for that class only).

    Returns:
        The JSON response with the status code and headers of the error.
    """
    assert isinstance(exc, ApiError)
    return JSONResponse(
        {"detail": exc.detail, "code": exc.code, "params": exc.params},
        status_code=exc.status_code,
        headers=exc.headers,
    )


async def damaged_file(request: Request, exc: Exception) -> JSONResponse:
    """Exception handler: HTTP 500 with the readable text of a damaged data file.

    Args:
        request: The request that failed (unused).
        exc: The :class:`~activelabel.storage.DamagedFile` raised.

    Returns:
        The JSON response (code ``file_damaged``, the file name in ``params``).
    """
    return JSONResponse(
        {"detail": str(exc), "code": "file_damaged", "params": {"file": getattr(exc, "file", "")}},
        status_code=500,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Answer an :class:`ApiError` with its code and a damaged data file with HTTP 500 and a readable text."""
    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(DamagedFile, damaged_file)
