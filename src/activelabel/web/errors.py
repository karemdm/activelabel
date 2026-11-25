"""Errors the whole API answers the same way.

Every error body has ``detail`` (a readable English text, as before). Errors the interface should
translate also have ``code`` (stable) and ``params`` (values for the translated text)::

    {"detail": "A project named 'Cats' already exists.", "code": "project_name_taken",
     "params": {"name": "Cats"}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from activelabel.errors import CodedError
from activelabel.storage import DamagedFile
from activelabel.textcheck import message_of


class ApiError(HTTPException):
    """An HTTP error with a stable ``code`` and ``params`` next to its English ``detail``."""

    def __init__(self, status_code: int, code: str, message: str, **params: Any) -> None:
        """Keep the code and the values next to the HTTP error.

        Args:
            status_code: HTTP status code.
            code: Stable code the interface translates.
            message: Readable English text (``detail``).
            **params: Values for the translated text.
        """
        super().__init__(status_code, message)
        self.code = code
        self.params = params


def bad_request(exc: Exception) -> HTTPException:
    """HTTP 400 for a rejected input (keeps the code of a :class:`~activelabel.errors.CodedError`).

    Args:
        exc: The error that rejected the input.

    Returns:
        A coded API error (HTTP 400) when ``exc`` has a code, else a plain HTTP 400.
    """
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


_PLAIN_CODES = {404: "not_found", 405: "method_not_allowed"}
_SIMPLE = (str, int, float, bool)


def _problem(entry: dict[str, Any]) -> dict[str, Any]:
    """One validation problem reduced to what the interface needs (never the value the person sent).

    Args:
        entry: An error of a :class:`~fastapi.exceptions.RequestValidationError`.

    Returns:
        ``loc`` (where), ``type`` (name of the problem), ``msg`` (English) and the simple ``ctx`` values.
    """
    ctx = {k: v for k, v in (entry.get("ctx") or {}).items() if isinstance(v, _SIMPLE)}
    return {"loc": list(entry["loc"]), "type": entry["type"], "msg": entry["msg"], "ctx": ctx}


async def validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Exception handler: HTTP 422 as ``{detail, code, params}`` (the problems are in ``params.errors``).

    Args:
        request: The request that failed (unused).
        exc: The :class:`~fastapi.exceptions.RequestValidationError` raised.

    Returns:
        The JSON response: ``detail`` is a readable English text, ``code`` is ``validation_error``.
    """
    assert isinstance(exc, RequestValidationError)
    errors = [_problem(e) for e in exc.errors()]
    text = "; ".join(
        f"{' > '.join(str(x) for x in e['loc'] if x not in ('body', 'query', 'path'))}: {e['msg']}".lstrip(
            ": "
        )
        for e in errors
    )
    return JSONResponse(
        {"detail": text, "code": "validation_error", "params": {"errors": errors}}, status_code=422
    )


async def plain_http_error(request: Request, exc: Exception) -> Response:
    """Exception handler: a plain HTTP error (unknown address, wrong method) as ``{detail, code, params}``.

    Args:
        request: The request that failed.
        exc: The HTTP error raised.

    Returns:
        The JSON response; the code is ``not_found``, ``method_not_allowed`` or ``http_error``.
    """
    assert isinstance(exc, StarletteHTTPException)
    if not isinstance(exc.detail, str):
        return await http_exception_handler(request, exc)
    code = _PLAIN_CODES.get(exc.status_code, "http_error")
    return JSONResponse(
        {"detail": exc.detail, "code": code, "params": {}}, status_code=exc.status_code, headers=exc.headers
    )


def install_error_handlers(app: FastAPI) -> None:
    """Give every API error the same body: coded errors, invalid requests (422) and plain HTTP errors.

    Args:
        app: The application to add the handlers to.
    """
    app.add_exception_handler(StarletteHTTPException, plain_http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(DamagedFile, damaged_file)
