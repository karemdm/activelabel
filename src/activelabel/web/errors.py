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
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from activelabel.annotate.base import ToolError
from activelabel.errors import CodedError
from activelabel.jobs import database_failure
from activelabel.optional import FeatureMissing, require_fiftyone
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


async def unexpected_error(request: Request, exc: Exception) -> Response:
    """Exception handler: any error no other handler took, as a coded answer instead of a bare text.

    The server still logs the traceback (Starlette re-raises after answering); the interface gets a code
    it translates. A FiftyOne database
    that did not answer is HTTP 503 ``fiftyone_unavailable`` (trying again can work); anything else is
    HTTP 500 ``internal_error``.

    Args:
        request: The request that failed (only its path is read).
        exc: The exception raised.

    Returns:
        The JSON response ``{detail, code, params}`` for an address under ``/api``; a plain text for a page.
    """
    if not request.url.path.startswith("/api/"):
        return PlainTextResponse("Internal Server Error", status_code=500)
    if database_failure(exc):
        return JSONResponse(
            {
                "detail": (
                    "The FiftyOne database did not respond: try again; "
                    "if it keeps failing, restart ActiveLabel."
                ),
                "code": "fiftyone_unavailable",
                "params": {},
            },
            status_code=503,
        )
    return JSONResponse(
        {
            "detail": "Unexpected error on the server (see the server log).",
            "code": "internal_error",
            "params": {},
        },
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


def tool_failure(exc: ToolError, tool: str, status: int = 400) -> ApiError:
    """An HTTP error for a problem of an annotation tool, with the code the tool's error carries.

    Args:
        exc: The error of the tool.
        tool: Name of the tool (``cvat``), for an error that has no code of its own.
        status: HTTP status of the answer.

    Returns:
        An :class:`ApiError` with the code and params of ``exc``; an error without a code becomes
        ``tool_error`` with the tool and the English text as params.
    """
    if exc.code:
        return ApiError(status, exc.code, str(exc), **exc.params)
    return ApiError(status, "tool_error", str(exc), tool=tool, reason=str(exc))


def need_fiftyone() -> None:
    """HTTP 501 ``fiftyone_not_installed`` when FiftyOne is not installed (the light install).

    Raises:
        ApiError: 501, with the install command as ``install``.
    """
    try:
        require_fiftyone()
    except FeatureMissing as exc:
        raise ApiError(501, exc.code, str(exc), **exc.params) from None


def install_error_handlers(app: FastAPI) -> None:
    """Give every API error the same body: coded, invalid (422), plain HTTP and unexpected errors.

    Args:
        app: The application to add the handlers to.
    """
    app.add_exception_handler(StarletteHTTPException, plain_http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(DamagedFile, damaged_file)
    app.add_exception_handler(Exception, unexpected_error)
