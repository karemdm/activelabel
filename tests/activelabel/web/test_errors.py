"""The error handlers: every API error has a detail, a stable code and params."""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from activelabel.errors import CodedError
from activelabel.storage import DamagedFile
from activelabel.web.errors import ApiError, api_error, bad_request, damaged_file, install_error_handlers


def _app() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/coded")
    def coded() -> None:
        raise ApiError(409, "busy", "Wait.", name="x")

    @app.get("/damaged")
    def damaged() -> None:
        raise DamagedFile("broken", "projects.json")

    return TestClient(app, raise_server_exceptions=False)


def test_api_error_answers_with_code_and_params() -> None:
    response = _app().get("/coded")
    assert response.status_code == 409
    assert response.json() == {"detail": "Wait.", "code": "busy", "params": {"name": "x"}}


def test_damaged_file_answers_500_with_the_file_name() -> None:
    response = _app().get("/damaged")
    assert response.status_code == 500
    assert response.json()["code"] == "file_damaged"
    assert response.json()["params"] == {"file": "projects.json"}


def test_handlers_are_plain_functions_of_the_module() -> None:
    answer = asyncio.run(api_error(None, ApiError(404, "gone", "Gone.")))  # type: ignore[arg-type]
    assert answer.status_code == 404
    assert json.loads(answer.body)["code"] == "gone"
    assert asyncio.run(damaged_file(None, DamagedFile("x", "a.json"))).status_code == 500  # type: ignore[arg-type]


def test_bad_request_keeps_the_code_of_a_coded_error() -> None:
    coded = bad_request(CodedError("some_code", "Readable.", n=1))
    assert isinstance(coded, ApiError) and coded.code == "some_code" and coded.params == {"n": 1}
    plain = bad_request(ValueError("nope"))
    assert type(plain) is HTTPException and plain.status_code == 400
