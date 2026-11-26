"""The error handlers: every API error has a detail, a stable code and params."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from activelabel import optional
from activelabel.errors import CodedError
from activelabel.storage import DamagedFile
from activelabel.web.errors import (
    ApiError,
    api_error,
    bad_request,
    damaged_file,
    install_error_handlers,
    need_fiftyone,
)


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


def test_a_request_with_wrong_types_answers_422_with_a_code_and_the_problems() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/n")
    def numbers(body: dict[str, int]) -> None:
        return None

    @app.get("/q")
    def query(page: int) -> None:
        return None

    client = TestClient(app, raise_server_exceptions=False)
    body = client.post("/n", json={"a": "x"}).json()
    assert body["code"] == "validation_error" and isinstance(body["detail"], str)
    assert body["params"]["errors"][0]["type"] == "int_parsing"
    assert body["params"]["errors"][0]["loc"] == ["body", "a"]
    assert "a" in body["detail"]
    missing = client.get("/q").json()
    assert missing["code"] == "validation_error" and missing["params"]["errors"][0]["type"] == "missing"
    assert client.post("/n", content=b"{bad", headers={"content-type": "application/json"}).status_code == 422


def test_unknown_address_and_method_answer_json_with_a_code() -> None:
    client = _app()
    gone = client.get("/nowhere")
    assert gone.status_code == 404
    assert gone.json() == {"detail": "Not Found", "code": "not_found", "params": {}}
    wrong = client.post("/coded")
    assert wrong.status_code == 405 and wrong.json()["code"] == "method_not_allowed"


class ServiceListenTimeout(Exception):
    """Same name as FiftyOne's error when its database service cannot start."""


def _boom() -> None:
    raise RuntimeError("a bug")


def _database_down() -> None:
    raise ServiceListenTimeout("DatabaseService failed to bind to port")


def _unexpected_app() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)
    app.add_api_route("/api/boom", _boom)
    app.add_api_route("/api/database", _database_down)
    app.add_api_route("/page", _boom)
    return TestClient(app, raise_server_exceptions=False)


def test_an_unexpected_error_answers_a_code_not_a_bare_text() -> None:
    """#535: the interface showed "HTTP 500" because the body was the plain text "Internal Server Error"."""
    client = _unexpected_app()
    boom = client.get("/api/boom")
    assert boom.status_code == 500
    assert boom.json() == {
        "detail": "Unexpected error on the server (see the server log).",
        "code": "internal_error",
        "params": {},
    }
    assert "a bug" not in boom.text  # the exception text stays in the log
    down = client.get("/api/database")
    assert down.status_code == 503 and down.json()["code"] == "fiftyone_unavailable"
    page = client.get("/page")
    assert page.status_code == 500 and page.text == "Internal Server Error"


def test_need_fiftyone_passes_when_installed_and_answers_501_when_not(monkeypatch) -> None:
    monkeypatch.setattr(optional, "FIFTYONE", True)
    assert need_fiftyone() is None
    monkeypatch.setattr(optional, "FIFTYONE", False)
    with pytest.raises(ApiError) as caught:
        need_fiftyone()
    assert caught.value.status_code == 501 and caught.value.code == "fiftyone_not_installed"
    assert caught.value.params == {"install": optional.FIFTYONE_INSTALL}
