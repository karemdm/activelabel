"""Every error the API sends the interface carries a stable code (see ``web/errors.py``)."""

from __future__ import annotations

import ast
from pathlib import Path

from api_client import make_client

import activelabel.web as web_package

WEB_DIR = Path(web_package.__file__).parent
TOOL_TEXT = "the text comes from the remote tool (a ToolError), which has no code yet"
# (module, function) -> why a plain HTTPException with a string detail may stay
ALLOWED = {
    ("annotate_api.py", "_tool"): TOOL_TEXT,
    ("annotate_api.py", "send"): TOOL_TEXT,
    ("annotate_api.py", "close_session"): TOOL_TEXT,
    ("errors.py", "bad_request"): "fallback for a ValueError that has no code",
}


def _client_errors(path: Path) -> list[tuple[str, int]]:
    """(function, line) of every ``HTTPException(4xx, <text>)`` in a module."""
    found: list[tuple[str, int]] = []

    def walk(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else function
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "HTTPException"
                and isinstance(child.args[0], ast.Constant)
                and 400 <= child.args[0].value < 500
                and not (len(child.args) > 1 and isinstance(child.args[1], ast.Dict))
            ):
                found.append((function, child.lineno))
            walk(child, name)

    walk(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return found


def test_no_client_error_without_a_code() -> None:
    bare = [
        f"{path.name}:{line} in {function}"
        for path in sorted(WEB_DIR.glob("*.py"))
        for function, line in _client_errors(path)
        if (path.name, function) not in ALLOWED
    ]
    assert not bare, "use ApiError (code + params) instead of HTTPException with a text: " + ", ".join(bare)


def _code(response, status: int) -> str:
    assert response.status_code == status, response.text
    return response.json()["code"]


def test_training_and_model_errors_have_codes(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a"]})
    runs = "/api/projects/1/runs"
    assert _code(client.get("/api/training/catalog", params={"type": "x"}), 400) == "type_invalid"
    assert _code(client.get("/api/training/environment", params={"model": "x"}), 404) == "model_not_found"
    assert _code(client.post(runs, json={"dataset": 1}), 400) == "config_choose"
    assert _code(client.post(runs, json={"dataset": 1, "model": "x", "path": "/tmp"}), 400) == (
        "config_unknown_options"
    )
    bad_device = {"dataset": 1, "model": "x", "device": "gpu"}
    assert _code(client.post(runs, json=bad_device), 400) == "device_invalid"
    assert _code(client.post(runs, json={"dataset": 7, "model": "x"}), 404) == "dataset_not_found"
    assert _code(client.get(f"{runs}/9"), 404) == "run_not_found"
    assert _code(client.request("DELETE", f"{runs}/9"), 400) == "confirm_required"
    models = "/api/projects/1/models"
    assert _code(client.get(f"{models}/9"), 404) == "model_not_found"
    assert _code(client.post(models, json={"run": 9}), 404) == "run_not_found"
    assert _code(client.post(f"{models}/public", json={"public": "x", "mapping": {}}), 404) == (
        "public_model_not_found"
    )
    assert _code(client.get(f"{models}/9/download/zip"), 404) == "download_unknown"
    body = client.get(f"{models}/9/download/zip").json()
    assert body["params"] == {"kind": "zip"} and body["detail"]
