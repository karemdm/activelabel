"""Only the server's own pages, or a local script, may call it (Host and Origin/Referer checks)."""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from activelabel.config import Settings
from activelabel.web.app import create_app
from activelabel.web.guard import LocalOnly, origin_address, split_host

BASE = "http://127.0.0.1:8095"
NEW_PROJECT = {"name": "Cats", "task": "classification", "classes": ["cat"]}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A client of the whole application, at a loopback address and port like the real server."""
    return TestClient(create_app(Settings(workspace=tmp_path / "ws")), base_url=BASE)


def test_split_host_reads_names_ports_and_ipv6() -> None:
    assert split_host("127.0.0.1:8090") == ("127.0.0.1", 8090)
    assert split_host("[::1]:8090") == ("::1", 8090)
    assert split_host("LocalHost") == ("localhost", 80)
    assert split_host("localhost", "https") == ("localhost", 443)
    for bad in ("", "127.0.0.1:port", "127.0.0.1:99999", "user@127.0.0.1", "127.0.0.1/x"):
        assert split_host(bad) is None, bad


def test_origin_address_reads_origins_and_referers() -> None:
    assert origin_address("http://127.0.0.1:8090") == ("127.0.0.1", 8090)
    assert origin_address("http://localhost:8090/projects/1?x=1") == ("localhost", 8090)
    assert origin_address("https://evil.example") == ("evil.example", 443)
    for bad in ("null", "", "file:///etc", "127.0.0.1:8090", "http://[::1"):
        assert origin_address(bad) is None, bad


def test_accepts_loopback_own_and_allowed_names_on_the_server_port() -> None:
    guard = LocalOnly(app=None, allowed_hosts=[" My-PC.local ", ""])  # type: ignore[arg-type]
    server = ("192.168.0.5", 8090)
    for name in ("127.0.0.1", "localhost", "::1", "192.168.0.5", "my-pc.local"):
        assert guard.accepts((name, 8090), server), name
    assert not guard.accepts(("evil.example", 8090), server)
    assert not guard.accepts(("127.0.0.1", 8091), server)
    assert not guard.accepts(None, server)
    assert guard.accepts(("localhost", 1234), None)  # no server address: only the name is checked


@pytest.mark.parametrize("host", ["127.0.0.1:8095", "localhost:8095", "[::1]:8095"])
def test_the_loopback_names_on_the_server_port_are_served(client: TestClient, host: str) -> None:
    assert client.get("/api/projects", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example:8095", "127.0.0.1:8096", "evil.example", "", "a:b"])
def test_another_host_is_refused_with_a_coded_403(client: TestClient, host: str) -> None:
    """DNS rebinding: a page of another domain resolved to 127.0.0.1 cannot read nor change anything."""
    for answer in (
        client.get("/api/projects", headers={"host": host}),
        client.get("/", headers={"host": host}),
        client.post("/api/projects", json=NEW_PROJECT, headers={"host": host}),
    ):
        assert answer.status_code == 403
        body = answer.json()
        assert body["code"] == "host_not_allowed" and body["params"] == {"host": host}
        assert "127.0.0.1" in body["detail"]
    assert client.get("/api/projects").json() == []


def test_a_name_of_allowed_hosts_is_served(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path / "ws", allowed_hosts=("my-pc.local",))
    with TestClient(create_app(settings), base_url=BASE) as client:
        assert client.get("/api/projects", headers={"host": "my-pc.local:8095"}).status_code == 200
        assert client.get("/api/projects", headers={"host": "other.local:8095"}).status_code == 403


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("header", ["origin", "referer"])
def test_a_change_from_another_site_is_refused_for_every_method(
    client: TestClient, method: str, header: str
) -> None:
    """CSRF: a page of another site (or another port of this machine) cannot change data."""
    for source in ("https://evil.example", "http://127.0.0.1:3000/page", "null"):
        answer = client.request(method, "/api/projects/1", json={}, headers={header: source})
        assert answer.status_code == 403, (method, header, source)
        assert answer.json()["code"] == "origin_not_allowed"
        assert answer.json()["params"] == {"origin": source}


def test_the_origin_is_checked_before_the_referer(client: TestClient) -> None:
    headers = {"origin": "https://evil.example", "referer": f"{BASE}/"}
    assert client.post("/api/projects", json=NEW_PROJECT, headers=headers).status_code == 403


def test_a_change_from_the_own_pages_or_a_script_goes_through(client: TestClient) -> None:
    created = client.post("/api/projects", json=NEW_PROJECT, headers={"origin": BASE})
    assert created.status_code == 200, created.text
    pid = created.json()["id"]
    renamed = client.patch(
        f"/api/projects/{pid}", json={"name": "Dogs"}, headers={"referer": f"{BASE}/projects/{pid}"}
    )
    assert renamed.status_code == 200
    other_name = client.patch(
        f"/api/projects/{pid}", json={"name": "Birds"}, headers={"origin": "http://localhost:8095"}
    )
    assert other_name.status_code == 200
    script = client.post("/api/projects", json={**NEW_PROJECT, "name": "Fish"})  # no Origin nor Referer
    assert script.status_code == 200, script.text


def test_reading_ignores_the_origin(client: TestClient) -> None:
    """A GET changes nothing; another site cannot read the answer anyway (no CORS headers)."""
    answer = client.get("/api/projects", headers={"origin": "https://evil.example"})
    assert answer.status_code == 200
    assert "access-control-allow-origin" not in answer.headers


async def _never_called(scope: Scope, receive: Receive, send: Send) -> None:
    raise AssertionError("the request should not reach the application")


async def _no_message() -> Message:
    return {"type": "websocket.connect"}


def test_a_websocket_from_another_site_is_closed_before_the_application() -> None:
    sent: list[Message] = []
    scope: Scope = {
        "type": "websocket",
        "server": ("127.0.0.1", 8095),
        "headers": [(b"host", b"127.0.0.1:8095"), (b"origin", b"https://evil.example")],
    }
    assert LocalOnly(_never_called).problem(scope).code == "origin_not_allowed"  # type: ignore[union-attr]
    asyncio.run(LocalOnly(_never_called)(scope, _no_message, _collect(sent)))
    assert sent == [{"type": "websocket.close", "code": 1008}]


def _collect(sent: list[Message]) -> Send:
    """A send channel that keeps the messages in ``sent``."""
    return functools.partial(_append, sent)


async def _append(sent: list[Message], message: Message) -> None:
    sent.append(message)


def test_the_lifespan_passes_through() -> None:
    sent: list[Message] = []
    with pytest.raises(AssertionError, match="should not reach"):
        asyncio.run(LocalOnly(_never_called)({"type": "lifespan"}, _no_message, _collect(sent)))
