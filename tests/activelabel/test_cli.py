import pytest

from activelabel import cli


@pytest.fixture
def served(monkeypatch):
    """Replace the server, the app and the browser with recorders."""
    calls: dict[str, object] = {"opened": []}
    monkeypatch.setattr(cli, "create_app", lambda: "the-app")
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: calls.update(app=app, **kw))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: calls["opened"].append(url))  # type: ignore[attr-defined]
    return calls


def test_serve_runs_the_app_on_the_local_address_by_default(served, capsys):
    assert cli.main(["serve"]) == 0
    assert served["app"] == "the-app"
    assert (served["host"], served["port"]) == ("127.0.0.1", 8090)
    assert served["opened"] == []
    assert "activelabel: http://127.0.0.1:8090" in capsys.readouterr().err


def test_serve_takes_host_and_port(served, capsys):
    cli.main(["serve", "--host", "192.168.1.5", "--port", "9000"])
    assert (served["host"], served["port"]) == ("192.168.1.5", 9000)
    assert "http://192.168.1.5:9000" in capsys.readouterr().err


def test_listening_on_every_address_shows_a_link_that_opens(served, capsys):
    cli.main(["serve", "--host", "0.0.0.0", "--port", "8123"])
    assert served["host"] == "0.0.0.0"
    assert "http://127.0.0.1:8123" in capsys.readouterr().err


def test_open_starts_the_browser_on_the_url(served):
    cli.main(["serve", "--port", "8123", "--open"])
    assert served["opened"] == ["http://127.0.0.1:8123"]


def test_a_command_is_required(served, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2
    assert "app" not in served


def test_the_port_must_be_a_number(served):
    with pytest.raises(SystemExit):
        cli.main(["serve", "--port", "x"])
