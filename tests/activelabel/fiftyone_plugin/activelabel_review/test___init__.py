import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from activelabel.fiftyone_plugin import activelabel_review as plugin


class Ops:
    def __init__(self):
        self.notes = []

    def notify(self, text, variant="info"):
        self.notes.append((variant, text))


class View:
    def __init__(self, name):
        self.name = name


class Dataset:
    def __init__(self, info):
        self.info = info


class Context:
    def __init__(self, info, params=None, view_name=None):
        self.dataset = Dataset(info)
        self.params = params or {}
        self.ops = Ops()
        self.view = View(view_name)
        self.results = {}


@pytest.fixture
def server():
    """A fake ActiveLabel server on a free local port: ``routes`` maps (method, path) to (status, JSON)."""
    routes: dict = {}
    seen: list = []

    class Handler(BaseHTTPRequestHandler):
        def answer(self):
            seen.append((self.command, self.path))
            status, body = routes.get((self.command, self.path), (404, {"detail": "no such route"}))
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = answer

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", routes, seen
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(plugin.webbrowser, "open", opened.append)
    monkeypatch.setattr(plugin.time, "sleep", lambda s: None)
    monkeypatch.delenv("ACTIVELABEL_LANG", raising=False)
    return opened


def session(sid, status="open", tool="fiftyone"):
    return {"id": sid, "tool": tool, "status": status, "scope": "all"}


def info_for(base, project=2):
    return {"project_id": project, plugin.URL_KEY: base}


def warning_label(prop):
    return prop.type.properties["warning"].view.label


@pytest.mark.parametrize("raw", ["http://127.0.0.1:8167", "http://localhost:8167/", "http://[::1]:8167"])
def test_a_local_server_address_is_accepted(raw):
    assert plugin.server_url({plugin.URL_KEY: raw}) is not None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        42,
        "",
        "https://127.0.0.1:8167",
        "http://example.com:8167",
        "http://192.168.1.5:8090",
        "http://127.0.0.1",
        "http://127.0.0.1:8167/api",
        "http://user:pw@127.0.0.1:8167",
        "http://127.0.0.1:8167?x=1",
        "http://127.0.0.1:99999",
        "file:///etc/passwd",
        "http://127.0.0.1.evil.com:8167",
    ],
)
def test_any_other_address_is_refused(raw):
    assert plugin.server_url({plugin.URL_KEY: raw}) is None


def test_the_address_is_rebuilt_without_the_original_text():
    assert plugin.server_url({plugin.URL_KEY: "http://localhost:8167/"}) == "http://localhost:8167"
    assert plugin.server_url({plugin.URL_KEY: "http://[::1]:8167"}) == "http://[::1]:8167"


def test_only_a_dataset_of_activelabel_has_a_project():
    assert plugin.project_of({"project_id": 3}) == 3
    assert plugin.project_of({}) is None
    assert plugin.project_of({"project_id": True}) is None
    assert plugin.project_of({"project_id": "3"}) is None


def test_the_session_of_a_view():
    assert plugin.session_of_view("annotate - session s1") == "s1"
    assert plugin.session_of_view("annotate - session ") is None
    assert plugin.session_of_view("task-4") is None
    assert plugin.session_of_view(None) is None


def test_texts_are_pure_and_the_language_follows_activelabel(monkeypatch):
    assert plugin.TEXT["pt"].keys() == plugin.TEXT["en"].keys()
    assert plugin.say("label") == "Enviar ao ActiveLabel"
    monkeypatch.setenv("ACTIVELABEL_LANG", "en")
    assert plugin.say("label") == "Send to ActiveLabel"
    assert plugin.say("down", url="http://x") == "Could not reach http://x. Start ActiveLabel and try again."


def test_a_server_that_is_down_raises():
    with pytest.raises(plugin.ServerError):
        plugin.call("http://127.0.0.1:1", "GET", "/api/x")


def test_an_error_answer_keeps_its_message(server):
    base, routes, _ = server
    routes[("GET", "/api/x")] = (409, {"detail": "busy"})
    with pytest.raises(plugin.ServerError, match="busy"):
        plugin.call(base, "GET", "/api/x")


def test_only_live_fiftyone_sessions_are_listed(server):
    base, routes, _ = server
    sessions = [session("a"), session("b", "applied"), session("c", "pulled"), session("d", tool="cvat")]
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": sessions})
    assert [s["id"] for s in plugin.live_sessions(base, 2)] == ["a", "c"]


def test_pull_waits_for_the_job(server):
    base, routes, seen = server
    routes[("POST", "/api/projects/2/annotate/a/pull")] = (200, {"id": "j1", "status": "queued"})
    routes[("GET", "/api/jobs/j1")] = (200, {"id": "j1", "status": "done", "result": {"diff": {}}})
    assert plugin.pull(base, 2, "a") == {"diff": {}}
    assert seen == [("POST", "/api/projects/2/annotate/a/pull"), ("GET", "/api/jobs/j1")]


def test_pull_reports_a_failed_job(server):
    base, routes, _ = server
    failed = {"id": "j1", "status": "failed", "error": "boom"}
    routes[("POST", "/api/projects/2/annotate/a/pull")] = (200, failed)
    with pytest.raises(plugin.ServerError, match="boom"):
        plugin.pull(base, 2, "a")


def test_pull_gives_up_after_the_wait(server):
    base, routes, _ = server
    routes[("POST", "/api/projects/2/annotate/a/pull")] = (200, {"id": "j1", "status": "running"})
    routes[("GET", "/api/jobs/j1")] = (200, {"id": "j1", "status": "running"})
    with pytest.raises(plugin.ServerError, match=plugin.TIMED_OUT):
        plugin.pull(base, 2, "a", wait=-1)


def test_the_session_id_cannot_change_the_path(server):
    base, routes, seen = server
    routes[("POST", "/api/projects/2/annotate/a%2F..%2Fb/pull")] = (200, {"id": "j", "status": "done"})
    plugin.pull(base, 2, "a/../b")
    assert seen == [("POST", "/api/projects/2/annotate/a%2F..%2Fb/pull")]


def test_the_review_page_is_the_annotations_step():
    url = plugin.review_url("http://127.0.0.1:8167", 2, "a b")
    assert url == "http://127.0.0.1:8167/projects/2?review=a%20b#annotations"


def test_the_dialog_refuses_a_dataset_that_is_not_from_activelabel():
    prop = plugin.SendToActiveLabel().resolve_input(Context({}))
    assert warning_label(prop) == plugin.TEXT["pt"]["not_ours_title"]


def test_the_dialog_refuses_a_server_address_that_is_not_local():
    info = {"project_id": 2, plugin.URL_KEY: "http://example.com:80"}
    prop = plugin.SendToActiveLabel().resolve_input(Context(info))
    assert warning_label(prop) == plugin.TEXT["pt"]["no_url_title"]


def test_the_dialog_says_when_the_server_is_not_reachable():
    prop = plugin.SendToActiveLabel().resolve_input(Context(info_for("http://127.0.0.1:1")))
    assert warning_label(prop) == plugin.TEXT["pt"]["down_title"]


def test_the_dialog_says_when_there_is_no_session(server):
    base, routes, _ = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": []})
    prop = plugin.SendToActiveLabel().resolve_input(Context(info_for(base)))
    assert warning_label(prop) == plugin.TEXT["pt"]["none_title"]


def test_one_session_needs_no_choice_and_several_default_to_the_open_view(server):
    base, routes, _ = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a")]})
    one = plugin.SendToActiveLabel().resolve_input(Context(info_for(base)))
    assert "session" not in one.type.properties
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a"), session("b")]})
    many = plugin.SendToActiveLabel().resolve_input(Context(info_for(base)))
    assert many.type.properties["session"].default == "a"
    chosen = plugin.SendToActiveLabel().resolve_input(
        Context(info_for(base), view_name="annotate - session b")
    )
    assert chosen.type.properties["session"].default == "b"


def test_running_it_pulls_the_session_and_opens_the_review(server, no_browser):
    base, routes, seen = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a"), session("b")]})
    routes[("POST", "/api/projects/2/annotate/b/pull")] = (200, {"id": "j1", "status": "done", "result": {}})
    ctx = Context(info_for(base), params={"session": "b"})
    result = plugin.SendToActiveLabel().execute(ctx)
    assert result == {"url": f"{base}/projects/2?review=b#annotations"}
    assert no_browser == [f"{base}/projects/2?review=b#annotations"]
    assert ctx.ops.notes == [("success", plugin.TEXT["pt"]["done"])]
    assert ("POST", "/api/projects/2/annotate/b/pull") in seen


def test_without_a_choice_the_session_of_the_open_view_is_used(server):
    base, routes, seen = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a"), session("b")]})
    routes[("POST", "/api/projects/2/annotate/b/pull")] = (200, {"id": "j1", "status": "done"})
    plugin.SendToActiveLabel().execute(Context(info_for(base), view_name="annotate - session b"))
    assert ("POST", "/api/projects/2/annotate/b/pull") in seen


def test_a_failure_is_shown_in_the_app_and_nothing_is_opened(server, no_browser):
    base, routes, _ = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a")]})
    routes[("POST", "/api/projects/2/annotate/a/pull")] = (409, {"detail": "project busy"})
    ctx = Context(info_for(base))
    assert plugin.SendToActiveLabel().execute(ctx) == {}
    assert ctx.ops.notes[0][0] == "error"
    assert "project busy" in ctx.ops.notes[0][1]
    assert no_browser == []


def test_a_timeout_has_its_own_text(server, monkeypatch):
    base, routes, _ = server
    routes[("GET", "/api/projects/2/annotate")] = (200, {"sessions": [session("a")]})
    routes[("POST", "/api/projects/2/annotate/a/pull")] = (200, {"id": "j1", "status": "running"})
    routes[("GET", "/api/jobs/j1")] = (200, {"id": "j1", "status": "running"})
    monkeypatch.setattr(plugin, "PULL_WAIT", -1)
    ctx = Context(info_for(base))
    plugin.SendToActiveLabel().execute(ctx)
    assert ctx.ops.notes == [("error", plugin.TEXT["pt"]["timeout"])]


def test_it_does_not_run_for_a_dataset_that_is_not_from_activelabel(no_browser):
    ctx = Context({"project_id": 2, plugin.URL_KEY: "http://example.com:80"})
    assert plugin.SendToActiveLabel().execute(ctx) == {}
    assert ctx.ops.notes[0][0] == "error"
    assert no_browser == []


def test_the_output_links_to_the_review():
    ctx = Context({})
    ctx.results = {"url": "http://127.0.0.1:8167/projects/2?review=a#annotations"}
    assert plugin.SendToActiveLabel().resolve_output(ctx).type.properties
    ctx.results = {}
    assert not plugin.SendToActiveLabel().resolve_output(ctx).type.properties


def test_the_operator_is_registered_and_placed_in_the_grid():
    registered = []

    class Plugin:
        register = registered.append

    plugin.register(Plugin)
    assert registered == [plugin.SendToActiveLabel]
    op = plugin.SendToActiveLabel()
    assert op.config.name == "send_to_activelabel"
    assert op.resolve_placement(Context({})).view.label == "Enviar ao ActiveLabel"
