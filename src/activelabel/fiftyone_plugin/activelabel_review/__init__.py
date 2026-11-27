"""FiftyOne plugin: send the edits made in the App to ActiveLabel and open the review.

The operator "Enviar ao ActiveLabel" (grid toolbar and operator browser, key ``) reads the edit session
of the dataset open in the App ("Revisar e aplicar" of ActiveLabel): it asks the ActiveLabel server to
read the edits back (the "pull" of the session) and opens the review page in the browser.

Safety: it only works for datasets made by ActiveLabel and only calls the server address ActiveLabel
recorded in ``dataset.info`` itself, and only when that address is on this machine (127.0.0.1, localhost
or ::1). It never calls any other address.

FiftyOne loads this file on its own, outside the ``activelabel`` package, so it imports nothing from it.
"""

from __future__ import annotations

import json
import os
import time
import webbrowser
from typing import Any
from urllib import error, request
from urllib.parse import quote, urlsplit

import fiftyone.operators as foo
import fiftyone.operators.types as types

URL_KEY = "activelabel_url"  # written by ActiveLabel (activelabel.fiftyone_plugin.use_plugin)
PROJECT_KEY = "project_id"  # written by ActiveLabel (fiftyone_bridge.sync_dataset)
VIEW_PREFIX = "annotate - session "  # saved view of a session (annotate.fiftyone_tool.view_name)
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
LIVE = ("open", "pulled")  # sessions that can be pulled
TIMEOUT = 10.0  # seconds for one call to the server
PULL_WAIT = 300.0  # seconds to wait for the pull job
FINISHED = ("done", "failed", "cancelled")  # states of a finished job (activelabel.constants)
TIMED_OUT = "timeout"

TEXT: dict[str, dict[str, str]] = {
    "pt": {
        "label": "Enviar ao ActiveLabel",
        "not_ours_title": "Este dataset não é do ActiveLabel",
        "not_ours": "Abra o FiftyOne pelo ActiveLabel (botão do FiftyOne no projeto) para usar este botão.",
        "no_url_title": "Endereço do ActiveLabel desconhecido",
        "no_url": "Este dataset não guarda o endereço do ActiveLabel (ou não é um endereço local). "
        "Abra o FiftyOne de novo pelo ActiveLabel.",
        "down_title": "O servidor do ActiveLabel não responde",
        "down": "Não foi possível falar com {url}. Inicie o ActiveLabel e tente de novo.",
        "none_title": "Nenhuma sessão de edição aberta",
        "none": "No ActiveLabel, envie imagens ao FiftyOne (Anotações) para criar uma sessão de edição.",
        "pick": "Sessão de edição",
        "session_item": "{id} ({status}, {scope})",
        "intro_title": "Trazer as edições e abrir a revisão",
        "intro": "O ActiveLabel lê o que você editou aqui e abre a página de revisão: lá você confere e "
        "aplica ou descarta. Nada é aplicado antes disso.",
        "failed": "Não foi possível trazer as edições: {why}",
        "timeout": "O ActiveLabel demorou demais para ler as edições. Abra a revisão e veja o andamento.",
        "done": "Edições lidas. A revisão foi aberta no ActiveLabel.",
        "link": "Abrir a revisão no ActiveLabel",
    },
    "en": {
        "label": "Send to ActiveLabel",
        "not_ours_title": "This dataset is not from ActiveLabel",
        "not_ours": "Open FiftyOne from ActiveLabel (the FiftyOne button of the project) to use this button.",
        "no_url_title": "ActiveLabel address unknown",
        "no_url": "This dataset does not keep the ActiveLabel address (or it is not a local address). "
        "Open FiftyOne again from ActiveLabel.",
        "down_title": "The ActiveLabel server is not reachable",
        "down": "Could not reach {url}. Start ActiveLabel and try again.",
        "none_title": "No open edit session",
        "none": "In ActiveLabel, send images to FiftyOne (Annotations) to create an edit session.",
        "pick": "Edit session",
        "session_item": "{id} ({status}, {scope})",
        "intro_title": "Bring the edits back and open the review",
        "intro": "ActiveLabel reads what you edited here and opens the review page: there you check and "
        "apply or discard. Nothing is applied before that.",
        "failed": "The edits could not be brought back: {why}",
        "timeout": "ActiveLabel took too long to read the edits. Open the review and check its progress.",
        "done": "Edits read. The review was opened in ActiveLabel.",
        "link": "Open the review in ActiveLabel",
    },
}


class ServerError(Exception):
    """The ActiveLabel server could not be reached, or it refused the call."""


def say(key: str, **values: Any) -> str:
    """A text of the plugin in the language of ActiveLabel (``ACTIVELABEL_LANG``; Portuguese by default).

    Args:
        key: the name of the text.
        **values: values to put in the text.

    Returns:
        The text, in one language only.
    """
    lang = "en" if os.environ.get("ACTIVELABEL_LANG", "").strip().lower().startswith("en") else "pt"
    return TEXT[lang][key].format(**values)


def server_url(info: dict[str, Any]) -> str | None:
    """The ActiveLabel server recorded in the dataset, when it is a local one.

    Args:
        info: ``dataset.info``.

    Returns:
        ``http://host:port`` without a path; None when nothing is recorded, or the address is not ``http``,
        not on this machine, has no port, or carries anything besides host and port.
    """
    raw = info.get(URL_KEY)
    if not isinstance(raw, str):
        return None
    try:
        parts = urlsplit(raw.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    bare = parts.path in ("", "/") and not (parts.query or parts.fragment or parts.username or parts.password)
    if parts.scheme != "http" or host not in LOOPBACK or port is None or not bare:
        return None
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def project_of(info: dict[str, Any]) -> int | None:
    """The ActiveLabel project of the dataset.

    Args:
        info: ``dataset.info``.

    Returns:
        The project id; None when the dataset was not made by ActiveLabel.
    """
    value = info.get(PROJECT_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def session_of_view(name: str | None) -> str | None:
    """The session a saved view belongs to.

    Args:
        name: the name of the view open in the App (None for the whole dataset).

    Returns:
        The session id, or None when the view is not the view of a session.
    """
    if name and name.startswith(VIEW_PREFIX):
        return name[len(VIEW_PREFIX) :] or None
    return None


def call(base: str, method: str, path: str) -> Any:
    """One call to the ActiveLabel server (never through a proxy).

    Args:
        base: the server address, from :func:`server_url`.
        method: ``GET`` or ``POST``.
        path: the path of the API, starting with ``/api/``.

    Returns:
        The JSON answer.

    Raises:
        ServerError: the server did not answer, or answered with an error (its message is kept).
    """
    opener = request.build_opener(request.ProxyHandler({}))
    req = request.Request(f"{base}{path}", method=method, data=b"" if method == "POST" else None)
    try:
        with opener.open(req, timeout=TIMEOUT) as answer:
            return json.loads(answer.read().decode("utf-8"))
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise ServerError(str(detail or exc)) from exc
    except (OSError, ValueError) as exc:  # URLError and timeouts are OSErrors
        raise ServerError(str(exc)) from exc


def live_sessions(base: str, project: int) -> list[dict[str, Any]]:
    """The FiftyOne edit sessions of the project that can still be read back, newest first.

    Args:
        base: the server address.
        project: the ActiveLabel project id.

    Returns:
        The sessions.
    """
    answer = call(base, "GET", f"/api/projects/{project}/annotate")
    found = answer.get("sessions", []) if isinstance(answer, dict) else []
    return [s for s in found if s.get("tool") == "fiftyone" and s.get("status") in LIVE]


def pull(base: str, project: int, session: str, wait: float | None = None) -> dict[str, Any]:
    """Ask the server to read the session's edits back and wait until it has.

    Args:
        base: the server address.
        project: the ActiveLabel project id.
        session: the session id.
        wait: seconds to wait for the job (``PULL_WAIT`` when not given).

    Returns:
        The result of the job (``{session, diff}``).

    Raises:
        ServerError: the call or the job failed, or the job did not finish in time (message ``timeout``).
    """
    job = call(base, "POST", f"/api/projects/{project}/annotate/{quote(session, safe='')}/pull")
    deadline = time.monotonic() + (PULL_WAIT if wait is None else wait)
    while job.get("status") not in FINISHED:
        if time.monotonic() > deadline:
            raise ServerError(TIMED_OUT)
        time.sleep(1.0)
        job = call(base, "GET", f"/api/jobs/{quote(str(job.get('id', '')), safe='')}")
    if job["status"] != "done":
        raise ServerError(str(job.get("error") or job["status"]))
    return job.get("result") or {}


def review_url(base: str, project: int, session: str) -> str:
    """The ActiveLabel page where the pulled edits are reviewed and applied.

    Args:
        base: the server address.
        project: the ActiveLabel project id.
        session: the session that was pulled (its card is opened on the page).

    Returns:
        The address of the project's Annotations step, with the session to show.
    """
    return f"{base}/projects/{project}?review={quote(session, safe='')}#annotations"


def warning(inputs: Any, title: str, text: str) -> Any:
    """A dialog that only shows a warning.

    Args:
        inputs: the dialog's ``types.Object``.
        title: the heading.
        text: the explanation.

    Returns:
        The property to return from ``resolve_input``.
    """
    inputs.view("warning", types.Warning(label=title, description=text))
    return types.Property(inputs)


def session_choice(found: list[dict[str, Any]], preferred: str | None) -> tuple[Any, str]:
    """The dropdown to choose between several open sessions.

    Args:
        found: the open sessions, newest first.
        preferred: the session of the view open in the App, when it is one of them.

    Returns:
        The ``Choices`` of the dropdown and the default session (the preferred one, else the newest).
    """
    group = types.Choices()
    for s in found:
        group.add_choice(
            s["id"], label=say("session_item", **{k: s.get(k, "") for k in ("id", "status", "scope")})
        )
    ids = [s["id"] for s in found]
    return group, (preferred if preferred and preferred in ids else ids[0])


class SendToActiveLabel(foo.Operator):
    """Read the edits of the open edit session back into ActiveLabel and open its review."""

    @property
    def config(self) -> Any:
        """Operator settings."""
        return foo.OperatorConfig(name="send_to_activelabel", label=say("label"), dynamic=True)

    def resolve_placement(self, ctx: Any) -> Any:
        """Button in the grid toolbar.

        Args:
            ctx: the operator context.

        Returns:
            The placement.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS, types.Button(label=say("label"), prompt=True)
        )

    def resolve_input(self, ctx: Any) -> Any:
        """The dialog: a warning when the dataset or the server is not usable, else the session to read.

        Args:
            ctx: the operator context.

        Returns:
            The dialog.
        """
        inputs = types.Object()
        info = dict(ctx.dataset.info or {})
        project = project_of(info)
        if project is None:
            return warning(inputs, say("not_ours_title"), say("not_ours"))
        base = server_url(info)
        if base is None:
            return warning(inputs, say("no_url_title"), say("no_url"))
        try:
            found = live_sessions(base, project)
        except ServerError:
            return warning(inputs, say("down_title"), say("down", url=base))
        if not found:
            return warning(inputs, say("none_title"), say("none"))
        inputs.view("intro", types.Notice(label=say("intro_title"), description=say("intro")))
        if len(found) > 1:
            group, default = session_choice(found, session_of_view(getattr(ctx.view, "name", None)))
            inputs.enum(
                "session", group.values(), default=default, required=True, label=say("pick"), view=group
            )
        return types.Property(inputs)

    def execute(self, ctx: Any) -> dict[str, Any]:
        """Pull the session and open the review in the browser.

        Args:
            ctx: the operator context.

        Returns:
            ``{url}`` of the review page; empty when nothing was done.
        """
        info = dict(ctx.dataset.info or {})
        project, base = project_of(info), server_url(info)
        if project is None or base is None:
            ctx.ops.notify(say("no_url"), variant="error")
            return {}
        try:
            ids = [s["id"] for s in live_sessions(base, project)]
            wanted = ctx.params.get("session") or session_of_view(getattr(ctx.view, "name", None))
            if not ids:
                ctx.ops.notify(say("none"), variant="error")
                return {}
            session = str(wanted) if wanted in ids else ids[0]
            pull(base, project, session)
        except ServerError as exc:
            why = say("timeout") if str(exc) == TIMED_OUT else say("failed", why=str(exc))
            ctx.ops.notify(why, variant="error")
            return {}
        url = review_url(base, project, session)
        try:
            webbrowser.open(url)
        except webbrowser.Error:
            pass  # the link in the output still works
        ctx.ops.notify(say("done"), variant="success")
        return {"url": url}

    def resolve_output(self, ctx: Any) -> Any:
        """A link to the review, for when the browser did not open it.

        Args:
            ctx: the operator context.

        Returns:
            The output description.
        """
        outputs = types.Object()
        url = (ctx.results or {}).get("url")
        if url:
            outputs.md(f"[{say('link')}]({url})")
        return types.Property(outputs)


def register(p: Any) -> None:
    """Register the operator with FiftyOne.

    Args:
        p: the FiftyOne plugin context.
    """
    p.register(SendToActiveLabel)
