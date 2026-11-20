"""HTTP API of annotation in external tools (see docs/ANNOTATION.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.annotate import credentials, sessions
from activelabel.annotate import diff as diffing
from activelabel.annotate.base import TOOLS, ToolError, ToolSettings
from activelabel.annotate.registry import get_tool
from activelabel.jobs import Progress
from activelabel.labels import Label, load_annotations
from activelabel.projects import Project
from activelabel.storage import read_json, write_json
from activelabel.sync import engine as sync_engine
from activelabel.sync.registry import get_remote
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
SYNC_TOOLS = ("cvat", "labelstudio")
STATE = Depends(get_state)


# -- bodies -------------------------------------------------------------------
class ToolBody(BaseModel):
    """Body of PUT /annotation-tools/{tool}. A missing ``secret`` keeps the stored one."""

    url: str = ""
    username: str = ""
    secret: str | None = None


class SendBody(BaseModel):
    """Body of POST /projects/{id}/annotate."""

    tool: str
    scope: str = "all"
    ids: list[str] = []
    issue: str = ""


class CloseBody(BaseModel):
    """Body of DELETE /projects/{id}/annotate/{session}."""

    delete_remote: bool = False


# -- dataset access (replaced in tests, FiftyOne is slow to start) --------------
def prepare_dataset(project: Project, state: AppState) -> Any:
    """Refresh the project's FiftyOne dataset from the project and return it."""
    from activelabel import fiftyone_bridge

    manifest = state.manifest(project.id)
    return fiftyone_bridge.sync_dataset(project, manifest, load_annotations(state.store.dir(project.id)))


def open_dataset(project: Project) -> Any:
    """The project's FiftyOne dataset (None when it does not exist)."""
    import fiftyone as fo

    from activelabel import fiftyone_bridge

    name = fiftyone_bridge.dataset_name(project.id)
    return fo.load_dataset(name) if fo.dataset_exists(name) else None


# -- helpers ------------------------------------------------------------------
def _tool_name(tool: str) -> str:
    if tool not in TOOLS:
        raise HTTPException(404, f"Unknown annotation tool '{tool}'.")
    return tool


def _settings(state: AppState, tool: str) -> ToolSettings:
    settings = credentials.load_settings(state.settings.workspace, tool)
    if not credentials.is_configured(settings):
        raise HTTPException(400, f"Configure {tool} first (URL, user, password/token).")
    return settings


def _tool(name: str) -> Any:
    try:
        return get_tool(name)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc


def _session(state: AppState, project_id: int, session_id: str) -> dict[str, Any]:
    state.project(project_id)
    try:
        return sessions.get(state.store.dir(project_id), session_id)
    except sessions.SessionNotFound:
        raise HTTPException(
            404, "This annotation session does not exist (it may have been closed)."
        ) from None


def _pulled_file(state: AppState, project_id: int, session_id: str) -> Any:
    return sessions.session_dir(state.store.dir(project_id), session_id) / "pulled.json"


# -- tools --------------------------------------------------------------------
@router.get("/annotation-tools")
def list_tools(state: AppState = STATE) -> list[dict[str, Any]]:
    """Every tool with its settings (no secrets)."""
    return [credentials.public_view(credentials.load_settings(state.settings.workspace, t)) for t in TOOLS]


@router.put("/annotation-tools/{tool}")
def save_tool(tool: str, body: ToolBody, state: AppState = STATE) -> dict[str, Any]:
    """Save URL and user (and the secret, in the system keyring, when given)."""
    _tool_name(tool)
    if tool != "fiftyone" and not body.url.strip():
        raise HTTPException(400, "Give the URL of the tool.")
    try:
        settings = credentials.save_settings(
            state.settings.workspace, tool, body.url, body.username, body.secret or None
        )
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc
    return credentials.public_view(settings)


@router.post("/annotation-tools/{tool}/test")
def test_tool(tool: str, state: AppState = STATE) -> dict[str, Any]:
    """Check the connection and login."""
    _tool_name(tool)
    settings = _settings(state, tool)
    try:
        return {"ok": True, "message": _tool(tool).check(settings)}
    except ToolError as exc:
        return {"ok": False, "message": str(exc)}


@router.delete("/annotation-tools/{tool}")
def forget_tool(tool: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget the settings and the secret of a tool."""
    _tool_name(tool)
    credentials.forget(state.settings.workspace, tool)
    return {"removed": True}


# -- sessions -----------------------------------------------------------------
@router.get("/projects/{project_id}/annotate")
def list_sessions(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Annotation sessions of a project, newest first."""
    state.project(project_id)
    return {"sessions": [sessions.public(s) for s in sessions.list_sessions(state.store.dir(project_id))]}


@router.post("/projects/{project_id}/annotate")
def send(project_id: int, body: SendBody, state: AppState = STATE) -> dict[str, Any]:
    """Send images to a tool (job; result ``{session, links}``)."""
    project = state.project(project_id)
    manifest = state.manifest_with_images(project_id)
    state.busy(project_id)
    tool = _tool(_tool_name(body.tool))
    settings = _settings(state, body.tool)
    try:
        image_ids = sessions.select_images(manifest, body.scope, body.ids, body.issue)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_dir = state.store.dir(project_id)
    remote = None
    if body.tool in SYNC_TOOLS:
        try:
            remote = get_remote(body.tool)
            detail = sync_engine.existing_remote(state.store, project, body.tool, settings, remote)
        except ToolError as exc:
            raise HTTPException(400, str(exc)) from exc
        if detail is not None:
            raise HTTPException(409, detail)

    def run(progress: Progress) -> dict[str, Any]:
        progress.update(message="Preparing the FiftyOne dataset")
        dataset = prepare_dataset(project, state)
        progress.update(message=f"Sending to {body.tool}")
        session_id = sessions.new_id()
        sent = tool.send(dataset, image_ids, project, settings, session_id)
        session = sessions.create(
            project_dir, body.tool, body.scope, image_ids, sent.links, sent.ref, session_id=session_id
        )
        if remote is not None:
            try:  # the link is a convenience: a failure here must not undo a send that worked
                sync_engine.link_after_send(
                    state.store, project, body.tool, settings, remote, sent.ref, image_ids
                )
            except Exception:  # noqa: BLE001, S110
                pass
        return {"session": sessions.public(session), "links": sent.links, "step": "annotate"}

    return state.jobs.submit(project_id, "annotate_send", run).as_dict()


@router.post("/projects/{project_id}/annotate/{session_id}/pull")
def pull(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Read the annotations back and summarise the changes (job; result ``{session, diff}``)."""
    session = _session(state, project_id, session_id)
    if session["status"] in (sessions.APPLIED, sessions.DISCARDED):
        raise HTTPException(409, f"This session was already {session['status']}. Start a new one.")
    state.busy(project_id)
    project = state.project(project_id)
    tool = _tool(session["tool"])
    settings = _settings(state, session["tool"])
    project_dir = state.store.dir(project_id)
    image_ids: list[str] = session["image_ids"]

    def run(progress: Progress) -> dict[str, Any]:
        progress.update(message=f"Reading annotations from {session['tool']}")
        new_labels = tool.pull(open_dataset(project), image_ids, project, settings, session["ref"])
        summary = diffing.diff(project_dir, image_ids, new_labels, project)
        write_json(
            _pulled_file(state, project_id, session_id),
            {i: [lb.as_dict() for lb in labels] for i, labels in new_labels.items()},
        )
        updated = sessions.update(project_dir, session_id, status=sessions.PULLED, diff=summary)
        return {"session": sessions.public(updated), "diff": summary}

    return state.jobs.submit(project_id, "annotate_pull", run).as_dict()


@router.post("/projects/{project_id}/annotate/{session_id}/apply")
def apply_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Make the pulled annotations the ground truth of the session images."""
    session = _session(state, project_id, session_id)
    state.busy(project_id)
    data = read_json(_pulled_file(state, project_id, session_id))
    if session["status"] != sessions.PULLED or data is None:
        raise HTTPException(400, "Bring the annotations back first (Pull), then review and apply them.")
    new_labels = {i: [Label.from_dict(x) for x in items] for i, items in data.items()}
    counts = diffing.apply(
        state.store, project_id, state.manifest(project_id), session["image_ids"], new_labels
    )
    sessions.update(state.store.dir(project_id), session_id, status=sessions.APPLIED, applied=counts)
    return {"applied": counts}


@router.post("/projects/{project_id}/annotate/{session_id}/discard")
def discard_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw away the pulled annotations (the project does not change)."""
    _session(state, project_id, session_id)
    sessions.update(state.store.dir(project_id), session_id, status=sessions.DISCARDED)
    return {"discarded": True}


@router.delete("/projects/{project_id}/annotate/{session_id}")
def close_session(
    project_id: int, session_id: str, body: CloseBody | None = None, state: AppState = STATE
) -> dict[str, bool]:
    """Forget a session (and delete the remote tasks when asked)."""
    session = _session(state, project_id, session_id)
    state.busy(project_id)
    project = state.project(project_id)
    tool = _tool(session["tool"])
    settings = credentials.load_settings(state.settings.workspace, session["tool"])
    delete_remote = bool(body and body.delete_remote)
    try:
        tool.close(open_dataset(project), settings, session.get("ref", {}), delete_remote)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc
    sessions.close(state.store.dir(project_id), session_id)
    return {"closed": True}
