"""HTTP API of annotation in external tools (see docs/ANNOTATION.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.annotate import credentials, sessions
from activelabel.annotate import diff as diffing
from activelabel.annotate.base import TOOLS, ToolError, ToolSettings
from activelabel.annotate.registry import get_tool
from activelabel.errors import CodedError
from activelabel.jobs import Progress
from activelabel.labels import Label, annotations_lock, load_annotations
from activelabel.projects import Project
from activelabel.storage import read_json, write_json
from activelabel.sync.links import label_hash
from activelabel.sync.registry import get_remote
from activelabel.web.errors import ApiError, bad_request
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
SESSION_TOOLS = ("fiftyone",)  # the tools that open edit sessions (CVAT is reviewed through the sync)
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


# -- dataset access (replaced in tests, FiftyOne is slow to start) --------------
def prepare_dataset(project: Project, state: AppState) -> Any:
    """Refresh the project's FiftyOne dataset from the project and return it."""
    from activelabel import fiftyone_bridge

    manifest = state.manifest(project.id)
    return fiftyone_bridge.sync_dataset(project, manifest, load_annotations(state.store.dir(project.id)))


def open_dataset(project: Project, state: AppState) -> Any:
    """The project's FiftyOne dataset (None when it does not exist)."""
    from activelabel import fiftyone_bridge

    return fiftyone_bridge.existing_dataset(state.store.dir(project.id), project.id)


# -- helpers ------------------------------------------------------------------
def _tool_name(tool: str) -> str:
    if tool not in TOOLS:
        raise ApiError(404, "tool_unknown", f"Unknown annotation tool '{tool}'.", tool=tool)
    return tool


def _settings(state: AppState, tool: str) -> ToolSettings:
    settings = credentials.load_settings(state.settings.workspace, tool)
    if not credentials.is_configured(settings):
        raise ApiError(
            400, "tool_not_configured", f"Configure {tool} first (URL, user, password/token).", tool=tool
        )
    return settings


def _tool(name: str) -> Any:
    try:
        return get_tool(name)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc


def _session_tool(name: str) -> Any:
    """The tool of an edit session.

    Raises:
        ApiError: ``tool_unsupported`` (400) for a session of a tool that has no edit sessions any more
            (CVAT: workspaces made before that may still hold such sessions; they can only be closed).
    """
    if name not in SESSION_TOOLS:
        raise ApiError(
            400,
            "tool_unsupported",
            f"Sessions of '{name}' are no longer supported: close this session.",
            tool=name,
        )
    return _tool(name)


def _remote(tool: str) -> Any:
    try:
        return get_remote(tool)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc


def _session(state: AppState, project_id: int, session_id: str) -> dict[str, Any]:
    state.project(project_id)
    try:
        return sessions.get(state.store.dir(project_id), session_id)
    except sessions.SessionNotFound:
        raise ApiError(
            404, "session_not_found", "This annotation session does not exist (it may have been closed)."
        ) from None


def _local_hashes(project_dir: Path, image_ids: list[str]) -> dict[str, str]:
    """Hash of the local labels of each image (to notice later that they changed)."""
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    return {i: label_hash(images.get(i, [])) for i in image_ids}


def _pulled_file(state: AppState, project_id: int, session_id: str) -> Path:
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
        raise ApiError(400, "url_required", "Give the URL of the tool.")
    try:
        settings = credentials.save_settings(
            state.settings.workspace, tool, body.url, body.username, body.secret or None
        )
    except (ToolError, CodedError) as exc:
        raise bad_request(exc) from exc
    return credentials.public_view(settings)


@router.post("/annotation-tools/{tool}/test")
def test_tool(tool: str, state: AppState = STATE) -> dict[str, Any]:
    """Check the connection and login."""
    _tool_name(tool)
    settings = _settings(state, tool)
    try:
        checker = _remote(tool) if tool == "cvat" else _tool(tool)
        return {"ok": True, "message": checker.check(settings)}
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
    _tool_name(body.tool)
    tool = _session_tool(body.tool)
    if body.scope not in sessions.SCOPES:
        raise ApiError(
            400,
            "scope_unknown",
            f"Unknown scope '{body.scope}' (use: {', '.join(sessions.SCOPES)}).",
            scope=body.scope,
        )
    manifest = state.manifest_with_images(project_id)
    state.busy(project_id)
    settings = _settings(state, body.tool)
    try:
        image_ids = sessions.select_images(manifest, body.scope, body.ids, body.issue)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_dir = state.store.dir(project_id)

    def run(progress: Progress) -> dict[str, Any]:
        progress.update(message="Preparing the FiftyOne dataset")
        kept = len(sessions.protected_images(project_dir) & set(image_ids))  # their edits stay in FiftyOne
        dataset = prepare_dataset(project, state)
        progress.update(message=f"Sending to {body.tool}")
        session_id = sessions.new_id()
        sent = tool.send(dataset, image_ids, project, settings, session_id)
        session = sessions.create(
            project_dir, body.tool, body.scope, image_ids, sent.links, sent.ref, session_id=session_id
        )
        result: dict[str, Any] = {
            "session": sessions.public(session),
            "links": sent.links,
            "step": "annotate",
        }
        if kept:
            result["kept_fiftyone_edits"] = kept
        return result

    return state.jobs.submit(project_id, "annotate_send", run).as_dict()


@router.post("/projects/{project_id}/annotate/{session_id}/pull")
def pull(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Read the annotations back and summarise the changes (job; result ``{session, diff}``)."""
    session = _session(state, project_id, session_id)
    if session["status"] in (sessions.APPLIED, sessions.DISCARDED):
        raise ApiError(
            409,
            "session_finished",
            f"This session was already {session['status']}. Start a new one.",
            status=session["status"],
        )
    state.busy(project_id)
    project = state.project(project_id)
    tool = _session_tool(session["tool"])
    settings = _settings(state, session["tool"])
    project_dir = state.store.dir(project_id)
    image_ids: list[str] = session["image_ids"]

    def run(progress: Progress) -> dict[str, Any]:
        progress.update(message=f"Reading annotations from {session['tool']}")
        new_labels = tool.pull(open_dataset(project, state), image_ids, project, settings, session["ref"])
        summary = diffing.diff(project_dir, image_ids, new_labels, project)
        write_json(
            _pulled_file(state, project_id, session_id),
            {i: [lb.as_dict() for lb in labels] for i, labels in new_labels.items()},
        )
        updated = sessions.update(
            project_dir,
            session_id,
            status=sessions.PULLED,
            diff=summary,
            base=_local_hashes(project_dir, image_ids),
        )
        return {"session": sessions.public(updated), "diff": summary}

    return state.jobs.submit(project_id, "annotate_pull", run).as_dict()


@router.post("/projects/{project_id}/annotate/{session_id}/apply")
def apply_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Make the pulled annotations the ground truth of the session images."""
    session = _session(state, project_id, session_id)
    state.busy(project_id)
    try:
        data = read_json(_pulled_file(state, project_id, session_id))
    except ValueError:  # damaged file: the annotations have to be brought back again
        data = None
    if session["status"] != sessions.PULLED or data is None:
        raise ApiError(
            400, "pull_first", "Bring the annotations back first (Pull), then review and apply them."
        )
    new_labels = {i: [Label.from_dict(x) for x in items] for i, items in data.items()}
    project_dir = state.store.dir(project_id)
    with annotations_lock(project_dir):  # the check and the write see the same labels
        session = _session(state, project_id, session_id)  # again: a second click may have applied it
        if session["status"] == sessions.APPLIED:
            raise ApiError(409, "already_applied", "These annotations were already applied.")
        base: dict[str, str] = session.get("base") or {}
        current = _local_hashes(project_dir, session["image_ids"])
        changed = sum(1 for i, h in current.items() if base.get(i) != h)
        if changed:
            raise ApiError(
                409,
                "annotations_changed",
                f"The annotations of {changed} images changed since you brought them back: "
                "bring them back again.",
                count=changed,
            )
        counts = diffing.apply(
            state.store, project_id, state.manifest(project_id), session["image_ids"], new_labels
        )
        sessions.update(project_dir, session_id, status=sessions.APPLIED, applied=counts)
    return {"applied": counts}


@router.post("/projects/{project_id}/annotate/{session_id}/discard")
def discard_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw away the pulled annotations (the project does not change)."""
    session = _session(state, project_id, session_id)
    if session["status"] == sessions.APPLIED:
        raise ApiError(
            409, "already_applied", "These annotations were already applied: there is nothing to discard."
        )
    sessions.update(state.store.dir(project_id), session_id, status=sessions.DISCARDED)
    return {"discarded": True}


@router.delete("/projects/{project_id}/annotate/{session_id}")
def close_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget a session.

    A session of a tool that has no edit sessions any more (an old workspace may still hold one) is only
    forgotten: no tool is contacted.
    """
    session = _session(state, project_id, session_id)
    state.busy(project_id)
    if session["tool"] in SESSION_TOOLS:
        settings = credentials.load_settings(state.settings.workspace, session["tool"])
        try:
            _tool(session["tool"]).close(
                open_dataset(state.project(project_id), state), settings, session.get("ref", {})
            )
        except ToolError as exc:
            raise HTTPException(400, str(exc)) from exc
    sessions.close(state.store.dir(project_id), session_id)
    return {"closed": True}
