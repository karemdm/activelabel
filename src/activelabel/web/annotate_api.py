"""HTTP API of annotation in external tools (see docs/ANNOTATION.md)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel import fiftyone_bridge
from activelabel.annotate import credentials, sessions
from activelabel.annotate import diff as diffing
from activelabel.annotate.base import TOOLS, Pulled, ToolError, ToolSettings
from activelabel.annotate.registry import get_tool
from activelabel.constants import DATASET_STEPS, PROGRESS_STEPS
from activelabel.errors import CodedError
from activelabel.jobs import Progress
from activelabel.labels import Label, annotations_lock, load_annotations
from activelabel.projects import Project
from activelabel.storage import now, read_json, write_json
from activelabel.sync.links import label_hash
from activelabel.sync.registry import get_remote
from activelabel.tasks import Task
from activelabel.tasks_store import TaskNotFound, TaskStore
from activelabel.web.errors import ApiError, bad_request, tool_failure
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
SESSION_TOOLS = ("fiftyone",)  # the tools that open edit sessions (CVAT is reviewed through the sync)
STATE = Depends(get_state)
log = logging.getLogger(__name__)


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
    task: int | None = None  # scope "task": the task whose active images are sent
    job: int | None = None  # scope "task": only this job of the task


# -- dataset access (replaced in tests, FiftyOne is slow to start) --------------
def prepare_dataset(
    project: Project, state: AppState, on_progress: Callable[[int, int], None] | None = None
) -> Any:
    """Refresh the project's FiftyOne dataset from the project and return it.

    Args:
        project: The project to refresh the dataset of.
        state: The shared state of the application.
        on_progress: Called with ``(images handled, images in all)`` while the dataset is refreshed.

    Returns:
        The FiftyOne dataset of the project.
    """
    manifest = state.manifest(project.id)
    annotations = load_annotations(state.store.dir(project.id))
    return fiftyone_bridge.sync_dataset(project, manifest, annotations, on_progress)


def open_dataset(project: Project, state: AppState) -> Any:
    """The project's FiftyOne dataset (None when it does not exist).

    Args:
        project: The project.
        state: The shared state of the application.

    Returns:
        The FiftyOne dataset, or ``None`` when the project has none.
    """
    return fiftyone_bridge.existing_dataset(state.store.dir(project.id), project.id)


# -- helpers ------------------------------------------------------------------
def _tool_name(tool: str) -> str:
    """The name of a known annotation tool, or HTTP 404.

    Args:
        tool: Name of the tool.

    Returns:
        The same name, when it is a known annotation tool.
    """
    if tool not in TOOLS:
        raise ApiError(404, "tool_unknown", f"Unknown annotation tool '{tool}'.", tool=tool)
    return tool


def _settings(state: AppState, tool: str) -> ToolSettings:
    """The saved settings of a tool, or HTTP 400 when it is not configured.

    Args:
        state: The shared state of the application.
        tool: Name of the tool.

    Returns:
        The saved settings of the tool, when it is configured.
    """
    settings = credentials.load_settings(state.settings.workspace, tool)
    if not credentials.is_configured(settings):
        raise ApiError(
            400, "tool_not_configured", f"Configure {tool} first (URL, user, password/token).", tool=tool
        )
    return settings


def _tool(name: str) -> Any:
    """The adapter of a tool, or an HTTP error when it cannot be created.

    Args:
        name: Name of the tool.

    Returns:
        The tool adapter.

    Raises:
        ApiError: 501 ``fiftyone_not_installed`` (the light install, #218), or 400 with the tool's code.
        HTTPException: 400 for a problem without a code.
    """
    try:
        return get_tool(name)
    except ToolError as exc:
        if exc.code:  # keep the code: the interface says it in its own language (it was English before)
            status = 501 if exc.code == "fiftyone_not_installed" else 400
            raise ApiError(status, exc.code, str(exc), **exc.params) from exc
        raise HTTPException(400, str(exc)) from exc


def _session_tool(name: str) -> Any:
    """The tool of an edit session.

    Args:
        name: Name of the tool of the session.

    Returns:
        The tool adapter.

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
    """The remote project adapter of a tool, or HTTP 400 when it cannot be created.

    Args:
        tool: Name of the remote tool.

    Returns:
        The remote project adapter of the tool.
    """
    try:
        return get_remote(tool)
    except ToolError as exc:
        raise ApiError(400, "tool_error", str(exc), tool=tool, reason=str(exc)) from exc


def _session(state: AppState, project_id: int, session_id: str) -> dict[str, Any]:
    """A session of the project, or HTTP 404.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        session_id: Id of the annotation session.

    Returns:
        The session record.
    """
    state.project(project_id)
    try:
        return sessions.get(state.store.dir(project_id), session_id)
    except sessions.SessionNotFound:
        raise ApiError(
            404, "session_not_found", "This annotation session does not exist (it may have been closed)."
        ) from None


def _local_hashes(project_dir: Path, image_ids: list[str]) -> dict[str, str]:
    """Hash of the local labels of each image (to notice later that they changed).

    Args:
        project_dir: Folder of the project.
        image_ids: The images to hash the labels of.

    Returns:
        Hash of the local labels of each image, by image id.
    """
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    return {i: label_hash(images.get(i, [])) for i in image_ids}


def _tell_tool(state: AppState, project_id: int, session: dict[str, Any], applied: bool) -> None:
    """Let the tool bring its copy of the session images in line with the project (apply or discard).

    Best effort: the project is already settled when this runs, so a failure is logged, not raised (the
    images then keep what the tool shows, as before).

    Args:
        state: the application state.
        project_id: id of the project.
        session: the session that was applied or discarded.
        applied: True after an apply (the images now equal the project), False after a discard (they go
            back to the project's labels).
    """
    if session["tool"] not in SESSION_TOOLS:
        return
    project = state.project(project_id)
    annotations = load_annotations(state.store.dir(project_id))
    labels = annotations.images if annotations else {}
    try:
        tool = _tool(session["tool"])
        hook = tool.applied if applied else tool.discarded
        hook(open_dataset(project, state), session["image_ids"], project, labels)
    except Exception:  # noqa: BLE001 - never fail an apply or a discard that already happened
        log.warning("Could not update the tool after the session %s", session["id"], exc_info=True)


def _pulled_file(state: AppState, project_id: int, session_id: str) -> Path:
    """Where the annotations read back for a session are kept.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        session_id: Id of the annotation session.

    Returns:
        Path of the file that holds the annotations read back for the session.
    """
    return sessions.session_dir(state.store.dir(project_id), session_id) / "pulled.json"


# -- tools --------------------------------------------------------------------
@router.get("/annotation-tools")
def list_tools(state: AppState = STATE) -> list[dict[str, Any]]:
    """Every tool with its settings (no secrets).

    Args:
        state: The shared state of the application.

    Returns:
        Every tool with its settings, without secrets.
    """
    return [credentials.public_view(credentials.load_settings(state.settings.workspace, t)) for t in TOOLS]


@router.put("/annotation-tools/{tool}")
def save_tool(tool: str, body: ToolBody, state: AppState = STATE) -> dict[str, Any]:
    """Save URL and user (and the secret, in the system keyring, when given).

    Args:
        tool: Name of the annotation tool.
        body: The URL, user and (optionally) the new secret.
        state: The shared state of the application.

    Returns:
        The settings of the tool, without the secret.
    """
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
    """Check the connection and login.

    Args:
        tool: Name of the annotation tool.
        state: The shared state of the application.

    Returns:
        ``{ok, message}``: whether the connection and login worked, and a readable text. For CVAT also
        ``code`` and ``params``: ``test_connected`` (with ``url`` and ``user``) or the code of the failure.
    """
    _tool_name(tool)
    settings = _settings(state, tool)
    try:
        checker = _remote(tool) if tool == "cvat" else _tool(tool)
        message = checker.check(settings)
    except ToolError as exc:
        failure = tool_failure(exc, tool)
        return {"ok": False, "message": str(exc), "code": failure.code, "params": failure.params}
    if tool == "cvat":
        return {
            "ok": True,
            "message": message,
            "code": "test_connected",
            "params": {"url": settings.url.rstrip("/"), "user": settings.username},
        }
    return {"ok": True, "message": message}


@router.delete("/annotation-tools/{tool}")
def forget_tool(tool: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget the settings and the secret of a tool.

    Args:
        tool: Name of the annotation tool.
        state: The shared state of the application.

    Returns:
        ``{removed: true}``.
    """
    _tool_name(tool)
    credentials.forget(state.settings.workspace, tool)
    return {"removed": True}


# -- sessions -----------------------------------------------------------------
@router.get("/projects/{project_id}/annotate")
def list_sessions(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Annotation sessions of a project, newest first.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{sessions}``: the sessions of the project, newest first.
    """
    state.project(project_id)
    return {"sessions": [sessions.public(s) for s in sessions.list_sessions(state.store.dir(project_id))]}


def _send_job(
    state: AppState,
    project: Project,
    tool: Any,
    settings: ToolSettings,
    body: SendBody,
    image_ids: list[str],
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``annotate_send`` job: refresh the FiftyOne dataset, send the images, record the session.

    Args:
        state: The shared state of the application.
        project: The project the images belong to.
        tool: The annotation tool that receives the images.
        settings: The saved settings (URL, user, secret) of that tool.
        body: The request (tool name and scope).
        image_ids: The images to send.
        progress: Where the job reports its progress.

    Returns:
        ``{session, links, step}``, plus ``kept_fiftyone_edits`` when some images keep edits made in FiftyOne.
        An open session with the same scope and images is reused, not duplicated.
    """
    project_dir = state.store.dir(project.id)
    progress.update(done=0, total=PROGRESS_STEPS, message="Preparing the FiftyOne dataset")
    dataset = prepare_dataset(
        project, state, progress.share(0, DATASET_STEPS, "Preparing the FiftyOne dataset")
    )
    progress.update(done=DATASET_STEPS + 5, message=f"Sending to {body.tool}")
    task = body.task if body.scope == "task" else None
    job = body.job if body.scope == "task" else None
    existing = sessions.find_open(project_dir, body.tool, body.scope, image_ids, task, job)
    session_id = existing["id"] if existing else sessions.new_id()  # the same images: the same session (#699)
    sent = tool.send(dataset, image_ids, project, settings, session_id)
    if existing:
        session = sessions.update(project_dir, session_id, links=sent.links, ref=sent.ref)
    else:
        session = sessions.create(
            project_dir,
            body.tool,
            body.scope,
            image_ids,
            sent.links,
            sent.ref,
            session_id=session_id,
            task=task,
            job=job,
        )
    result: dict[str, Any] = {
        "session": sessions.public(session),
        "links": sent.links,
        "step": "annotate",
    }
    if sent.kept:  # images that still show edits of an earlier session (not brought back yet)
        result["kept_fiftyone_edits"] = sent.kept
    progress.update(done=PROGRESS_STEPS)
    return result


@router.post("/projects/{project_id}/annotate")
def send(project_id: int, body: SendBody, state: AppState = STATE) -> dict[str, Any]:
    """Send images to a tool (job; result ``{session, links}``).

    Args:
        project_id: Id of the project.
        body: The tool, the scope and the images or issue to send.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.

    Raises:
        ApiError: ``fiftyone_keypoints_not_editable`` (400) for a keypoints project, which the FiftyOne
            App cannot edit, plus the usual errors of the tool, scope and images.
    """
    project = state.project(project_id)
    _tool_name(body.tool)
    tool = _session_tool(body.tool)
    if project.task is Task.KEYPOINTS:  # the FiftyOne App cannot edit keypoints
        raise ApiError(
            400,
            "fiftyone_keypoints_not_editable",
            "The FiftyOne App does not edit keypoints: edit them in CVAT.",
        )
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
    if body.scope == "task" and body.task is not None:
        try:
            TaskStore(state.store.dir(project_id)).get(body.task)
        except TaskNotFound:
            raise ApiError(404, "task_not_found", "This task does not exist.") from None
    try:
        image_ids = sessions.select_images(
            manifest, body.scope, body.ids, body.issue, task=body.task, job=body.job
        )
    except ValueError as exc:
        raise bad_request(exc) from exc
    job = partial(_send_job, state, project, tool, settings, body, image_ids)
    return state.jobs.submit(project_id, "annotate_send", job).as_dict()


def _pull_job(
    state: AppState,
    project: Project,
    tool: Any,
    settings: ToolSettings,
    session: dict[str, Any],
    session_id: str,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``annotate_pull`` job: read the annotations back, summarise and keep them for applying.

    Args:
        state: The shared state of the application.
        project: The project of the session.
        tool: The annotation tool to read from.
        settings: The saved settings (URL, user, secret) of that tool.
        session: The session record (tool, images and reference of the send).
        session_id: Id of the session.
        progress: Where the job reports its progress.

    Returns:
        ``{session, diff}``: the updated session and the summary of the changes.
    """
    project_dir = state.store.dir(project.id)
    image_ids: list[str] = session["image_ids"]
    progress.update(message=f"Reading annotations from {session['tool']}")
    pulled = tool.pull(open_dataset(project, state), image_ids, project, settings, session["ref"])
    annotations = load_annotations(project_dir)
    new_labels, conflicts = diffing.settle(pulled, annotations.images if annotations else {})
    problems = pulled.problems if isinstance(pulled, Pulled) else {}
    summary = diffing.diff(project_dir, image_ids, new_labels, project, conflicts, problems)
    write_json(
        _pulled_file(state, project.id, session_id),
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


@router.post("/projects/{project_id}/annotate/{session_id}/pull")
def pull(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Read the annotations back and summarise the changes (job; result ``{session, diff}``).

    Args:
        project_id: Id of the project.
        session_id: Id of the annotation session.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
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
    job = partial(_pull_job, state, project, tool, settings, session, session_id)
    return state.jobs.submit(project_id, "annotate_pull", job).as_dict()


@router.post("/projects/{project_id}/annotate/{session_id}/apply")
def apply_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Make the pulled annotations the ground truth of the session images.

    Args:
        project_id: Id of the project.
        session_id: Id of the annotation session.
        state: The shared state of the application.

    Returns:
        ``{applied}``: the counts of what was applied.
    """
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
        session = _session(state, project_id, session_id)  # again: a second click or a discard came first
        if session["status"] == sessions.APPLIED:
            raise ApiError(409, "already_applied", "These annotations were already applied.")
        if session["status"] == sessions.DISCARDED:
            raise ApiError(409, "already_discarded", "These annotations were already discarded.")
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
            state.store,
            project_id,
            state.manifest(project_id),
            session["image_ids"],
            new_labels,
            last_session={"tool": session["tool"], "session": session_id, "at": now()},
        )
        sessions.update(project_dir, session_id, status=sessions.APPLIED, applied=counts)
        _tell_tool(state, project_id, session, applied=True)
    return {"applied": counts}


@router.post("/projects/{project_id}/annotate/{session_id}/discard")
def discard_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw away the pulled annotations (the project does not change) and undo the edits in the tool.

    The check and the change of status happen under the lock of the annotations, like in apply, so of
    an apply and a discard of the same session exactly one wins and the other is refused (409).

    Args:
        project_id: Id of the project.
        session_id: Id of the annotation session.
        state: The shared state of the application.

    Returns:
        ``{discarded: true}``.
    """
    _session(state, project_id, session_id)
    project_dir = state.store.dir(project_id)
    with annotations_lock(project_dir):
        session = _session(state, project_id, session_id)
        if session["status"] == sessions.APPLIED:
            raise ApiError(
                409, "already_applied", "These annotations were already applied: there is nothing to discard."
            )
        sessions.update(project_dir, session_id, status=sessions.DISCARDED)
        _tell_tool(state, project_id, session, applied=False)
    return {"discarded": True}


@router.delete("/projects/{project_id}/annotate/{session_id}")
def close_session(project_id: int, session_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget a session.

    A session of a tool that has no edit sessions any more (an old workspace may still hold one) is only
    forgotten: no tool is contacted.

    Args:
        project_id: Id of the project.
        session_id: Id of the annotation session.
        state: The shared state of the application.

    Returns:
        ``{closed: true}``.
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
