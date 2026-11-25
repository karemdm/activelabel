"""HTTP API of synchronisation with CVAT (see docs/SYNC.md)."""

from __future__ import annotations

from functools import partial
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.annotate import credentials
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.jobs import Progress
from activelabel.sync import engine, links
from activelabel.sync.registry import get_remote
from activelabel.sync.remote import RemoteProject
from activelabel.web.errors import ApiError, tool_failure
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
SYNC_TOOLS = ("cvat",)


class PlanBody(BaseModel):
    """Body of POST /projects/{id}/sync/{tool}."""

    create: bool = False
    remote_id: str | int | None = None
    scope: str = "project"  # project, task or job (docs/TASKS.md)
    direction: str = "both"  # both, pull (the tool wins) or push (this project wins)
    task: int | None = None
    job: int | None = None


class ApplyBody(BaseModel):
    """Body of POST /projects/{id}/sync/{tool}/{plan_id}/apply."""

    resolutions: dict[str, str] = {}
    default: str | None = None
    delete_tasks: bool = False  # delete in the tool the tasks deleted in ActiveLabel


def _tool(tool: str) -> str:
    """The name of a tool that can be synchronised, or HTTP 404.

    Args:
        tool: Name of the tool.

    Returns:
        The same name, when the tool can be synchronised.
    """
    if tool not in SYNC_TOOLS:
        raise ApiError(404, "tool_unknown", f"Unknown tool '{tool}' (use cvat).", tool=tool)
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


def _remote(tool: str) -> RemoteProject:
    """The remote project adapter of a tool, or HTTP 400 when it cannot be created.

    Args:
        tool: Name of the tool.

    Returns:
        The remote project adapter of the tool.
    """
    try:
        return get_remote(tool)
    except ToolError as exc:
        raise tool_failure(exc, tool) from exc


@router.get("/cvat/audit")
def cvat_audit(state: AppState = STATE) -> dict[str, Any]:
    """Read-only list of the CVAT tasks per project of the configured server (nothing is changed).

    The tasks in no project come grouped by name pattern; a project linked to one of ours says which.

    Args:
        state: The shared state of the application.

    Returns:
        ``{url, ...}``: the CVAT tasks per project and the loose ones, grouped.
    """
    settings = _settings(state, "cvat")
    try:
        found = _remote("cvat").audit(settings)
    except ToolError as exc:
        raise tool_failure(exc, "cvat") from exc
    linked: dict[str, dict[str, Any]] = {}
    for project in state.store.list():
        link = links.get_link(state.store.dir(project.id), "cvat")
        if link and str(link.get("url", "")).rstrip("/") == settings.url.rstrip("/"):
            linked[str(link["remote_id"])] = {"id": project.id, "name": project.name}
    for entry in found["projects"]:
        entry["linked_to"] = linked.get(entry["id"])
    return {"url": settings.url.rstrip("/"), **found}


@router.get("/projects/{project_id}/links")
def list_links(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Links of a project with remote projects.

    A link written for a tool that cannot be synchronised any more (an old workspace) is listed with
    ``supported: false``: it can only be unlinked.

    Args:
        project_id: Id of the project.
        state: The shared state of the application.

    Returns:
        ``{links}``.
    """
    state.project(project_id)
    found = links.load_links(state.store.dir(project_id))
    return {
        "links": [
            {**links.public_view(tool, link), "supported": tool in SYNC_TOOLS} for tool, link in found.items()
        ]
    }


@router.delete("/projects/{project_id}/links/{tool}")
def unlink(project_id: int, tool: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget the link with a tool (the remote project is not touched). Refused (409) while a job runs.

    Any link can be forgotten, also one of a tool that cannot be synchronised any more.

    Args:
        project_id: Id of the project.
        tool: Name of the remote tool (``cvat``).
        state: The shared state of the application.

    Returns:
        ``{unlinked: true}``.
    """
    state.project(project_id)
    state.busy(project_id)  # a running comparison or sync could write the link again
    if not links.remove_link(state.store.dir(project_id), tool):
        raise ApiError(404, "not_linked", f"This project is not linked to {tool}.", tool=tool)
    return {"unlinked": True}


def _plan_job(
    state: AppState,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    body: PlanBody,
    remote_id: str | None,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``sync_plan`` job: compare the project with the remote one and save the comparison.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        tool: Name of the remote tool.
        settings: The saved settings (URL, user, secret) of the tool.
        remote: The remote project adapter of the tool.
        body: The request (create, scope, task, job and direction).
        remote_id: Id of the remote project to link to, when given.
        progress: Where the job reports its progress.

    Returns:
        The saved comparison (id and summary of the differences).
    """
    return engine.make_plan(
        state.store,
        project_id,
        tool,
        settings,
        remote,
        progress,
        body.create,
        remote_id,
        scope=body.scope,
        task_id=body.task,
        job=body.job,
        direction=body.direction,
    )


@router.post("/projects/{project_id}/sync/{tool}")
def plan(project_id: int, tool: str, body: PlanBody | None = None, state: AppState = STATE) -> dict[str, Any]:
    """Compare the project with its remote project (job ``sync_plan``; result: the plan).

    Args:
        project_id: Id of the project.
        tool: Name of the remote tool (``cvat``).
        body: Direction, scope, task, job and whether to create the remote project.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    state.project(project_id)
    _tool(tool)
    body = body or PlanBody()
    if body.direction not in engine.DIRECTIONS:
        raise ApiError(400, "direction_invalid", "The direction must be both, pull or push.")
    if body.scope not in engine.SCOPES:
        raise ApiError(400, "scope_invalid", "The scope must be project, task or job.")
    if body.scope != "project" and body.task is None:
        raise ApiError(400, "task_required", "Choose the task.")
    if body.scope == "job" and body.job is None:
        raise ApiError(400, "job_required", "Choose the job.")
    state.busy(project_id)  # the input is checked first: a wrong request is wrong whatever the state
    settings = _settings(state, tool)
    remote = _remote(tool)
    remote_id = str(body.remote_id) if body.remote_id not in (None, "") else None
    job = partial(_plan_job, state, project_id, tool, settings, remote, body, remote_id)
    return state.jobs.submit(project_id, "sync_plan", job).as_dict()


def _apply_job(
    state: AppState,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    plan_id: str,
    body: ApplyBody,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``sync_apply`` job: carry out a saved comparison.

    Args:
        state: The shared state of the application.
        project_id: Id of the project.
        tool: Name of the remote tool.
        settings: The saved settings (URL, user, secret) of the tool.
        remote: The remote project adapter of the tool.
        plan_id: Id of the saved comparison.
        body: The request (resolutions, default choice and whether to delete tasks).
        progress: Where the job reports its progress.

    Returns:
        The counts of what was applied.
    """
    return engine.apply_plan(
        state.store,
        project_id,
        tool,
        settings,
        remote,
        plan_id,
        body.resolutions,
        body.default,
        progress,
        delete_tasks=body.delete_tasks,
    )


@router.delete("/projects/{project_id}/sync/{tool}/{plan_id}")
def discard(project_id: int, tool: str, plan_id: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw a comparison away: it cannot be applied afterwards (404), from any tab.

    Args:
        project_id: Id of the project.
        tool: The remote tool (``cvat``).
        plan_id: Id of the comparison.
        state: The shared state of the application.

    Returns:
        ``{discarded: True}``.

    Raises:
        ApiError: 404 ``plan_not_found`` when the comparison does not exist.
    """
    state.project(project_id)
    _tool(tool)
    if not engine.discard_plan(state.store.dir(project_id), plan_id):
        raise ApiError(
            404, "plan_not_found", "This comparison does not exist (it may be old). Synchronise again."
        )
    return {"discarded": True}


@router.post("/projects/{project_id}/sync/{tool}/{plan_id}/apply")
def apply(
    project_id: int, tool: str, plan_id: str, body: ApplyBody | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Carry out a comparison (job ``sync_apply``; result: counts).

    Args:
        project_id: Id of the project.
        tool: Name of the remote tool (``cvat``).
        plan_id: Id of the saved comparison.
        body: The resolutions, the default choice and whether to delete tasks.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    state.project(project_id)
    _tool(tool)
    if engine.load_plan(state.store.dir(project_id), plan_id) is None:
        raise ApiError(
            404, "plan_not_found", "This comparison does not exist (it may be old). Synchronise again."
        )
    state.busy(project_id)
    settings = _settings(state, tool)
    remote = _remote(tool)
    body = body or ApplyBody()
    if body.default not in (None, "", "local", "remote") or any(
        v not in engine.CHOICES for v in body.resolutions.values()
    ):
        raise ApiError(400, "resolution_invalid", "Resolutions and default must be 'local' or 'remote'.")
    job = partial(_apply_job, state, project_id, tool, settings, remote, plan_id, body)
    return state.jobs.submit(project_id, "sync_apply", job).as_dict()
