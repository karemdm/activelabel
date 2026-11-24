"""HTTP API of synchronisation with CVAT and Label Studio (see docs/SYNC.md)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from activelabel.annotate import credentials
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.jobs import Progress
from activelabel.sync import engine, links
from activelabel.sync.registry import get_remote
from activelabel.sync.remote import RemoteProject
from activelabel.web.errors import ApiError
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
SYNC_TOOLS = ("cvat", "labelstudio")


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
    if tool not in SYNC_TOOLS:
        raise ApiError(404, "tool_unknown", f"Unknown tool '{tool}' (use cvat or labelstudio).", tool=tool)
    return tool


def _settings(state: AppState, tool: str) -> ToolSettings:
    settings = credentials.load_settings(state.settings.workspace, tool)
    if not credentials.is_configured(settings):
        raise ApiError(
            400, "tool_not_configured", f"Configure {tool} first (URL, user, password/token).", tool=tool
        )
    return settings


def _remote(tool: str) -> RemoteProject:
    try:
        return get_remote(tool)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/projects/{project_id}/links")
def list_links(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Links of a project with remote projects."""
    state.project(project_id)
    found = links.load_links(state.store.dir(project_id))
    return {"links": [links.public_view(tool, link) for tool, link in found.items()]}


@router.delete("/projects/{project_id}/links/{tool}")
def unlink(project_id: int, tool: str, state: AppState = STATE) -> dict[str, bool]:
    """Forget the link with a tool (the remote project is not touched). Refused (409) while a job runs."""
    state.project(project_id)
    _tool(tool)
    state.busy(project_id)  # a running comparison or sync could write the link again
    if not links.remove_link(state.store.dir(project_id), tool):
        raise ApiError(404, "not_linked", f"This project is not linked to {tool}.", tool=tool)
    return {"unlinked": True}


@router.post("/projects/{project_id}/sync/{tool}")
def plan(project_id: int, tool: str, body: PlanBody | None = None, state: AppState = STATE) -> dict[str, Any]:
    """Compare the project with its remote project (job ``sync_plan``; result: the plan)."""
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
    store = state.store

    def run(progress: Progress) -> dict[str, Any]:
        return engine.make_plan(
            store,
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

    return state.jobs.submit(project_id, "sync_plan", run).as_dict()


@router.post("/projects/{project_id}/sync/{tool}/{plan_id}/apply")
def apply(
    project_id: int, tool: str, plan_id: str, body: ApplyBody | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Carry out a comparison (job ``sync_apply``; result: counts)."""
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
    store = state.store

    def run(progress: Progress) -> dict[str, Any]:
        return engine.apply_plan(
            store,
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

    return state.jobs.submit(project_id, "sync_apply", run).as_dict()
