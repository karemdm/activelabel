"""HTTP API of importing a project from CVAT (see docs/IMPORT.md)."""

from __future__ import annotations

from functools import partial
from types import ModuleType
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from activelabel.annotate import credentials
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.importing import cvat_import
from activelabel.importing.common import RemoteNotFound
from activelabel.jobs import Progress
from activelabel.sync import engine as sync_engine
from activelabel.sync.registry import get_remote
from activelabel.tasks import Task
from activelabel.web.errors import ApiError, need_fiftyone, tool_failure
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
IMPORTERS: dict[str, ModuleType] = {"cvat": cvat_import}
KINDS = {"cvat": ("project", "task")}


class ImportBody(BaseModel):
    """Body of POST /import."""

    tool: str
    remote_id: str | int
    kind: str = "project"
    name: str = ""
    description: str = ""
    task: Task = Task.DETECTION


def _importer(tool: str) -> ModuleType:
    """The importer of a tool, or HTTP 404.

    Args:
        tool: Name of the tool.

    Returns:
        The importer module of the tool.
    """
    if tool not in IMPORTERS:
        raise ApiError(404, "tool_unknown", f"Unknown tool '{tool}' (use cvat).", tool=tool)
    return IMPORTERS[tool]


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
        raise ApiError(400, "tool_not_configured", f"Configure {tool} first", tool=tool)
    return settings


def _marked(
    importer: ModuleType, state: AppState, settings: ToolSettings, search: str
) -> list[dict[str, Any]]:
    """The importable items, each project that already has a local project linked to it marked.

    Args:
        importer: The importer module of the tool.
        state: The shared state of the application.
        settings: The saved settings of the tool.
        search: Text to search in the names.

    Returns:
        The items of the importer; a project with a local one gets ``linked_to: {id, name}``.
    """
    items: list[dict[str, Any]] = importer.list_remote(settings, search)
    if not hasattr(importer, "linked_projects"):
        return items
    known = importer.linked_projects(state.store, settings)
    for item in items:
        project = known.get(str(item["id"])) if item.get("kind") == "project" else None
        if project is not None:
            item["linked_to"] = {"id": project.id, "name": project.name}
    return items


@router.get("/import/{tool}/projects")
def list_remote(tool: str, search: str = "", state: AppState = STATE) -> dict[str, Any]:
    """Projects (and CVAT tasks outside projects) that can be imported, newest first.

    Args:
        tool: Name of the remote tool (``cvat``).
        search: Text to search in the names.
        state: The shared state of the application.

    Returns:
        ``{items}``: what can be imported.
    """
    importer = _importer(tool)
    settings = _settings(state, tool)
    try:
        return {"items": _marked(importer, state, settings, search)}
    except ToolError as exc:
        raise tool_failure(exc, tool) from exc


def _import_job(
    importer: ModuleType,
    state: AppState,
    settings: ToolSettings,
    body: ImportBody,
    remote_id: str,
    progress: Progress,
) -> dict[str, Any]:
    """Body of the ``import_project`` job: import the remote project, then link it to the tool.

    Args:
        importer: The importer module of the tool.
        state: The shared state of the application.
        settings: The saved settings (URL, user, secret) of the tool.
        body: The request (tool, kind, name, description and task type).
        remote_id: Id of the remote project or task, as text.
        progress: Where the job reports its progress.

    Returns:
        The import result, with ``linked`` telling whether the new project is linked to the remote one.
    """
    result: dict[str, Any] = importer.import_project(
        state.store,
        settings,
        state.settings.workspace,
        kind=body.kind,
        remote_id=remote_id,
        name=body.name,
        description=body.description,
        task=body.task,
        progress=progress,
    )
    if "linked" in result:  # the importer linked it itself (CVAT project: tasks and jobs)
        return result
    try:  # the link is a convenience: a failure here must not undo an import that worked
        linked = sync_engine.link_after_import(
            state.store, int(result["project_id"]), body.tool, settings, get_remote(body.tool)
        )
    except Exception:  # noqa: BLE001
        linked = False
    result["linked"] = linked
    return result


@router.post("/import")
def start_import(body: ImportBody, state: AppState = STATE) -> dict[str, Any]:
    """Import a remote project as a new project (job ``import_project``, project id 0).

    Args:
        body: The request body.
        state: The shared state of the application.

    Returns:
        The queued job (id and state); its result is read from the job when it finishes.
    """
    importer = _importer(body.tool)
    need_fiftyone()  # the annotations are read with FiftyOne's CVAT reader
    if body.kind not in KINDS[body.tool]:
        raise ApiError(
            400,
            "import_kind_invalid",
            f"kind must be one of: {', '.join(KINDS[body.tool])}.",
            kinds=list(KINDS[body.tool]),
        )
    settings = _settings(state, body.tool)
    remote_id = str(body.remote_id)
    guard = getattr(importer, "already_imported", None)  # a project that already has a local one linked
    if guard is not None and body.kind == "project" and (again := guard(state.store, settings, remote_id)):
        raise tool_failure(again, body.tool, 409) from again
    try:
        importer.check_remote(settings, body.kind, remote_id)
    except RemoteNotFound as exc:
        raise tool_failure(exc, body.tool, 404) from exc
    except ToolError as exc:
        raise tool_failure(exc, body.tool) from exc
    job = partial(_import_job, importer, state, settings, body, remote_id)
    return state.jobs.submit(0, "import_project", job).as_dict()
