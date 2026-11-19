"""Import a Label Studio project as an ActiveLabel project.

Everything goes through the REST API with the login of the annotation tool (personal access token
exchanged for a Bearer token, see :mod:`activelabel.annotate.labelstudio_tool`): the project list,
the JSON export, and the image of every task (uploaded files and local files through the server,
absolute URLs directly and without the token). The export is parsed with the existing Label Studio
parser against the new image records, so annotations are matched by file name. The token never
appears in logs, files or messages.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.annotate.labelstudio_tool import _Client
from activelabel.annotations.context import ParseContext
from activelabel.annotations.labelstudio import _task_image, image_candidates, parse_labelstudio
from activelabel.annotations.models import Problem
from activelabel.images.ingest import IMAGE_EXTENSIONS, ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.importing import common
from activelabel.importing.common import MAX_ITEMS, Origin, RemoteNotFound
from activelabel.jobs import Progress
from activelabel.projects import ProjectStore
from activelabel.tasks import Task

PAGE = 100
EXPORT_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 120
CHUNK = 1024 * 1024
LABEL_TAGS = ("Labels", "RectangleLabels", "PolygonLabels", "KeyPointLabels", "BrushLabels", "EllipseLabels")
FATAL = ("refused the token", "Cannot reach")  # failures that make every image fail: stop


def label_names(config: str, for_import: bool = False) -> list[str]:
    """Class names of a labeling config (``Label`` and ``Choice`` values).

    Args:
        config: labeling config XML (invalid XML gives no names).
        for_import: leave out keypoint names (they are not classes).
    """
    try:
        root = ET.fromstring(config or "<View/>")
    except ET.ParseError:
        return []
    names: list[str] = []
    for el in root.iter():
        if el.tag in LABEL_TAGS or el.tag == "Choices":
            if for_import and el.tag == "KeyPointLabels":
                continue
            names += [c.get("value", "") for c in el if c.tag in ("Label", "Choice")]
    return list(dict.fromkeys(n for n in names if n))


def _results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("results", [])
    return [x for x in data if isinstance(x, dict)]


def list_remote(settings: ToolSettings, search: str = "") -> list[dict[str, Any]]:
    """Projects, newest first (at most 200): ``[{id, kind, name, images, created, classes}]``."""
    client = _Client(settings)
    items: list[dict[str, Any]] = []
    page = 1
    while len(items) < MAX_ITEMS:
        data = client.request(
            "GET", f"/api/projects?page_size={PAGE}&page={page}&ordering=-created_at"
        ).json()
        batch = _results(data)
        items += batch
        if not batch or not isinstance(data, dict) or not data.get("next"):
            break
        page += 1
    needle = search.strip().lower()
    items = [p for p in items if needle in str(p.get("title", "")).lower()]
    items.sort(key=lambda p: (str(p.get("created_at") or ""), int(p.get("id") or 0)), reverse=True)
    return [
        {
            "id": p["id"],
            "kind": "project",
            "name": str(p.get("title", "")),
            "images": int(p.get("task_number") or 0),
            "created": str(p.get("created_at") or ""),
            "classes": label_names(str(p.get("label_config") or "")),
        }
        for p in items[:MAX_ITEMS]
    ]


def _info(client: _Client, kind: str, remote_id: str) -> dict[str, Any]:
    if kind != "project":
        raise ToolError("Label Studio imports projects: kind must be 'project'.")
    if not str(remote_id).isdigit():
        raise RemoteNotFound(f"Label Studio has no project '{remote_id}'.")
    try:
        info = client.request("GET", f"/api/projects/{int(remote_id)}").json()
    except ToolError as err:
        if "not found" in str(err):
            raise RemoteNotFound(f"Label Studio has no project '{remote_id}'.") from None
        raise
    return {
        "name": str(info.get("title") or f"Label Studio project {remote_id}"),
        "classes": label_names(str(info.get("label_config") or ""), for_import=True),
    }


def check_remote(settings: ToolSettings, kind: str, remote_id: str) -> dict[str, Any]:
    """Name and declared classes of a Label Studio project.

    Raises:
        RemoteNotFound: no such project.
        ToolError: bad token, unreachable server.
    """
    return _info(_Client(settings), kind, remote_id)


# -- images ---------------------------------------------------------------------------
def _local_name(value: str, used: set[str], index: int) -> str:
    """File name (possibly with sub-folders) for the image of a task; unique among ``used``."""
    raw = image_candidates(value)[0]
    parts = [p for p in raw.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    name = "/".join(parts) or f"task_{index}"
    suffix = Path(name).suffix.lower()
    stem = name[: len(name) - len(suffix)] if suffix else name
    candidate, n = name, 1
    while candidate.lower() in used:
        n += 1
        candidate = f"{stem}_{n}{suffix}"
    used.add(candidate.lower())
    return candidate


def _fetch(client: _Client, value: str, target: Path) -> str:
    """Download ``value`` to ``target``; returns the Content-Type. Raises ToolError."""
    base = client.url
    if value.startswith(base):
        value = value[len(base) :]
    if value.startswith("/"):
        response = client.request("GET", value, timeout=DOWNLOAD_TIMEOUT, stream=True)
    elif urlsplit(value).scheme in ("http", "https"):
        try:
            response = requests.get(value, timeout=DOWNLOAD_TIMEOUT, stream=True)  # no token for other hosts
        except requests.RequestException:
            raise ToolError("could not download the image") from None
        if response.status_code >= 400:
            raise ToolError(f"image server answered {response.status_code}")
    else:
        raise ToolError("unsupported image address")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as out:
            for chunk in response.iter_content(CHUNK):
                out.write(chunk)
    except requests.RequestException:
        raise ToolError("the download was interrupted") from None
    return str(response.headers.get("Content-Type", ""))


def _with_extension(name: str, content_type: str) -> str:
    if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
        return name
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
    return name + (".jpg" if ext == ".jpe" else ext)


def _download_images(
    client: _Client, tasks: list[dict[str, Any]], folder: Path, progress: Progress
) -> tuple[dict[int, str], list[Problem]]:
    """Download the image of every task into ``folder``; returns ``{task index: file name}``."""
    got: dict[int, str] = {}
    failed: list[str] = []
    used: set[str] = set()
    no_image = 0
    for index, task in enumerate(tasks):
        progress.update(index, len(tasks), "Downloading images")
        value = _task_image(task)
        if not value:
            no_image += 1
            continue
        name = _local_name(value, used, index)
        target = folder / name
        try:
            content_type = _fetch(client, value, target)
        except ToolError as err:
            target.unlink(missing_ok=True)
            if any(m in str(err) for m in FATAL):
                raise
            failed.append(value.split("?", 1)[0].rsplit("/", 1)[-1] or value)
            continue
        final = _with_extension(name, content_type)
        if final != name:
            target.replace(folder / final)
        got[index] = final
    progress.update(len(tasks), len(tasks), "Downloading images")
    problems: list[Problem] = []
    if failed:
        problems.append(
            Problem(
                "warning",
                "image_download_failed",
                "Some images could not be downloaded.",
                len(failed),
                failed[:5],
            )
        )
    if no_image:
        problems.append(Problem("warning", "task_without_image", "Some tasks have no image.", no_image))
    return got, problems


def _rename_in_export(tasks: list[dict[str, Any]], got: dict[int, str], names: dict[str, str]) -> None:
    """Point each task at the name its image has in the project (``names``: file name -> record name)."""
    for index, file_name in got.items():
        record_name = names.get(file_name)
        data = tasks[index].get("data")
        if record_name is None or not isinstance(data, dict):
            continue
        data["image"] = "/data/local-files/?d=" + quote(record_name)


def import_project(
    store: ProjectStore,
    settings: ToolSettings,
    workspace: Path,
    *,
    kind: str,
    remote_id: str,
    name: str,
    description: str,
    task: Task,
    progress: Progress,
) -> dict[str, Any]:
    """Run the whole import (see docs/IMPORT.md); returns the job result.

    Raises:
        ToolError: login, network or Label Studio problems (readable message).
        Cancelled: cancelled by the user; the half-created project is removed.
    """
    progress.update(message="Listing Label Studio data")
    client = _Client(settings)
    remote = _info(client, kind, remote_id)
    origin = Origin("labelstudio", client.url, str(remote_id), kind, remote["name"])
    export = client.request(
        "GET",
        f"/api/projects/{int(remote_id)}/export?exportType=JSON&download_all_tasks=true",
        timeout=EXPORT_TIMEOUT,
    ).json()
    tasks = [t for t in export if isinstance(t, dict)] if isinstance(export, list) else []
    if not tasks:
        raise ToolError("This Label Studio project has no tasks to import.")
    work = workspace / "tmp" / f"import-{uuid.uuid4().hex[:10]}"
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    try:
        got, problems = _download_images(client, tasks, images, progress)
        if not got:
            raise ToolError("No image of this Label Studio project could be downloaded.")
        with common.new_project(store, origin, name, description, task) as project:
            progress.update(message="Creating the project")
            manifest = Manifest(store.dir(project.id))
            report = ingest_folder(
                manifest, images, work, common.FixedMessage(progress, "Creating the project")
            )
            progress.update(len(tasks), len(tasks), "Reading annotations")
            records = manifest.load()
            ids = {f: common.file_id(images / f) for f in got.values()}
            names = {f: records[i].name for f, i in ids.items() if i in records}
            progress.check()
            _rename_in_export(tasks, got, names)
            path = work / "export.json"
            path.write_text(json.dumps(tasks), encoding="utf-8")
            ctx = ParseContext("labelstudio", manifest.active())
            parse_labelstudio(ctx, [path])
            result = ctx.result()
            problems += [p for p in result.problems if p.code != "unmatched_entries"]
            return common.finish(
                store, project, result.labels, list(remote["classes"]), origin, report, problems, task
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
