"""Import a CVAT project (or a task outside any project) as an ActiveLabel project.

Listing uses CVAT's REST API with basic auth. The import uses FiftyOne's
``fiftyone.utils.cvat.import_annotations`` on a temporary FiftyOne dataset with
``download_media=True``: images go to a temporary folder of the workspace, are stored in the new
project exactly like uploads, and every label field the import created is converted to internal
labels. The temporary dataset and folder are always removed. The password is only passed to
FiftyOne as an argument; it is never logged, stored or put in a message.
"""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiftyone as fo
import requests
from fiftyone.utils import cvat as fouc

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.annotate.fo_labels import labels_from_fo
from activelabel.annotations.models import Problem
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.importing import common
from activelabel.importing.common import MAX_ITEMS, Origin, RemoteNotFound
from activelabel.jobs import Progress
from activelabel.labels import Label, load_annotations
from activelabel.projects import ProjectStore, clean_classes
from activelabel.storage import write_json
from activelabel.sync import engine as sync_engine
from activelabel.sync.registry import get_remote
from activelabel.sync.remote import RemoteProject
from activelabel.tasks import Task

TIMEOUT = 30
PAGE = 100
MAX_TASKS = 1000  # tasks read to add up the images of projects
LABEL_CLASSES = ("Classification", "Classifications", "Detections", "Polylines", "Keypoints")
KINDS = ("project", "task")


@dataclass
class SampleLabels:
    """One downloaded image: its file and the label values (FiftyOne labels) of every label field."""

    filepath: Path
    values: list[Any]


class _Http:
    """Minimal CVAT REST client (basic auth). Failures become readable :class:`ToolError` messages."""

    def __init__(self, settings: ToolSettings) -> None:
        """Prepare a session against the CVAT server.

        Args:
            settings: Server address, username and password of CVAT.

        Raises:
            ToolError: the address, username or password is missing, or the address is not http(s).
        """
        if not settings.url or not settings.username or not settings.secret:
            raise ToolError(
                "CVAT needs the server address, the username and the password.", "cvat_config_missing"
            )
        self.url = settings.url.strip().rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise ToolError(
                "CVAT address must start with http:// or https:// (e.g. https://app.cvat.ai).",
                "cvat_url_invalid",
            )
        self._http = requests.Session()
        self._auth = (settings.username, settings.secret)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON document of the API.

        Args:
            path: Path of the endpoint, starting with ``/api``.
            params: Query parameters, or None.

        Returns:
            The decoded JSON body.

        Raises:
            RemoteNotFound: CVAT answered 404.
            ToolError: the server is unreachable, the login is refused, CVAT is rate limiting or failing,
                or the answer is not JSON.
        """
        try:
            r = self._http.get(f"{self.url}{path}", params=params, auth=self._auth, timeout=TIMEOUT)
        except requests.RequestException:
            raise common.unreachable(self.url) from None
        if r.status_code in (401, 403):
            raise common.login_failed(self.url)
        if r.status_code == 404:
            raise common.not_found(path)
        if r.status_code == 429:
            raise common.rate_limited()
        if r.status_code >= 400:
            raise common.http_error(r.status_code, path)
        try:
            return r.json()
        except ValueError:  # e.g. the address of a web page that is not CVAT
            raise ToolError(
                f"{self.url} did not answer as the CVAT API: check the address.", "cvat_not_api", url=self.url
            ) from None

    def paged(self, path: str, params: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        """Items of a paginated list, up to ``limit``.

        Args:
            path: Path of the list endpoint.
            params: Query parameters (the page is added).
            limit: Most items to return.

        Returns:
            The items of the pages read, at most ``limit``.
        """
        out: list[dict[str, Any]] = []
        page = 1
        while len(out) < limit:
            data = self.get(path, {**params, "page_size": PAGE, "page": page})
            if isinstance(data, list):
                return data[:limit]
            results = list(data.get("results") or [])
            out += results
            if not results or not data.get("next"):
                break
            page += 1
        return out[:limit]

    def label_names(self, item: dict[str, Any], param: str) -> list[str]:
        """Label names of a project/task (inline list, or fetched through ``/api/labels``).

        Args:
            item: The project or task as CVAT returns it.
            param: Filter of ``/api/labels`` that selects it (``project_id`` or ``task_id``).

        Returns:
            The distinct top-level label names, in order (empty when they cannot be read).
        """
        labels = item.get("labels")
        try:
            if not isinstance(labels, list):
                labels = self.paged("/api/labels", {param: item["id"]}, 200)
        except ToolError:
            return []
        names = [
            str(x["name"]) for x in labels if isinstance(x, dict) and x.get("name") and not x.get("parent_id")
        ]
        return list(dict.fromkeys(names))


def _has_status(low: str, code: int) -> bool:
    """Whether the text names an HTTP status as a number of its own (not inside an id or a path).

    Args:
        low: The lower-cased error text.
        code: The HTTP status code to look for.

    Returns:
        True when ``code`` appears as a separate number.
    """
    return re.search(rf"(?<![\w/.-]){code}(?![\w.-])", low) is not None


def explain(err: Exception, url: str = "") -> ToolError:
    """Turn CVAT / FiftyOne / network errors into a readable :class:`ToolError`.

    Args:
        err: The error that was raised.
        url: Address of the server, added to the message when given.

    Returns:
        ``err`` itself when it already is a ToolError, else a ToolError with a readable message.
    """
    if isinstance(err, ToolError):
        return err
    text = str(err)
    low = text.lower()
    name = type(err).__name__.lower()
    if (
        "maximum number of" in low or "upgrade" in low
    ):  # a plan limit often comes as a 403: not a login problem
        return common.refused(text)
    if "auth/login" in low or _has_status(low, 401) or "unauthorized" in low:
        return common.login_failed(url)
    if _has_status(low, 403) or "forbidden" in low:
        return common.refused("")
    if _has_status(low, 429) or "too many requests" in low:
        return common.rate_limited()
    if "connection" in name or "timeout" in name or "max retries" in low or "name or service" in low:
        return common.unreachable(url)
    if _has_status(low, 404) or "not found" in low or "invalidurl" in name or "missingschema" in name:
        return common.not_found(url or "the server")
    reason = text.splitlines()[0] if text else type(err).__name__
    return ToolError(f"CVAT error: {reason}", "cvat_error_other", reason=reason)


def _created(item: dict[str, Any]) -> str:
    """Creation date of a remote project or task.

    Args:
        item: The project or task as CVAT returns it.

    Returns:
        Its ``created_date`` (or ``created_at``), or an empty text.
    """
    return str(item.get("created_date") or item.get("created_at") or "")


def list_remote(settings: ToolSettings, search: str = "") -> list[dict[str, Any]]:
    """Projects, and tasks that are in no project, newest first (at most 200).

    Args:
        settings: Connection settings of CVAT.
        search: Text the name must contain (ignoring case); empty lists everything.

    Returns:
        ``[{id, kind, name, images, created, classes}]``; ``images`` of a project is the sum of the
        frames of its tasks.
    """
    http = _Http(settings)
    projects = http.paged("/api/projects", {"sort": "-id"}, MAX_ITEMS)
    tasks = http.paged("/api/tasks", {"sort": "-id"}, MAX_TASKS)
    frames: dict[Any, int] = {}
    for t in tasks:
        frames[t.get("project_id")] = frames.get(t.get("project_id"), 0) + int(t.get("size") or 0)
    items: list[tuple[str, dict[str, Any], int, str]] = []
    for p in projects:
        items.append(("project", p, frames.get(p["id"], 0), "project_id"))
    for t in tasks:
        if t.get("project_id") is None:
            items.append(("task", t, int(t.get("size") or 0), "task_id"))
    needle = search.strip().lower()
    chosen = [i for i in items if needle in str(i[1].get("name", "")).lower()]
    chosen.sort(key=lambda i: (_created(i[1]), int(i[1]["id"])), reverse=True)
    return [
        {
            "id": item["id"],
            "kind": kind,
            "name": str(item.get("name", "")),
            "images": images,
            "created": _created(item),
            "classes": http.label_names(item, param),
        }
        for kind, item, images, param in chosen[:MAX_ITEMS]
    ]


def check_remote(settings: ToolSettings, kind: str, remote_id: str) -> dict[str, Any]:
    """Name and labels of one remote project or task (a task also says its ``project_id``, or None).

    Args:
        settings: Connection settings of CVAT.
        kind: ``project`` or ``task``.
        remote_id: Id of the project or task (digits).

    Returns:
        ``{name, classes}``, plus ``project_id`` (text or None) for a task.

    Raises:
        RemoteNotFound: it does not exist.
        ToolError: bad login, unreachable server or a malformed id.
    """
    if kind not in KINDS:
        raise ToolError("kind must be 'project' or 'task'.")
    if not str(remote_id).isdigit():
        raise RemoteNotFound(
            f"CVAT has no {kind} '{remote_id}'.", "cvat_item_not_found", what=kind, id=str(remote_id)
        )
    http = _Http(settings)
    item = http.get(f"/api/{kind}s/{int(remote_id)}")
    param = "project_id" if kind == "project" else "task_id"
    found: dict[str, Any] = {
        "name": str(item.get("name") or f"CVAT {kind} {remote_id}"),
        "classes": http.label_names(item, param),
    }
    if kind == "task":
        found["project_id"] = str(item["project_id"]) if item.get("project_id") is not None else None
    return found


# -- FiftyOne side (replaced in unit tests) ---------------------------------------
def _is_label_field(field: Any) -> bool:
    """Whether a FiftyOne field holds labels that can be imported.

    Args:
        field: A field of the FiftyOne dataset schema.

    Returns:
        True when its document type is one of ``LABEL_CLASSES``.
    """
    doc = getattr(field, "document_type", None)
    return getattr(doc, "__name__", "") in LABEL_CLASSES


def _download(settings: ToolSettings, kind: str, remote_id: str, data_dir: Path) -> list[SampleLabels]:
    """Download images and annotations with FiftyOne into ``data_dir``; read them back.

    The temporary FiftyOne dataset is deleted before returning.

    Args:
        settings: Connection settings of CVAT.
        kind: ``project`` or ``task``.
        remote_id: Id of the project or task.
        data_dir: Folder the images are downloaded into.

    Returns:
        One entry per downloaded image, with its file and the values of its label fields.

    Raises:
        ToolError: the download failed (the error is turned into a readable message).
    """
    where: dict[str, Any] = (
        {"project_id": int(remote_id)} if kind == "project" else {"task_ids": [int(remote_id)]}
    )
    url = settings.url.strip().rstrip("/")
    dataset = fo.Dataset(f"activelabel-import-{uuid.uuid4().hex[:10]}")
    try:
        try:
            fouc.import_annotations(
                dataset,
                data_path=str(data_dir),
                download_media=True,
                url=url,
                username=settings.username,
                password=settings.secret,
                **where,
            )
        except Exception as err:
            raise explain(err, url) from None
        fields = [n for n, f in dataset.get_field_schema().items() if _is_label_field(f)]
        return [SampleLabels(Path(s.filepath), [s[n] for n in fields]) for s in dataset.iter_samples()]
    finally:
        try:
            dataset.delete()
        except Exception:  # noqa: S110 - a leftover temporary dataset is harmless
            pass


def _labels_of(values: list[Any]) -> list[Label]:
    """Internal labels of the FiftyOne label values of one image.

    Args:
        values: The values of its label fields (None values are skipped).

    Returns:
        Every label, whatever its kind (the task is chosen later).
    """
    out: list[Label] = []
    for value in values:
        if value is not None:
            out += labels_from_fo(value, Task.DETECTION)  # keep every class: the task is chosen later
    return out


def _tmp_dir(workspace: Path) -> Path:
    """A new, unique temporary folder path for one import.

    Args:
        workspace: The workspace folder.

    Returns:
        ``<workspace>/tmp/import-<random>`` (not created).
    """
    return workspace / "tmp" / f"import-{uuid.uuid4().hex[:10]}"


def _walk_labels(
    samples: list[SampleLabels], manifest: Manifest, progress: Progress
) -> Iterator[tuple[str, list[Label]]]:
    """Labels of the downloaded images that the project stored.

    Args:
        samples: The downloaded images with their label values.
        manifest: Manifest of the new project (tells which images it holds).
        progress: Progress handle, checked for cancellation at every image.

    Yields:
        ``(image id, labels)`` for each image file that exists and is in the project.
    """
    known = manifest.load()
    for sample in samples:
        progress.check()
        path = sample.filepath
        if not path.is_file():
            continue
        image_id = common.file_id(path)
        if image_id in known:
            yield image_id, _labels_of(sample.values)


def _read_labels(
    remote: RemoteProject, settings: ToolSettings, rid: str, only: list[str] | None
) -> dict[str, list[Label]]:
    """Labels per remote item key, to choose the task of the project (tags when there are no shapes).

    Args:
        remote: The CVAT adapter.
        settings: Connection settings of CVAT.
        rid: Id of the remote project.
        only: Ids of the remote tasks to read, or None for all.

    Returns:
        ``{item key: labels}`` (the classification tags when the items have no shapes).
    """
    found = remote.items(settings, rid, Task.DETECTION, only)
    labels = {it.key: list(it.labels) for it in found}
    if found and not any(labels.values()):
        tags = {it.key: list(it.labels) for it in remote.items(settings, rid, Task.CLASSIFICATION, only)}
        if any(tags.values()):
            return tags
    return labels


def _import_with_tasks(
    store: ProjectStore,
    settings: ToolSettings,
    origin: Origin,
    rid: str,
    only: list[str] | None,
    declared: list[str],
    name: str,
    description: str,
    hint: Task,
    progress: Progress,
) -> dict[str, Any]:
    """Import a CVAT project (or ``only`` some of its tasks) with the sync engine: tasks, jobs, link.

    Args:
        store: Projects of the workspace.
        settings: Connection settings of CVAT.
        origin: Where the project comes from.
        rid: Id of the remote project.
        only: Ids of the remote tasks to import, or None for the whole project.
        declared: Class names the remote project declares.
        name: Wanted name of the new project.
        description: Description of the new project.
        hint: Task used when there are no annotations.
        progress: Progress handle of the job.

    Returns:
        The job result described in docs/IMPORT.md, with ``tasks`` and ``linked`` added.

    Raises:
        ToolError: CVAT returned no images.
    """
    remote = get_remote("cvat")
    progress.update(message="Reading annotations")
    labels = _read_labels(remote, settings, rid, only)
    if not labels:
        raise ToolError(
            "CVAT returned no images for this " + origin.kind + ".", "cvat_no_images", what=origin.kind
        )
    plan = common.plan_task(labels, hint)
    names = list(declared) if plan.task is not Task.KEYPOINTS else []
    for items in plan.labels.values():
        names += [lb.label for lb in items]
    progress.check()
    with common.new_project(store, origin, name, description, hint) as project:
        store.update(project.id, task=plan.task, classes=clean_classes(names))
        summary = sync_engine.make_plan(
            store,
            project.id,
            "cvat",
            settings,
            remote,
            common.FixedMessage(progress, "Comparing with CVAT"),
            create=False,
            remote_id=rid,
            scope="project",
            remote_tasks=only,
        )
        progress.update(0, len(labels), "Downloading images")
        applied = sync_engine.apply_plan(
            store, project.id, "cvat", settings, remote, summary["plan_id"], {}, None, progress
        )
        if not applied["tasks_imported"]:
            raise ToolError(
                "CVAT returned no images for this " + origin.kind + ".", "cvat_no_images", what=origin.kind
            )
        project_dir = store.dir(project.id)
        write_json(project_dir / common.ORIGIN_FILE, origin.as_dict())
        done = store.get(project.id)
        saved = load_annotations(project_dir)
        added = len(Manifest(project_dir).all())
        problems = list(plan.problems)
        if added < len(labels):
            message = "Identical or unreadable images were not added."
            problems.append(Problem("info", "images_missing", message, len(labels) - added))
        return {
            "project_id": project.id,
            "name": done.name,
            "task": done.task.value,
            "images_added": added,
            "images_rejected": 0,
            "labels": sum(len(v) for v in saved.images.values()) if saved else 0,
            "classes": list(done.classes),
            "problems": [p.as_dict() for p in problems],
            "tasks": applied["tasks_imported"],
            "linked": True,
        }


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

    A project, or a task that belongs to one, goes through the synchronisation engine (one local task
    per CVAT task, linked, nothing left to do in the next sync); a task outside any project is
    downloaded with FiftyOne into a single local task.

    Args:
        store: Projects of the workspace.
        settings: Connection settings of CVAT.
        workspace: Workspace folder (temporary downloads go under it).
        kind: ``project`` or ``task``.
        remote_id: Id of the project or task in CVAT.
        name: Wanted name of the new project (blank uses the remote name).
        description: Description of the new project.
        task: Task used when the remote data has no annotations.
        progress: Progress handle of the job.

    Returns:
        The job result described in docs/IMPORT.md.

    Raises:
        ToolError: login, network or CVAT problems (readable message).
        Cancelled: cancelled by the user; the half-created project is removed.
    """
    progress.update(message="Listing CVAT data")
    remote = check_remote(settings, kind, remote_id)
    origin = Origin("cvat", settings.url.strip().rstrip("/"), str(remote_id), kind, remote["name"])
    parent = str(remote_id) if kind == "project" else remote.get("project_id")
    if parent:
        only = None if kind == "project" else [str(remote_id)]
        return _import_with_tasks(
            store, settings, origin, parent, only, remote["classes"], name, description, task, progress
        )
    data_dir = _tmp_dir(workspace)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        progress.update(0, 0, "Downloading images")
        samples = _download(settings, kind, str(remote_id), data_dir)
        progress.check()
        if not samples:
            raise ToolError("CVAT returned no images for this " + kind + ".", "cvat_no_images", what=kind)
        with common.new_project(store, origin, name, description, task) as project:
            progress.update(message="Creating the project")
            manifest = Manifest(store.dir(project.id))
            report = ingest_folder(
                manifest, data_dir, data_dir, common.FixedMessage(progress, "Creating the project")
            )
            progress.update(0, len(samples), "Reading annotations")
            labels: dict[str, list[Label]] = {}
            for image_id, items in _walk_labels(samples, manifest, progress):
                labels.setdefault(image_id, []).extend(items)
            labels = {k: v for k, v in labels.items() if v}  # CVAT does not say "annotated, empty"
            problems: list[Problem] = []
            return common.finish(
                store, project, labels, list(remote["classes"]), origin, report, problems, task
            )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
