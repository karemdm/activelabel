"""Synchronise a project with its remote project (see docs/SYNC.md): plan, then apply.

The plan compares, per image, three versions of the labels: ``base`` (hash stored in the link at the
last synchronisation), ``local`` (project ground truth) and ``remote`` (read from the tool). Remote
labels are cleaned like any pulled labels (:func:`activelabel.annotate.diff.clean_labels`) before they
are hashed, so both sides are compared in the project's own terms. An image without a known base
(matched by file name for the first time) counts as changed on both sides unless one side has no
labels, in which case the other side wins; equal labels are never a conflict.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from activelabel.annotate import diff as diffing
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.backups import make_backup
from activelabel.images.ingest import ingest_files
from activelabel.images.manifest import ACTIVE, EXCLUDED, ImageRecord, Manifest
from activelabel.images.video import VideoStore, probe
from activelabel.importing.common import RemoteNotFound, read_origin
from activelabel.jobs import Progress
from activelabel.labels import Label, annotations_lock, load_annotations, save_annotations
from activelabel.projects import Project, ProjectStore
from activelabel.storage import now, read_json, write_json
from activelabel.sync import links
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteProject, RemoteTask
from activelabel.tasks_store import ImageTask, TaskStore
from activelabel.tasks_store import jobs as task_jobs

log = logging.getLogger(__name__)

PULL, PUSH, CONFLICT, SAME, ADD = "pull", "push", "conflict", "same", "add"
REMOVE_REMOTE, REMOVE_LOCAL = "remove_remote", "remove_local"  # image removed on the other side
RESTORE_REMOTE, RESTORE_LOCAL = "restore_remote", "restore_local"  # image restored on the other side
GONE = "gone"  # removed on both sides
ACTIONS = (PULL, PUSH, CONFLICT, SAME, ADD, REMOVE_REMOTE, REMOVE_LOCAL, RESTORE_REMOTE, RESTORE_LOCAL, GONE)
SCOPES = ("project", "task", "job")
BOTH, PULL_ONLY, PUSH_ONLY = "both", "pull", "push"  # direction of a synchronisation
DIRECTIONS = (BOTH, PULL_ONLY, PUSH_ONLY)
CHOICES = ("local", "remote")
MAX_EXAMPLES = 10
KEEP_PLANS = 10
_PLAN_ID = re.compile(r"^[0-9a-f]{6,32}$")
CHANGED_REMOTE = "The remote project changed since the comparison. Synchronise again."
CHANGED_LOCAL = "The project's annotations changed since the comparison. Synchronise again."
CHANGED_LINK = (
    "The link with the tool changed since the comparison (another synchronisation ran). Synchronise again."
)


class SyncError(ToolError):
    """A readable problem of the synchronisation."""


class RemoteChanged(SyncError):
    """The remote project (or the project) changed after the plan was made."""


# -- helpers --------------------------------------------------------------------
class FrameCounter:
    """Counts the frames downloaded for new tasks and reports them as the job's progress."""

    def __init__(self, progress: Progress, total: int) -> None:
        """Start at zero of ``total`` frames.

        Args:
            progress: The job's progress handle (also checked for cancellation at every frame).
            total: How many frames will be downloaded.
        """
        self.progress = progress
        self.total = total
        self.done = 0
        progress.update(0, total, "Downloading images")

    def __call__(self) -> None:
        """One more frame was downloaded: report ``done / total`` and stop here when cancelled."""
        self.done += 1
        self.progress.update(self.done, self.total, "Downloading images")
        self.progress.check()


def remote_name(project: Project) -> str:
    """Name the annotation tools give the remote project of a project.

    Args:
        project: The project.

    Returns:
        ``ActiveLabel <id> - <name>``.
    """
    return f"ActiveLabel {project.id} - {project.name}"


def _plan_project_rename(
    project: Project, link: dict[str, Any] | None, info: RemoteInfo | None, direction: str
) -> tuple[dict[str, str] | None, dict[str, str] | None, str]:
    """Compare the name of the project with its remote project (docs/SYNC.md, *Names*).

    Three-way against the remote name recorded in the link at the last synchronisation: only here changed
    -> sent; only there changed -> taken; both changed to different names -> kept and reported. ``push``
    always sends the local name, ``pull`` always takes the remote one. A project made here is called
    ``ActiveLabel <id> - <name>`` in the tool; a project imported from it keeps the plain name there (the
    style is the one of the recorded remote name).

    Args:
        project: The project.
        link: The existing link, or None (nothing to compare before the first synchronisation).
        info: The remote project, or None when it does not exist yet.
        direction: ``both``, ``pull`` or ``push``.

    Returns:
        ``(rename, conflict, recorded)``: ``rename`` is ``{old, new, direction}`` (``push``: the remote name
        ``old`` becomes ``new``; ``pull``: the local name does) or None; ``conflict`` is ``{local, remote}``
        when both sides were renamed to different names, else None; ``recorded`` is the remote name to keep
        in the link after the apply (the one both sides agree on, or the old one on a conflict).
    """
    if not link or info is None:
        return None, None, info.name if info else ""
    base = link.get("remote_name") or None
    prefix = f"ActiveLabel {project.id} - "
    styled = (base if base is not None else info.name).startswith(prefix)
    mine = prefix + project.name if styled else project.name
    if info.name == mine:
        return None, None, info.name
    here = base is not None and mine != base
    there = base is not None and info.name != base
    if direction == PULL_ONLY or (direction == BOTH and there and not here):
        new = info.name.removeprefix(prefix) if info.name.startswith(prefix) else info.name
        if new == project.name:
            return None, None, info.name
        return {"old": project.name, "new": new, "direction": PULL_ONLY}, None, info.name
    if direction == PUSH_ONLY or not there:
        return {"old": info.name, "new": mine, "direction": PUSH_ONLY}, None, mine
    return None, {"local": mine, "remote": info.name}, str(base)


def plans_dir(project_dir: Path) -> Path:
    """Folder with the saved plans.

    Args:
        project_dir: Folder of the project.

    Returns:
        ``<project_dir>/sync``.
    """
    return project_dir / "sync"


def plan_file(project_dir: Path, plan_id: str) -> Path | None:
    """File of a saved plan (None when the id is malformed).

    Args:
        project_dir: Folder of the project.
        plan_id: Id of the plan (6 to 32 hex characters).

    Returns:
        The plan file, or None when the id is malformed.
    """
    return plans_dir(project_dir) / f"{plan_id}.json" if _PLAN_ID.match(plan_id) else None


def load_plan(project_dir: Path, plan_id: str) -> dict[str, Any] | None:
    """A saved plan, or None.

    Args:
        project_dir: Folder of the project.
        plan_id: Id of the plan.

    Returns:
        The plan, or None when it is missing, malformed or the id is invalid.
    """
    path = plan_file(project_dir, plan_id)
    data = read_json(path) if path is not None else None
    return data if isinstance(data, dict) else None


def discard_plan(project_dir: Path, plan_id: str) -> bool:
    """Delete a saved plan, so that it can no longer be applied (from any tab).

    Args:
        project_dir: Folder of the project.
        plan_id: Id of the plan.

    Returns:
        True when the plan existed and was deleted; False when it was missing or the id is invalid.
    """
    path = plan_file(project_dir, plan_id)
    if path is None or not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def _prune(project_dir: Path) -> None:
    """Delete the oldest saved plans, keeping the last ``KEEP_PLANS``.

    Args:
        project_dir: Folder of the project.
    """
    files = sorted(plans_dir(project_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_PLANS:]:
        old.unlink(missing_ok=True)


def _side(labels: list[Label]) -> dict[str, Any]:
    """Count the labels of one side of a comparison.

    Args:
        labels: The labels of one side.

    Returns:
        ``{labels, per_class}``: the total and the count by class.
    """
    per_class: dict[str, int] = {}
    for lb in labels:
        per_class[lb.label] = per_class.get(lb.label, 0) + 1
    return {"labels": len(labels), "per_class": per_class}


def _clean(project: Project, labels: list[Label]) -> list[Label]:
    """Labels cleaned like any pulled labels (classes and geometry of the project).

    Args:
        project: The project whose classes and task apply.
        labels: The labels to clean.

    Returns:
        The labels that are valid in the project (the problems found are dropped).
    """
    return diffing.clean_labels(project, labels)[0]


def _theirs(
    project: Project, labels: list[Label], mine: list[Label], base: list[Label] | None
) -> list[Label]:
    """The labels read from the tool, cleaned, as this project compares them.

    Pose objects whose points did not change take the box this side has for them (the tool may have no
    box for a pose, see :func:`links.carry_pose_boxes`): the project's box first, then the last sync's.

    Args:
        project: The project whose classes and task apply.
        labels: The labels of the item in the tool.
        mine: The labels of the image in the project.
        base: The labels at the last synchronisation, or None when unknown.

    Returns:
        The cleaned labels, with the boxes of unchanged poses carried over.
    """
    return links.carry_pose_boxes([*mine, *(base or [])], _clean(project, labels))


def _before_renames(labels: dict[str, list[Label]], renames: list[dict[str, str]]) -> dict[str, list[Label]]:
    """The labels read from the tool with the classes renamed there given their names here again.

    The review lists the renamed classes on their own line, so the labels of a renamed class are neither
    a new class nor an edited label in the comparison of the annotations (#519).

    Args:
        labels: The labels read from the tool, by image id.
        renames: ``[{"from": class here, "to": name there}]`` (:func:`renamed_classes`).

    Returns:
        ``labels`` itself when nothing was renamed, else a copy with the old names.
    """
    if not renames:
        return labels
    back = {r["to"].casefold(): r["from"] for r in renames}
    return {
        image_id: [replace(lb, label=back.get(lb.label.casefold(), lb.label)) for lb in items]
        for image_id, items in labels.items()
    }


def renamed_classes(
    project: Project, before: dict[str, str] | None, now: dict[str, str]
) -> list[dict[str, str]]:
    """Classes renamed in the tool since the last synchronisation, found by their id there.

    A class counts as renamed when the tool still has its id under another name (not only another case),
    the old name is a class of this project and the new one is not (else it would be two classes merged,
    which is left to the person).

    Args:
        project: The project (its classes).
        before: ``{class id: name}`` stored in the link at the last synchronisation, or None when unknown.
        now: ``{class id: name}`` as the tool has them now.

    Returns:
        ``[{"from": class here, "to": name there}]``, in the order of the project's classes.
    """
    mine = {c.casefold(): c for c in project.classes}
    out: list[dict[str, str]] = []
    for cid, old in (before or {}).items():
        new = now.get(cid)
        if new is None or new.casefold() == old.casefold():
            continue
        if old.casefold() in mine and new.casefold() not in mine:
            out.append({"from": mine[old.casefold()], "to": new})
    order = {c: n for n, c in enumerate(project.classes)}
    return sorted(out, key=lambda r: order[r["from"]])


def _rename_classes(store: ProjectStore, project_id: int, renames: list[dict[str, str]]) -> str:
    """Rename classes of the project and of every label of its ground truth (after a backup).

    Args:
        store: The projects.
        project_id: The project.
        renames: ``[{"from": old name, "to": new name}]``.

    Returns:
        The name of the backup of ``annotations.json`` ("" when there was nothing to back up).
    """
    names = {r["from"]: r["to"] for r in renames}
    project_dir = store.dir(project_id)
    backup = ""
    with annotations_lock(project_dir):
        found = load_annotations(project_dir)
        if found is not None:
            backup = make_backup(project_dir, "rename")
            for labels in found.images.values():
                for lb in labels:
                    lb.label = names.get(lb.label, lb.label)
            found.classes = [names.get(c, c) for c in found.classes]
            save_annotations(project_dir, found)
        project = store.get(project_id)
        store.update(project_id, classes=[names.get(c, c) for c in project.classes])
    return backup


def not_converted(items: list[RemoteItem]) -> dict[str, int]:
    """What the tool has but the comparison cannot see, summed over the items read (by kind, sorted).

    Args:
        items: The remote items read for the plan.

    Returns:
        ``{kind: count}`` (for example ``{"ellipse": 2, "tracks": 1}``); empty when everything was read.
    """
    out: dict[str, int] = {}
    for it in items:
        for kind, n in it.not_converted.items():
            out[kind] = out.get(kind, 0) + int(n)
    return dict(sorted(out.items()))


def _local_labels(project_dir: Path) -> dict[str, list[Label]]:
    """Labels of the project by image id.

    Args:
        project_dir: Folder of the project.

    Returns:
        ``{image_id: labels}``; empty when there are no annotations.
    """
    annotations = load_annotations(project_dir)
    return annotations.images if annotations else {}


def _info_dict(info: RemoteInfo | None, name: str) -> dict[str, Any]:
    """JSON-ready form of a remote project (an empty one when it does not exist yet).

    Args:
        info: The remote project, or None when it will be created.
        name: Name shown when there is no remote project.

    Returns:
        ``{id, name, url, images}``.
    """
    if info is None:
        return {"id": "", "name": name, "url": "", "images": 0}
    return {"id": info.id, "name": info.name, "url": info.url, "images": info.images}


def _info_obj(data: dict[str, Any]) -> RemoteInfo:
    """Remote project from its JSON-ready form (see :func:`_info_dict`).

    Args:
        data: ``{id, name, url, images?}`` as stored in a plan.

    Returns:
        The remote project.
    """
    return RemoteInfo(str(data["id"]), str(data["name"]), str(data["url"]), int(data.get("images", 0)))


def _decide(entry: dict[str, Any] | None, local: list[Label], remote: list[Label]) -> str:
    """Pull / push / same / conflict from the labels at the last sync, here and there (geometry tolerant).

    Args:
        entry: Link entry of the image (its base labels or hash), or None when never synchronised.
        local: The labels of the image in the project.
        remote: The cleaned labels of the item in the tool.

    Returns:
        One of ``SAME``, ``PULL``, ``PUSH`` or ``CONFLICT``.
    """
    same = links.same_labels
    base = links.base_labels(entry) if entry else None
    if base is None and entry and entry.get("base"):
        # older links keep only the hash: compare exactly
        bh, lh, rh = entry["base"], links.label_hash(local), links.label_hash(remote)
        local_changed, remote_changed = lh != bh, rh != bh
    elif base is None:
        if same(local, remote):
            return SAME
        if not local:
            return PULL
        if not remote:
            return PUSH
        return CONFLICT
    else:
        local_changed, remote_changed = not same(local, base), not same(remote, base)
    if not local_changed and not remote_changed:
        return SAME
    if not local_changed:
        return PULL
    if not remote_changed:
        return PUSH
    return SAME if same(local, remote) else CONFLICT


def _push_summary(
    project: Project, names: dict[str, str], changes: dict[str, tuple[list[Label], list[Label]]]
) -> dict[str, Any]:
    """Diff of what a push does: ``{image_id: (remote labels now, labels after the push)}``.

    Args:
        project: The project (its task decides how labels are compared).
        names: File name of every image, by image id.
        changes: ``(remote labels now, labels after the push)`` of every image the push touches.

    Returns:
        The number of images changed, the added / removed / edited labels (total and by class)
        and a few examples.
    """
    per_class: dict[str, dict[str, int]] = {}
    totals = {"added": 0, "removed": 0, "edited": 0}
    examples: list[dict[str, Any]] = []
    changed = 0
    for image_id, (old, new) in changes.items():
        found = diffing.compare_image(project.task, old, new)
        if not found:
            continue
        changed += 1
        for kind in totals:
            items: list[Label] = getattr(found, kind)
            totals[kind] += len(items)
            for lb in items:
                per_class.setdefault(lb.label, {"added": 0, "removed": 0, "edited": 0})[kind] += 1
        if len(examples) < diffing.MAX_EXAMPLES:
            examples.append(
                {
                    "image_id": image_id,
                    "name": names.get(image_id, image_id),
                    "added": len(found.added),
                    "removed": len(found.removed),
                    "edited": len(found.edited),
                }
            )
    return {
        "images_changed": changed,
        **totals,
        "per_class": per_class,
        "unknown_classes": [],
        "problems": [],
        "examples": examples,
    }


# -- plan -----------------------------------------------------------------------
def _find_remote(
    remote: RemoteProject,
    settings: ToolSettings,
    project: Project,
    link: dict[str, Any] | None,
    remote_id: str | None,
    create: bool,
) -> RemoteInfo | None:
    """The remote project to synchronise with: the linked one, the given one, or the one found by name.

    Args:
        remote: The tool's remote adapter.
        settings: Login of the tool.
        project: The project.
        link: The existing link, or None.
        remote_id: Id of the remote project to use when there is no link.
        create: Whether a missing remote project is acceptable (it will be created).

    Returns:
        The remote project, or None when it does not exist and ``create`` is set.

    Raises:
        SyncError: there is no remote project and ``create`` is not set.
        RemoteNotFound: the linked remote project was deleted in the tool (code ``remote_gone``).
    """
    if link:
        try:
            return remote.get(settings, str(link["remote_id"]))
        except RemoteNotFound:
            raise RemoteNotFound(
                "The remote project linked to this project no longer exists.", "remote_gone"
            ) from None
    if remote_id:
        return remote.get(settings, str(remote_id))
    info = remote.find(settings, remote_name(project))
    if info is None and not create:
        raise SyncError(
            "There is no remote project to synchronise with. Send the images first.", "sync_no_remote"
        )
    return info


def _key_task(key: str) -> str:
    """Remote task of an item key (``"<task>:<frame>"`` in CVAT).

    Args:
        key: Key of a remote item.

    Returns:
        The part of the key before the colon.
    """
    return str(key).partition(":")[0]


def _key_frame(key: str) -> int:
    """Frame number of an item key (``"<task>:<frame>"`` in CVAT).

    Args:
        key: Key of a remote item.

    Returns:
        The part of the key after the colon (0 when there is none).
    """
    return int(str(key).partition(":")[2] or 0)


def _scope(
    tasks: list[ImageTask], scope: str, task_id: int | None, job: int | None
) -> tuple[list[ImageTask], list[str]]:
    """Tasks touched by the scope and the image ids it covers (task order).

    Args:
        tasks: Every task of the project.
        scope: ``project``, ``task`` or ``job``.
        task_id: The task, for the ``task`` and ``job`` scopes.
        job: Number of the job, for the ``job`` scope.

    Returns:
        The tasks in scope and the ids of their images.

    Raises:
        SyncError: the scope is unknown, or the task or job does not exist.
    """
    if scope not in SCOPES:
        raise SyncError("The scope must be project, task or job.", "scope_unknown", scope=str(scope))
    if scope == "project":
        return tasks, [i for t in tasks for i in t.images]
    found = next((t for t in tasks if t.id == task_id), None)
    if found is None:
        raise SyncError("This task does not exist (it may have been deleted).", "task_not_found")
    if scope == "task":
        return [found], list(found.images)
    chosen = next((j for j in task_jobs(found) if j.n == job), None)
    if chosen is None:
        raise SyncError("This job does not exist in the task.", "job_not_found")
    return [found], list(chosen.image_ids)


def _task_mapping(tasks: list[ImageTask], tmap: dict[str, Any], link_items: dict[str, Any]) -> dict[str, str]:
    """Local task id -> remote task id, from the link (or inferred from the keys of an older link).

    Args:
        tasks: The local tasks.
        tmap: The ``tasks`` of the link (``{local task id: {remote_task, ...}}``).
        link_items: The ``items`` of the link (image id -> entry with the remote key).

    Returns:
        ``{local task id: remote task id}`` for the tasks whose remote task is known.
    """
    out: dict[str, str] = {}
    for t in tasks:
        known = (tmap.get(str(t.id)) or {}).get("remote_task")
        if known:
            out[str(t.id)] = str(known)
        else:
            found = {
                _key_task(link_items[i]["key"]) for i in t.images if (link_items.get(i) or {}).get("key")
            }
            if len(found) == 1:
                out[str(t.id)] = found.pop()
    return out


def make_plan(
    store: ProjectStore,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    progress: Progress,
    create: bool = False,
    remote_id: str | None = None,
    scope: str = "project",
    task_id: int | None = None,
    job: int | None = None,
    remote_tasks: list[str] | None = None,
    direction: str = BOTH,
) -> dict[str, Any]:
    """Compare the project (or one task / one job) with its remote project and save the plan.

    Images and annotations are compared, in both directions: tasks and images added, removed or
    restored, labels added / modified / removed (docs/TASKS.md, docs/SYNC.md).

    Args:
        store: projects.
        project_id: the project.
        tool: ``cvat``.
        settings: login of the tool.
        remote: the tool's remote adapter.
        progress: job progress.
        create: when there is no link and no remote project with the project's name, plan to create
            it (it is created on apply, never while planning).
        remote_id: remote project to link when there is no link yet.
        scope: ``project``, ``task`` or ``job``.
        task_id: the task (scopes task and job).
        job: the job number in the task (scope job).
        remote_tasks: project scope only: limit ``tasks_new`` (tasks added in the tool) to these remote
            task ids, and read only those. Used by the import of one CVAT task of a project.
        direction: ``both`` (each change goes the way it was made; conflicts are asked), ``pull``
            (the tool wins: this project becomes like it; what exists only here is left alone and
            listed) or ``push`` (this project wins: the tool becomes like it; what exists only there
            is left alone and listed).

    Returns:
        The plan summary (docs/SYNC.md).

    Raises:
        SyncError: nothing to synchronise with, unknown task/job.
        ToolError: the tool's own problems (readable message).
    """
    if direction not in DIRECTIONS:
        raise SyncError("The direction must be both, pull or push.", "direction_invalid")
    project = store.get(project_id)
    project_dir = store.dir(project_id)
    manifest = Manifest(project_dir)
    local_tasks = TaskStore(project_dir).load()
    scoped_tasks, scope_ids = _scope(local_tasks, scope, task_id, job)
    link = links.get_link(project_dir, tool)
    link_items: dict[str, dict[str, Any]] = (link or {}).get("items", {})
    tmap: dict[str, Any] = (link or {}).get("tasks", {})
    progress.update(message=f"Reading {tool}")
    info = _find_remote(remote, settings, project, link, remote_id, create)
    rtasks = remote.tasks(settings, info.id) if info else []
    renames = (
        renamed_classes(project, link.get("classes"), remote.classes(settings, info.id))
        if link and info
        else []
    )
    rtask_ids = {rt.id for rt in rtasks}
    mapping = _task_mapping(local_tasks, tmap, link_items)

    # which remote tasks are read: all of them for the project, else the ones of the scope
    read: set[str] | None = None  # None: everything
    if scope != "project":
        read = {mapping[str(t.id)] for t in scoped_tasks if str(t.id) in mapping}
        read |= {_key_task(link_items[i]["key"]) for i in scope_ids if (link_items.get(i) or {}).get("key")}
        read &= rtask_ids
    elif remote_tasks is not None:
        read = {str(t) for t in remote_tasks} & rtask_ids
    if info is None or (read is not None and not read):
        items: list[RemoteItem] = []
    else:
        items = remote.items(settings, info.id, project.task, sorted(read) if read is not None else None)
    # tasks deleted here that the person chose to keep in the tool: left alone, not offered again (#684)
    kept_remote = {str(m.get("remote_task")) for m in tmap.values() if m.get("kept_remote")}
    items = [it for it in items if _key_task(it.key) not in kept_remote]
    progress.update(message="Comparing")

    records = manifest.load()
    local = _local_labels(project_dir)
    by_key = {it.key: it for it in items}
    known_tasks = rtask_ids if read is None else read
    gone_tasks: set[str] = set()  # remote tasks the link knows and the tool no longer has
    if info is not None:
        # remote tasks the link knows but the tool no longer has: deleted there, so their images
        # count as removed (for the tasks of the scope)
        scoped = {str(t.id) for t in scoped_tasks}
        linked_tasks = {
            str((m or {}).get("remote_task"))
            for tid, m in tmap.items()
            if scope == "project" or tid in scoped
        }
        linked_tasks |= {
            _key_task(link_items[i]["key"]) for i in scope_ids if (link_items.get(i) or {}).get("key")
        }
        if scope == "project":
            linked_tasks |= {_key_task(e["key"]) for e in link_items.values() if e.get("key")}
        gone_tasks = linked_tasks - rtask_ids - {"None", ""}
        known_tasks = set(known_tasks) | gone_tasks

    # candidates: images of the scope, plus (project/task scope) linked images deleted locally
    in_tasks = {i for t in local_tasks for i in t.images}
    candidates = list(scope_ids)
    if scope != "job":
        for image_id, link_entry in link_items.items():
            key = link_entry.get("key", "")
            if image_id not in in_tasks and key and _key_task(key) in known_tasks:
                candidates.append(image_id)
    linked = {i for i in candidates if (link_items.get(i) or {}).get("key")}
    used = {link_items[i]["key"] for i in linked}
    free_records = [records[i] for i in candidates if i not in linked and i in records]
    by_name = links.match_by_name(free_records, [it for it in items if it.key not in used])

    decisions: dict[str, dict[str, Any]] = {}
    remote_labels: dict[str, list[dict[str, Any]]] = {}
    pull_new: dict[str, list[Label]] = {}
    push_changes: dict[str, tuple[list[Label], list[Label]]] = {}
    conflicts: list[dict[str, Any]] = []
    names = {i: r.name for i, r in records.items()}
    unsynced: list[str] = []
    for image_id in candidates:
        progress.check()
        rec = records.get(image_id)
        entry = link_items.get(image_id)
        key = (entry or {}).get("key", "")
        if key and _key_task(key) not in known_tasks:
            continue
        item = by_key.get(key) if key else by_name.get(image_id)
        if not key and item is None:
            unsynced.append(image_id)
            continue
        key = key or item.key  # type: ignore[union-attr]
        here = rec is not None and rec.status == ACTIVE
        there = item is not None
        base_removed = bool((entry or {}).get("removed"))
        mine = local.get(image_id, []) if rec is not None else []
        base = links.base_labels(entry) if entry else None
        theirs = _theirs(project, item.labels, mine, base) if item is not None else []
        kind = ""
        if here and there:
            action = _decide(None if base_removed else entry, mine, theirs)
            kind = "labels" if action == CONFLICT else ""
        elif not here and not there:
            action = SAME if base_removed else GONE
        elif here:  # missing in the tool
            if base_removed:
                action = RESTORE_REMOTE
            elif base is not None and not links.same_labels(mine, base):
                action, kind = CONFLICT, "removed_remote"
            else:
                action = REMOVE_LOCAL
        else:  # removed here, present in the tool
            if base_removed and rec is None:
                action = SAME  # deleted for good here
            elif base_removed:
                action = RESTORE_LOCAL
            elif base is not None and not links.same_labels(theirs, base):
                action, kind = CONFLICT, "removed_local"
            else:
                action = REMOVE_REMOTE
        if direction != BOTH:  # one side wins: no conflicts, every difference goes one way
            action, kind = _directed(action, kind, direction, rec is not None, there), ""
            if action == RESTORE_REMOTE and _key_task(key) not in rtask_ids:
                action = SAME  # its task no longer exists in the tool: nothing to restore there
        decisions[image_id] = {
            "key": key,
            "action": action,
            "kind": kind,
            "here": here,
            "there": there,
            "local_hash": links.label_hash(mine),
            "remote_hash": links.label_hash(theirs),
        }
        if action in (PULL, CONFLICT, RESTORE_LOCAL) and item is not None:
            remote_labels[key] = [lb.as_dict() for lb in theirs]
        if action in (PULL, RESTORE_LOCAL):
            pull_new[image_id] = theirs
        elif action in (PUSH, RESTORE_REMOTE):
            push_changes[image_id] = (theirs, mine)
        elif action == CONFLICT:
            conflicts.append(
                {
                    "image_id": image_id,
                    "name": names.get(image_id, image_id),
                    "kind": kind,
                    "local": {**_side(mine), "removed": not here},
                    "remote": {**_side(theirs), "removed": not there},
                }
            )

    # tasks to create in the tool (never sent), images to add to a task already there (older links)
    unsynced_set = set(unsynced)
    tasks_create: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []  # never sent: empty, or still receiving images
    only_here_tasks: list[dict[str, Any]] = []  # pull: never sent, left alone
    added: list[str] = []
    sent_new: list[str] = []  # images sent for the first time (new tasks and added images)
    for t in scoped_tasks:
        sent = str(t.id) in mapping or any((link_items.get(i) or {}).get("key") for i in t.images)
        sent = sent or any(i in decisions for i in t.images)
        if not sent and direction == PULL_ONLY:
            only_here_tasks.append({"task": t.id, "name": t.name, "images": len(t.images)})
            unsynced_set.difference_update(t.images)
        elif not sent and (why := _skip_reason(t, records)):
            skipped_tasks.append({"task": t.id, "name": t.name, "reason": why, "images": len(t.images)})
            unsynced_set.difference_update(t.images)  # not sent one by one either
        elif not sent:
            tasks_create.append(
                {"task": t.id, "name": t.name, "job_size": t.job_size, "images": len(t.images)}
            )
            for image_id in t.images:
                unsynced_set.discard(image_id)
                if image_id in records and records[image_id].status == ACTIVE:
                    push_changes[image_id] = ([], local.get(image_id, []))
                    sent_new.append(image_id)
    only_here = 0
    for image_id in unsynced:
        if image_id in unsynced_set and image_id in records and records[image_id].status == ACTIVE:
            if direction == PULL_ONLY:  # the tool wins: images only here are left alone
                only_here += 1
                continue
            added.append(image_id)
            sent_new.append(image_id)
            mine = local.get(image_id, [])
            push_changes[image_id] = ([], mine)
            decisions[image_id] = {"key": "", "action": ADD, "kind": "", "local_hash": links.label_hash(mine)}

    # tasks deleted in the tool, with the images that go with them (the review names them)
    tasks_removed_remote = [
        {
            "task": t.id,
            "name": t.name,
            "images": sum(1 for i in t.images if (decisions.get(i) or {}).get("action") == REMOVE_LOCAL),
        }
        for t in scoped_tasks
        if mapping.get(str(t.id)) in gone_tasks
    ]
    tasks_removed_remote = [x for x in tasks_removed_remote if x["images"]]

    # tasks deleted here (still in the tool) and tasks added in the tool (project scope)
    local_ids = {str(t.id) for t in local_tasks}
    tasks_delete = [
        {"task": int(tid), "remote_task": str(m.get("remote_task")), "name": str(m.get("name", ""))}
        for tid, m in tmap.items()
        if direction != PULL_ONLY  # the tool wins: its task stays
        and not m.get("kept_remote")
        and tid not in local_ids
        and str(m.get("remote_task")) in rtask_ids
        and (read is None or str(m.get("remote_task")) in read)
    ]
    mapped_remote = (
        set(mapping.values()) | {_key_task(d["key"]) for d in decisions.values() if d["key"]} | kept_remote
    )
    tasks_new: list[dict[str, Any]] = []
    if scope == "project" and direction != PUSH_ONLY:  # push: the tool's extra tasks stay
        tasks_new = [
            {"remote_task": rt.id, "name": rt.name, "job_size": rt.job_size, "images": rt.size}
            for rt in rtasks
            if rt.id not in mapped_remote and (read is None or rt.id in read)
        ]
    tasks_rename, tasks_names, tasks_rename_conflicts = _plan_renames(
        scoped_tasks, mapping, tmap, rtasks, read, direction
    )
    project_rename, project_conflict, project_name = (
        _plan_project_rename(project, link, info, direction) if scope == "project" else (None, None, "")
    )
    new_remote = {t["remote_task"] for t in tasks_new}
    leftovers = [it for it in items if it.key not in {d["key"] for d in decisions.values()}]
    unmatched = [it for it in leftovers if _key_task(it.key) not in new_remote]
    if scope == "job":  # the other jobs of the task are outside the scope, not "only in the tool"
        unmatched = []

    plan_id = uuid.uuid4().hex[:10]
    remote_only = {"count": len(unmatched), "examples": [it.file_name for it in unmatched[:MAX_EXAMPLES]]}
    count = {a: sum(1 for d in decisions.values() if d["action"] == a) for a in ACTIONS}
    plan: dict[str, Any] = {
        "plan_id": plan_id,
        "tool": tool,
        "created": now(),
        "create": info is None,
        "scope": {"scope": scope, "task": task_id, "job": job},
        "direction": direction,
        "remote": _info_dict(info, remote_name(project)),
        "decisions": decisions,
        "remote_labels": remote_labels,
        "remote_only": remote_only,
        "tasks_create": tasks_create,
        "skipped_tasks": skipped_tasks,
        "tasks_new": tasks_new,
        "tasks_delete": tasks_delete,
        "tasks_removed_remote": tasks_removed_remote,
        "tasks_rename": tasks_rename,
        "tasks_names": tasks_names,
        "tasks_rename_conflicts": tasks_rename_conflicts,
        "project_rename": project_rename,
        "project_rename_conflict": project_conflict,
        "project_name": project_name,
        "link": _link_token(link),
        "new_items": {rt: _new_task_items(remote, settings, project, rt, items) for rt in new_remote},
        "renamed_classes": renames,
    }
    write_json(plans_dir(project_dir) / f"{plan_id}.json", plan)
    _prune(project_dir)
    return {
        "plan_id": plan_id,
        "tool": tool,
        "remote": plan["remote"],
        "create": plan["create"],
        "scope": plan["scope"],
        "direction": direction,
        "only_here": {"images": only_here, "tasks": only_here_tasks},
        "pull": count[PULL],
        "push": count[PUSH],
        "add": len(added),
        "images_new": len(sent_new),
        "labels_new": sum(len(local.get(i, [])) for i in sent_new),
        "remove_remote": sum(  # images of tasks deleted here are counted in tasks_delete
            1
            for d in decisions.values()
            if d["action"] == REMOVE_REMOTE
            and _key_task(d["key"]) not in {str(t["remote_task"]) for t in tasks_delete}
        ),
        "remove_local": count[REMOVE_LOCAL],
        "restore_remote": count[RESTORE_REMOTE],
        "restore_local": count[RESTORE_LOCAL],
        "tasks_create": tasks_create,
        "skipped_tasks": skipped_tasks,
        "tasks_new": tasks_new,
        "tasks_delete": tasks_delete,
        "tasks_removed_remote": tasks_removed_remote,
        "tasks_rename": tasks_rename,
        "tasks_rename_conflicts": tasks_rename_conflicts,
        "project_rename": project_rename,
        "project_rename_conflict": project_conflict,
        "conflicts": conflicts,
        "remote_only": remote_only,
        "unchanged": count[SAME] + count[GONE],
        "diff_pull": diffing.diff(project_dir, list(pull_new), _before_renames(pull_new, renames), project),
        "diff_push": _push_summary(project, names, push_changes),
        "renamed_classes": renames,
        "not_converted": not_converted(items),
    }


def _skip_reason(task: ImageTask, records: dict[str, ImageRecord]) -> str:
    """Why a task is not created in the tool: ``open`` (still receiving images), ``no_images``, or ``""``.

    Args:
        task: The local task.
        records: Every image of the project by id.

    Returns:
        The reason, or an empty text when the task can be created.
    """
    if task.open:
        return "open"
    if not any(i in records for i in task.images):
        return "no_images"
    return ""


def _link_token(link: dict[str, Any] | None) -> list[Any] | None:
    """What identifies the state of a link (None: no link), kept in the plan to notice changes.

    Args:
        link: The link with the tool, or None.

    Returns:
        ``[revision, last_sync]`` of the link, or None when there is no link.
    """
    return None if not link else [int(link.get("revision", 0)), str(link.get("last_sync", ""))]


def _directed(action: str, kind: str, direction: str, here_exists: bool, there: bool) -> str:
    """What a per-image action becomes when one side wins (``pull``: the tool, ``push``: this project).

    Args:
        action: The action decided by the comparison.
        kind: Kind of conflict (``labels``, ``removed_local``, ``removed_remote``), used for a conflict.
        direction: ``pull`` (the tool wins) or ``push`` (this project wins).
        here_exists: Whether the image is still in the project.
        there: Whether the item exists in the tool (not used by the decision).

    Returns:
        The action to carry out.
    """
    if action == CONFLICT:
        side = "remote" if direction == PULL_ONLY else "local"
        action = RESOLVE[kind or "labels"][side]
    if direction == PULL_ONLY:
        action = {PUSH: PULL, REMOVE_REMOTE: RESTORE_LOCAL, RESTORE_REMOTE: REMOVE_LOCAL}.get(action, action)
        if action == RESTORE_LOCAL and not here_exists:
            action = SAME  # deleted here for good: cannot be brought back from the record
    else:
        action = {PULL: PUSH, REMOVE_LOCAL: RESTORE_REMOTE, RESTORE_LOCAL: REMOVE_REMOTE}.get(action, action)
    return action


# -- apply ----------------------------------------------------------------------
def _choice(value: Any, what: str) -> str | None:
    """Validate a side chosen to resolve conflicts.

    Args:
        value: ``local``, ``remote``, or None / empty for no choice.
        what: Name of the value, used in the error message.

    Returns:
        ``local`` or ``remote``, or None when there is no choice.

    Raises:
        SyncError: the value is something else.
    """
    if value in (None, ""):
        return None
    if value not in CHOICES:
        raise SyncError(f"{what} must be 'local' or 'remote'.", "resolution_invalid")
    return str(value)


# what a resolved conflict becomes: kind -> {choice: action}
RESOLVE = {
    "labels": {"local": PUSH, "remote": PULL},
    "removed_local": {"local": REMOVE_REMOTE, "remote": RESTORE_LOCAL},  # removed here, edited there
    "removed_remote": {"local": RESTORE_REMOTE, "remote": REMOVE_LOCAL},  # edited here, removed there
}


def _check_unchanged(
    decisions: dict[str, dict[str, Any]],
    current: dict[str, RemoteItem],
    project: Project,
    local: dict[str, list[Label]],
    records: dict[str, ImageRecord],
    ids: list[str],
    link_items: dict[str, dict[str, Any]],
) -> None:
    """Check that the images to carry out still match the plan (nothing changed on either side).

    Args:
        decisions: The decisions of the plan by image id (with the hashes seen then).
        current: The remote items as they are now, by key.
        project: The project.
        local: The labels of the project now, by image id.
        records: Every image of the project by id.
        ids: Ids of the images to check.
        link_items: The ``items`` of the link (labels at the last sync), as when the plan was made.

    Raises:
        RemoteChanged: the item or the image changed or appeared / disappeared since the plan.
    """
    for image_id in ids:
        d = decisions[image_id]
        item = current.get(d["key"])
        if (item is not None) != d.get("there", item is not None):
            raise RemoteChanged(CHANGED_REMOTE, "sync_remote_changed")
        rec = records.get(image_id)
        mine = local.get(image_id, []) if rec is not None else []
        entry = link_items.get(image_id)
        theirs = (
            _theirs(project, item.labels, mine, links.base_labels(entry) if entry else None)
            if item is not None
            else []
        )
        if item is not None and links.label_hash(theirs) != d["remote_hash"]:
            raise RemoteChanged(CHANGED_REMOTE, "sync_remote_changed")
        here = rec is not None and rec.status == ACTIVE
        if here != d.get("here", here):
            raise RemoteChanged(CHANGED_LOCAL, "sync_local_changed")
        if rec is not None and links.label_hash(local.get(image_id, [])) != d["local_hash"]:
            raise RemoteChanged(CHANGED_LOCAL, "sync_local_changed")


def _ids_of(final_action: dict[str, str], *actions: str) -> list[str]:
    """Images whose final action is one of ``actions``.

    Args:
        final_action: The action carried out, by image id.
        *actions: The actions to look for.

    Returns:
        The ids of the images with one of those actions.
    """
    return [i for i, a in final_action.items() if a in actions]


def apply_plan(
    store: ProjectStore,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    plan_id: str,
    resolutions: dict[str, str],
    default: str | None,
    progress: Progress,
    delete_tasks: bool = False,
) -> dict[str, Any]:
    """Carry out a saved plan (job ``sync_apply``).

    Conflicts take ``resolutions[image_id]`` (``"local"`` keeps ActiveLabel's side, ``"remote"`` the
    tool's), else ``default``; a conflict with neither is left as it is. Tasks deleted in ActiveLabel
    are deleted in the tool only with ``delete_tasks``.

    Args:
        store: projects.
        project_id: the project.
        tool: ``cvat``.
        settings: login of the tool.
        remote: the tool's remote adapter.
        plan_id: the saved plan to carry out.
        resolutions: ``{image_id: "local" | "remote"}``, the side chosen for each conflict.
        default: side for the conflicts not in ``resolutions`` (None leaves them as they are).
        progress: progress handle of the job (also checked for cancellation).
        delete_tasks: whether tasks deleted here are also deleted in the tool.

    Returns:
        Counts: ``pulled``, ``pushed``, ``added``, ``images_new`` and ``labels_new`` (images and labels
        sent for the first time, in new tasks or added), ``removed_remote``, ``removed_local``,
        ``restored_remote``, ``restored_local``, ``tasks_created``, ``tasks_imported``,
        ``tasks_deleted``, ``tasks_renamed``, ``project_renamed`` (whether the project's name was sent or
        taken), ``conflicts_left``, ``unchanged``, ``new_classes``, ``renamed_classes`` (classes renamed in
        the tool, renamed here too: ``[{from, to}]``), ``backup``, ``skipped``, ``warnings``,
        ``videos_as_images`` (names of the imported tasks whose original video could not be used, so they
        stay as images).

    Raises:
        RemoteChanged: the remote project or the project changed since the plan (nothing was written).
        SyncError: unknown plan or invalid choices.
    """
    project_dir = store.dir(project_id)
    plan = load_plan(project_dir, plan_id)
    if plan is None or plan.get("tool") != tool:
        raise SyncError("This comparison no longer exists. Synchronise again.", "plan_not_found")
    if "link" in plan and plan["link"] != _link_token(links.get_link(project_dir, tool)):
        # another synchronisation (or a send) ran after this comparison
        raise RemoteChanged(CHANGED_LINK, "sync_link_changed")
    fallback = _choice(default, "The default")
    decisions: dict[str, dict[str, Any]] = plan["decisions"]
    final_action: dict[str, str] = {}
    conflicts_left = 0
    for image_id, d in decisions.items():
        action = d["action"]
        if action == CONFLICT:
            choice = _choice(resolutions.get(image_id), "A resolution") or fallback
            if choice is None:
                conflicts_left += 1
                continue
            action = RESOLVE[d.get("kind") or "labels"][choice]
        final_action[image_id] = action

    # tasks deleted here: without delete_tasks their remote task is left exactly as it is
    planned_drop = {str(t["remote_task"]) for t in plan.get("tasks_delete") or []}
    if not delete_tasks:
        for image_id in [
            i
            for i, a in final_action.items()
            if a == REMOVE_REMOTE and _key_task(decisions[i]["key"]) in planned_drop
        ]:
            del final_action[image_id]

    project = store.get(project_id)
    manifest = Manifest(project_dir)
    tstore = TaskStore(project_dir)
    records = manifest.load()
    local = _local_labels(project_dir)
    rid = str(plan["remote"]["id"])
    to_create = bool(plan.get("create")) and not rid
    warnings: list[str] = []
    progress.update(message=f"Reading {tool}")
    current: dict[str, RemoteItem] = {}
    if to_create:
        if remote.find(settings, remote_name(project)) is not None:
            raise RemoteChanged(CHANGED_REMOTE, "sync_remote_changed")
    else:
        wanted = sorted({_key_task(d["key"]) for d in decisions.values() if d["key"]})
        if wanted:
            current = {it.key: it for it in remote.items(settings, rid, project.task, wanted)}
    checked = [i for i, a in final_action.items() if a not in (SAME, GONE, ADD)]
    link_now = links.get_link(project_dir, tool) or {}
    _check_unchanged(decisions, current, project, local, records, checked, link_now.get("items", {}))
    dropping = _safe_drops(remote, settings, plan, decisions, final_action, delete_tasks, warnings)
    for image_id in _ids_of(final_action, ADD):
        if (
            image_id not in records
            or links.label_hash(local.get(image_id, [])) != decisions[image_id]["local_hash"]
        ):
            raise RemoteChanged(CHANGED_LOCAL, "sync_local_changed")

    # classes renamed in the tool (checked again now): renamed here too, so they are not created again
    old_link = links.get_link(project_dir, tool) or {}
    renames: list[dict[str, str]] = []
    if not to_create and old_link.get("classes"):
        renames = renamed_classes(project, old_link["classes"], remote.classes(settings, rid))
    if renames:
        _rename_classes(store, project_id, renames)
        project = store.get(project_id)
        local = _local_labels(project_dir)

    pull_ids = _ids_of(final_action, PULL, RESTORE_LOCAL)
    new_labels = {
        i: links.carry_predictions(
            local.get(i, []), [Label.from_dict(x) for x in plan["remote_labels"].get(decisions[i]["key"], [])]
        )
        for i in pull_ids
    }
    found = diffing.diff(project_dir, pull_ids, new_labels, project) if pull_ids else {}
    classes_after = [*project.classes, *found.get("unknown_classes", [])]

    progress.update(message=f"Writing to {tool}")
    info = _info_obj(plan["remote"])
    if to_create:
        info = remote.create(settings, project)
        rid = info.id
    old = links.get_link(project_dir, tool)
    keep = old is not None and str(old["remote_id"]) == rid
    items: dict[str, dict[str, Any]] = dict(old["items"]) if old and keep else {}
    tmap: dict[str, Any] = dict(old.get("tasks", {})) if old and keep else {}
    push_ids = _ids_of(final_action, PUSH, RESTORE_REMOTE)
    if push_ids or _ids_of(final_action, ADD) or plan.get("tasks_create"):
        remote.ensure_classes(settings, rid, classes_after)

    # tasks never sent: created whole, in order, with jobs of job_size; excluded images -> deleted frames
    created = 0
    sent_images = 0
    sent_labels = 0
    for spec in plan.get("tasks_create") or []:
        progress.check()
        task = next((t for t in tstore.load() if t.id == spec["task"]), None)
        if task is None:
            continue
        ordered = [i for i in task.images if i in records]
        uploads = [(records[i], manifest.file_path(records[i]), local.get(i, [])) for i in ordered]
        rtask, keys = remote.add_task(
            settings, rid, project, task.name, task.job_size, uploads, _video_of(project_dir, task)
        )
        hidden = [_key_frame(keys[i]) for i in ordered if i in keys and records[i].status != ACTIVE]
        if hidden:
            remote.set_frames(settings, rtask.id, hidden, [])
        tmap[str(task.id)] = {
            "remote_task": rtask.id,
            "name": task.name,
            "jobs": {str(j.n): j.id for j in rtask.jobs},
        }
        for image_id in ordered:
            if image_id in keys:
                removed = records[image_id].status != ACTIVE
                if not removed:
                    sent_images += 1
                    sent_labels += len(local.get(image_id, []))
                items[image_id] = links.base_entry(keys[image_id], local.get(image_id, []), removed)
        created += 1
    new_keys: dict[str, str] = {}
    if _ids_of(final_action, ADD):
        uploads = [
            (records[i], manifest.file_path(records[i]), local.get(i, [])) for i in _ids_of(final_action, ADD)
        ]
        new_keys = remote.add(settings, rid, project, uploads)
        sent_images += len(new_keys)
        sent_labels += sum(len(local.get(i, [])) for i in new_keys)
    if push_ids:
        remote.write(settings, rid, project, {decisions[i]["key"]: local.get(i, []) for i in push_ids})

    # frames removed / restored in the tool (not for tasks that are deleted whole)
    frames: dict[str, tuple[list[int], list[int]]] = {}
    for image_id in _ids_of(final_action, REMOVE_REMOTE):
        key = decisions[image_id]["key"]
        if _key_task(key) not in dropping:
            frames.setdefault(_key_task(key), ([], []))[0].append(_key_frame(key))
    for image_id in _ids_of(final_action, RESTORE_REMOTE):
        key = decisions[image_id]["key"]
        frames.setdefault(_key_task(key), ([], []))[1].append(_key_frame(key))
    for rtask_id, (delete, restore) in frames.items():
        remote.set_frames(settings, rtask_id, delete, restore)
    deleted_tasks = 0
    for spec in plan.get("tasks_delete") or []:
        if str(spec["remote_task"]) in dropping:
            remote.delete_task(settings, str(spec["remote_task"]))
            deleted_tasks += 1
            tmap.pop(str(spec["task"]), None)

    renamed_tasks = _apply_renames(tstore, remote, settings, plan, tmap, warnings)
    recorded_name, renamed_project = _apply_project_rename(
        store, project_id, remote, settings, plan, rid, warnings
    )

    # Writing tracked objects makes the tool re-interpolate their neighbouring frames (and a tool may
    # give a track a new id): read the touched tasks again and take their tracked frames as they are now
    refreshed = _refresh_tracked(
        remote, settings, rid, project, decisions, push_ids, items, local, pull_ids, final_action
    )
    if refreshed:
        new_labels.update(refreshed)
        pull_ids = [*pull_ids, *[i for i in refreshed if i not in pull_ids]]

    progress.update(message="Updating the project")
    counts: dict[str, Any] = {"new_classes": [], "backup": "", "skipped": 0}
    if pull_ids:
        counts = diffing.apply(store, project_id, manifest, pull_ids, new_labels, "sync")
    removed_here = _ids_of(final_action, REMOVE_LOCAL)
    if removed_here:
        manifest.set_status(removed_here, EXCLUDED, f"deleted in {tool}")
    restored_here = [i for i in _ids_of(final_action, RESTORE_LOCAL) if i in records]
    if restored_here:
        manifest.set_status(restored_here, ACTIVE)

    imported = 0
    videos_as_images: list[str] = []
    remote_jobs: dict[str, dict[str, str]] = {}
    if plan.get("tasks_new"):
        remote_jobs = {rt.id: {str(j.n): j.id for j in rt.jobs} for rt in remote.tasks(settings, rid)}
    new_tasks = plan.get("tasks_new") or []
    counter = (
        FrameCounter(progress, sum(len(plan["new_items"].get(spec["remote_task"], [])) for spec in new_tasks))
        if new_tasks
        else None
    )
    for spec in plan.get("tasks_new") or []:
        progress.check()
        imported += _import_remote_task(
            store,
            project_id,
            tool,
            settings,
            remote,
            spec,
            plan["new_items"].get(spec["remote_task"], []),
            items,
            tmap,
            remote_jobs.get(str(spec["remote_task"]), {}),
            warnings,
            counter,
            videos_as_images,
        )

    final = _local_labels(project_dir)
    records = manifest.load()
    for image_id, action in final_action.items():
        key = new_keys.get(image_id, "") if action == ADD else str(decisions[image_id]["key"])
        if not key:
            continue
        rec = records.get(image_id)
        if (
            action == SAME
            and rec is not None
            and links.label_hash(final.get(image_id, [])) != decisions[image_id]["local_hash"]
        ):
            continue  # edited here after the comparison: the old base stays, so the next plan sends it
        removed = action in (REMOVE_LOCAL, REMOVE_REMOTE, GONE) or (
            action == SAME and bool((items.get(image_id) or {}).get("removed"))
        )
        labels = (
            final.get(image_id, []) if rec is not None else links.base_labels(items.get(image_id, {})) or []
        )
        items[image_id] = links.base_entry(key, labels, removed)
    for image_id in refreshed:  # tracked frames re-read after the push: their new base
        if image_id not in final_action and (items.get(image_id) or {}).get("key"):
            items[image_id] = links.base_entry(str(items[image_id]["key"]), final.get(image_id, []))
    for image_id, d in decisions.items():  # conflicts left as they are keep their old base
        if image_id not in final_action and d["key"]:
            items.setdefault(image_id, {"key": d["key"]})
    if not delete_tasks:  # tasks deleted here and kept in the tool: remembered, so the next plan leaves them
        for spec in plan.get("tasks_delete") or []:
            tid = str(spec["task"])
            if tid in tmap:
                tmap[tid] = {**tmap[tid], "kept_remote": True}
                dropping = {*dropping, str(spec["remote_task"])}
    if dropping:  # images of tasks gone on both sides, or left in the tool: nothing here to link
        items = {i: e for i, e in items.items() if _key_task(str(e.get("key", ""))) not in dropping}
    if not recorded_name:  # a comparison of one task or job leaves the recorded name as it was
        recorded_name = str((old or {}).get("remote_name") or "") if old and keep else ""
    record = links.new_link(info.url, rid, recorded_name or info.name, items)
    record["tasks"] = tmap
    record["classes"] = remote.classes(settings, rid)  # to notice a class renamed there next time
    if old and keep:
        record["created"] = old.get("created", record["created"])
    links.save_link(project_dir, tool, record)
    path = plan_file(project_dir, plan_id)
    if path is not None:
        path.unlink(missing_ok=True)
    return {
        "pulled": len(_ids_of(final_action, PULL)),
        "pushed": len(_ids_of(final_action, PUSH)),
        "added": len(new_keys),
        "images_new": sent_images,
        "labels_new": sent_labels,
        "removed_remote": len(_ids_of(final_action, REMOVE_REMOTE)),
        "removed_local": len(removed_here),
        "restored_remote": len(_ids_of(final_action, RESTORE_REMOTE)),
        "restored_local": len(restored_here),
        "tasks_created": created,
        "tasks_imported": imported,
        "tasks_deleted": deleted_tasks,
        "tasks_renamed": renamed_tasks,
        "project_renamed": renamed_project,
        "conflicts_left": conflicts_left,
        "unchanged": len(_ids_of(final_action, SAME, GONE)),
        "new_classes": counts["new_classes"],
        "renamed_classes": renames,
        "backup": counts["backup"],
        "skipped": counts["skipped"],
        "warnings": warnings,
        "videos_as_images": videos_as_images,
    }


def _plan_renames(
    scoped_tasks: list[ImageTask],
    mapping: dict[str, str],
    tmap: dict[str, Any],
    rtasks: list[RemoteTask],
    read: set[str] | None,
    direction: str,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Compare the name of each linked task with its remote task (docs/TASKS.md).

    Like the rest of the synchronisation it is three-way, against the name recorded in the link at the
    last synchronisation: only here changed -> sent; only there changed -> pulled; both changed to
    different names -> kept as they are and reported. ``push`` always sends the local name, ``pull`` always
    takes the remote one.

    Args:
        scoped_tasks: The local tasks in scope.
        mapping: Local task id -> remote task id.
        tmap: The ``tasks`` of the link (the recorded ``name``).
        rtasks: The remote tasks.
        read: The remote tasks that are read (None: all).
        direction: ``both``, ``pull`` or ``push``.

    Returns:
        ``(renames, names, conflicts)``: ``renames`` is ``[{task, remote_task, old, new, direction}]``
        (``direction`` is ``push`` or ``pull``); ``names`` is ``{local task id: name to record in the
        link}`` for the tasks whose names agree after the apply; ``conflicts`` is ``[{task, remote_task,
        local, remote}]`` for the tasks renamed on both sides.
    """
    by_id = {rt.id: rt for rt in rtasks}
    renames: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for t in scoped_tasks:
        rid = mapping.get(str(t.id))
        rt = by_id.get(rid) if rid else None
        if rt is None or (read is not None and rt.id not in read):
            continue
        if rt.name == t.name:
            names[str(t.id)] = t.name
            continue
        base = (tmap.get(str(t.id)) or {}).get("name")
        here, there = t.name != base and base is not None, rt.name != base and base is not None
        if direction == PULL_ONLY or (direction == BOTH and there and not here):
            way, old, new = PULL_ONLY, t.name, rt.name
        elif direction == PUSH_ONLY or not there:
            way, old, new = PUSH_ONLY, rt.name, t.name
        else:
            conflicts.append({"task": t.id, "remote_task": rt.id, "local": t.name, "remote": rt.name})
            continue
        renames.append({"task": t.id, "remote_task": rt.id, "old": old, "new": new, "direction": way})
        names[str(t.id)] = new
    return renames, names, conflicts


def _apply_renames(
    tstore: TaskStore,
    remote: RemoteProject,
    settings: ToolSettings,
    plan: dict[str, Any],
    tmap: dict[str, Any],
    warnings: list[str],
) -> int:
    """Carry out the planned task renames; one that fails is a warning and keeps its old recorded name.

    Args:
        tstore: The tasks of the project.
        remote: The tool's remote adapter.
        settings: Login of the tool.
        plan: The saved plan (``tasks_rename``, ``tasks_names``, ``tasks_rename_conflicts``).
        tmap: The ``tasks`` of the new link; receives the recorded names.
        warnings: Receives the readable warnings.

    Returns:
        The number of tasks renamed (here or in the tool).
    """
    failed: set[str] = set()
    done = 0
    for spec in plan.get("tasks_rename") or []:
        try:
            if spec["direction"] == PUSH_ONLY:
                remote.rename_task(settings, str(spec["remote_task"]), str(spec["new"]))
            else:
                tstore.rename(int(spec["task"]), str(spec["new"]))
            done += 1
        except (ToolError, ValueError, KeyError) as err:  # CodedError and TaskNotFound are among these
            failed.add(str(spec["task"]))
            warnings.append(f"The task '{spec['old']}' could not be renamed to '{spec['new']}': {err}")
    for conflict in plan.get("tasks_rename_conflicts") or []:
        warnings.append(
            f"The task '{conflict['local']}' was renamed here and in the tool ('{conflict['remote']}'): "
            "both names are kept."
        )
    for tid, name in (plan.get("tasks_names") or {}).items():
        if tid in tmap and tid not in failed:
            tmap[tid] = {**tmap[tid], "name": name}
    return done


def _apply_project_rename(
    store: ProjectStore,
    project_id: int,
    remote: RemoteProject,
    settings: ToolSettings,
    plan: dict[str, Any],
    rid: str,
    warnings: list[str],
) -> tuple[str, bool]:
    """Carry out the planned rename of the project, in the tool or here; a failure is a warning.

    Args:
        store: The projects.
        project_id: The project.
        remote: The tool's remote adapter.
        settings: Login of the tool.
        plan: The saved plan (``project_rename``, ``project_rename_conflict``, ``project_name``).
        rid: Id of the remote project.
        warnings: Receives the readable warnings.

    Returns:
        ``(recorded, renamed)``: the remote name to record in the link (empty when the plan does not
        compare names: a task or job scope, or a project created now) and whether a rename was done.
    """
    recorded = str(plan.get("project_name") or "")
    spec = plan.get("project_rename")
    renamed = False
    if spec:
        try:
            if spec["direction"] == PUSH_ONLY:
                remote.rename_project(settings, rid, str(spec["new"]))
            else:
                store.update(project_id, name=str(spec["new"]))
            renamed = True
        except (ToolError, ValueError, KeyError) as err:  # CodedError and ProjectNotFound are among these
            warnings.append(f"The project could not be renamed to '{spec['new']}': {err}")
            recorded = str(spec["old"] if spec["direction"] == PUSH_ONLY else plan["remote"]["name"])
    conflict = plan.get("project_rename_conflict")
    if conflict:
        warnings.append(
            f"The project was renamed here ('{conflict['local']}') and in the tool ('{conflict['remote']}'): "
            "both names are kept."
        )
    return recorded, renamed


def _safe_drops(
    remote: RemoteProject,
    settings: ToolSettings,
    plan: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    final_action: dict[str, str],
    delete_tasks: bool,
    warnings: list[str],
) -> set[str]:
    """Remote tasks that can really be deleted now (``delete_tasks``, planned, nothing of them is kept).

    A task is kept (with a warning) when any of its images ends otherwise than removed or equal on both
    sides, e.g. a conflict resolved or left as "remote". A task that has frames the plan knows nothing
    about (added in the tool after the link) is never deleted: nothing is written at all.

    Args:
        remote: The tool's remote adapter.
        settings: Login of the tool.
        plan: The saved plan (its ``tasks_delete``).
        decisions: The decisions of the plan by image id.
        final_action: The action finally taken for each image id.
        delete_tasks: Whether deleting tasks was asked for at all.
        warnings: List that receives a note for each task that is kept.

    Returns:
        The ids of the remote tasks to delete.

    Raises:
        RemoteChanged: a task has frames added in the tool that the plan does not know.
    """
    if not delete_tasks:
        return set()
    drop: set[str] = set()
    for spec in plan.get("tasks_delete") or []:
        rt = str(spec["remote_task"])
        mine = [i for i, d in decisions.items() if d["key"] and _key_task(d["key"]) == rt]
        if any(final_action.get(i) not in (REMOVE_REMOTE, SAME, GONE) for i in mine):
            name = spec.get("name") or rt
            warnings.append(
                f"The task '{name}' was not deleted in the tool: some of its images are kept there."
            )
            continue
        try:
            frames = remote.frames(settings, rt)
        except RemoteNotFound:
            drop.add(rt)  # already gone: nothing to check
            continue
        known = {_key_frame(decisions[i]["key"]) for i in mine}
        extra = [f for f, (_, deleted) in enumerate(frames) if not deleted and f not in known]
        if extra:
            raise RemoteChanged(
                f"The task '{spec.get('name') or rt}' in the tool has {len(extra)} image(s) that were added "
                "after the last synchronisation, so it is not deleted. Synchronise again.",
                "sync_task_has_new_images",
                name=str(spec.get("name") or rt),
                count=len(extra),
            )
        drop.add(rt)
    return drop


def _refresh_tracked(
    remote: RemoteProject,
    settings: ToolSettings,
    rid: str,
    project: Project,
    decisions: dict[str, dict[str, Any]],
    push_ids: list[str],
    items: dict[str, dict[str, Any]],
    local: dict[str, list[Label]],
    pull_ids: list[str],
    final_action: dict[str, str],
) -> dict[str, list[Label]]:
    """Tracked frames of the tasks a push touched, as the tool has them now: ``{image_id: labels}``.

    Only frames with a tracked object (here or there) are taken, and only when they had no pending
    change here (pushed now, or equal to their labels at the last synchronisation), so nothing made
    here is lost.

    Args:
        remote: The tool's remote adapter.
        settings: Login of the tool.
        rid: Id of the remote project.
        project: The project.
        decisions: The decisions of the plan by image id.
        push_ids: Ids of the images pushed.
        items: The ``items`` of the link (image id -> entry with the remote key).
        local: The labels of the project, by image id.
        pull_ids: Ids of the images being pulled.
        final_action: The action finally taken for each image id.

    Returns:
        ``{image_id: labels}`` for the frames to take from the tool; empty when nothing was pushed.
    """
    if not push_ids:
        return {}
    touched = sorted({_key_task(decisions[i]["key"]) for i in push_ids if decisions[i].get("key")})
    if not touched:
        return {}
    fresh = {it.key: it for it in remote.items(settings, rid, project.task, touched)}
    pushed = set(push_ids)
    out: dict[str, list[Label]] = {}
    for image_id, entry in items.items():
        key = str((entry or {}).get("key", ""))
        if _key_task(key) not in touched or image_id in pull_ids or entry.get("removed"):
            continue
        if final_action.get(image_id) not in (None, PUSH, SAME, RESTORE_REMOTE):
            continue
        item = fresh.get(key)
        if item is None:
            continue
        mine = local.get(image_id, [])
        theirs = _theirs(project, item.labels, mine, links.base_labels(entry))
        if not any(lb.track is not None for lb in [*mine, *theirs]):
            continue
        unchanged_here = image_id in pushed or links.same_labels(mine, links.base_labels(entry) or [])
        if unchanged_here and not links.same_labels(mine, theirs):
            out[image_id] = links.carry_predictions(mine, theirs)
    return out


def _video_of(project_dir: Path, task: ImageTask) -> tuple[Path, dict[str, Any]] | None:
    """The original video and selection of a task made from a video (None for an image task).

    Args:
        project_dir: Folder of the project.
        task: The local task.

    Returns:
        ``(video file, selection with the video name)``, or None when the task is not made from a
        video or the video is no longer there.
    """
    source = task.source or {}
    if not source.get("video"):
        return None
    store = VideoStore(project_dir)
    try:
        info = store.get(str(source["video"]))
        path = store.file(info)
    except Exception:  # noqa: BLE001 - video gone: send the frames as images
        return None
    return (path, {**source, "name": info.name}) if path.exists() else None


def _new_task_items(
    remote: RemoteProject, settings: ToolSettings, project: Project, rt: str, items: list[RemoteItem]
) -> list[dict[str, Any]]:
    """Frames of a task added in the tool, in order, deleted ones included (flagged ``deleted``).

    Deleted frames are brought too, as excluded images, so the task keeps the tool's positions and
    its jobs match one to one.

    Args:
        remote: The tool's remote adapter.
        settings: Login of the tool.
        project: The project (its task and classes clean the labels).
        rt: Id of the remote task.
        items: The remote items read now.

    Returns:
        ``{key, name, labels, deleted?}`` of every frame of the task, in frame order.
    """
    out: list[dict[str, Any]] = [
        {"key": it.key, "name": it.file_name, "labels": [lb.as_dict() for lb in _clean(project, it.labels)]}
        for it in items
        if _key_task(it.key) == rt
    ]
    present = {x["key"] for x in out}
    for frame, (name, deleted) in enumerate(remote.frames(settings, rt)):
        key = f"{rt}:{frame}"
        if deleted and key not in present:
            out.append({"key": key, "name": name, "labels": [], "deleted": True})
    return sorted(out, key=lambda x: _key_frame(x["key"]))


def _import_remote_task(
    store: ProjectStore,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    spec: dict[str, Any],
    new_items: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    tmap: dict[str, Any],
    remote_jobs: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    on_frame: Callable[[], None] | None = None,
    as_images: list[str] | None = None,
) -> int:
    """A task added in the tool becomes a local task: download its frames, labels, link. Returns 1 or 0.

    ``remote_jobs`` is ``{job number: remote job id}`` for the link's tasks mapping. A video task also
    brings its original video (``remote.video_source``): the local task is then made from the video
    (docs/VIDEO.md); when that is not possible the task stays as images and ``warnings`` gets a note.

    Args:
        store: projects.
        project_id: the project.
        tool: ``cvat``.
        settings: login of the tool.
        remote: the tool's remote adapter.
        spec: the planned task (``remote_task``, ``name``, ``job_size``).
        new_items: the frames of the remote task (see :func:`_new_task_items`).
        items: the ``items`` of the link; receives the entries of the imported images.
        tmap: the ``tasks`` of the link; receives the new local task.
        remote_jobs: ``{job number: remote job id}`` of the task.
        warnings: list that receives notes about problems that do not stop the import.
        on_frame: called after each frame is downloaded (the job's progress).
        as_images: list that receives the name of the task when its video was expected but not used.

    Returns:
        1 when a local task was created, 0 when there was nothing to import.
    """
    project_dir = store.dir(project_id)
    manifest = Manifest(project_dir)
    tstore = TaskStore(project_dir)
    ordered = sorted(new_items, key=lambda x: _key_frame(x["key"]))
    if not ordered:
        return 0
    files: list[tuple[str, io.BytesIO]] = []
    for x in ordered:
        files.append(
            (str(x["name"]), io.BytesIO(remote.frame(settings, spec["remote_task"], _key_frame(x["key"]))))
        )
        if on_frame is not None:
            on_frame()
    with tstore.locked():
        tstore.ensure()  # a project without tasks.json would put the new images in a "Task 1" of its own
        report = ingest_files(manifest, files)
        ids = list(report.ids)
        if not ids:
            return 0
        task = tstore.create(str(spec["name"] or "") or None, spec.get("job_size"), ids)
        placed = _placed(ordered, report.per_file)
        _attach_video(
            project_dir, tstore, manifest, remote, settings, spec, task, ids, placed, warnings, as_images
        )
    pulled: dict[str, list[Label]] = {}
    removed: list[str] = []
    for image_id, x in placed.items():
        if image_id not in ids:  # a frame identical to an image the project already has: linked, not pulled
            if image_id not in items and not x.get("deleted"):
                items[image_id] = links.base_entry(str(x["key"]), [Label.from_dict(lb) for lb in x["labels"]])
            continue
        if x.get("deleted"):
            removed.append(image_id)
            items[image_id] = links.base_entry(str(x["key"]), [], removed=True)
            continue
        pulled[image_id] = [Label.from_dict(lb) for lb in x["labels"]]
        items[image_id] = links.base_entry(str(x["key"]), pulled[image_id])
    if removed:  # deleted frames stay in the task (same positions, same jobs) as excluded images
        manifest.set_status(removed, EXCLUDED, f"deleted in {tool}")
    if any(pulled.values()):
        diffing.apply(store, project_id, manifest, list(pulled), pulled, "sync")
    tmap[str(task.id)] = {
        "remote_task": str(spec["remote_task"]),
        "name": task.name,
        "jobs": dict(remote_jobs or {}),
    }
    return 1


def _placed(ordered: list[dict[str, Any]], per_file: list[str | None]) -> dict[str, dict[str, Any]]:
    """Remote items by the image each one became (items are paired by position, never by file name).

    Two frames with the same base name in different folders stay apart, and a frame identical to an
    image the project already has is paired with that image. When two items give the same image the
    first one wins.

    Args:
        ordered: The remote items of the task, in frame order.
        per_file: The image id each item became (None when it was not stored), same order.

    Returns:
        ``{image_id: item}``.
    """
    placed: dict[str, dict[str, Any]] = {}
    for item, image_id in zip(ordered, per_file, strict=True):
        if image_id:
            placed.setdefault(image_id, item)
    return placed


def _attach_video(
    project_dir: Path,
    tstore: TaskStore,
    manifest: Manifest,
    remote: RemoteProject,
    settings: ToolSettings,
    spec: dict[str, Any],
    task: ImageTask,
    ids: list[str],
    placed: dict[str, dict[str, Any]],
    warnings: list[str] | None,
    as_images: list[str] | None = None,
) -> None:
    """Bring the original video of a remote video task and make the local task a task from a video.

    Never fails the import: without the video (not a video task, tool without the feature, download or
    probe problem, selection that does not fit) the task stays as images, with a warning when a video
    was expected.

    Args:
        project_dir: Folder of the project.
        tstore: Tasks of the project.
        manifest: Images of the project.
        remote: The tool's remote adapter.
        settings: Login of the tool.
        spec: The planned task (``remote_task``, ``name``).
        task: The local task that was created.
        ids: Ids of the images of that task.
        placed: The remote item of every image (see :func:`_placed`).
        warnings: List that receives notes when the video is not used; None to skip them.
        as_images: List that receives the task name when the video was expected but not used (the import
            report turns it into the ``video_as_images`` problem); None to skip it.
    """
    name = str(spec.get("name") or spec["remote_task"])
    folder = Path(tempfile.mkdtemp(prefix="activelabel-video-"))
    try:
        try:
            found = remote.video_source(settings, str(spec["remote_task"]), folder)
        except Exception as err:  # noqa: BLE001 - the frames are already imported
            log.warning("video of remote task %s not downloaded: %s", spec["remote_task"], err)
            found = None
            if as_images is not None:
                as_images.append(name)
            if warnings is not None:
                warnings.append(
                    f"The original video of task '{name}' could not be downloaded: it stays as images."
                )
        if found is None:
            return
        path, sel = found
        try:
            start, stop, step = int(sel["start"]), int(sel["stop"]), max(1, int(sel["step"]))
            fps, frames, _, _ = probe(path)
            if not 0 <= start <= stop < frames:
                raise ValueError("selection outside the video")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            video_id = digest.hexdigest()[:16]
            videos = VideoStore(project_dir)
            videos.folder.mkdir(parents=True, exist_ok=True)
            tmp = videos.folder / f".upload-{uuid.uuid4().hex}{path.suffix}"
            shutil.copyfile(path, tmp)
            info = videos.add(path.name, path.suffix.lower() or ".mp4", tmp, video_id, path.stat().st_size)
        except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as err:
            log.warning("video of remote task %s not usable: %s", spec["remote_task"], err)
            if as_images is not None:
                as_images.append(name)
            if warnings is not None:
                warnings.append(f"The original video of task '{name}' could not be used: it stays as images.")
            return
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    changes: dict[str, dict[str, Any]] = {}
    for image_id in ids:
        x = placed.get(image_id)
        if x is None:
            continue
        index = start + _key_frame(x["key"]) * step
        changes[image_id] = {"video": info.id, "frame": index, "time": index / info.fps}
    manifest.update(changes)
    tstore.set_source(
        task.id, {"video": info.id, "start": start, "stop": stop, "step": step, "fps": info.fps}
    )


# -- links made by send and import ------------------------------------------------
def link_from_items(
    store: ProjectStore,
    project_id: int,
    tool: str,
    info: RemoteInfo,
    items: list[RemoteItem],
    only: list[str] | None = None,
) -> int:
    """Create (or extend) the link with ``info``, mapping project images to remote items by file name.

    Args:
        store: projects.
        project_id: the project.
        tool: ``cvat``.
        info: the remote project.
        items: its items right now.
        only: image ids to map (every image when None).

    Returns:
        How many images were mapped. The base of each is the hash of the remote labels now (cleaned like
        a pull), so anything the tool did not keep shows up as a change in the next comparison.
    """
    project_dir = store.dir(project_id)
    records: list[ImageRecord] = Manifest(project_dir).all()
    if only is not None:
        wanted = set(only)
        records = [r for r in records if r.id in wanted]
    project = store.get(project_id)
    pairs = links.match_by_name(records, items)
    old = links.get_link(project_dir, tool)
    keep = old is not None and str(old["remote_id"]) == str(info.id)
    entries: dict[str, dict[str, Any]] = dict(old["items"]) if old and keep else {}
    for image_id, item in pairs.items():
        entries[image_id] = links.base_entry(item.key, _clean(project, item.labels))
    record = links.new_link(info.url, info.id, info.name, entries)
    if old and keep:
        record["created"] = old.get("created", record["created"])
    links.save_link(project_dir, tool, record)
    return len(pairs)


def link_after_import(
    store: ProjectStore, project_id: int, tool: str, settings: ToolSettings, remote: RemoteProject
) -> bool:
    """Link an imported project to the remote project (or task) it came from; returns whether it did.

    Args:
        store: projects.
        project_id: the imported project.
        tool: ``cvat``.
        settings: login of the tool.
        remote: the tool's remote adapter.

    Returns:
        True when a link was made; False when the project has no origin in that tool or
        no image could be matched.
    """
    project = store.get(project_id)
    origin = read_origin(store.dir(project_id))
    if not origin or origin.get("tool") != tool:
        return False
    rid = str(origin["remote_id"])
    try:
        info = remote.get(settings, rid)
    except ToolError:
        info = RemoteInfo(rid, str(origin.get("name", "")), str(origin.get("url", "")), 0)
    items = remote.items(settings, rid, project.task)
    return link_from_items(store, project_id, tool, info, items) > 0
