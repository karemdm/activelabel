"""Synchronise a project with its remote project (see docs/SYNC.md): plan, then apply.

The plan compares, per image, three versions of the labels: ``base`` (hash stored in the link at the
last synchronisation), ``local`` (project ground truth) and ``remote`` (read from the tool). Remote
labels are cleaned like any pulled labels (:func:`activelabel.annotate.diff.clean_labels`) before they
are hashed, so both sides are compared in the project's own terms. An image without a known base
(matched by file name for the first time) counts as changed on both sides unless one side has no
labels, in which case the other side wins; equal labels are never a conflict.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from activelabel.annotate import diff as diffing
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import ACTIVE, ImageRecord, Manifest
from activelabel.importing.common import read_origin
from activelabel.jobs import Progress
from activelabel.labels import Label, load_annotations
from activelabel.projects import Project, ProjectStore
from activelabel.storage import now, read_json, write_json
from activelabel.sync import links
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteProject

PULL, PUSH, CONFLICT, SAME, ADD = "pull", "push", "conflict", "same", "add"
CHOICES = ("local", "remote")
MAX_EXAMPLES = 10
KEEP_PLANS = 10
_PLAN_ID = re.compile(r"^[0-9a-f]{6,32}$")
CHANGED_REMOTE = "The remote project changed since the comparison. Synchronise again."
CHANGED_LOCAL = "The project's annotations changed since the comparison. Synchronise again."


class SyncError(ToolError):
    """A readable problem of the synchronisation."""


class RemoteChanged(SyncError):
    """The remote project (or the project) changed after the plan was made."""


# -- helpers --------------------------------------------------------------------
def remote_name(project: Project) -> str:
    """Name the annotation tools give the remote project of a project."""
    return f"ActiveLabel {project.id} - {project.name}"


def plans_dir(project_dir: Path) -> Path:
    """Folder with the saved plans."""
    return project_dir / "sync"


def plan_file(project_dir: Path, plan_id: str) -> Path | None:
    """File of a saved plan (None when the id is malformed)."""
    return plans_dir(project_dir) / f"{plan_id}.json" if _PLAN_ID.match(plan_id) else None


def load_plan(project_dir: Path, plan_id: str) -> dict[str, Any] | None:
    """A saved plan, or None."""
    path = plan_file(project_dir, plan_id)
    data = read_json(path) if path is not None else None
    return data if isinstance(data, dict) else None


def _prune(project_dir: Path) -> None:
    files = sorted(plans_dir(project_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_PLANS:]:
        old.unlink(missing_ok=True)


def _side(labels: list[Label]) -> dict[str, Any]:
    per_class: dict[str, int] = {}
    for lb in labels:
        per_class[lb.label] = per_class.get(lb.label, 0) + 1
    return {"labels": len(labels), "per_class": per_class}


def _clean(project: Project, labels: list[Label]) -> list[Label]:
    return diffing.clean_labels(project, labels)[0]


def _local_labels(project_dir: Path) -> dict[str, list[Label]]:
    annotations = load_annotations(project_dir)
    return annotations.images if annotations else {}


def _info_dict(info: RemoteInfo | None, name: str) -> dict[str, Any]:
    if info is None:
        return {"id": "", "name": name, "url": "", "images": 0}
    return {"id": info.id, "name": info.name, "url": info.url, "images": info.images}


def _info_obj(data: dict[str, Any]) -> RemoteInfo:
    return RemoteInfo(str(data["id"]), str(data["name"]), str(data["url"]), int(data.get("images", 0)))


def _decide(entry: dict[str, Any] | None, local: list[Label], remote: list[Label]) -> str:
    """Pull / push / same / conflict from the labels at the last sync, here and there (geometry tolerant)."""
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
    """Diff of what a push does: ``{image_id: (remote labels now, labels after the push)}``."""
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
    if link:
        return remote.get(settings, str(link["remote_id"]))
    if remote_id:
        return remote.get(settings, str(remote_id))
    info = remote.find(settings, remote_name(project))
    if info is None and not create:
        raise SyncError("There is no remote project to synchronise with. Send the images first.")
    return info


def make_plan(
    store: ProjectStore,
    project_id: int,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    progress: Progress,
    create: bool = False,
    remote_id: str | None = None,
) -> dict[str, Any]:
    """Compare the project with its remote project and save the plan (job ``sync_plan``).

    Args:
        store: projects.
        project_id: the project.
        tool: ``cvat`` or ``labelstudio``.
        settings: login of the tool.
        remote: the tool's remote adapter.
        progress: job progress.
        create: when there is no link and no remote project with the project's name, plan to create
            it (it is created on apply, never while planning).
        remote_id: remote project to link when there is no link yet.

    Returns:
        The plan as described in docs/SYNC.md.

    Raises:
        SyncError: nothing to synchronise with.
        ToolError: the tool's own problems (readable message).
    """
    project = store.get(project_id)
    project_dir = store.dir(project_id)
    manifest = Manifest(project_dir)
    link = links.get_link(project_dir, tool)
    progress.update(message=f"Reading {tool}")
    info = _find_remote(remote, settings, project, link, remote_id, create)
    items = remote.items(settings, info.id, project.task) if info else []
    progress.update(message="Comparing")

    records = {r.id: r for r in manifest.all()}
    local = _local_labels(project_dir)
    link_items: dict[str, dict[str, Any]] = (link or {}).get("items", {})
    by_key = {it.key: it for it in items}
    mapped: dict[str, RemoteItem] = {}
    for image_id, entry in link_items.items():
        item = by_key.get(entry.get("key", ""))
        if image_id in records and item is not None:
            mapped[image_id] = item
    used = {it.key for it in mapped.values()}
    free_records = [r for i, r in records.items() if i not in mapped]
    mapped.update(links.match_by_name(free_records, [it for it in items if it.key not in used]))
    used = {it.key for it in mapped.values()}
    unmatched = [it for it in items if it.key not in used]

    decisions: dict[str, dict[str, Any]] = {}
    remote_labels: dict[str, list[dict[str, Any]]] = {}
    pull_new: dict[str, list[Label]] = {}
    push_changes: dict[str, tuple[list[Label], list[Label]]] = {}
    conflicts: list[dict[str, Any]] = []
    names = {i: r.name for i, r in records.items()}
    for image_id, item in mapped.items():
        progress.check()
        if records[image_id].status != ACTIVE:
            continue  # excluded locally: left alone
        mine = local.get(image_id, [])
        theirs = _clean(project, item.labels)
        action = _decide(link_items.get(image_id), mine, theirs)
        decisions[image_id] = {
            "key": item.key,
            "action": action,
            "local_hash": links.label_hash(mine),
            "remote_hash": links.label_hash(theirs),
        }
        if action in (PULL, CONFLICT):
            remote_labels[item.key] = [lb.as_dict() for lb in theirs]
        if action == PULL:
            pull_new[image_id] = theirs
        elif action == PUSH:
            push_changes[image_id] = (theirs, mine)
        elif action == CONFLICT:
            conflicts.append(
                {"image_id": image_id, "name": names[image_id], "local": _side(mine), "remote": _side(theirs)}
            )
    added = [r.id for r in records.values() if r.status == ACTIVE and r.id not in mapped]
    for image_id in added:
        mine = local.get(image_id, [])
        push_changes[image_id] = ([], mine)
        decisions[image_id] = {"key": "", "action": ADD, "local_hash": links.label_hash(mine)}

    plan_id = uuid.uuid4().hex[:10]
    remote_only = {"count": len(unmatched), "examples": [it.file_name for it in unmatched[:MAX_EXAMPLES]]}
    pushes = sum(1 for d in decisions.values() if d["action"] == PUSH)
    plan: dict[str, Any] = {
        "plan_id": plan_id,
        "tool": tool,
        "created": now(),
        "create": info is None,
        "remote": _info_dict(info, remote_name(project)),
        "decisions": decisions,
        "remote_labels": remote_labels,
        "remote_only": remote_only,
    }
    write_json(plans_dir(project_dir) / f"{plan_id}.json", plan)
    _prune(project_dir)
    return {
        "plan_id": plan_id,
        "tool": tool,
        "remote": plan["remote"],
        "create": plan["create"],
        "pull": len(pull_new),
        "push": pushes,
        "add": len(added),
        "conflicts": conflicts,
        "remote_only": remote_only,
        "unchanged": sum(1 for d in decisions.values() if d["action"] == SAME),
        "diff_pull": diffing.diff(project_dir, list(pull_new), pull_new, project),
        "diff_push": _push_summary(project, names, push_changes),
    }


# -- apply ----------------------------------------------------------------------
def _choice(value: Any, what: str) -> str | None:
    if value in (None, ""):
        return None
    if value not in CHOICES:
        raise SyncError(f"{what} must be 'local' or 'remote'.")
    return str(value)


def _check_unchanged(
    plan: dict[str, Any],
    current: dict[str, RemoteItem],
    project: Project,
    local: dict[str, list[Label]],
    needed: list[str],
) -> None:
    for image_id in needed:
        entry = plan["decisions"][image_id]
        item = current.get(entry["key"])
        if item is None or links.label_hash(_clean(project, item.labels)) != entry["remote_hash"]:
            raise RemoteChanged(CHANGED_REMOTE)
        if links.label_hash(local.get(image_id, [])) != entry["local_hash"]:
            raise RemoteChanged(CHANGED_LOCAL)


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
) -> dict[str, Any]:
    """Carry out a saved plan (job ``sync_apply``).

    Conflicts take ``resolutions[image_id]`` (``"local"`` keeps ActiveLabel's labels, ``"remote"``
    brings the remote ones), else ``default``; a conflict with neither is left as it is.

    Returns:
        Counts: ``pulled``, ``pushed``, ``added``, ``conflicts_left``, ``unchanged``, ``new_classes``,
        ``backup``, ``skipped``.

    Raises:
        RemoteChanged: the remote project or the project changed since the plan (nothing was written).
        SyncError: unknown plan or invalid choices.
    """
    project_dir = store.dir(project_id)
    plan = load_plan(project_dir, plan_id)
    if plan is None or plan.get("tool") != tool:
        raise SyncError("This comparison no longer exists. Synchronise again.")
    fallback = _choice(default, "The default")
    decisions: dict[str, dict[str, Any]] = plan["decisions"]
    pull_ids = [i for i, d in decisions.items() if d["action"] == PULL]
    push_ids = [i for i, d in decisions.items() if d["action"] == PUSH]
    add_ids = [i for i, d in decisions.items() if d["action"] == ADD]
    conflicts_left = 0
    for image_id, d in decisions.items():
        if d["action"] != CONFLICT:
            continue
        choice = _choice(resolutions.get(image_id), "A resolution") or fallback
        if choice == "remote":
            pull_ids.append(image_id)
        elif choice == "local":
            push_ids.append(image_id)
        else:
            conflicts_left += 1

    project = store.get(project_id)
    manifest = Manifest(project_dir)
    records = manifest.load()
    local = _local_labels(project_dir)
    rid = str(plan["remote"]["id"])
    to_create = bool(plan.get("create")) and not rid
    progress.update(message=f"Reading {tool}")
    current: dict[str, RemoteItem] = {}
    if to_create:
        if remote.find(settings, remote_name(project)) is not None:
            raise RemoteChanged(CHANGED_REMOTE)
    else:
        current = {it.key: it for it in remote.items(settings, rid, project.task)}
    _check_unchanged(plan, current, project, local, pull_ids + push_ids)
    for image_id in add_ids:
        if (
            image_id not in records
            or links.label_hash(local.get(image_id, [])) != decisions[image_id]["local_hash"]
        ):
            raise RemoteChanged(CHANGED_LOCAL)

    new_labels = {
        i: [Label.from_dict(x) for x in plan["remote_labels"][decisions[i]["key"]]] for i in pull_ids
    }
    found = diffing.diff(project_dir, pull_ids, new_labels, project) if pull_ids else {}
    classes_after = [*project.classes, *found.get("unknown_classes", [])]

    progress.update(message=f"Writing to {tool}")
    info = _info_obj(plan["remote"])
    if to_create:
        info = remote.create(settings, project)
        rid = info.id
    new_keys: dict[str, str] = {}
    if push_ids or add_ids:
        remote.ensure_classes(settings, rid, classes_after)
    if push_ids:
        remote.write(settings, rid, project, {decisions[i]["key"]: local.get(i, []) for i in push_ids})
    if add_ids:
        uploads = [(records[i], manifest.file_path(records[i]), local.get(i, [])) for i in add_ids]
        new_keys = remote.add(settings, rid, project, uploads)

    progress.update(message="Updating the project")
    counts: dict[str, Any] = {"new_classes": [], "backup": "", "skipped": 0}
    if pull_ids:
        counts = diffing.apply(store, project_id, manifest, pull_ids, new_labels)
    final = _local_labels(project_dir)
    old = links.get_link(project_dir, tool)
    keep = old is not None and str(old["remote_id"]) == rid
    items: dict[str, dict[str, str]] = dict(old["items"]) if old and keep else {}
    for image_id, d in decisions.items():
        key = new_keys.get(image_id, "") if d["action"] == ADD else str(d["key"])
        if not key:
            continue
        if d["action"] == CONFLICT and image_id not in pull_ids and image_id not in push_ids:
            items.setdefault(image_id, {"key": key})
        else:
            items[image_id] = links.base_entry(key, final.get(image_id, []))
    record = links.new_link(info.url, rid, info.name, items)
    if old and keep:
        record["created"] = old.get("created", record["created"])
    links.save_link(project_dir, tool, record)
    path = plan_file(project_dir, plan_id)
    if path is not None:
        path.unlink(missing_ok=True)
    return {
        "pulled": len(pull_ids),
        "pushed": len(push_ids),
        "added": len(new_keys),
        "conflicts_left": conflicts_left,
        "unchanged": sum(1 for d in decisions.values() if d["action"] == SAME),
        "new_classes": counts["new_classes"],
        "backup": counts["backup"],
        "skipped": counts["skipped"],
    }


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
        tool: ``cvat`` or ``labelstudio``.
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


def remote_gone(err: ToolError) -> bool:
    """Whether a tool error means the remote project does not exist (HTTP 404), not a network problem."""
    return "not found" in str(err).lower()


def existing_remote(
    store: ProjectStore, project: Project, tool: str, settings: ToolSettings, remote: RemoteProject
) -> dict[str, Any] | None:
    """The 409 detail of a send when the remote project already exists (None when it does not).

    A link counts, and so does a remote project called like the project in the tool (or like the one
    the project was imported from).
    """
    project_dir = store.dir(project.id)
    link = links.get_link(project_dir, tool)
    info: RemoteInfo | None = None
    if link:
        try:
            info = remote.get(settings, str(link["remote_id"]))
        except ToolError as err:
            if remote_gone(err):  # deleted in the tool: the link means nothing any more
                links.remove_link(project_dir, tool)
                link = None
            else:  # unreachable: the link still stands
                name, url = str(link.get("remote_name", "")), str(link.get("url", ""))
                info = RemoteInfo(str(link["remote_id"]), name, url, len(link.get("items", {})))
    if link is None:
        info = remote.find(settings, remote_name(project))
        origin = read_origin(project_dir)
        if info is None and origin and origin.get("tool") == tool and origin.get("name"):
            info = remote.find(settings, str(origin["name"]))
    if info is None:
        return None
    return {
        "code": "remote_exists",
        "tool": tool,
        "remote": _info_dict(info, info.name),
        "linked": link is not None,
        "options": ["cancel", "sync"],
    }


def link_after_send(
    store: ProjectStore,
    project: Project,
    tool: str,
    settings: ToolSettings,
    remote: RemoteProject,
    ref: dict[str, Any],
    image_ids: list[str],
) -> bool:
    """Link the project to the remote project a send just created. Returns whether a link was made.

    The remote project id comes from the tool's ``ref`` (``project_id``) when it has one, else the
    project is looked up by the name the tool gave it (``project_name`` or the usual one).
    """
    if ref.get("project_id"):
        info: RemoteInfo | None = remote.get(settings, str(ref["project_id"]))
    else:
        info = remote.find(settings, str(ref.get("project_name") or remote_name(project)))
    if info is None:
        return False
    items = remote.items(settings, info.id, project.task)
    return link_from_items(store, project.id, tool, info, items, only=image_ids) > 0


def link_after_import(
    store: ProjectStore, project_id: int, tool: str, settings: ToolSettings, remote: RemoteProject
) -> bool:
    """Link an imported project to the remote project (or task) it came from; returns whether it did."""
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
