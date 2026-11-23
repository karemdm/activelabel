"""Links between a project and one remote project in CVAT or Label Studio (``<project>/links.json``).

Format (docs/SYNC.md)::

    {"<tool>": {"url", "remote_id", "remote_name", "created", "last_sync",
                "items": {"<image_id>": {"key": "<remote item key>", "base": "<label hash>"}}}}

The ``base`` of an image is the :func:`label_hash` of its labels at the last synchronisation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.storage import now, read_json, write_json
from activelabel.sync.remote import RemoteItem

LINKS_FILE = "links.json"
DECIMALS = 4
_lock = threading.RLock()


# -- hash -----------------------------------------------------------------------
def _round(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return round(float(value), DECIMALS)


def _canonical(label: Label) -> str:
    data: dict[str, Any] = {"label": label.label}
    if label.bbox:
        data["bbox"] = _round(label.bbox)
    if label.polygons:
        data["polygons"] = _round(label.polygons)
    if label.keypoints:
        data["keypoints"] = _round(label.keypoints)
    if label.track is not None:
        data["track"] = label.track
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def label_hash(labels: list[Label]) -> str:
    """Stable SHA-256 (hex) of the labels of one image in the internal format.

    Every label is written as canonical JSON (class, bbox, polygons, keypoints, track id when there is
    one; the score is left out) with coordinates rounded to 4 decimals, and the list is sorted, so the
    order of the labels and geometry noise below 0.00005 of the image size do not change the hash. No
    labels give the hash of the empty list.
    """
    body = "\n".join(sorted(_canonical(lb) for lb in labels))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


TOLERANCE = 0.002  # 0.2 % of the image side: about one pixel, the noise of a round trip through a tool


def _close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b, strict=True))
    return a == b


def _label_close(a: Label, b: Label, tol: float) -> bool:
    if a.label != b.label or a.track != b.track:
        return False
    return all(_close(getattr(a, f), getattr(b, f), tol) for f in ("bbox", "polygons", "keypoints"))


def same_labels(a: list[Label], b: list[Label], tol: float = TOLERANCE) -> bool:
    """True when both lists have the same labels, geometry within ``tol`` (relative), in any order."""
    if len(a) != len(b):
        return False
    left = list(b)
    for label in a:
        match = next((i for i, other in enumerate(left) if _label_close(label, other, tol)), None)
        if match is None:
            return False
        left.pop(match)
    return True


def _geometry_close(a: Label, b: Label, tol: float) -> bool:
    return a.label == b.label and all(
        _close(getattr(a, f), getattr(b, f), tol) for f in ("bbox", "polygons", "keypoints")
    )


def carry_predictions(old: list[Label], new: list[Label], tol: float = TOLERANCE) -> list[Label]:
    """``new`` with the ``score`` and ``model`` of the pre-annotations of ``old`` they still match.

    A label of ``new`` with the same class and geometry (within ``tol``, track id ignored) as a model
    pre-annotation of ``old`` (one with a score or a model) is that same untouched label that went
    through a tool that does not keep scores, so it keeps them. A label that matches nothing was made
    or edited by a person: it stays without score (reviewed). Every old label is used at most once.
    """
    free = [lb for lb in old if lb.score is not None or lb.model is not None]
    out: list[Label] = []
    for label in new:
        match = next((i for i, o in enumerate(free) if _geometry_close(o, label, tol)), None)
        if match is None or label.score is not None:
            out.append(label)
            continue
        source = free.pop(match)
        out.append(Label(**{**label.__dict__, "score": source.score, "model": source.model}))
    return out


def base_entry(key: str, labels: list[Label], removed: bool = False) -> dict[str, Any]:
    """Link entry of one image: its remote key, its labels at this synchronisation (and their hash).

    ``removed``: the image was removed on both sides at this synchronisation (excluded or deleted here,
    deleted frame there), so a later restore on one side is seen as a change.
    """
    entry: dict[str, Any] = {
        "key": key,
        "base": label_hash(labels),
        "labels": [lb.as_dict() for lb in labels],
    }
    if removed:
        entry["removed"] = True
    return entry


def base_labels(entry: dict[str, Any]) -> list[Label] | None:
    """Labels of an image at the last synchronisation, or None when unknown."""
    if "labels" not in entry:
        return None
    return [Label.from_dict(x) for x in entry["labels"]]


# -- store ----------------------------------------------------------------------
def _file(project_dir: Path) -> Path:
    return project_dir / LINKS_FILE


def load_links(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Every link of a project by tool."""
    data = read_json(_file(project_dir), {})
    return data if isinstance(data, dict) else {}


def get_link(project_dir: Path, tool: str) -> dict[str, Any] | None:
    """The link of a project with one tool, or None."""
    link = load_links(project_dir).get(tool)
    return link if isinstance(link, dict) else None


def save_link(project_dir: Path, tool: str, link: dict[str, Any]) -> None:
    """Create or replace the link with one tool (its ``revision`` goes up by one: plans notice it)."""
    with _lock:
        data = load_links(project_dir)
        before = data.get(tool)
        link["revision"] = (int(before.get("revision", 0)) if isinstance(before, dict) else 0) + 1
        data[tool] = link
        write_json(_file(project_dir), data)


def remove_link(project_dir: Path, tool: str) -> bool:
    """Forget the link with one tool (the remote project is untouched). Returns whether there was one."""
    with _lock:
        data = load_links(project_dir)
        if tool not in data:
            return False
        del data[tool]
        write_json(_file(project_dir), data)
        return True


def new_link(
    url: str, remote_id: str, remote_name: str, items: dict[str, dict[str, str]] | None = None
) -> dict[str, Any]:
    """A link record, created and synchronised now."""
    stamp = now()
    return {
        "url": url,
        "remote_id": str(remote_id),
        "remote_name": remote_name,
        "created": stamp,
        "last_sync": stamp,
        "items": dict(items or {}),
    }


def public_view(tool: str, link: dict[str, Any]) -> dict[str, Any]:
    """What the API shows for a link."""
    return {
        "tool": tool,
        "url": link.get("url", ""),
        "remote_id": link.get("remote_id", ""),
        "remote_name": link.get("remote_name", ""),
        "last_sync": link.get("last_sync", ""),
        "images": len(link.get("items", {})),
    }


# -- matching -------------------------------------------------------------------
def match_by_name(records: list[ImageRecord], items: list[RemoteItem]) -> dict[str, RemoteItem]:
    """Pair remote items with project images by file name: ``{image_id: item}``.

    The full name is tried first (sub-folders included), then the base name ignoring case. A name is
    paired only when it is unique on both sides (two images or two items with the same name are
    ambiguous and stay unpaired, never silently collapsed). Every image and every item is used at
    most once.
    """
    by_name: dict[str, list[ImageRecord]] = {}
    by_base: dict[str, list[ImageRecord]] = {}
    for r in records:
        by_name.setdefault(r.name, []).append(r)
        by_base.setdefault(Path(r.name.replace("\\", "/")).name.lower(), []).append(r)
    item_names: dict[str, int] = {}
    for item in items:
        item_names[item.file_name] = item_names.get(item.file_name, 0) + 1
    pairs: dict[str, RemoteItem] = {}
    rest: list[RemoteItem] = []
    for item in items:
        same = by_name.get(item.file_name, [])
        if len(same) == 1 and item_names[item.file_name] == 1 and same[0].id not in pairs:
            pairs[same[0].id] = item
        else:
            rest.append(item)
    # images sent from ActiveLabel keep the stored file name "<image id>.<ext>" in the remote tool
    by_id = {r.id: r for r in records}
    bases: dict[str, int] = {}
    for item in rest:
        base = Path(item.file_name.replace("\\", "/")).name.lower()
        bases[base] = bases.get(base, 0) + 1
    for item in rest:
        base = Path(item.file_name.replace("\\", "/")).name.lower()
        record = by_id.get(Path(base).stem)
        if record is not None and record.id not in pairs:
            pairs[record.id] = item
            continue
        candidates = [r for r in by_base.get(base, []) if r.id not in pairs]
        if len(candidates) == 1 and bases[base] == 1:
            pairs[candidates[0].id] = item
    return pairs
