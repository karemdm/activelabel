"""Dataset versions: a frozen split of images and labels (see docs/DATASETS.md).

A version lives in ``<project>/datasets/v<N>.json`` and never changes once written. The split keeps
groups of related images (frames of one video, near duplicates) in one part and balances the classes
across parts with a seeded, deterministic iterative stratification.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.curation.issues import curation_path, load_duplicate_of
from activelabel.errors import CodedError
from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.images.video import VideoStore
from activelabel.labels import (
    BOX_INVALID,
    KEYPOINTS_COUNT_MISMATCH,
    MULTIPLE_CLASSES,
    NO_GEOMETRY,
    POLYGON_TOO_FEW_POINTS,
    Label,
    geometry_problem,
    load_annotations,
    validate_labels,
)
from activelabel.projects import Project
from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore

PARTS = ("train", "val", "test")
DEFAULT_SPLIT = {"train": 70, "val": 20, "test": 10}
MIN_IMAGES = 5  # a part with fewer images gets a warning
DRIFT = 5.0  # percentage points between requested and real share that trigger a warning
MAX_NOTE = 1000
VIDEO_MODES = ("auto", "whole")
MAX_GAP = 3600.0  # seconds
SEGMENTS_PER_VIDEO = 10  # a split video gives about this many segments...
SEGMENT_MIN_FRAMES = 2  # ...of at least this many frames...
SEGMENT_MAX_FRAMES = 30  # ...and at most this many (long videos give more segments)
MIN_SPLIT_FRAMES = 4  # a video with fewer frames always stays whole

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per folder, shared by every store object of the same project."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


class DatasetNotFound(KeyError):
    """No dataset version with this number."""


@dataclass
class DatasetOptions:
    """What the person chose for a version.

    Attributes:
        tasks: ids of the tasks that go in (empty = every task).
        split: percentages ``{"train", "val", "test"}`` (whole numbers, sum 100).
        kfold: None, or K (2..10): K folds of the non-test part (``val`` is then ignored).
        background: include active images without annotations (never for classification).
        seed: seed of the random generator.
        videos: "auto" (a video that is too big to reach the requested proportions is cut into
            segments spread over the parts) or "whole" (a video always stays in one part).
        gap_seconds: frames of a train part closer than this to a val/test segment of the same
            video are left out of the version (0 = keep them).
    """

    tasks: list[int] = field(default_factory=list)
    split: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPLIT))
    kfold: int | None = None
    background: bool = False
    seed: int = 42
    videos: str = "auto"
    gap_seconds: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "tasks": list(self.tasks),
            "split": dict(self.split),
            "kfold": self.kfold,
            "background": self.background,
            "seed": self.seed,
            "videos": self.videos,
            "gap_seconds": self.gap_seconds,
        }


def _whole(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodedError("not_whole_number", f"{what} must be a whole number.", what=what)
    return value


def parse_options(data: dict[str, Any] | None, existing_tasks: list[int]) -> DatasetOptions:
    """Validate the options given by a person (missing ones take the defaults).

    Raises:
        CodedError: percentages not whole numbers or not summing to 100, train empty, K-fold outside
            2..10, seed not a whole number, unknown task id, unknown ``videos`` mode, bad gap.
    """
    data = data or {}
    split = {**DEFAULT_SPLIT, **(data.get("split") or {})}
    unknown = set(split) - set(PARTS)
    if unknown:
        raise CodedError(
            "split_unknown_part", f"Unknown part in the split: {sorted(unknown)[0]}.", part=sorted(unknown)[0]
        )
    split = {k: _whole(split[k], f"The {k} percentage") for k in PARTS}
    if any(v < 0 or v > 100 for v in split.values()):
        raise CodedError("split_percentage_range", "Percentages must be between 0 and 100.")
    kfold = data.get("kfold")
    if kfold is not None:
        kfold = _whole(kfold, "K")
        if not 2 <= kfold <= 10:
            raise CodedError("kfold_range", "K-fold needs between 2 and 10 folds.")
        if split["test"] >= 100:
            raise CodedError("kfold_test_too_big", "The test percentage must leave images for the folds.")
    else:
        if sum(split.values()) != 100:
            raise CodedError("split_sum", "The percentages must add up to 100.")
        if split["train"] <= 0:
            raise CodedError("split_train_empty", "The training part needs more than 0%.")
    seed = _whole(data.get("seed", 42), "The seed")
    background = data.get("background", False)
    if not isinstance(background, bool):
        raise CodedError("background_not_bool", "Background must be true or false.")
    videos = data.get("videos", "auto")
    if videos not in VIDEO_MODES:
        raise CodedError("videos_mode", 'Videos must be "auto" or "whole".')
    gap = data.get("gap_seconds", 0.5)
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap):
        raise CodedError("gap_not_number", "The safety gap must be a number of seconds.")
    if not 0 <= gap <= MAX_GAP:
        raise CodedError(
            "gap_range", f"The safety gap must be between 0 and {MAX_GAP:g} seconds.", max=MAX_GAP
        )
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise CodedError("tasks_not_list", "Tasks must be a list of task ids.")
    tasks = [_whole(t, "A task id") for t in tasks]
    missing = [t for t in tasks if t not in existing_tasks]
    if missing:
        raise CodedError("task_not_found", f"Task {missing[0]} does not exist.", task=missing[0])
    return DatasetOptions(list(dict.fromkeys(tasks)), split, kfold, background, seed, videos, float(gap))


@dataclass
class Group:
    """Images that must stay together, with the classes they hold.

    A group with ``video`` set is one *segment* (contiguous frames, number ``index``) of a split video.
    """

    images: list[str]
    classes: Counter[str] = field(default_factory=Counter)  # class -> images of the group that have it
    video: str = ""
    index: int = -1


class _DisjointSets:
    """Union-find over strings: items join the structure when first seen."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """The representative of the set that holds ``item`` (a new item is a set of its own)."""
        parent = self._parent
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        """Merge the sets of ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def build_groups(
    candidates: list[str],
    videos: dict[str, str],
    duplicate_of: dict[str, str],
    classes_of: dict[str, set[str]],
    segment_of: dict[str, tuple[str, int]] | None = None,
) -> list[Group]:
    """Groups of candidate images: same video, or linked (transitively) as near duplicates.

    ``videos`` maps image ids (candidates or not) to their video. ``segment_of`` maps the images of
    videos being split to ``(video, segment number)``: those join their segment instead of the whole
    video, and near-duplicate links between two frames of the same split video are ignored.
    """
    segment_of = segment_of or {}
    split_videos = {v for v, _ in segment_of.values()}
    sets = _DisjointSets()
    for image_id in candidates:
        sets.find(image_id)
        if image_id in segment_of:
            video, number = segment_of[image_id]
            sets.union(image_id, f"segment:{video}#{number}")
        elif videos.get(image_id):
            sets.union(image_id, "video:" + videos[image_id])
    for dup, kept in duplicate_of.items():
        if videos.get(dup) in split_videos and videos.get(dup) == videos.get(kept):
            continue  # neighbouring frames are similar by nature: the safety gap handles the borders
        sets.union(dup, kept)  # through images that are not candidates too: they still link the others
    roots: dict[str, Group] = {}
    for image_id in candidates:  # keeps candidate order
        group = roots.setdefault(sets.find(image_id), Group([]))
        group.images.append(image_id)
        group.classes.update(classes_of.get(image_id, ()))
    for group in roots.values():
        keys = {segment_of.get(i) for i in group.images}
        only = next(iter(keys))
        if len(keys) == 1 and only is not None:  # a pure segment (not glued to other images)
            group.video, group.index = only
    return list(roots.values())


class _Placement:
    """The state of one ``assign`` call: what every part has got so far, and how to choose the next part."""

    def __init__(
        self, groups: list[Group], fractions: dict[str, float], rng: random.Random, rest: str | None
    ) -> None:
        self.parts = list(fractions)
        self.fractions = fractions
        self.rng = rng
        self.rest = rest
        self.out: dict[str, list[Group]] = {p: [] for p in self.parts}
        total_images = sum(len(g.images) for g in groups)
        self.totals: Counter[str] = Counter()
        for g in groups:
            self.totals.update(g.classes)
        self.want_images = {p: total_images * fractions[p] for p in self.parts}
        self.got_images = dict.fromkeys(self.parts, 0)
        self.got: dict[str, Counter[str]] = {p: Counter() for p in self.parts}

    def score(self, p: str, names: list[str]) -> tuple[float, float, float]:
        """How far part ``p`` is below its targets: for the rarest class, the other classes, the images."""
        deficits = [self.totals[c] * self.fractions[p] - self.got[p][c] for c in names]
        return (
            round(deficits[0], 6) if deficits else 0.0,
            round(sum(deficits[1:]), 6),
            round(self.want_images[p] - self.got_images[p], 6),
        )

    def pick(self, g: Group, candidates: list[str]) -> str:
        """The candidate part furthest below its targets for ``g`` (ties broken with the random generator)."""
        names = sorted(g.classes, key=lambda c: (self.totals[c], c))
        best = max(self.score(p, names) for p in candidates)
        return self.rng.choice([p for p in candidates if self.score(p, names) == best])

    def put(self, g: Group, p: str) -> None:
        """Give group ``g`` to part ``p``."""
        self.out[p].append(g)
        self.got_images[p] += len(g.images)
        self.got[p].update(g.classes)

    def place_video(self, segs: list[Group]) -> None:
        """Share the segments of one video (in time order) among the parts."""
        rest, parts, want_images, got_images = self.rest, self.parts, self.want_images, self.got_images
        m = len(segs)
        movable = [p for p in parts if p != rest]
        if not movable:
            for g in segs:
                self.put(g, parts[0])
            return
        need = {p: max(0.0, want_images[p] - got_images[p]) for p in movable}
        if rest is None:  # every segment goes to some part
            weights = need if sum(need.values()) > 0 else dict.fromkeys(movable, 1.0)
            raw = {p: m * weights[p] / sum(weights.values()) for p in movable}
            quota = {p: int(raw[p]) for p in movable}
            by_remainder = sorted(movable, key=lambda p: (quota[p] - raw[p], movable.index(p)))
            for p in by_remainder[: m - sum(quota.values())]:
                quota[p] += 1
            chosen = list(range(m))
        else:
            length = sum(len(g.images) for g in segs) / m
            quota = {p: int(need[p] / length + 0.5) for p in movable}
            for p in movable:  # a requested part is never left empty when it can have a segment
                if quota[p] == 0 and need[p] > 0 and got_images[p] == 0:
                    quota[p] = 1
            while sum(quota.values()) > m - 1:  # the rest keeps at least one segment
                quota[max(movable, key=lambda p: (quota[p], movable.index(p)))] -= 1
            total = sum(quota.values())
            chosen = sorted({int((k + 0.5) * m / total) for k in range(total)})
        remaining = dict(quota)
        last: tuple[int, str] | None = None
        taken: set[int] = set()
        for pos in chosen:
            candidates = [p for p in movable if remaining[p] > 0]
            if not candidates:
                break
            if last and segs[pos].index - last[0] == 1 and len(candidates) > 1:
                candidates = [p for p in candidates if p != last[1]] or candidates
            p = self.pick(segs[pos], candidates)
            remaining[p] -= 1
            self.put(segs[pos], p)
            taken.add(pos)
            last = (segs[pos].index, p)
        for pos, g in enumerate(segs):
            if pos not in taken:
                self.put(g, rest if rest is not None else movable[0])


def assign(
    groups: list[Group], fractions: dict[str, float], rng: random.Random, rest: str | None = None
) -> dict[str, list[Group]]:
    """Give each group to a part: iterative stratification, rarest classes first.

    A group goes to the part furthest below its target for the group's rarest class, then below its
    targets for the group's other classes, then below its image target; remaining ties are broken
    with ``rng``. Groups without classes (background) come last and are spread by image count.

    Segments of split videos (groups with ``video``) are placed first, video by video: each part
    other than ``rest`` gets as many segments as it still needs (proportional to its missing images),
    taken evenly spread over the video's timeline and never adjacent when avoidable; the other
    segments go to ``rest``. With ``rest=None`` (K-fold) every segment is shared among the parts the
    same way, neighbours going to different parts. Which part gets which chosen segment follows the
    class stratification.
    """
    parts = list(fractions)
    by_video: dict[str, list[Group]] = {}
    regular: list[Group] = []
    for g in groups:
        if g.video:
            by_video.setdefault(g.video, []).append(g)
        else:
            regular.append(g)
    for video in [v for v, segs in by_video.items() if len(segs) < 2]:
        regular += by_video.pop(video)
    placement = _Placement(groups, fractions, rng, rest)
    for segs in sorted(by_video.values(), key=lambda s: (-sum(len(g.images) for g in s), s[0].video)):
        placement.place_video(sorted(segs, key=lambda g: g.index))

    order = list(regular)
    rng.shuffle(order)
    inf = float("inf")
    totals = placement.totals
    order.sort(key=lambda g: (min((totals[c] for c in g.classes), default=inf), -len(g.images)))
    for g in order:
        placement.put(g, placement.pick(g, parts))
    return placement.out


def fingerprint(
    parts: dict[str, list[str]], folds: list[list[str]] | None, labels: dict[str, list[dict[str, Any]]]
) -> str:
    """SHA-256 of a canonical JSON of the parts, folds and labels."""
    canonical = json.dumps(
        {"parts": parts, "folds": folds, "labels": labels}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Plan:
    """The outcome of a split, before it is saved."""

    options: DatasetOptions
    type: str
    classes: list[str]
    parts: dict[str, list[str]]
    folds: list[list[str]] | None
    labels: dict[str, list[dict[str, Any]]]
    stats: dict[str, Any]
    warnings: list[dict[str, Any]]
    duplicates_analysed: bool
    videos: list[dict[str, Any]] = field(default_factory=list)
    groups: int = 0
    foldable: int = 0

    def empty_parts(self) -> list[str]:
        """Parts that were asked to have images (a share above 0) and have none (``fold`` for a fold)."""
        if self.folds is not None:
            empty = ["fold"] * any(not f for f in self.folds)
            return (["test"] if self.options.split["test"] > 0 and not self.parts["test"] else []) + empty
        return [p for p in PARTS if self.options.split[p] > 0 and not self.parts[p]]

    def max_kfold(self) -> int:
        """Largest K the images allow (at most 10): the groups left for the folds once the test part is taken.

        Below 2 K-fold is not possible.
        """
        return min(10, self.foldable)

    def sizes(self) -> dict[str, Any]:
        """Images per part (``{test, folds: [...]}`` with K-fold)."""
        if self.folds is not None:
            return {"test": len(self.parts["test"]), "folds": [len(f) for f in self.folds]}
        return {p: len(self.parts[p]) for p in PARTS}

    def fingerprint(self) -> str:
        """SHA-256 of the parts (or folds) and the labels."""
        return fingerprint(self.parts, self.folds, self.labels)

    def preview(self) -> dict[str, Any]:
        """What the interface shows before saving."""
        return {
            "type": self.type,
            "options": self.options.as_dict(),
            "sizes": self.sizes(),
            "stats": self.stats,
            "warnings": self.warnings,
            "duplicates_analysed": self.duplicates_analysed,
            "videos": self.videos,
            "empty_parts": self.empty_parts(),
            "groups": self.groups,
            "max_kfold": self.max_kfold(),
        }


def _part_stats(ids: list[str], labels: dict[str, list[Label]]) -> dict[str, Any]:
    objects: Counter[str] = Counter()
    background = 0
    for image_id in ids:
        found = labels.get(image_id) or []
        background += not found
        objects.update(lb.label for lb in found)
    return {"images": len(ids), "background": background, "objects": dict(sorted(objects.items()))}


def _slot_warnings(
    key: dict[str, Any],
    ids: list[str],
    fraction: float,
    total: int,
    biggest: tuple[int, str],
    classes: list[str],
    labels: dict[str, list[Label]],
) -> list[dict[str, Any]]:
    """Warnings of one part: a part with a share above 0 is expected to have images and every class."""
    if fraction <= 0 or not total:
        return []
    out: list[dict[str, Any]] = []
    present = {lb.label for i in ids for lb in labels.get(i) or []}
    out += [{"code": "class_missing", **key, "class": c} for c in classes if c not in present]
    if len(ids) < MIN_IMAGES:
        out.append({"code": "few_images", **key, "images": len(ids)})
    target = total * fraction
    real = 100.0 * len(ids) / total
    drifted = abs(real - 100.0 * fraction) > DRIFT
    size, cause = biggest
    if size > max(1, target) and drifted:  # a big group only matters when it moved the proportions
        out.append(
            {"code": "group_too_big", **key, "images": size, "target": round(target, 1), "cause": cause}
        )
    if drifted:
        out.append(
            {"code": "ratio_drift", **key, "expected": round(100.0 * fraction, 1), "actual": round(real, 1)}
        )
    return out


def _flatten(groups: list[Group], position: dict[str, int]) -> list[str]:
    return sorted((i for g in groups for i in g.images), key=position.__getitem__)


def _chunks(frames: int) -> list[tuple[int, int]]:
    """Bounds ``(start, stop)`` of the contiguous segments of a video with ``frames`` frames.

    Length = ``frames / 10`` rounded up, at least 2 and at most 30 frames (so a video gives about
    10 segments, more when it is long); a last piece of 1 frame joins the previous segment.
    """
    length = min(SEGMENT_MAX_FRAMES, max(SEGMENT_MIN_FRAMES, -(-frames // SEGMENTS_PER_VIDEO)))
    out = [(a, min(frames, a + length)) for a in range(0, frames, length)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < SEGMENT_MIN_FRAMES:
        out[-2:] = [(out[-2][0], frames)]
    return out


def _video_meta(project_dir: Path, tasks: list[Any]) -> tuple[dict[str, str], dict[str, float]]:
    """Names and frame rates of the project's videos (``{id: name}``, ``{id: fps}``)."""
    names: dict[str, str] = {}
    fps: dict[str, float] = {}
    try:
        for info in VideoStore(project_dir).list():
            names[info.id] = info.name
            fps[info.id] = info.fps
    except Exception:  # noqa: BLE001 - no video store: names stay empty, fps comes from the tasks
        pass
    for task in tasks:
        source = task.source or {}
        video = str(source.get("video", ""))
        if video and source.get("fps") and video not in fps:
            fps[video] = float(source["fps"])
    return names, fps


def _gap_ids(
    frames_of: dict[str, list[str]],
    times: dict[str, float | None],
    protected: set[str],
    droppable: set[str],
    seconds: float,
) -> dict[str, set[str]]:
    """Per video, the droppable frames within ``seconds`` of a protected frame (needs every frame's time)."""
    out: dict[str, set[str]] = {}
    if seconds <= 0:
        return out
    for video, ids in frames_of.items():
        at: dict[str, float] = {}
        for i in ids:
            t = times.get(i)
            if t is None:
                break
            at[i] = t
        if len(at) < len(ids):
            continue  # no time for some frame: nothing to measure the gap with
        near = sorted(at[i] for i in ids if i in protected)
        drop: set[str] = set()
        for i in ids if near else ():
            if i not in droppable:
                continue
            k = bisect_left(near, at[i])
            closest = min(abs(at[i] - near[j]) for j in (k - 1, k) if 0 <= j < len(near))
            if closest <= seconds + 1e-9:
                drop.add(i)
        if drop:
            out[video] = drop
    return out


def _segments(
    video: str, ids: list[str], segment_of: dict[str, tuple[str, int]], where: dict[str, dict[str, Any]]
) -> list[tuple[list[str], dict[str, Any]]]:
    """Segments of a cut video as ``(frames in time order, part key)``; every frame of one is in one part."""
    found: dict[int, list[str]] = {}
    for i in ids:
        if i in segment_of and segment_of[i][0] == video:
            found.setdefault(segment_of[i][1], []).append(i)
    return [(images, where[images[0]]) for _, images in sorted(found.items())]


def _too_few(slots: list[tuple[dict[str, Any], list[str], float]], total: int) -> bool:
    """True when a requested part is empty or has less than half of its share."""
    return any(share > 0 and len(ids) / total < share / 2 for _, ids, share in slots)


@dataclass
class _Attempt:
    """One way of splitting (videos whole, or some of them cut into segments)."""

    parts: dict[str, list[str]]
    folds: list[list[str]] | None
    slots: list[tuple[dict[str, Any], list[str], float]]
    biggest: tuple[int, str]
    groups: int
    foldable: int
    split: set[str]
    segment_of: dict[str, tuple[str, int]]


def common_count(sizes: Iterable[int]) -> int | None:
    """The most frequent of the numbers of points of pose labels (None when there are none)."""
    found = Counter(n for n in sizes if n)
    return found.most_common(1)[0][0] if found else None


def usable_labels(
    task: Task, labels: dict[str, list[Label]]
) -> tuple[dict[str, list[Label]], dict[str, Any]]:
    """Keep the labels a task can train on (see :func:`activelabel.labels.geometry_problem`).

    Labels without the shape the task needs, with a polygon of fewer than 3 points, with an invalid
    box or (pose) with another number of points than most are dropped; a classification image keeps
    only its first label (the class training uses). Images left without labels are not in the result.

    Returns:
        The usable labels per image and ``{"unusable": labels dropped, "keypoints": expected number of
        points or None, "warnings": [...]}`` with one warning per problem
        (``labels_without_geometry``, ``keypoints_count_mismatch``, ``multiple_classes``) giving ``count`` and
        up to 5 ``examples`` (image ids).
    """
    expected = (
        common_count(len(lb.keypoints or []) for items in labels.values() for lb in items)
        if task is Task.KEYPOINTS
        else None
    )
    check = validate_labels(task, labels, None, expected)
    kept: dict[str, list[Label]] = {}
    for image_id, items in labels.items():
        good = [lb for lb in items if geometry_problem(task, lb, expected) is None]
        if task is Task.CLASSIFICATION:
            good = good[:1]
        if good:
            kept[image_id] = good
    shape = (NO_GEOMETRY, BOX_INVALID, POLYGON_TOO_FEW_POINTS)
    warnings: list[dict[str, Any]] = []
    missing = sum(check.counts[p] for p in shape)
    if missing:
        examples = [i for p in shape for i in check.examples.get(p, [])][:5]
        warnings.append({"code": "labels_without_geometry", "count": missing, "examples": examples})
    if check.counts[KEYPOINTS_COUNT_MISMATCH]:
        warnings.append(
            {
                "code": "keypoints_count_mismatch",
                "count": check.counts[KEYPOINTS_COUNT_MISMATCH],
                "examples": check.examples.get(KEYPOINTS_COUNT_MISMATCH, []),
                "expected": expected,
            }
        )
    if check.counts[MULTIPLE_CLASSES]:
        warnings.append(
            {
                "code": "multiple_classes",
                "count": check.counts[MULTIPLE_CLASSES],
                "examples": check.examples.get(MULTIPLE_CLASSES, []),
            }
        )
    unusable = missing + check.counts[KEYPOINTS_COUNT_MISMATCH]
    return kept, {"unusable": unusable, "keypoints": expected, "warnings": warnings}


@dataclass
class _Splitter:
    """What every attempt of ``plan_split`` shares: the candidates, their groups' inputs and the targets."""

    options: DatasetOptions
    candidates: list[str]
    position: dict[str, int]
    classes_of: dict[str, set[str]]
    video_of: dict[str, str]
    duplicate_of: dict[str, str]
    frames_of: dict[str, list[str]]
    shares: dict[str, float]
    test_share: float

    def attempt(self, split_ids: set[str]) -> _Attempt:
        """Split with the videos in ``split_ids`` cut into segments (the others stay whole)."""
        options, candidates, shares, test_share = self.options, self.candidates, self.shares, self.test_share
        position, frames_of = self.position, self.frames_of
        segment_of: dict[str, tuple[str, int]] = {}
        for video in sorted(split_ids):
            for number, (a, b) in enumerate(_chunks(len(frames_of[video]))):
                for i in frames_of[video][a:b]:
                    segment_of[i] = (video, number)
        groups = build_groups(candidates, self.video_of, self.duplicate_of, self.classes_of, segment_of)
        rng = random.Random(options.seed)
        folds: list[list[str]] | None = None
        foldable = len(groups)
        slots: list[tuple[dict[str, Any], list[str], float]]
        if options.kfold is None:
            placed = assign(groups, shares, rng, rest="train")
            if test_share > 0:  # the groups K-fold would have left for its folds
                foldable = len(
                    assign(
                        groups,
                        {"pool": 1 - test_share, "test": test_share},
                        random.Random(options.seed),
                        rest="pool",
                    )["pool"]
                )
            parts = {p: _flatten(placed.get(p, []), position) for p in PARTS}
            slots = [({"part": p}, parts[p], options.split[p] / 100) for p in PARTS]
        else:
            pool = groups
            test_groups: list[Group] = []
            if test_share > 0:
                placed = assign(groups, {"pool": 1 - test_share, "test": test_share}, rng, rest="pool")
                pool, test_groups = placed["pool"], placed["test"]
            foldable = len(pool)
            if len(pool) < options.kfold:
                raise CodedError(
                    "kfold_too_few_groups",
                    f"There are only {len(pool)} groups of images for {options.kfold} folds.",
                    groups=len(pool),
                    folds=options.kfold,
                )
            k_share = (1 - test_share) / options.kfold
            by_fold = assign(pool, {str(n): 1 / options.kfold for n in range(options.kfold)}, rng)
            folds = [_flatten(by_fold[str(n)], position) for n in range(options.kfold)]
            parts = {"test": _flatten(test_groups, position)}
            slots = [({"part": "test"}, parts["test"], test_share)]
            slots += [({"part": "fold", "fold": n}, folds[n], k_share) for n in range(options.kfold)]
        largest = max(groups, key=lambda g: len(g.images))
        cause = "video" if any(self.video_of.get(i) for i in largest.images) else "duplicates"
        return _Attempt(
            parts,
            folds,
            slots,
            (len(largest.images), cause),
            len(groups),
            foldable,
            set(split_ids),
            segment_of,
        )


def _seconds(record: Any, fps: dict[str, float]) -> float | None:
    """The time of an image in its video (seconds), or None when it cannot be known."""
    if record.time is not None:
        return float(record.time)
    if record.frame is not None and fps.get(record.video):
        return record.frame / fps[record.video]
    return None


def _frame_no(record: Any, position: int) -> int:
    """The frame number of an image: the one in the manifest, or its position in the video's order."""
    return position if record.frame is None else record.frame


def plan_split(project_dir: Path, project: Project, options: DatasetOptions) -> Plan:
    """Compute a split (nothing is saved).

    Raises:
        CodedError: no image to split, or fewer image groups than K-fold needs.
    """
    tasks = TaskStore(project_dir).load()
    wanted = set(options.tasks) if options.tasks else {t.id for t in tasks}
    manifest = Manifest(project_dir).load()
    annotations = load_annotations(project_dir)
    people = {
        k: [lb for lb in v if lb.score is None]
        for k, v in (annotations.images if annotations else {}).items()
    }
    classification = project.task == Task.CLASSIFICATION
    chosen: list[str] = list(
        dict.fromkeys(
            i
            for task in tasks
            if task.id in wanted
            for i in task.images
            if i in manifest and manifest[i].status == ACTIVE
        )
    )
    labels, problems = usable_labels(project.task, {i: people[i] for i in chosen if people.get(i)})
    candidates = [i for i in chosen if labels.get(i) or (options.background and not people.get(i))]
    if classification:
        candidates = [i for i in candidates if labels.get(i)]
    if not candidates:
        if problems["unusable"]:
            raise CodedError(
                "no_usable_labels",
                f"None of the labels has the shape a {project.task.value} project needs.",
                task=project.task.value,
            )
        raise CodedError(
            "nothing_to_split", "There are no images to split: choose tasks with annotated images."
        )
    total = len(candidates)
    position = {i: n for n, i in enumerate(candidates)}
    classes_of = {i: {lb.label for lb in labels.get(i, [])} for i in candidates}
    video_of = {i: manifest[i].video for i in manifest}
    duplicate_of = load_duplicate_of(project_dir)
    names, fps = _video_meta(project_dir, tasks)
    inf = float("inf")
    frames_of: dict[str, list[str]] = {}
    for i in candidates:
        if manifest[i].video:
            frames_of.setdefault(manifest[i].video, []).append(i)
    for ids in frames_of.values():  # frame order = time order
        ids.sort(
            key=lambda i: (
                inf if manifest[i].frame is None else manifest[i].frame,
                inf if manifest[i].time is None else manifest[i].time,
                position[i],
            )
        )

    test_share = options.split["test"] / 100
    shares = {p: options.split[p] / 100 for p in PARTS if options.split[p] > 0}
    if options.kfold is None:
        targets = [total * s for p, s in shares.items() if p != "train"]
    else:
        targets = ([total * test_share] if test_share > 0 else []) + [
            total * (1 - test_share) / options.kfold
        ]

    big: set[str] = set()
    if options.videos == "auto" and targets:
        big = {v for v, ids in frames_of.items() if len(ids) >= MIN_SPLIT_FRAMES and len(ids) > min(targets)}
    splitter = _Splitter(
        options, candidates, position, classes_of, video_of, duplicate_of, frames_of, shares, test_share
    )
    final: _Attempt | None
    try:
        final = splitter.attempt(set())
    except ValueError:
        if not big:
            raise
        final = None
    if big and (final is None or _too_few(final.slots, total)):
        try:
            final = splitter.attempt(big)
        except ValueError:
            if final is None:
                raise
    assert final is not None
    split = final.split

    # frames of a train part (or of the folds) right next to a val/test segment are left out
    if final.folds is None:
        protected = set(final.parts["val"]) | set(final.parts["test"])
        droppable = set(final.parts["train"])
    else:
        protected = set(final.parts["test"])
        droppable = {i for f in final.folds for i in f}
    gap = _gap_ids(
        {v: frames_of[v] for v in split},
        {i: _seconds(manifest[i], fps) for i in candidates if manifest[i].video in split},
        protected,
        droppable,
        options.gap_seconds,
    )
    dropped = set().union(*gap.values())
    parts = {p: [i for i in ids if i not in dropped] for p, ids in final.parts.items()}
    folds = None if final.folds is None else [[i for i in f if i not in dropped] for f in final.folds]
    slots = [(key, [i for i in ids if i not in dropped], share) for key, ids, share in final.slots]

    found = sorted({c for s in classes_of.values() for c in s})
    ordered = [c for c in project.classes if c in found] + [c for c in found if c not in project.classes]
    warnings: list[dict[str, Any]] = list(problems["warnings"])
    for key, ids, share in slots:
        warnings += _slot_warnings(key, ids, share, total - len(dropped), final.biggest, ordered, labels)

    where = {i: key for key, ids, _ in final.slots for i in ids}  # every image's part (before the gap)
    videos: list[dict[str, Any]] = []
    for video, ids in frames_of.items():
        index = {i: n for n, i in enumerate(ids)}

        segs = _segments(video, ids, final.segment_of, where)
        parts_used = {tuple(key.items()) for _, key in segs}
        cut = video in split and len(parts_used) >= 2
        entry = {
            "video": video,
            "name": names.get(video, video),
            "frames": len(ids),
            "split": cut,
            "segments": [
                {
                    **key,
                    "first_frame": _frame_no(manifest[images[0]], index[images[0]]),
                    "last_frame": _frame_no(manifest[images[-1]], index[images[-1]]),
                }
                for images, key in segs
            ]
            if cut
            else [],
            "gap": len(gap.get(video, ())),
        }
        videos.append(entry)
        if cut:
            spread = [p for p in ("val", "test", "fold") if any(key["part"] == p for _, key in segs)]
            warnings.append(
                {
                    "code": "video_split",
                    "video": video,
                    "name": entry["name"],
                    "segments": len(segs),
                    "parts": spread,
                    "gap": entry["gap"],
                }
            )

    analysed = read_json(curation_path(project_dir), {}).get("duplicate_of") is not None
    if not analysed:
        warnings.append({"code": "no_curation"})
    if folds is None:
        stats: dict[str, Any] = {p: _part_stats(parts[p], labels) for p in PARTS}
    else:
        stats = {"test": _part_stats(parts["test"], labels), "folds": [_part_stats(f, labels) for f in folds]}
    if any(v["split"] for v in videos):
        stats["gap"] = len(dropped)
    used = {i for ids in parts.values() for i in ids} | {i for f in folds or [] for i in f}
    return Plan(
        options=options,
        type=project.task.value,
        classes=list(project.classes) + [c for c in found if c not in project.classes],
        parts=parts,
        folds=folds,
        labels={i: [lb.as_dict() for lb in labels[i]] for i in candidates if i in used and labels.get(i)},
        stats=stats,
        warnings=warnings,
        duplicates_analysed=analysed,
        videos=videos,
        groups=final.groups,
        foldable=final.foldable,
    )


def _summary(data: dict[str, Any], manifest_ids: set[str]) -> dict[str, Any]:
    """Summary of a stored version (without parts and labels)."""
    folds = data.get("folds")
    if folds is not None:
        sizes: dict[str, Any] = {"test": len(data["parts"].get("test", [])), "folds": [len(f) for f in folds]}
    else:
        sizes = {p: len(data["parts"].get(p, [])) for p in PARTS}
    ids = {i for v in data["parts"].values() for i in v} | {i for f in folds or [] for i in f}
    return {
        "version": data["version"],
        "created": data["created"],
        "note": data.get("note", ""),
        "type": data["type"],
        "classes": data["classes"],
        "options": data["options"],
        "sizes": sizes,
        "stats": data["stats"],
        "warnings": data["warnings"],
        "videos": data.get("videos", []),
        "incomplete": bool(ids - manifest_ids),
    }


class DatasetStore:
    """Dataset versions of one project, saved in ``<project>/datasets/v<N>.json``."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.dir = project_dir / "datasets"
        self._lock = _lock_for(self.dir)

    def _file(self, number: int) -> Path:
        return self.dir / f"v{number}.json"

    def _numbers(self) -> list[int]:
        if not self.dir.exists():
            return []
        return sorted(int(p.stem[1:]) for p in self.dir.glob("v*.json") if p.stem[1:].isdigit())

    def _manifest_ids(self) -> set[str]:
        return set(Manifest(self.project_dir).load())

    def create(self, plan: Plan, note: str = "") -> dict[str, Any]:
        """Save a plan as the next version (a number is never reused). Returns its summary.

        Raises:
            CodedError: note too long.
        """
        note = note.strip()
        if len(note) > MAX_NOTE:
            raise CodedError(
                "note_too_long", f"The note is too long (max {MAX_NOTE} characters).", max=MAX_NOTE
            )
        with self._lock:
            counter_file = self.dir / "counter.json"
            counter = int((read_json(counter_file, {}) or {}).get("next", 1))
            number = max([counter, *[n + 1 for n in self._numbers()]])
            data = {
                "version": number,
                "created": now(),
                "note": note,
                "type": plan.type,
                "classes": plan.classes,
                "options": plan.options.as_dict(),
                "parts": plan.parts,
                "folds": plan.folds,
                "labels": plan.labels,
                "stats": plan.stats,
                "warnings": plan.warnings,
                "videos": plan.videos,
                "fingerprint": plan.fingerprint(),
            }
            write_json(self._file(number), data)
            write_json(counter_file, {"next": number + 1})
        return _summary(data, self._manifest_ids())

    def versions(self) -> list[dict[str, Any]]:
        """Summaries of every version, newest first."""
        ids = self._manifest_ids()
        out = []
        for n in reversed(self._numbers()):
            data = read_json(self._file(n))
            if data:
                out.append(_summary(data, ids))
        return out

    def read(self, number: int) -> dict[str, Any]:
        """The stored file of a version, with the label copy.

        Raises:
            DatasetNotFound: unknown number.
        """
        data = read_json(self._file(number))
        if not data:
            raise DatasetNotFound(number)
        return dict(data)

    def get(self, number: int) -> dict[str, Any]:
        """A version without the label copy, plus its summary fields (``sizes``, ``incomplete``).

        Raises:
            DatasetNotFound: unknown number.
        """
        data = self.read(number)
        data.pop("labels", None)
        return {**data, **_summary(data, self._manifest_ids())}

    def delete(self, number: int) -> None:
        """Remove a version (its number is not reused).

        Raises:
            DatasetNotFound: unknown number.
        """
        with self._lock:
            if not self._file(number).exists():
                raise DatasetNotFound(number)
            self._file(number).unlink()

    def image_ids(self, number: int, part: str | None = None, fold: int | None = None) -> list[str]:
        """Image ids of a version, of one part or fold (all of them without ``part``/``fold``).

        With K-fold, ``fold`` i and part ``val`` give fold i, part ``train`` gives the other folds;
        ``fold`` alone means its validation part.

        Raises:
            DatasetNotFound: unknown number.
            CodedError: a part or fold that does not fit the version.
        """
        data = self.read(number)
        parts, folds = data["parts"], data.get("folds")
        if part is not None and part not in PARTS:
            raise CodedError("part_unknown", "The part must be train, val or test.")
        if folds is None:
            if fold is not None:
                raise CodedError("version_has_no_folds", "This version has no folds.")
            if part is None:
                return [i for p in PARTS for i in parts.get(p, [])]
            return list(parts.get(part, []))
        if part == "test":
            return list(parts.get("test", []))
        if part is None and fold is None:
            return [i for f in folds for i in f] + list(parts.get("test", []))
        if fold is None:
            raise CodedError("fold_required", "Give the fold (this version uses K-fold).")
        if not 0 <= fold < len(folds):
            raise CodedError("fold_not_found", "This fold does not exist.")
        if part == "train":
            return [i for n, f in enumerate(folds) if n != fold for i in f]
        return list(folds[fold])
