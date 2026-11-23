"""Dataset versions: a frozen split of images and labels (see docs/DATASETS.md).

A version lives in ``<project>/datasets/v<N>.json`` and never changes once written. The split keeps
groups of related images (frames of one video, near duplicates) in one part and balances the classes
across parts with a seeded, deterministic iterative stratification.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activelabel.curation.issues import curation_path, load_duplicate_of
from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.labels import Label, load_annotations
from activelabel.projects import Project
from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore

PARTS = ("train", "val", "test")
DEFAULT_SPLIT = {"train": 70, "val": 20, "test": 10}
MIN_IMAGES = 5  # a part with fewer images gets a warning
DRIFT = 5.0  # percentage points between requested and real share that trigger a warning
MAX_NOTE = 1000

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
    """

    tasks: list[int] = field(default_factory=list)
    split: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPLIT))
    kfold: int | None = None
    background: bool = False
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "tasks": list(self.tasks),
            "split": dict(self.split),
            "kfold": self.kfold,
            "background": self.background,
            "seed": self.seed,
        }


def _whole(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be a whole number.")
    return value


def parse_options(data: dict[str, Any] | None, existing_tasks: list[int]) -> DatasetOptions:
    """Validate the options given by a person (missing ones take the defaults).

    Raises:
        ValueError: percentages not whole numbers or not summing to 100, train empty, K-fold outside
            2..10, seed not a whole number, unknown task id.
    """
    data = data or {}
    split = {**DEFAULT_SPLIT, **(data.get("split") or {})}
    unknown = set(split) - set(PARTS)
    if unknown:
        raise ValueError(f"Unknown part in the split: {sorted(unknown)[0]}.")
    split = {k: _whole(split[k], f"The {k} percentage") for k in PARTS}
    if any(v < 0 or v > 100 for v in split.values()):
        raise ValueError("Percentages must be between 0 and 100.")
    kfold = data.get("kfold")
    if kfold is not None:
        kfold = _whole(kfold, "K")
        if not 2 <= kfold <= 10:
            raise ValueError("K-fold needs between 2 and 10 folds.")
        if split["test"] >= 100:
            raise ValueError("The test percentage must leave images for the folds.")
    else:
        if sum(split.values()) != 100:
            raise ValueError("The percentages must add up to 100.")
        if split["train"] <= 0:
            raise ValueError("The training part needs more than 0%.")
    seed = _whole(data.get("seed", 42), "The seed")
    background = data.get("background", False)
    if not isinstance(background, bool):
        raise ValueError("Background must be true or false.")
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a list of task ids.")
    tasks = [_whole(t, "A task id") for t in tasks]
    missing = [t for t in tasks if t not in existing_tasks]
    if missing:
        raise ValueError(f"Task {missing[0]} does not exist.")
    return DatasetOptions(list(dict.fromkeys(tasks)), split, kfold, background, seed)


@dataclass
class Group:
    """Images that must stay together, with the classes they hold."""

    images: list[str]
    classes: Counter[str] = field(default_factory=Counter)  # class -> images of the group that have it


def build_groups(
    candidates: list[str],
    videos: dict[str, str],
    duplicate_of: dict[str, str],
    classes_of: dict[str, set[str]],
) -> list[Group]:
    """Groups of candidate images: same video, or linked (transitively) as near duplicates."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for image_id in candidates:
        find(image_id)
        if videos.get(image_id):
            union(image_id, "video:" + videos[image_id])
    for dup, kept in duplicate_of.items():
        union(dup, kept)  # through images that are not candidates too: they still link the others
    roots: dict[str, Group] = {}
    for image_id in candidates:  # keeps candidate order
        group = roots.setdefault(find(image_id), Group([]))
        group.images.append(image_id)
        group.classes.update(classes_of.get(image_id, ()))
    return list(roots.values())


def assign(groups: list[Group], fractions: dict[str, float], rng: random.Random) -> dict[str, list[Group]]:
    """Give each group to a part: iterative stratification, rarest classes first.

    A group goes to the part furthest below its target for the group's rarest class, then below its
    targets for the group's other classes, then below its image target; remaining ties are broken
    with ``rng``. Groups without classes (background) come last and are spread by image count.
    """
    parts = list(fractions)
    out: dict[str, list[Group]] = {p: [] for p in parts}
    total_images = sum(len(g.images) for g in groups)
    totals: Counter[str] = Counter()
    for g in groups:
        totals.update(g.classes)
    want_images = {p: total_images * fractions[p] for p in parts}
    got_images = dict.fromkeys(parts, 0)
    got: dict[str, Counter[str]] = {p: Counter() for p in parts}
    order = list(groups)
    rng.shuffle(order)
    inf = float("inf")
    order.sort(key=lambda g: (min((totals[c] for c in g.classes), default=inf), -len(g.images)))
    for g in order:
        names = sorted(g.classes, key=lambda c: (totals[c], c))

        def score(p: str, names: list[str] = names) -> tuple[float, float, float]:
            deficits = [totals[c] * fractions[p] - got[p][c] for c in names]
            return (
                round(deficits[0], 6) if deficits else 0.0,
                round(sum(deficits[1:]), 6),
                round(want_images[p] - got_images[p], 6),
            )

        best = max(score(p) for p in parts)
        chosen = rng.choice([p for p in parts if score(p) == best])
        out[chosen].append(g)
        got_images[chosen] += len(g.images)
        got[chosen].update(g.classes)
    return out


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
    biggest: int,
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
    if biggest > target:
        out.append({"code": "group_too_big", **key, "images": biggest, "target": round(target, 1)})
    real = 100.0 * len(ids) / total
    if abs(real - 100.0 * fraction) > DRIFT:
        out.append(
            {"code": "ratio_drift", **key, "expected": round(100.0 * fraction, 1), "actual": round(real, 1)}
        )
    return out


def _flatten(groups: list[Group], position: dict[str, int]) -> list[str]:
    return sorted((i for g in groups for i in g.images), key=position.__getitem__)


def plan_split(project_dir: Path, project: Project, options: DatasetOptions) -> Plan:
    """Compute a split (nothing is saved).

    Raises:
        ValueError: no image to split, or fewer image groups than K-fold needs.
    """
    tasks = TaskStore(project_dir).load()
    wanted = set(options.tasks) if options.tasks else {t.id for t in tasks}
    manifest = Manifest(project_dir).load()
    annotations = load_annotations(project_dir)
    labels = {
        k: [lb for lb in v if lb.score is None]
        for k, v in (annotations.images if annotations else {}).items()
    }
    classification = project.task == Task.CLASSIFICATION
    chosen: list[str] = []
    for task in tasks:
        if task.id in wanted:
            chosen += [i for i in task.images if i in manifest and manifest[i].status == ACTIVE]
    candidates = [
        i for i in dict.fromkeys(chosen) if labels.get(i) or (options.background and not classification)
    ]
    if not candidates:
        raise ValueError("There are no images to split: choose tasks with annotated images.")
    position = {i: n for n, i in enumerate(candidates)}
    classes_of = {i: {lb.label for lb in labels.get(i, [])} for i in candidates}
    groups = build_groups(
        candidates, {i: manifest[i].video for i in candidates}, load_duplicate_of(project_dir), classes_of
    )
    biggest = max(len(g.images) for g in groups)
    rng = random.Random(options.seed)
    test_share = options.split["test"] / 100

    parts: dict[str, list[str]]
    folds: list[list[str]] | None = None
    slots: list[tuple[dict[str, Any], list[str], float]]
    if options.kfold is None:
        shares = {p: options.split[p] / 100 for p in PARTS if options.split[p] > 0}
        placed = assign(groups, shares, rng)
        parts = {p: _flatten(placed.get(p, []), position) for p in PARTS}
        slots = [({"part": p}, parts[p], options.split[p] / 100) for p in PARTS]
    else:
        pool = groups
        test_groups: list[Group] = []
        if test_share > 0:
            placed = assign(groups, {"pool": 1 - test_share, "test": test_share}, rng)
            pool, test_groups = placed["pool"], placed["test"]
        if len(pool) < options.kfold:
            raise ValueError(f"There are only {len(pool)} groups of images for {options.kfold} folds.")
        k_share = (1 - test_share) / options.kfold
        by_fold = assign(pool, {str(n): 1 / options.kfold for n in range(options.kfold)}, rng)
        folds = [_flatten(by_fold[str(n)], position) for n in range(options.kfold)]
        parts = {"test": _flatten(test_groups, position)}
        slots = [({"part": "test"}, parts["test"], test_share)]
        slots += [({"part": "fold", "fold": n}, folds[n], k_share) for n in range(options.kfold)]

    found = sorted({c for s in classes_of.values() for c in s})
    ordered = [c for c in project.classes if c in found] + [c for c in found if c not in project.classes]
    warnings: list[dict[str, Any]] = []
    for key, ids, share in slots:
        warnings += _slot_warnings(key, ids, share, len(candidates), biggest, ordered, labels)
    analysed = read_json(curation_path(project_dir), {}).get("duplicate_of") is not None
    if not analysed:
        warnings.append({"code": "no_curation"})
    if folds is None:
        stats: dict[str, Any] = {p: _part_stats(parts[p], labels) for p in PARTS}
    else:
        stats = {"test": _part_stats(parts["test"], labels), "folds": [_part_stats(f, labels) for f in folds]}
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
            ValueError: note too long.
        """
        note = note.strip()
        if len(note) > MAX_NOTE:
            raise ValueError(f"The note is too long (max {MAX_NOTE} characters).")
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
            ValueError: a part or fold that does not fit the version.
        """
        data = self.read(number)
        parts, folds = data["parts"], data.get("folds")
        if part is not None and part not in PARTS:
            raise ValueError("The part must be train, val or test.")
        if folds is None:
            if fold is not None:
                raise ValueError("This version has no folds.")
            if part is None:
                return [i for p in PARTS for i in parts.get(p, [])]
            return list(parts.get(part, []))
        if part == "test":
            return list(parts.get("test", []))
        if part is None and fold is None:
            return [i for f in folds for i in f] + list(parts.get("test", []))
        if fold is None:
            raise ValueError("Give the fold (this version uses K-fold).")
        if not 0 <= fold < len(folds):
            raise ValueError("This fold does not exist.")
        if part == "train":
            return [i for n, f in enumerate(folds) if n != fold for i in f]
        return list(folds[fold])
