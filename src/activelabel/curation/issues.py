"""Quality issues flagged from the measures, with thresholds saved per project.

``<project>/curation.json`` holds ``{"thresholds": {...overrides...}, "duplicate_of": {id: id}}``.
``ImageRecord`` has no field for the image a duplicate repeats, so that map lives in this file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.jobs import Progress
from activelabel.storage import read_json, write_json

DARK, BRIGHT, LOW_CONTRAST, BLURRY = "dark", "bright", "low_contrast", "blurry"
SMALL, ODD_ASPECT, GRAYSCALE, NEAR_DUPLICATE = "small", "odd_aspect", "grayscale", "near_duplicate"
#: The stored file is gone or is not the imported one any more (#620): found by :func:`flag_files`.
FILE_MISSING, FILE_CHANGED = "file_missing", "file_changed"
FILE_ISSUES = (FILE_MISSING, FILE_CHANGED)
ISSUES = (DARK, BRIGHT, LOW_CONTRAST, BLURRY, SMALL, ODD_ASPECT, GRAYSCALE, NEAR_DUPLICATE, *FILE_ISSUES)


@dataclass
class Thresholds:
    """Limits used to flag issues.

    Attributes:
        dark: brightness below this is ``dark`` (0..255).
        bright: brightness above this is ``bright`` (overexposed).
        low_contrast: contrast (gray standard deviation) below this is ``low_contrast``.
        blur_relative: sharpness below this fraction of the project median can be ``blurry``.
        blur_ceiling: ...but only if the sharpness is also below this value, so a project of
            uniformly soft images is not flagged as a whole.
        blur_floor: sharpness below this is always ``blurry``, whatever the rest of the project.
        min_side: images whose shortest side is below this (pixels) are ``small``.
        max_aspect: long side / short side above this is ``odd_aspect``.
        phash_distance: perceptual hashes this close (Hamming, 0..64) are near duplicates.
    """

    dark: float = 40.0
    bright: float = 215.0
    low_contrast: float = 20.0
    blur_relative: float = 0.3
    blur_ceiling: float = 150.0
    blur_floor: float = 10.0
    min_side: int = 224
    max_aspect: float = 3.0
    phash_distance: int = 6

    def __post_init__(self) -> None:
        """Check every value (finite, within its limits) and make the whole-number ones ``int``.

        Raises:
            CodedError: a value that is not a number, not finite, outside its limits or not whole
                (``curation_threshold_not_number``, ``_range`` or ``_not_whole``).
        """
        for name, (low, high) in LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CodedError("curation_threshold_not_number", f"{name} must be a number.", threshold=name)
            if not math.isfinite(value) or not low <= value <= high:
                raise CodedError(
                    "curation_threshold_range",
                    f"{name} must be between {low:g} and {high:g}.",
                    threshold=name,
                    min=low,
                    max=high,
                )
            if name in WHOLE:
                if value != int(value):
                    raise CodedError(
                        "curation_threshold_not_whole", f"{name} must be a whole number.", threshold=name
                    )
                setattr(self, name, int(value))
            else:
                setattr(self, name, float(value))


LIMITS: dict[str, tuple[float, float]] = {
    "dark": (0, 255),
    "bright": (0, 255),
    "low_contrast": (0, 255),
    "blur_relative": (0, 1),
    "blur_ceiling": (0, 1_000_000),
    "blur_floor": (0, 1_000_000),
    "min_side": (1, 100_000),
    "max_aspect": (1, 1_000),
    "phash_distance": (0, 64),
}
WHOLE = ("min_side", "phash_distance")


def curation_path(project_dir: Path) -> Path:
    """File with the thresholds and duplicate map of a project.

    Args:
        project_dir: folder of the project.

    Returns:
        ``<project_dir>/curation.json`` (it may not exist).
    """
    return project_dir / "curation.json"


def load_thresholds(project_dir: Path) -> Thresholds:
    """Thresholds of a project: the defaults with the saved overrides applied.

    Args:
        project_dir: folder of the project.

    Returns:
        The thresholds; saved values that are unknown or no longer valid are ignored.
    """
    saved = read_json(curation_path(project_dir), {}).get("thresholds", {})
    values: dict[str, Any] = {}
    for key, value in (saved if isinstance(saved, dict) else {}).items():
        if key not in LIMITS:
            continue
        try:
            Thresholds(**{key: value})  # a bad saved value (an older version allowed them) is ignored
        except (TypeError, ValueError):
            continue
        values[key] = value
    return Thresholds(**values)


def save_thresholds(project_dir: Path, thresholds: Thresholds) -> None:
    """Save only the values that differ from the defaults (keeps the duplicate map).

    Args:
        project_dir: folder of the project.
        thresholds: the limits to save.
    """
    data = read_json(curation_path(project_dir), {})
    default = asdict(Thresholds())
    data["thresholds"] = {k: v for k, v in asdict(thresholds).items() if v != default[k]}
    write_json(curation_path(project_dir), data)


def load_duplicate_of(project_dir: Path) -> dict[str, str]:
    """``{duplicate id: id of the image it repeats}`` from the last :func:`flag_issues` run.

    Args:
        project_dir: folder of the project.

    Returns:
        The map (empty when the images were never analysed).
    """
    return dict(read_json(curation_path(project_dir), {}).get("duplicate_of", {}))


def _save_duplicate_of(project_dir: Path, duplicate_of: dict[str, str]) -> None:
    """Save the duplicate map in ``curation.json`` (keeps the thresholds).

    Args:
        project_dir: folder of the project.
        duplicate_of: duplicate image id -> id of the image it repeats.
    """
    data = read_json(curation_path(project_dir), {})
    data["duplicate_of"] = duplicate_of
    write_json(curation_path(project_dir), data)


def _find(parent: list[int], i: int) -> int:
    """The root of ``i`` in the union-find forest ``parent`` (compresses the path it walks).

    Args:
        parent: the parent of each node (a root is its own parent); updated by the compression.
        i: the node to look up.

    Returns:
        The root of ``i``.
    """
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _group_duplicates(ids: list[str], hashes: list[str], max_distance: int) -> list[list[str]]:
    """Groups (size >= 2) of images whose hashes are within ``max_distance`` of a group member.

    Args:
        ids: image ids.
        hashes: the perceptual hash of each image (16 hex digits), in the order of ``ids``.
        max_distance: the largest Hamming distance (0..64) that links two images.

    Returns:
        The groups of ids, in the order of their first image.
    """
    parent = list(range(len(ids)))
    values = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    # Pigeonhole: two 64-bit hashes that differ in at most d bits agree entirely on at least one of
    # d + 1 chunks, so only hashes sharing a chunk value are compared (not every pair).
    chunks = max_distance + 1
    edges = np.linspace(0, 64, chunks + 1).astype(int)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        key = (values >> np.uint64(low)) & np.uint64((1 << (high - low)) - 1)
        order = np.argsort(key, kind="stable")
        sorted_keys = key[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        for begin, end in zip(starts, np.r_[starts[1:], len(order)], strict=True):
            bucket = order[begin:end]
            for n in range(len(bucket) - 1):
                xor = values[bucket[n + 1 :]] ^ values[bucket[n]]
                bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
                for offset in np.nonzero(bits <= max_distance)[0]:
                    parent[_find(parent, int(bucket[n]))] = _find(parent, int(bucket[n + 1 + offset]))
    groups: dict[int, list[str]] = {}
    for i, image_id in enumerate(ids):
        groups.setdefault(_find(parent, i), []).append(image_id)
    return [g for g in groups.values() if len(g) > 1]


def flag_issues(manifest: Manifest, thresholds: Thresholds | None = None) -> dict[str, int]:
    """Flag issues of active, measured images and save them in ``record.issues``.

    Blurriness is relative to the project: an image is ``blurry`` when its sharpness is below
    ``blur_floor`` (absolute), or below ``blur_relative`` times the project median while also
    below ``blur_ceiling``. The relative part adapts to the kind of images (a dataset of
    night photos is softer than one of text), the ceiling avoids flagging sharp images of an
    unusually sharp project, the floor catches hopelessly blurry images in a blurry project.

    Near duplicates are grouped by perceptual hash; in each group the image with the largest
    resolution (then the sharpest) is kept unflagged and the others get ``near_duplicate``.
    Which image each one repeats is saved in ``curation.json`` (``duplicate_of``). The same
    file cannot be stored twice (ids are content hashes), but identical video frames can (their ids
    are salted): they are near duplicates at distance 0. Excluded and unmeasured images are neither
    flagged nor counted.

    Args:
        manifest: images of the project.
        thresholds: limits; the project's saved ones by default.

    Returns:
        Number of images per issue (every issue is present, possibly 0).
    """
    thresholds = thresholds or load_thresholds(manifest.project_dir)
    active = manifest.active()
    measured = [r for r in active if r.metrics]
    sharpness = [r.metrics["sharpness"] for r in measured if "sharpness" in r.metrics]
    typical = median(sharpness) if sharpness else 0.0
    # the file problems found by flag_files stay: thresholds do not change them
    flags: dict[str, list[str]] = {r.id: [i for i in r.issues if i in FILE_ISSUES] for r in measured}

    for record in measured:
        m, issues = record.metrics, flags[record.id]
        if m.get("brightness", 128) < thresholds.dark:
            issues.append(DARK)
        if m.get("brightness", 128) > thresholds.bright:
            issues.append(BRIGHT)
        if m.get("contrast", 255) < thresholds.low_contrast:
            issues.append(LOW_CONTRAST)
        sharp = m.get("sharpness")
        if sharp is not None and (
            sharp < thresholds.blur_floor
            or (sharp < thresholds.blur_relative * typical and sharp < thresholds.blur_ceiling)
        ):
            issues.append(BLURRY)
        short, long = sorted((record.width, record.height))
        if short < thresholds.min_side:
            issues.append(SMALL)
        if short > 0 and long / short > thresholds.max_aspect:
            issues.append(ODD_ASPECT)
        if m.get("grayscale"):
            issues.append(GRAYSCALE)

    duplicate_of: dict[str, str] = {}
    hashed = [r for r in measured if r.phash]
    by_id = {r.id: r for r in hashed}
    groups = _group_duplicates([r.id for r in hashed], [r.phash for r in hashed], thresholds.phash_distance)
    for group in groups:
        best = max(
            group, key=lambda i: (by_id[i].width * by_id[i].height, by_id[i].metrics.get("sharpness", 0), i)
        )
        for image_id in group:
            if image_id != best:
                duplicate_of[image_id] = best
                flags[image_id].append(NEAR_DUPLICATE)

    current = {r.id: r.issues for r in measured}
    manifest.update({i: {"issues": v} for i, v in flags.items() if v != current[i]})
    _save_duplicate_of(manifest.project_dir, duplicate_of)
    final = {**{r.id: r.issues for r in active}, **flags}  # unmeasured images keep their file problems
    return {issue: sum(issue in v for v in final.values()) for issue in ISSUES}


def flag_files(manifest: Manifest, progress: Progress | None = None) -> dict[str, int]:
    """Check that every active image file is the imported one (flags ``file_missing``, ``file_changed``).

    The content is hashed (see :meth:`Manifest.file_problem`), so a file replaced by another of the same size
    is found too. An image whose file is fine again loses the flag.

    Args:
        manifest: images of the project.
        progress: reports progress and can cancel.

    Returns:
        ``{file_missing, file_changed}``: how many images have each problem.
    """
    active = manifest.active()
    if progress:
        progress.update(done=0, total=len(active), message="Checking the image files")
    changes: dict[str, dict[str, Any]] = {}
    counts = dict.fromkeys(FILE_ISSUES, 0)
    for n, record in enumerate(active, 1):
        problem = manifest.file_problem(record, deep=True)
        flag = {"missing": FILE_MISSING, "changed": FILE_CHANGED}.get(problem or "")
        kept = [i for i in record.issues if i not in FILE_ISSUES] + ([flag] if flag else [])
        if flag:
            counts[flag] += 1
        if kept != record.issues:
            changes[record.id] = {"issues": kept}
        if progress and n % 50 == 0:
            progress.update(done=n)
    manifest.update(changes)
    return counts


def reset_curation(manifest: Manifest) -> int:
    """Forget the whole analysis: measures, issues and hashes of every image, and the duplicate map.

    The thresholds stay. Exclusions are not touched.

    Args:
        manifest: images of the project.

    Returns:
        How many images had something to clear.
    """
    records = manifest.all()
    cleared = [r.id for r in records if r.metrics or r.issues or r.phash]
    manifest.update({r.id: {"metrics": {}, "issues": [], "phash": ""} for r in records})
    data = read_json(curation_path(manifest.project_dir), {})
    data.pop("duplicate_of", None)  # "not analysed" again: a new data split warns about it
    write_json(curation_path(manifest.project_dir), data)
    return len(cleared)
