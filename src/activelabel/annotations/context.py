"""Shared state of one parse run: image lookup by name, problems, classes and geometry helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from activelabel.annotations.models import PROBLEMS, ParseResult, Problem
from activelabel.annotations.target import VideoTarget
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label, Point

MAX_UNMATCHED_EXAMPLES = 20
EPS = 1e-6


def norm_name(name: str) -> str:
    """Name with ``/`` separators and no leading ``./`` or ``/``."""
    n = name.replace("\\", "/").strip()
    while n.startswith("./"):
        n = n[2:]
    return n.lstrip("/")


def stem_parts(name: str) -> list[str]:
    """Path parts of a name without the extension of the last part."""
    parts = [p for p in norm_name(name).split("/") if p]
    if parts and "." in parts[-1]:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return parts


def bbox_of_points(points: Iterable[Sequence[float]]) -> list[float] | None:
    """``[x, y, w, h]`` around points (relative); None when there are none or the box is empty."""
    pts = list(points)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


class ParseContext:
    """Collects labels, classes and problems while a format parser runs."""

    def __init__(self, fmt: str, images: list[ImageRecord], target: VideoTarget | None = None) -> None:
        self.fmt = fmt
        self.images = images
        self.target = target
        self.labels: dict[str, list[Label]] = {}
        self.classes: list[str] = []
        self.files: list[str] = []
        self.keypoint_names: list[str] = []
        self.unmatched: list[str] = []
        self.unmatched_count = 0
        self._problems: dict[str, Problem] = {}
        self._class_seen: set[str] = set()
        self._claimed: set[str] = set()
        self._exact: dict[str, list[ImageRecord]] = {}
        self._lower: dict[str, list[ImageRecord]] = {}
        self._base: dict[str, list[ImageRecord]] = {}
        self._stem_suffix: dict[str, list[ImageRecord]] = {}
        for rec in images:
            n = norm_name(rec.name)
            self._exact.setdefault(n, []).append(rec)
            self._lower.setdefault(n.lower(), []).append(rec)
            self._base.setdefault(n.rsplit("/", 1)[-1].lower(), []).append(rec)
            parts = stem_parts(rec.name)
            for k in range(len(parts)):
                self._stem_suffix.setdefault("/".join(parts[k:]).lower(), []).append(rec)

    # -- problems and classes ------------------------------------------------
    def problem(self, code: str, example: str | None = None, n: int = 1) -> None:
        """Record a problem (aggregated by code)."""
        level, message = PROBLEMS[code]
        prob = self._problems.setdefault(code, Problem(level, code, message))
        prob.add(example, n)

    def add_class(self, name: str) -> None:
        """Remember a class name (first seen first)."""
        if name not in self._class_seen:
            self._class_seen.add(name)
            self.classes.append(name)

    # -- image lookup ----------------------------------------------------------
    def _find(self, name: str) -> tuple[ImageRecord | None, bool]:
        n = norm_name(name)
        base = n.rsplit("/", 1)[-1].lower()
        for index, key in ((self._exact, n), (self._lower, n.lower()), (self._base, base)):
            found = index.get(key)
            if found:
                return (found[0], False) if len(found) == 1 else (None, True)
        return None, False

    def match(self, *names: str, example: str | None = None) -> ImageRecord | None:
        """Image for the first of ``names`` that matches (exact name, then case-insensitive basename).

        Records an unmatched or ambiguous problem when none matches.
        """
        ambiguous = False
        for name in names:
            rec, amb = self._find(name)
            if rec is not None:
                return rec
            ambiguous = ambiguous or amb
        shown = example or (names[0] if names else "")
        if ambiguous:
            self.problem("ambiguous_image_name", shown)
        else:
            self.unmatch(shown)
        return None

    def match_stem(self, parts: list[str], example: str) -> ImageRecord | None:
        """Image for a label file path without extension (YOLO): ``labels/`` -> ``images/`` and suffixes."""
        swapped = list(parts)
        for i in range(len(swapped) - 2, -1, -1):
            if swapped[i].lower() == "labels":
                swapped[i] = "images"
                break
        ambiguous = False
        for variant in (swapped, parts):
            for k in range(len(variant)):
                found = self._stem_suffix.get("/".join(variant[k:]).lower())
                if found:
                    if len(found) == 1:
                        return found[0]
                    ambiguous = True
                    break
        if ambiguous:
            self.problem("ambiguous_image_name", example)
        else:
            self.unmatch(example)
        return None

    def unmatch(self, name: str) -> None:
        """Count an annotation entry without an image."""
        self.unmatched_count += 1
        if len(self.unmatched) < MAX_UNMATCHED_EXAMPLES and name not in self.unmatched:
            self.unmatched.append(name)
        self.problem("unmatched_entries", name)

    def at_frame(self, frame: int, example: str | None = None, n: int = 1) -> ImageRecord | None:
        """Image of the video task at this frame; an unused frame is counted (``n`` labels) and skipped."""
        rec = self.target.frames.get(frame) if self.target else None
        if rec is None:
            self.problem("frames_not_in_task", example or f"frame {frame}", n)
        return rec

    def claim(self, rec: ImageRecord, example: str | None = None) -> bool:
        """Start an entry for an image; False (and a problem) when it already has one."""
        if rec.id in self._claimed:
            self.problem("duplicate_image_entry", example or rec.name)
            return False
        self._claimed.add(rec.id)
        self.labels.setdefault(rec.id, [])
        return True

    def add_label(self, rec: ImageRecord, label: Label) -> None:
        """Add a label to an image and remember its class."""
        self.labels.setdefault(rec.id, []).append(label)
        self.add_class(label.label)

    # -- geometry ---------------------------------------------------------------
    def box(
        self, x: float, y: float, w: float, h: float, width: float, height: float, where: str
    ) -> list[float] | None:
        """Relative ``[x, y, w, h]`` from absolute values (use width=height=1 for relative input).

        Small overflow is clipped (and reported); boxes without area are dropped.
        """
        if not all(math.isfinite(v) for v in (x, y, w, h)) or width <= 0 or height <= 0:
            self.problem("invalid_annotation", where)
            return None
        x0, y0, x1, y1 = x / width, y / height, (x + w) / width, (y + h) / height
        if x1 - x0 <= EPS or y1 - y0 <= EPS:
            self.problem("box_degenerate", where)
            return None
        if x0 < -EPS or y0 < -EPS or x1 > 1 + EPS or y1 > 1 + EPS:
            self.problem("box_clipped", where)
        cx0, cy0, cx1, cy1 = _clip(x0), _clip(y0), _clip(x1), _clip(y1)
        if cx1 - cx0 <= EPS or cy1 - cy0 <= EPS:
            self.problem("box_degenerate", where)
            return None
        return [cx0, cy0, cx1 - cx0, cy1 - cy0]

    def ring(
        self, points: Sequence[Sequence[float]], width: float, height: float, where: str
    ) -> list[Point] | None:
        """Relative polygon ring from absolute points; None (with a problem) when unusable."""
        if len(points) < 3:
            self.problem("polygon_too_few_points", where)
            return None
        if width <= 0 or height <= 0 or not all(math.isfinite(v) for p in points for v in p[:2]):
            self.problem("invalid_annotation", where)
            return None
        return [[_clip(p[0] / width), _clip(p[1] / height)] for p in points]

    def point(
        self, x: float, y: float, v: float, width: float, height: float, where: str
    ) -> list[float] | None:
        """Relative keypoint ``[x, y, v]``; None when the numbers are not finite."""
        if not all(math.isfinite(t) for t in (x, y, v)) or width <= 0 or height <= 0:
            self.problem("invalid_annotation", where)
            return None
        return [_clip(x / width), _clip(y / height), float(int(v))]

    def result(self) -> ParseResult:
        """Final result of the run."""
        return ParseResult(
            format=self.fmt,
            labels=self.labels,
            classes=self.classes,
            files=self.files,
            images_total=len(self.images),
            unmatched_count=self.unmatched_count,
            unmatched=self.unmatched,
            problems=list(self._problems.values()),
            keypoint_names=self.keypoint_names,
        )
