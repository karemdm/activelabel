"""Shared state of one parse run: image lookup by name, problems, classes and geometry helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from activelabel.annotations.models import PROBLEMS, ParseResult, Problem
from activelabel.annotations.target import VideoTarget
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label, Point, clamp01

MAX_UNMATCHED_EXAMPLES = 20
EPS = 1e-6


def norm_name(name: str) -> str:
    """Name with ``/`` separators and no leading ``./`` or ``/``.

    Args:
        name: a file name or relative path as written in an annotation file.

    Returns:
        The name with ``/`` separators, without leading ``./`` or ``/``.
    """
    n = name.replace("\\", "/").strip()
    while n.startswith("./"):
        n = n[2:]
    return n.lstrip("/")


def stem_parts(name: str) -> list[str]:
    """Path parts of a name without the extension of the last part.

    Args:
        name: a file name or relative path.

    Returns:
        The path parts, the last one without its extension.
    """
    parts = [p for p in norm_name(name).split("/") if p]
    if parts and "." in parts[-1]:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return parts


def bbox_of_points(points: Iterable[Sequence[float]]) -> list[float] | None:
    """``[x, y, w, h]`` around points (relative); None when there are none or the box is empty.

    Args:
        points: points with ``x`` and ``y`` first (relative coordinates); other values are ignored.

    Returns:
        ``[x, y, w, h]`` around the points, or None.
    """
    pts = list(points)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def declared_size(
    ctx: ParseContext, rec: ImageRecord, width: object, height: object, name: str
) -> tuple[float, float]:
    """Size of an image as a file declares it, checked against the stored image.

    A missing, non-numeric, non-finite or non-positive value means "not declared": the stored size
    is used. A declared size that differs from the stored one is reported (``size_mismatch``).

    Args:
        ctx: parse context that records ``size_mismatch``.
        rec: the stored image.
        width: width as read from the file (may be missing or not a number).
        height: height as read from the file (may be missing or not a number).
        name: image name, shown as the example of the problem.

    Returns:
        ``(width, height)`` in pixels: the declared size, or the stored size when none is declared.
    """
    try:
        w, h = float(width), float(height)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(rec.width), float(rec.height)
    if not (math.isfinite(w) and math.isfinite(h)) or w <= 0 or h <= 0:
        return float(rec.width), float(rec.height)
    if (int(w), int(h)) != (rec.width, rec.height):
        ctx.problem("size_mismatch", f"{name}: {int(w)}x{int(h)} vs {rec.width}x{rec.height}")
    return w, h


def _index(index: dict[str, list[ImageRecord]], key: str, rec: ImageRecord) -> None:
    """Add ``rec`` under ``key``, once (two names of one image must not make the lookup ambiguous).

    Args:
        index: The lookup table, key to the images with that key.
        key: The key.
        rec: The image.
    """
    found = index.setdefault(key, [])
    if rec not in found:
        found.append(rec)


class ParseContext:
    """Collects labels, classes and problems while a format parser runs."""

    def __init__(self, fmt: str, images: list[ImageRecord], target: VideoTarget | None = None) -> None:
        """Index the images by name and start empty results.

        Args:
            fmt: annotation format name, copied to the result.
            images: image records the annotations are matched to.
            target: the video task, for the formats that match by frame number; None for images.
        """
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
            for name in (rec.name, *rec.aliases):  # a repeated upload is also known by its other names
                n = norm_name(name)
                _index(self._exact, n, rec)
                _index(self._lower, n.lower(), rec)
                _index(self._base, n.rsplit("/", 1)[-1].lower(), rec)
                parts = stem_parts(name)
                for k in range(len(parts)):
                    _index(self._stem_suffix, "/".join(parts[k:]).lower(), rec)

    # -- problems and classes ------------------------------------------------
    def problem(self, code: str, example: str | None = None, n: int = 1) -> None:
        """Record a problem (aggregated by code).

        Args:
            code: key of ``PROBLEMS``.
            example: text shown as an example (up to 5 distinct ones are kept); None for none.
            n: how many occurrences to count.
        """
        level, message = PROBLEMS[code]
        prob = self._problems.setdefault(code, Problem(level, code, message))
        prob.add(example, n)

    def add_class(self, name: str) -> None:
        """Remember a class name (first seen first).

        Args:
            name: class name.
        """
        if name not in self._class_seen:
            self._class_seen.add(name)
            self.classes.append(name)

    # -- image lookup ----------------------------------------------------------
    def _find(self, name: str) -> tuple[ImageRecord | None, bool]:
        """Look a name up in the image indexes.

        Args:
            name: image name to find.

        Returns:
            ``(record, ambiguous)``: the record when exactly one image matches at the first level that finds
            any, ``(None, True)`` when that level finds several, ``(None, False)`` when none matches.
        """
        n = norm_name(name)
        base = n.rsplit("/", 1)[-1].lower()
        for index, key in ((self._exact, n), (self._lower, n.lower()), (self._base, base)):
            found = index.get(key)
            if found:
                return (found[0], False) if len(found) == 1 else (None, True)
        return None, False

    def match(self, *names: str, example: str | None = None) -> ImageRecord | None:
        """Image for the first of ``names`` that matches.

        For each name the lookup tries the exact relative name, then the same ignoring case, then the file
        name alone ignoring case; a lookup that finds several images is ambiguous and counts as no match.
        Records an unmatched or ambiguous problem when no name matches.

        Args:
            names: candidate names, tried in order.
            example: text shown in the problem when nothing matches; the first name when None.

        Returns:
            The matching image record, or None.
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
        """Image for a label file path without extension (YOLO): ``labels/`` -> ``images/`` and suffixes.

        Args:
            parts: path parts of the label file without extension (see ``stem_parts``).
            example: label file name, shown in the problem when nothing matches.

        Returns:
            The matching image record, or None.
        """
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
        """Count an annotation entry without an image.

        Args:
            name: entry name; up to 20 distinct ones are kept as examples.
        """
        self.unmatched_count += 1
        if len(self.unmatched) < MAX_UNMATCHED_EXAMPLES and name not in self.unmatched:
            self.unmatched.append(name)
        self.problem("unmatched_entries", name)

    def at_frame(self, frame: int, example: str | None = None, n: int = 1) -> ImageRecord | None:
        """Image of the video task at this frame; an unused frame is counted (``n`` labels) and skipped.

        Args:
            frame: video frame index.
            example: text shown in the problem; ``frame <n>`` when None.
            n: how many labels are lost when the frame is not used.

        Returns:
            The image record of that frame, or None.
        """
        rec = self.target.frames.get(frame) if self.target else None
        if rec is None:
            self.problem("frames_not_in_task", example or f"frame {frame}", n)
        return rec

    def claim(self, rec: ImageRecord, example: str | None = None) -> bool:
        """Start an entry for an image; False (and a problem) when it already has one.

        Args:
            rec: the image the entry is for.
            example: text shown in the problem; the image name when None.

        Returns:
            True when the entry was started, False when the image already had one.
        """
        if rec.id in self._claimed:
            self.problem("duplicate_image_entry", example or rec.name)
            return False
        self._claimed.add(rec.id)
        self.labels.setdefault(rec.id, [])
        return True

    def add_label(self, rec: ImageRecord, label: Label) -> None:
        """Add a label to an image and remember its class.

        Args:
            rec: the image the label belongs to.
            label: the label, with relative geometry.
        """
        self.labels.setdefault(rec.id, []).append(label)
        self.add_class(label.label)

    # -- geometry ---------------------------------------------------------------
    def box(
        self, x: float, y: float, w: float, h: float, width: float, height: float, where: str
    ) -> list[float] | None:
        """Relative ``[x, y, w, h]`` from absolute values (use width=height=1 for relative input).

        Small overflow is clipped (and reported); boxes without area are dropped.

        Args:
            x: left edge.
            y: top edge.
            w: box width.
            h: box height.
            width: image width in the unit of x and w.
            height: image height in the unit of y and h.
            where: image name, shown as the example of any problem.

        Returns:
            The relative ``[x, y, w, h]``, or None when the numbers are unusable or the box has no area.
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
        cx0, cy0, cx1, cy1 = clamp01(x0), clamp01(y0), clamp01(x1), clamp01(y1)
        if cx1 - cx0 <= EPS or cy1 - cy0 <= EPS:
            self.problem("box_degenerate", where)
            return None
        return [cx0, cy0, cx1 - cx0, cy1 - cy0]

    @staticmethod
    def _outside(x: float, y: float) -> bool:
        """Whether a relative position lies outside the image.

        Args:
            x: relative horizontal position.
            y: relative vertical position.

        Returns:
            True when it is outside ``[0, 1]`` by more than the tolerance.
        """
        return not (-EPS <= x <= 1 + EPS and -EPS <= y <= 1 + EPS)

    def ring(
        self, points: Sequence[Sequence[float]], width: float, height: float, where: str
    ) -> list[Point] | None:
        """Relative polygon ring from absolute points; None (with a problem) when unusable.

        Vertices outside the image are moved to its border and reported (``geometry_clipped``).

        Args:
            points: polygon ``(x, y)`` points in absolute units.
            width: image width.
            height: image height.
            where: image name, shown as the example of any problem.

        Returns:
            The relative ``[x, y]`` points, or None.
        """
        if len(points) < 3:
            self.problem("polygon_too_few_points", where)
            return None
        if width <= 0 or height <= 0 or not all(math.isfinite(v) for p in points for v in p[:2]):
            self.problem("invalid_annotation", where)
            return None
        ring = [[p[0] / width, p[1] / height] for p in points]
        if any(self._outside(x, y) for x, y in ring):
            self.problem("geometry_clipped", where)
        return [[clamp01(x), clamp01(y)] for x, y in ring]

    def point(
        self, x: float, y: float, v: float, width: float, height: float, where: str
    ) -> list[float] | None:
        """Relative keypoint ``[x, y, v]``; None when the numbers are not finite.

        A point outside the image is moved to its border and reported (``geometry_clipped``).

        Args:
            x: horizontal position.
            y: vertical position.
            v: visibility (0 not labelled, 1 hidden, 2 visible); the integer part is kept.
            width: image width in the unit of x.
            height: image height in the unit of y.
            where: image name, shown as the example of any problem.

        Returns:
            The relative ``[x, y, v]``, or None when a number is not finite.
        """
        if not all(math.isfinite(t) for t in (x, y, v)) or width <= 0 or height <= 0:
            self.problem("invalid_annotation", where)
            return None
        rx, ry = x / width, y / height
        if self._outside(rx, ry):
            self.problem("geometry_clipped", where)
        return [clamp01(rx), clamp01(ry), float(int(v))]

    def result(self) -> ParseResult:
        """Final result of the run.

        Returns:
            A ``ParseResult`` with everything collected so far.
        """
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
