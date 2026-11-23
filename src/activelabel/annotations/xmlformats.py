"""Pascal VOC XML (boxes only) and CVAT for images 1.1 XML (box, polygon, points, skeleton, tag).

VOC pixel coordinates are used as written (no 1-pixel shift). CVAT ``points`` shapes become one pose
object each (class from the label, all points visible unless ``occluded``); ``polyline``, ``ellipse``,
``mask`` and ``cuboid`` are skipped with ``unsupported_shape``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from activelabel.annotations.context import ParseContext, bbox_of_points, declared_size, norm_name
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label

UNSUPPORTED_SHAPES = {"polyline", "ellipse", "mask", "cuboid"}


def read_xml(path: Path) -> ET.Element:
    """Parse an XML file (entity definitions are refused).

    Raises:
        ValueError: unreadable, malformed or using entities.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(str(exc)) from exc
    if b"<!ENTITY" in raw:
        raise ValueError("XML entities are not allowed")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML ({exc})") from exc


def _text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def _f(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return float("nan")


def _size(ctx: ParseContext, rec: ImageRecord, width: float, height: float, name: str) -> tuple[float, float]:
    return declared_size(ctx, rec, width, height, name)


# -- Pascal VOC ----------------------------------------------------------------
def parse_voc(ctx: ParseContext, files: list[Path]) -> None:
    """Read VOC XML files (one per image) into ``ctx``."""
    for path in files:
        if path.suffix.lower() != ".xml":
            continue
        try:
            root = read_xml(path)
            if root.tag != "annotation":
                raise ValueError("not a VOC file")
        except ValueError as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        filename = _text(root, "filename")
        names = [n for n in (filename, norm_name(_text(root, "path")), path.stem) if n]
        rec = ctx.match(*names, example=filename or path.name)
        if rec is None or not ctx.claim(rec, filename or path.name):
            continue
        ctx.files.append(path.name)
        size = root.find("size")
        width, height = _size(
            ctx,
            rec,
            _f(_text(size, "width")) if size is not None else 0,
            _f(_text(size, "height")) if size is not None else 0,
            rec.name,
        )
        for obj in root.findall("object"):
            name, bnd = _text(obj, "name"), obj.find("bndbox")
            if not name or bnd is None:
                ctx.problem("unsupported_annotation", f"{rec.name}: object without name or box")
                continue
            x0, y0, x1, y1 = (_f(_text(bnd, k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            box = ctx.box(x0, y0, x1 - x0, y1 - y0, width, height, rec.name)
            if box is not None:
                ctx.add_label(rec, Label(name, bbox=box))


# -- CVAT ----------------------------------------------------------------------
def _points(text: str) -> list[tuple[float, float]]:
    pts = []
    for pair in text.split(";"):
        if pair.strip():
            a, _, b = pair.partition(",")
            pts.append((_f(a), _f(b)))
    return pts


def parse_cvat(ctx: ParseContext, files: list[Path]) -> None:
    """Read CVAT for images 1.1 XML files into ``ctx``."""
    for path in files:
        if path.suffix.lower() != ".xml":
            continue
        try:
            root = read_xml(path)
            if root.tag != "annotations" or root.find("image") is None:
                raise ValueError("not a CVAT for images file")
        except ValueError as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)
        for labels in root.iter("labels"):
            for label in labels.findall("label"):
                if _text(label, "name"):
                    ctx.add_class(_text(label, "name"))
        for image in root.findall("image"):
            name = image.get("name") or ""
            rec = ctx.match(name)
            if rec is None or not ctx.claim(rec, name):
                continue
            width, height = _size(ctx, rec, _f(image.get("width")), _f(image.get("height")), name)
            for shape in image:
                parsed = _shape(ctx, shape, width, height, name)
                if parsed is not None:
                    ctx.add_label(rec, parsed)


def _shape(ctx: ParseContext, shape: ET.Element, width: float, height: float, where: str) -> Label | None:
    kind, name = shape.tag, shape.get("label") or ""
    if kind in UNSUPPORTED_SHAPES:
        ctx.problem("unsupported_shape", f"{where}: {kind}")
        return None
    if kind == "tag":
        return Label(name) if name else None
    if kind == "box":
        x0, y0, x1, y1 = (_f(shape.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
        box = ctx.box(x0, y0, x1 - x0, y1 - y0, width, height, where)
        return Label(name, bbox=box) if box else None
    if kind == "polygon":
        ring = ctx.ring(_points(shape.get("points") or ""), width, height, where)
        return Label(name, bbox=bbox_of_points(ring), polygons=[ring]) if ring else None
    if kind == "points":
        pts = _points(shape.get("points") or "")
        v = 1 if shape.get("occluded") == "1" else 2
        return _pose(ctx, name, [(x, y, v) for x, y in pts], width, height, where)
    if kind == "skeleton":
        triples = []
        for sub in shape.findall("points"):
            pts = _points(sub.get("points") or "")
            x, y = pts[0] if pts else (0.0, 0.0)
            v = 0 if sub.get("outside") == "1" else (1 if sub.get("occluded") == "1" else 2)
            triples.append((x, y, v))
        return _pose(ctx, name, triples, width, height, where)
    return None


def _pose(
    ctx: ParseContext,
    name: str,
    triples: list[tuple[float, float, int]],
    width: float,
    height: float,
    where: str,
) -> Label | None:
    points = []
    for x, y, v in triples:
        pt = ctx.point(x, y, v, width, height, where)
        if pt is None:
            return None
        points.append(pt)
    if not points:
        ctx.problem("unsupported_annotation", where)
        return None
    return Label(name, bbox=bbox_of_points(p for p in points if p[2] > 0), keypoints=points)
