"""COCO JSON: boxes, polygons, RLE (box only) and keypoints.

RLE masks are not decoded: the object is counted (problem ``rle_bbox_only``) and imported with its
``bbox`` only, so a file with only RLE objects is read as detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from activelabel.annotations.context import ParseContext, bbox_of_points, declared_size
from activelabel.labels import Label


def parse_coco(ctx: ParseContext, files: list[Path]) -> None:
    """Read COCO files into ``ctx``.

    Args:
        ctx: parse context; receives the classes, the labels, the names of the files read (``ctx.files``) and
            the problems.
        files: staged files; only ``.json`` files are read, and one that is not a COCO file is reported as
            ``invalid_file``.
    """
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("images"), list):
                raise ValueError("not a COCO file")
            _parse_one(ctx, data)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            ctx.problem("invalid_file", f"{path.name}: {exc}")
            continue
        ctx.files.append(path.name)


def _parse_one(ctx: ParseContext, data: dict[str, Any]) -> None:
    """Read one decoded COCO document into ``ctx``: classes, image entries, then annotations.

    Args:
        ctx: parse context; images are matched by ``file_name`` and claimed, labels are added to it.
        data: the decoded COCO JSON: an object with ``images`` and optional ``categories`` and
            ``annotations``.
    """
    cats: dict[Any, dict[str, Any]] = {}
    for cat in data.get("categories") or []:
        cats[cat["id"]] = cat
        ctx.add_class(str(cat["name"]))
    images: dict[Any, tuple[Any, float, float]] = {}
    for im in data["images"]:
        name = str(im["file_name"])
        rec = ctx.match(name)
        if rec is None or not ctx.claim(rec, name):
            continue
        width, height = declared_size(ctx, rec, im.get("width"), im.get("height"), name)
        images[im["id"]] = (rec, width, height)
    for ann in data.get("annotations") or []:
        entry = images.get(ann.get("image_id"))
        if entry is None:
            continue
        rec, width, height = entry
        label = _annotation(ctx, ann, cats, width, height, rec.name)
        if label is not None:
            ctx.add_label(rec, label)


def _annotation(
    ctx: ParseContext,
    ann: dict[str, Any],
    cats: dict[Any, dict[str, Any]],
    width: float,
    height: float,
    where: str,
) -> Label | None:
    """Label for one COCO annotation, or None when it cannot be used.

    Args:
        ctx: parse context that records the problems found.
        ann: the COCO annotation: ``category_id``, ``bbox`` (``[x, y, w, h]`` in pixels), ``segmentation`` and
            ``keypoints``.
        cats: the file's categories by id.
        width: width in pixels of the image, to make coordinates relative.
        height: height in pixels of the image, to make coordinates relative.
        where: image name, shown as the example of any problem.

    Returns:
        A label with relative (0..1) geometry; the box comes from ``bbox`` or else from the polygons. None
        when the category is unknown, the annotation is malformed or has no usable geometry.
    """
    cat = cats.get(ann.get("category_id"))
    if cat is None:
        ctx.problem("unknown_class_id", f"{where}: category {ann.get('category_id')}")
        return None
    name = str(cat["name"])
    try:
        polygons = _polygons(ctx, ann.get("segmentation"), width, height, where)
        keypoints = _keypoints(ctx, ann.get("keypoints"), cat, width, height, where)
        bbox = None
        if ann.get("bbox"):
            x, y, w, h = (float(v) for v in ann["bbox"][:4])
            bbox = ctx.box(x, y, w, h, width, height, where)
    except (TypeError, ValueError, IndexError):
        ctx.problem("invalid_annotation", where)
        return None
    if bbox is None and polygons:
        bbox = bbox_of_points(p for ring in polygons for p in ring)
    if bbox is None and not polygons and not keypoints:
        if not ann.get("bbox") and not ann.get("segmentation") and not ann.get("keypoints"):
            ctx.problem("unsupported_annotation", where)
        return None
    return Label(name, bbox=bbox, polygons=polygons or None, keypoints=keypoints or None)


def _polygons(
    ctx: ParseContext, seg: Any, width: float, height: float, where: str
) -> list[list[list[float]]] | None:
    """Relative polygon rings of a COCO ``segmentation`` value.

    Args:
        ctx: parse context that records the problems found.
        seg: the ``segmentation`` value: a list of flat rings ``[x1, y1, x2, y2, ...]``, one flat ring, or an
            RLE dict.
        width: width in pixels of the image.
        height: height in pixels of the image.
        where: image name, shown as the example of any problem.

    Returns:
        The rings as lists of relative ``[x, y]`` points, or None when there are no usable ones (an RLE mask
        is reported as ``rle_bbox_only``).
    """
    if isinstance(seg, dict):  # RLE
        ctx.problem("rle_bbox_only", where)
        return None
    if not isinstance(seg, list) or not seg:
        return None
    rings_raw = [seg] if all(isinstance(v, (int, float)) for v in seg) else seg
    rings = []
    for raw in rings_raw:
        pts = [(float(a), float(b)) for a, b in zip(raw[0::2], raw[1::2], strict=False)]
        ring = ctx.ring(pts, width, height, where)
        if ring is not None:
            rings.append(ring)
    return rings or None


def _keypoints(
    ctx: ParseContext, flat: Any, cat: dict[str, Any], width: float, height: float, where: str
) -> list[list[float]] | None:
    """Relative keypoints of a COCO ``keypoints`` value.

    Args:
        ctx: parse context that records the problems found.
        flat: flat list ``[x1, y1, v1, x2, y2, v2, ...]`` with pixel coordinates and visibility.
        cat: the category of the object; its ``keypoints`` list gives the expected number of points.
        width: width in pixels of the image.
        height: height in pixels of the image.
        where: image name, shown as the example of any problem.

    Returns:
        Relative ``[x, y, v]`` per point, or None when there are none or a point is unusable.

    Raises:
        ValueError: the list is not made of x, y, v triples.
    """
    if not flat:
        return None
    if len(flat) % 3:
        raise ValueError("keypoints are not x, y, v triples")
    expected = len(cat.get("keypoints") or [])
    if expected and expected != len(flat) // 3:
        ctx.problem("keypoint_count_mismatch", f"{where}: {len(flat) // 3} of {expected}")
    out = []
    for i in range(0, len(flat), 3):
        pt = ctx.point(float(flat[i]), float(flat[i + 1]), float(flat[i + 2]), width, height, where)
        if pt is None:
            return None
        out.append(pt)
    return out
