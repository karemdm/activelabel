"""Pre-annotation: labels made by a saved model, to be reviewed by a person (see docs/PREANNOTATE.md).

A pre-annotated label is an ordinary label with a ``score`` (the model's confidence) and the
``model`` that made it (``m1``…). It stays a pre-annotation until a person reviews it: anything that
comes back from CVAT or FiftyOne replaces it, and *accept* keeps it as it is but marks it reviewed
(score and model removed). Dataset versions ignore labels with a score, so a pre-annotation never
trains a model by accident.

Rules that keep people's work safe:

- a pre-annotation never replaces a label made by a person;
- ``unlabeled`` fills only images without labels; ``unreviewed`` redoes only images whose labels are
  all pre-annotations (for example with a better model).
"""

from __future__ import annotations

import builtins
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from activelabel.datasets import common_count
from activelabel.errors import CodedError
from activelabel.images.manifest import ACTIVE, Manifest
from activelabel.jobs import QUEUED
from activelabel.labels import (
    AnnotationSet,
    Label,
    annotations_lock,
    geometry_problem,
    load_annotations,
    save_annotations,
)
from activelabel.projects import Project
from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore

SCOPES = ("unlabeled", "unreviewed")
#: Kinds of task a model can pre-annotate now (segmentation masks become polygons: see ``mask_labels``).
KINDS = ("classification", "detection", "keypoints", "instance", "semantic")
#: The project type whose labels each kind of model makes.
KIND_TASK = {
    "classification": Task.CLASSIFICATION,
    "detection": Task.DETECTION,
    "keypoints": Task.KEYPOINTS,
    "instance": Task.SEGMENTATION,
    "semantic": Task.SEGMENTATION,
}
MIN_REGION = 0.0002  # a mask region smaller than this share of the image is noise, not an object
SIMPLIFY = 0.004  # polygon simplification: tolerance as a share of the region's perimeter (min. 1 pixel)

_LOCKS: dict[Path, threading.RLock] = {}
_GUARD = threading.Lock()


def _lock(path: Path) -> threading.RLock:
    with _GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


def is_preannotation(label: Label) -> bool:
    """True for a label made by a model and not reviewed yet."""
    return label.score is not None


def image_state(images: dict[str, list[Label]], image_id: str) -> str:
    """``none`` (not in the annotations), ``pre`` (only pre-annotations) or ``human``.

    An image listed with no labels is ``human``: a person reviewed it and found nothing (a background
    image), for example after deleting a model's wrong box; it is never pre-annotated again.
    """
    if image_id not in images:
        return "none"
    labels = images[image_id]
    if labels and all(is_preannotation(lb) for lb in labels):
        return "pre"
    return "human"


def decode_mask(mask: dict[str, Any]) -> Any:
    """Boolean 2-D array of a run-length mask (``size`` ``[h, w]``, alternating ``counts``, zeros first)."""
    height, width = mask["size"]
    values = np.arange(len(mask["counts"])) % 2 == 1
    flat = np.repeat(values, mask["counts"])
    if flat.size != height * width:
        raise ValueError("A mask does not fit its size.")
    return flat.reshape(height, width)


def mask_polygons(mask: Any) -> list[list[list[float]]]:
    """Outer outlines of the regions of a boolean mask as relative rings (holes are not kept).

    Regions below ``MIN_REGION`` of the image are dropped and every outline is simplified
    (Douglas-Peucker, tolerance ``SIMPLIFY`` of its perimeter, at least 1 pixel).
    """
    height, width = mask.shape
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in found:
        if cv2.contourArea(contour) < max(2.0, MIN_REGION * width * height):
            continue
        tolerance = max(1.0, SIMPLIFY * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
        if len(points) >= 3:
            rings.append([[round((x + 0.5) / width, 6), round((y + 0.5) / height, 6)] for x, y in points])
    return rings


def _box_of(rings: list[list[list[float]]]) -> list[float]:
    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def mask_labels(items: list[dict[str, Any]]) -> list[Label]:
    """Labels from the predictions of a segmentation model (``mask`` in each item, others as they are).

    An item with a box is one object (an instance): all the regions of its mask are its rings. An
    item without a box (the mask of a whole class, from a semantic model) becomes one object per
    region. Items whose mask has no region left are dropped.
    """
    out: list[Label] = []
    for item in items:
        if "mask" not in item:
            out.append(Label.from_dict(item))
            continue
        rings = mask_polygons(decode_mask(item["mask"]))
        if not rings:
            continue
        common = {"score": item.get("score"), "model": item.get("model")}
        if item.get("bbox"):
            out.append(Label(item["label"], bbox=_box_of(rings), polygons=rings, **common))
        else:
            out += [Label(item["label"], bbox=_box_of([r]), polygons=[r], **common) for r in rings]
    return out


def project_keypoints(project_dir: Path) -> int | None:
    """Points per object most pose labels of the project have (None when there are none)."""
    annotations = load_annotations(project_dir)
    labels = (lb for items in (annotations.images if annotations else {}).values() for lb in items)
    return common_count(len(lb.keypoints or []) for lb in labels if lb.score is None)


def fits(kind: str, project_task: Task) -> bool:
    """True when a model of this kind makes the labels a project of this type uses."""
    return KIND_TASK.get(kind) is project_task


def targets(project_dir: Path, task_id: int, scope: str) -> list[str]:
    """Active images of a task that a pre-annotation with ``scope`` would fill.

    Raises:
        CodedError: ``scope_invalid`` or ``task_not_found``.
    """
    if scope not in SCOPES:
        raise CodedError(
            "scope_invalid", f"The images must be one of: {', '.join(SCOPES)}.", scopes=list(SCOPES)
        )
    task = next((t for t in TaskStore(project_dir).load() if t.id == task_id), None)
    if task is None:
        raise CodedError("task_not_found", "This task does not exist.", task=task_id)
    manifest = Manifest(project_dir).load()
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    wanted = "none" if scope == "unlabeled" else "pre"
    return [
        i
        for i in task.images
        if i in manifest and manifest[i].status == ACTIVE and image_state(images, i) == wanted
    ]


def task_counts(project_dir: Path) -> list[dict[str, Any]]:
    """Per task: active images, images without labels and images with only pre-annotations."""
    manifest = Manifest(project_dir).load()
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    out = []
    for task in TaskStore(project_dir).load():
        active = [i for i in task.images if i in manifest and manifest[i].status == ACTIVE]
        states = [image_state(images, i) for i in active]
        models = sorted(
            {lb.model for i in active for lb in images.get(i, []) if is_preannotation(lb) and lb.model}
        )
        out.append(
            {
                "id": task.id,
                "name": task.name,
                "images": len(active),
                "unlabeled": states.count("none"),
                "unreviewed": states.count("pre"),
                "models": models,
            }
        )
    return out


def apply_predictions(
    project_dir: Path, project: Project, predictions: dict[str, list[Label]], scope: str
) -> dict[str, int]:
    """Write a model's labels, never over a person's labels.

    Images that changed since the run started are skipped: with ``unlabeled`` an image a person
    labelled (or reviewed as empty) meanwhile, with ``unreviewed`` an image that now has a reviewed
    label, and an image deleted or excluded meanwhile. Only labels of the project's CURRENT classes and
    with the shape its task needs are kept; the others are counted, not written.

    Returns:
        ``{"images": written, "objects": labels written, "skipped": images left as they were,
        "empty": images the model found nothing in, "dropped_class": labels of a class the project does
        not have, "dropped_geometry": labels without the shape of the task}``.
    """
    manifest = Manifest(project_dir).load()
    with annotations_lock(project_dir):
        annotations = load_annotations(project_dir) or AnnotationSet(
            project.task, list(project.classes), {}, {}
        )
        classes = set(project.classes)
        counts = {
            "images": 0,
            "objects": 0,
            "skipped": 0,
            "empty": 0,
            "dropped_class": 0,
            "dropped_geometry": 0,
        }
        wanted = "none" if scope == "unlabeled" else "pre"
        for image_id, labels in predictions.items():
            gone = image_id not in manifest or manifest[image_id].status != ACTIVE
            if gone or image_state(annotations.images, image_id) != wanted:
                counts["skipped"] += 1
                continue
            kept = []
            for lb in labels:
                if lb.score is None:
                    continue
                if lb.label not in classes:
                    counts["dropped_class"] += 1
                elif geometry_problem(project.task, lb):
                    counts["dropped_geometry"] += 1
                else:
                    kept.append(lb)
            if kept:
                annotations.images[image_id] = kept
                counts["images"] += 1
                counts["objects"] += len(kept)
            else:
                annotations.images.pop(image_id, None)  # an old pre-annotation goes, nothing replaces it
                counts["empty"] += 1
        save_annotations(project_dir, annotations)
    return counts


def accept(project_dir: Path, image_ids: list[str]) -> int:
    """Mark the pre-annotations of these images as reviewed (kept as they are). Returns how many images."""
    with annotations_lock(project_dir):
        annotations = load_annotations(project_dir)
        if not annotations:
            return 0
        changed = 0
        for image_id in image_ids:
            labels = annotations.images.get(image_id)
            if not labels or not any(is_preannotation(lb) for lb in labels):
                continue
            for lb in labels:
                lb.score = None
                lb.model = None
            changed += 1
        if changed:
            save_annotations(project_dir, annotations)
    return changed


def unreviewed_images(project_dir: Path, task_id: int | None = None) -> list[str]:
    """Active images whose labels are all pre-annotations (of one task, or of the project)."""
    if task_id is not None:
        return targets(project_dir, task_id, "unreviewed")
    manifest = Manifest(project_dir).load()
    annotations = load_annotations(project_dir)
    images = annotations.images if annotations else {}
    return [
        i
        for i in images
        if i in manifest and manifest[i].status == ACTIVE and image_state(images, i) == "pre"
    ]


class RunNotFound(KeyError):
    """No pre-annotation run with this number."""


class PreannotationStore:
    """Pre-annotation runs of a project: ``preannotations.json``."""

    def __init__(self, project_dir: Path) -> None:
        self.file = project_dir / "preannotations.json"
        self._lock = _lock(self.file)

    def _read(self) -> dict[str, Any]:
        return dict(read_json(self.file, {"next": 1, "runs": []}) or {"next": 1, "runs": []})

    def list(self) -> builtins.list[dict[str, Any]]:
        """Runs, newest first."""
        return sorted(self._read()["runs"], key=lambda r: r["number"], reverse=True)

    def get(self, number: int) -> dict[str, Any]:
        """One run.

        Raises:
            RunNotFound: unknown number.
        """
        for run in self._read()["runs"]:
            if run["number"] == number:
                return dict(run)
        raise RunNotFound(number)

    def create(self, **fields: Any) -> dict[str, Any]:
        """Store a new queued run."""
        with self._lock:
            data = self._read()
            number = int(data.get("next", 1))
            run = {
                "number": number,
                "status": QUEUED,
                "created": now(),
                "done": 0,
                "total": 0,
                "error": "",
                **fields,
            }
            data["runs"] = [*data["runs"], run]
            data["next"] = number + 1
            write_json(self.file, data)
        return run

    def update(self, number: int, **fields: Any) -> dict[str, Any]:
        """Change fields of a run.

        Raises:
            RunNotFound: unknown number.
        """
        with self._lock:
            data = self._read()
            for run in data["runs"]:
                if run["number"] == number:
                    run.update({k: v for k, v in fields.items() if k not in ("number", "created")})
                    write_json(self.file, data)
                    return dict(run)
        raise RunNotFound(number)
