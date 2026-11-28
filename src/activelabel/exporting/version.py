"""Export a dataset version as a zip in YOLO or COCO format (see docs/DATASETS.md).

Layout of the zip, with one *group* of images per part (``train``, ``val``, ``test``) or, for a K-fold
version, per fold (``fold_1``...) and the ``test`` part:

- **YOLO** (detection, segmentation, keypoints): ``images/<group>/<image>``, ``labels/<group>/<image>.txt``
  and ``data.yaml`` (a ``data_fold_<n>.yaml`` per fold for K-fold: fold *n* validates, the others train).
  Classification is ``<group>/<class>/<image>``, the folders of YOLO classification.
- **COCO** (detection, segmentation, keypoints): ``images/<group>/<image>`` and
  ``annotations/instances_<group>.json``.

A format that cannot hold the task is refused with a coded error: regression (neither format holds a number
per image) and classification in COCO (it has no label for a whole image).
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activelabel.annotations.context import norm_name
from activelabel.annotations.staging import IMAGE_SUFFIXES
from activelabel.datasets import PARTS, DatasetStore
from activelabel.errors import CodedError
from activelabel.exporting.coco import coco_document
from activelabel.exporting.yolo import data_yaml, label_lines
from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.labels import Label
from activelabel.tasks import Task

FORMATS = ("yolo", "coco")
_UNSAFE = re.compile(r'[<>:"\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class ExportItem:
    """One image of the export.

    Attributes:
        record: the image of the project.
        group: the part or fold it belongs to.
        name: its path inside the folder of the group (inside the folder of its class in classification).
        labels: its labels in the version.
    """

    record: ImageRecord
    group: str
    name: str
    labels: list[Label]


def _clean(name: str) -> str:
    """A relative file name that every system accepts.

    Args:
        name: a name as uploaded (``a/b.jpg``).

    Returns:
        The name with ``/`` separators, without ``..`` parts and without characters that Windows refuses.
    """
    parts = [_UNSAFE.sub("_", p).strip(" .") for p in norm_name(name).split("/")]
    return "/".join(p for p in parts if p and p != "..")


def _unique(name: str, taken: set[str], keep: str, file: bool = True) -> str:
    """The name, with ``keep`` added before its extension when ``taken`` has it already (ignoring case).

    A file is compared without its extension, because its label file (``a.txt``) is named after it: ``a.jpg``
    and ``a.png`` in one folder would share one.

    Args:
        name: the wanted relative name.
        taken: the lowercase names given so far; the result is added to it.
        keep: text that tells this name from the other (the image id).
        file: whether ``name`` is a file name (a folder name has no extension).

    Returns:
        A name nobody has.
    """
    stem, dot, suffix = name.rpartition(".") if file and "." in name.rsplit("/", 1)[-1] else (name, "", "")
    result = name
    if stem.lower() in taken:
        result = f"{stem}_{keep}{dot}{suffix}"
        stem = f"{stem}_{keep}"
    taken.add(stem.lower())
    return result


def _image_name(record: ImageRecord, flat: bool) -> str:
    """The name an image gets in the zip: the one it was uploaded with, with an image extension.

    Args:
        record: the image.
        flat: drop the sub-folders of the name (classification folders hold the images of one class).

    Returns:
        The relative name.
    """
    name = _clean(record.name) or record.id
    if flat:
        name = name.rsplit("/", 1)[-1]
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        name += Path(record.file).suffix or ".jpg"
    return name


def _class_folders(classes: list[str]) -> dict[str, str]:
    """The folder of each class in a classification export.

    Args:
        classes: the class names.

    Returns:
        The class name to a folder name that is safe in a path and differs from the others (ignoring case).
    """
    taken: set[str] = set()
    folders: dict[str, str] = {}
    for position, name in enumerate(classes):
        base = _UNSAFE.sub("_", name.replace("/", "_")).strip(" .") or "class"
        folders[name] = _unique(base, taken, str(position), file=False)
    return folders


def groups_of(data: dict[str, Any]) -> dict[str, list[str]]:
    """The groups of image ids a version is exported in (those with images).

    Args:
        data: the stored version.

    Returns:
        ``train``, ``val`` and ``test``; for a K-fold version ``fold_1``... and ``test``.
    """
    parts, folds = data["parts"], data.get("folds")
    groups = (
        {p: list(parts.get(p, [])) for p in PARTS}
        if folds is None
        else {
            **{f"fold_{n}": list(ids) for n, ids in enumerate(folds, 1)},
            "test": list(parts.get("test", [])),
        }
    )
    return {g: ids for g, ids in groups.items() if ids}


def _refuse(task: Task, fmt: str) -> None:
    """Refuse a format that cannot hold the task.

    Args:
        task: the task of the version.
        fmt: ``yolo`` or ``coco``.

    Raises:
        CodedError: the format is unknown (``export_format_invalid``), the task is regression
            (``export_regression``) or classification in COCO (``export_coco_classification``).
    """
    if fmt not in FORMATS:
        raise CodedError("export_format_invalid", "The format must be yolo or coco.")
    if task is Task.REGRESSION:
        raise CodedError(
            "export_regression",
            "YOLO and COCO have no place for a number per image: a regression version cannot be exported.",
        )
    if task is Task.CLASSIFICATION and fmt == "coco":
        raise CodedError(
            "export_coco_classification",
            "COCO has no label for a whole image: export a classification version as YOLO.",
        )


def _items(project_dir: Path, data: dict[str, Any], task: Task, with_images: bool) -> list[ExportItem]:
    """The images of a version with their names in the zip.

    Args:
        project_dir: folder of the project.
        data: the stored version.
        task: the task of the version.
        with_images: whether the image files are going into the zip (they must then exist).

    Returns:
        The items, group by group.

    Raises:
        CodedError: the version has no images (``export_empty``) or some of them are gone from the project
            (``export_images_missing``).
    """
    groups = groups_of(data)
    if not groups:
        raise CodedError("export_empty", "This version has no images to export.")
    manifest = Manifest(project_dir)
    records = manifest.load()
    missing = [
        i
        for ids in groups.values()
        for i in ids
        if i not in records or (with_images and not manifest.file_path(records[i]).is_file())
    ]
    if missing:
        raise CodedError(
            "export_images_missing",
            f"{len(missing)} image(s) of this version are no longer in the project.",
            count=len(missing),
        )
    classification = task is Task.CLASSIFICATION
    folder = _class_folders(data["classes"]) if classification else {}
    items: list[ExportItem] = []
    for group, ids in groups.items():
        taken: set[str] = set()
        for image_id in ids:
            labels = [Label.from_dict(d) for d in data["labels"].get(image_id, [])]
            record = records[image_id]
            name = _image_name(record, flat=classification)
            if classification:
                name = f"{folder[labels[0].label]}/{name}"
            items.append(ExportItem(record, group, _unique(name, taken, record.id[:8]), labels))
    return items


def _keypoints(items: list[ExportItem]) -> int:
    """Points per object of a keypoints version.

    Args:
        items: the images of the version.

    Returns:
        The number of points of the first object that has points.
    """
    return next((len(lb.keypoints) for it in items for lb in it.labels if lb.keypoints), 0)


def _yolo_files(task: Task, data: dict[str, Any], items: list[ExportItem]) -> dict[str, str]:
    """The text files of a YOLO export.

    Args:
        task: the task of the version.
        data: the stored version.
        items: the images of the version.

    Returns:
        Path in the zip to text: one label file per image and the ``data.yaml`` file(s); nothing for
        classification (its labels are the folders).
    """
    if task is Task.CLASSIFICATION:
        return {}
    classes = data["classes"]
    files = {
        f"labels/{it.group}/{it.name.rsplit('.', 1)[0]}.txt": "".join(
            f"{line}\n" for line in label_lines(task, it.labels, classes)
        )
        for it in items
    }
    groups = list(groups_of(data))
    points = _keypoints(items) if task is Task.KEYPOINTS else 0
    folds = [g for g in groups if g.startswith("fold_")]
    test = {"test": "images/test"} if "test" in groups else {}
    if not folds:
        files["data.yaml"] = data_yaml({g: f"images/{g}" for g in groups}, classes, points)
    for fold in folds:
        train = [f"images/{g}" for g in folds if g != fold]
        parts: dict[str, str | list[str]] = {"val": f"images/{fold}", **test}
        if train:
            parts = {"train": train if len(train) > 1 else train[0], **parts}
        files[f"data_{fold}.yaml"] = data_yaml(parts, classes, points)
    return files


def _coco_files(task: Task, data: dict[str, Any], items: list[ExportItem]) -> dict[str, str]:
    """The JSON files of a COCO export.

    Args:
        task: the task of the version.
        data: the stored version.
        items: the images of the version.

    Returns:
        Path in the zip to JSON text: one ``instances_<group>.json`` per group.
    """
    points = _keypoints(items) if task is Task.KEYPOINTS else 0
    files = {}
    for group in groups_of(data):
        rows = [(it.name, it.record.width, it.record.height, it.labels) for it in items if it.group == group]
        document = coco_document(task, rows, data["classes"], points)
        files[f"annotations/instances_{group}.json"] = json.dumps(document, ensure_ascii=False, indent=1)
    return files


def _prepare(
    project_dir: Path, number: int, fmt: str, with_images: bool
) -> tuple[dict[str, Any], Task, list[ExportItem]]:
    """Read a version and check that it can be exported.

    Args:
        project_dir: folder of the project.
        number: number of the dataset version.
        fmt: ``yolo`` or ``coco``.
        with_images: whether the image files go into the zip.

    Returns:
        The stored version, its task and its images.

    Raises:
        DatasetNotFound: unknown version.
        CodedError: see :func:`export_version`.
    """
    data = DatasetStore(project_dir).read(number)
    task = Task(data["type"])
    _refuse(task, fmt)
    return data, task, _items(project_dir, data, task, with_images)


def _summary(
    data: dict[str, Any], task: Task, items: list[ExportItem], fmt: str, with_images: bool
) -> dict[str, Any]:
    """What an export holds.

    Args:
        data: the stored version.
        task: its task.
        items: its images.
        fmt: ``yolo`` or ``coco``.
        with_images: whether the image files go into the zip.

    Returns:
        ``{format, type, images, groups, size_bytes}``: the images per group and the size of the image files
        (0 without images).
    """
    return {
        "format": fmt,
        "type": task.value,
        "images": len(items),
        "groups": {g: len(ids) for g, ids in groups_of(data).items()},
        "size_bytes": sum(it.record.size_bytes for it in items) if with_images else 0,
    }


def check_export(project_dir: Path, number: int, fmt: str, with_images: bool = True) -> dict[str, Any]:
    """What an export would contain, without writing it (and the same refusals).

    Args:
        project_dir: folder of the project.
        number: number of the dataset version.
        fmt: ``yolo`` or ``coco``.
        with_images: whether the image files go into the zip.

    Returns:
        ``{format, type, images, groups, size_bytes}``: the images per group and the size of the image files
        (0 without images).

    Raises:
        DatasetNotFound: unknown version.
        CodedError: see :func:`export_version`.
    """
    data, task, items = _prepare(project_dir, number, fmt, with_images)
    return _summary(data, task, items, fmt, with_images)


def export_version(
    project_dir: Path, number: int, fmt: str, target: Path, with_images: bool = True
) -> dict[str, Any]:
    """Write a dataset version as a zip.

    Args:
        project_dir: folder of the project.
        number: number of the dataset version.
        fmt: ``yolo`` or ``coco``.
        target: the zip file to write.
        with_images: put the image files in the zip (False: only the labels).

    Returns:
        The same summary as :func:`check_export`.

    Raises:
        DatasetNotFound: unknown version.
        CodedError: unknown format (``export_format_invalid``), a task the format cannot hold
            (``export_regression``, ``export_coco_classification``), a version without images
            (``export_empty``) or whose images left the project (``export_images_missing``).
    """
    data, task, items = _prepare(project_dir, number, fmt, with_images)
    manifest = Manifest(project_dir)
    texts = (_yolo_files if fmt == "yolo" else _coco_files)(task, data, items)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, text in texts.items():
            archive.writestr(path, text)
        if with_images:
            folder = "" if task is Task.CLASSIFICATION else "images/"
            for it in items:
                archive.write(
                    manifest.file_path(it.record), f"{folder}{it.group}/{it.name}", zipfile.ZIP_STORED
                )
    return _summary(data, task, items, fmt, with_images)
