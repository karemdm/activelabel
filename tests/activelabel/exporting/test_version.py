"""Exporting a dataset version: the zip layout, the refusals and the round trip (export, then import)."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from ann_helpers import make_manifest

from activelabel.annotations.parsing import parse
from activelabel.errors import CodedError
from activelabel.exporting.version import check_export, export_version, groups_of
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.storage import write_json

CLASSES = ["gato preto", "cão"]  # an accent and a space: names must survive the trip


def _version(
    project_dir: Path, task: str, parts: dict[str, list[str]], labels: dict[str, list[Label]], **extra: Any
):
    data = {
        "version": 1,
        "type": task,
        "classes": CLASSES,
        "parts": {"train": [], "val": [], "test": [], **parts},
        "folds": None,
        "labels": {i: [lb.as_dict() for lb in items] for i, items in labels.items() if items},
        **extra,
    }
    write_json(project_dir / "datasets" / "v1.json", data)


def _setup(tmp_path: Path, names: list[str]):
    project_dir = tmp_path / "p"
    manifest = make_manifest(project_dir, names, size=(64, 48))
    return project_dir, manifest, [r.id for r in manifest.all()]


def _extract(archive: Path, where: Path) -> list[Path]:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(where)
    return sorted(p for p in where.rglob("*") if p.is_file())


def _names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return zf.namelist()


def _flat(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [n for item in value for n in (_flat(item) if isinstance(item, list) else [item])]


def _same(got: list[Label], want: list[Label]) -> None:
    assert [lb.label for lb in got] == [lb.label for lb in want]
    for a, b in zip(got, want, strict=True):
        for field in ("bbox", "polygons", "keypoints"):
            assert _flat(getattr(a, field)) == pytest.approx(_flat(getattr(b, field)), abs=2e-6), field


def _reimport(archive: Path, fmt: str, where: Path, manifest) -> dict[str, list[Label]]:
    files = [p for p in _extract(archive, where) if p.suffix in {".txt", ".yaml", ".json"}]
    result = parse(files, fmt, manifest.active())
    assert not [p for p in result.problems if p.level == "error"], result.problems
    return result.labels


DETECTION = [Label("cão", bbox=[0.1, 0.2, 0.3, 0.4]), Label("gato preto", bbox=[0.5, 0.5, 0.25, 0.4])]
SEGMENTATION = [
    Label("cão", bbox=[0.1, 0.1, 0.5, 0.6], polygons=[[[0.1, 0.1], [0.6, 0.1], [0.6, 0.7], [0.1, 0.5]]])
]
KEYPOINTS = [
    Label("gato preto", bbox=[0.1, 0.1, 0.6, 0.7], keypoints=[[0.2, 0.2, 2], [0.5, 0.6, 1], [0.3, 0.3, 2]])
]
SHAPES = {"detection": DETECTION, "segmentation": SEGMENTATION, "keypoints": KEYPOINTS}


@pytest.mark.parametrize("fmt", ["yolo", "coco"])
@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_round_trip_gives_the_same_annotations(tmp_path, fmt, task):
    names = ["a b.png", "sub dir/cão.png", "c.png", "d.png", "e.png"]
    project_dir, manifest, ids = _setup(tmp_path, names)
    labels = {ids[0]: SHAPES[task], ids[1]: SHAPES[task][:1], ids[2]: SHAPES[task], ids[3]: SHAPES[task]}
    _version(
        project_dir, task, {"train": ids[:2], "val": [ids[2]], "test": [ids[3], ids[4]]}, labels
    )  # e: background
    archive = tmp_path / "out.zip"

    summary = export_version(project_dir, 1, fmt, archive)

    assert summary["images"] == 5 and summary["groups"] == {"train": 2, "val": 1, "test": 2}
    got = _reimport(archive, fmt, tmp_path / "x", manifest)
    for image_id in ids[:4]:
        _same(got[image_id], labels[image_id])
    assert got.get(ids[4], []) == []  # the background image has no objects
    assert "images/train/sub dir/cão.png" in _names(archive)


def test_segmentation_with_two_rings_becomes_two_yolo_objects(tmp_path):
    project_dir, manifest, ids = _setup(tmp_path, ["a.png"])
    ring = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]
    _version(project_dir, "segmentation", {"train": ids}, {ids[0]: [Label("cão", polygons=[ring, ring])]})
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "yolo", archive)
    assert len(_reimport(archive, "yolo", tmp_path / "x", manifest)[ids[0]]) == 2


def test_yolo_layout_and_data_yaml(tmp_path):
    project_dir, _, ids = _setup(tmp_path, ["a.png", "b.png", "c.png"])
    _version(
        project_dir, "detection", {"train": ids[:1], "val": ids[1:2], "test": ids[2:]}, {ids[0]: DETECTION}
    )
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "yolo", archive)
    assert sorted(_names(archive)) == sorted(
        ["data.yaml"]
        + [f"images/{g}/{n}.png" for g, n in (("train", "a"), ("val", "b"), ("test", "c"))]
        + [f"labels/{g}/{n}.txt" for g, n in (("train", "a"), ("val", "b"), ("test", "c"))]
    )
    with zipfile.ZipFile(archive) as zf:
        data = yaml.safe_load(zf.read("data.yaml"))
        assert zf.read("labels/val/b.txt") == b""
    assert data["train"] == "images/train" and data["val"] == "images/val" and data["test"] == "images/test"
    assert data["names"] == {0: "gato preto", 1: "cão"}


def test_empty_parts_are_not_exported(tmp_path):
    project_dir, _, ids = _setup(tmp_path, ["a.png", "b.png"])
    _version(project_dir, "detection", {"train": ids}, {ids[0]: DETECTION})
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "yolo", archive)
    with zipfile.ZipFile(archive) as zf:
        assert set(yaml.safe_load(zf.read("data.yaml"))) == {"path", "train", "names"}


def test_kfold_has_a_group_per_fold_and_a_data_yaml_per_fold(tmp_path):
    project_dir, manifest, ids = _setup(tmp_path, ["a.png", "b.png", "c.png", "d.png"])
    labels = {i: DETECTION for i in ids}
    _version(project_dir, "detection", {"test": ids[3:]}, labels, folds=[ids[:1], ids[1:2], ids[2:3]])
    assert list(groups_of(json.loads((project_dir / "datasets" / "v1.json").read_text()))) == [
        "fold_1",
        "fold_2",
        "fold_3",
        "test",
    ]
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "yolo", archive)
    with zipfile.ZipFile(archive) as zf:
        second = yaml.safe_load(zf.read("data_fold_2.yaml"))
    assert second["train"] == ["images/fold_1", "images/fold_3"]
    assert second["val"] == "images/fold_2" and second["test"] == "images/test"
    got = _reimport(archive, "yolo", tmp_path / "x", manifest)
    assert set(got) == set(ids)
    coco = tmp_path / "coco.zip"
    export_version(project_dir, 1, "coco", coco)
    assert {n for n in _names(coco) if n.endswith(".json")} == {
        f"annotations/instances_{g}.json" for g in ("fold_1", "fold_2", "fold_3", "test")
    }


def test_classification_is_a_folder_per_class_and_round_trips(tmp_path):
    names = [
        "x/a.png",
        "y/a.png",
        "c.png",
        "d.png",
    ]  # two images called a.png: they sit in different class folders
    project_dir, manifest, ids = _setup(tmp_path, names)
    labels = {
        ids[0]: [Label("cão")],
        ids[1]: [Label("gato preto")],
        ids[2]: [Label("cão")],
        ids[3]: [Label("cão")],
    }
    _version(project_dir, "classification", {"train": ids[:3], "val": ids[3:]}, labels)
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "yolo", archive)
    entries = sorted(_names(archive))
    assert entries == ["train/cão/a.png", "train/cão/c.png", "train/gato preto/a.png", "val/cão/d.png"]
    records = [ImageRecord(n, n, "x", 64, 48, 1, "t") for n in entries]
    result = parse([], "folders", records)
    assert {r: [lb.label for lb in v] for r, v in result.labels.items()} == {
        "train/cão/a.png": ["cão"],
        "train/cão/c.png": ["cão"],
        "train/gato preto/a.png": ["gato preto"],
        "val/cão/d.png": ["cão"],
    }


def test_names_that_collide_or_are_unsafe_are_kept_apart(tmp_path):
    names = ["a.png", "A.PNG", "bad:name?.png", "../up.png"]
    project_dir, _, ids = _setup(tmp_path, names)
    _version(project_dir, "detection", {"train": ids}, {i: DETECTION for i in ids})
    archive = tmp_path / "out.zip"
    export_version(project_dir, 1, "coco", archive)
    entries = [n for n in _names(archive) if n.startswith("images/")]
    assert len({n.lower() for n in entries}) == 4
    assert "images/train/bad_name_.png" in entries and "images/train/up.png" in entries


def test_labels_only_has_no_images_and_does_not_need_them(tmp_path):
    project_dir, manifest, ids = _setup(tmp_path, ["a.png"])
    _version(project_dir, "detection", {"train": ids}, {ids[0]: DETECTION})
    (project_dir / manifest.get(ids[0]).file).unlink()
    archive = tmp_path / "out.zip"
    summary = export_version(project_dir, 1, "yolo", archive, with_images=False)
    assert summary["size_bytes"] == 0 and sorted(_names(archive)) == ["data.yaml", "labels/train/a.txt"]


def test_check_reports_groups_and_size(tmp_path):
    project_dir, _, ids = _setup(tmp_path, ["a.png", "b.png"])
    _version(project_dir, "detection", {"train": ids[:1], "test": ids[1:]}, {ids[0]: DETECTION})
    got = check_export(project_dir, 1, "coco")
    assert got == {
        "format": "coco",
        "type": "detection",
        "images": 2,
        "groups": {"train": 1, "test": 1},
        "size_bytes": 2,
    }


@pytest.mark.parametrize(
    ("task", "fmt", "code"),
    [
        ("regression", "yolo", "export_regression"),
        ("regression", "coco", "export_regression"),
        ("classification", "coco", "export_coco_classification"),
        ("detection", "voc", "export_format_invalid"),
    ],
)
def test_a_format_that_cannot_hold_the_task_is_refused(tmp_path, task, fmt, code):
    project_dir, _, ids = _setup(tmp_path, ["a.png"])
    _version(project_dir, task, {"train": ids}, {ids[0]: [Label("cão")]})
    for call in (
        lambda: check_export(project_dir, 1, fmt),
        lambda: export_version(project_dir, 1, fmt, tmp_path / "o.zip"),
    ):
        with pytest.raises(CodedError) as caught:
            call()
        assert caught.value.code == code
    assert not (tmp_path / "o.zip").exists()


def test_an_empty_version_is_refused(tmp_path):
    project_dir, _, _ = _setup(tmp_path, ["a.png"])
    _version(project_dir, "detection", {}, {})
    with pytest.raises(CodedError) as caught:
        check_export(project_dir, 1, "yolo")
    assert caught.value.code == "export_empty"


def test_images_gone_from_the_project_are_refused_and_counted(tmp_path):
    project_dir, manifest, ids = _setup(tmp_path, ["a.png", "b.png"])
    _version(project_dir, "detection", {"train": ids}, {i: DETECTION for i in ids})
    (project_dir / manifest.get(ids[1]).file).unlink()
    with pytest.raises(CodedError) as caught:
        check_export(project_dir, 1, "yolo")
    assert caught.value.code == "export_images_missing" and caught.value.params == {"count": 1}
    manifest.delete([ids[0]])
    with pytest.raises(CodedError) as again:
        check_export(project_dir, 1, "yolo", with_images=False)
    assert again.value.params == {"count": 1}  # the deleted record is gone; the file check is for images only
