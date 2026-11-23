import json

import pytest

from activelabel.curation.issues import curation_path
from activelabel.datasets import DatasetStore, parse_options, plan_split
from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.projects import ProjectStore
from activelabel.storage import write_json
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore


def build(tmp_path, spec, task="detection", videos=None, tasks=None):
    """A project whose images are ``spec``: ``{image id: [class names]}`` (empty = background)."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("P", "", task, ["a", "b", "c", "rare"])
    pdir = store.dir(project.id)
    videos = videos or {}
    records = [
        ImageRecord(i, f"{i}.png", f"images/{i}.png", 8, 8, 1, "t", video=videos.get(i, "")) for i in spec
    ]
    for chunk in tasks or [list(spec)]:  # tasks first: unplaced images would become an extra task
        TaskStore(pdir).create(images=chunk)
    Manifest(pdir).add(records)
    kind = Task(task)
    images = {
        i: [Label(c) if kind == Task.CLASSIFICATION else Label(c, bbox=[0, 0, 0.5, 0.5]) for c in cs]
        for i, cs in spec.items()
        if cs
    }
    save_annotations(pdir, AnnotationSet(kind, project.classes, images, {}))
    return pdir, project


def balanced(n=100):
    """Up to 100 images, one class each: a (50), b (30), c (10), rare (10)."""
    classes = ["a"] * 50 + ["b"] * 30 + ["c"] * 10 + ["rare"] * 10
    return {f"i{k:03d}": [c] for k, c in enumerate(classes[:n])}


def plan(pdir, project, **options):
    return plan_split(pdir, project, parse_options(options, [t.id for t in TaskStore(pdir).load()]))


def part_of(p):
    return {i: name for name, ids in p.parts.items() for i in ids}


def test_options_validation():
    ok = parse_options({"split": {"train": 60, "val": 20, "test": 20}, "seed": 7}, [1])
    assert ok.split == {"train": 60, "val": 20, "test": 20} and ok.seed == 7 and ok.kfold is None
    for bad in (
        {"split": {"train": 60, "val": 20, "test": 10}},
        {"split": {"train": 0, "val": 50, "test": 50}},
        {"split": {"train": 70.5, "val": 19.5, "test": 10}},
        {"split": {"train": True, "val": 20, "test": 10}},
        {"split": {"train": 70, "val": 20, "test": 10, "x": 0}},
        {"kfold": 1},
        {"kfold": 11},
        {"seed": "a"},
        {"tasks": [9]},
        {"background": "yes"},
    ):
        with pytest.raises(ValueError):
            parse_options(bad, [1])
    kf = parse_options({"kfold": 5, "split": {"train": 0, "val": 0, "test": 10}}, [1])
    assert kf.kfold == 5  # val ignored with K-fold: percentages need not add up


def test_deterministic_and_seed_changes_result(tmp_path):
    pdir, project = build(tmp_path, balanced())
    one, two = plan(pdir, project, seed=1), plan(pdir, project, seed=1)
    assert one.parts == two.parts and one.fingerprint() == two.fingerprint()
    other = plan(pdir, project, seed=2)
    assert other.parts != one.parts and other.fingerprint() != one.fingerprint()


def test_ratios_and_stratification_with_rare_class(tmp_path):
    pdir, project = build(tmp_path, balanced())
    p = plan(pdir, project)
    assert p.sizes() == {"train": 70, "val": 20, "test": 10}
    for name in ("train", "val", "test"):
        assert set(p.stats[name]["objects"]) == {"a", "b", "c", "rare"}, name
    assert p.stats["test"]["objects"]["rare"] == 1 and p.stats["train"]["objects"]["rare"] == 7
    assert not [w for w in p.warnings if w["code"] == "class_missing"]
    assert sorted(i for ids in p.parts.values() for i in ids) == sorted(balanced())


def test_video_frames_never_split(tmp_path):
    videos = {f"i{k:03d}": "v1" for k in range(0, 8)} | {f"i{k:03d}": "v2" for k in range(50, 58)}
    pdir, project = build(tmp_path, balanced(60), videos=videos)
    for seed in range(5):
        where = part_of(plan(pdir, project, seed=seed))
        assert len({where[f"i{k:03d}"] for k in range(0, 8)}) == 1
        assert len({where[f"i{k:03d}"] for k in range(50, 58)}) == 1


def test_near_duplicates_never_split_transitively(tmp_path):
    pdir, project = build(tmp_path, balanced(60))
    write_json(
        curation_path(pdir),
        {"duplicate_of": {"i001": "i000", "i002": "i001", "i010": "i009", "i020": "i099"}},
    )
    for seed in range(5):
        p = plan(pdir, project, seed=seed)
        where = part_of(p)
        assert len({where[i] for i in ("i000", "i001", "i002")}) == 1
        assert where["i009"] == where["i010"]
        assert p.duplicates_analysed
        assert not [w for w in p.warnings if w["code"] == "no_curation"]
    other_dir, other = build(tmp_path / "x", balanced(20))
    assert any(w["code"] == "no_curation" for w in plan(other_dir, other).warnings)


def test_background_off_and_on(tmp_path):
    spec = balanced(40) | {f"bg{k}": [] for k in range(20)}
    pdir, project = build(tmp_path, spec)
    off = plan(pdir, project)
    assert sum(off.sizes().values()) == 40 and all(s["background"] == 0 for s in off.stats.values())
    on = plan(pdir, project, background=True)
    assert sum(on.sizes().values()) == 60
    assert sum(s["background"] for s in on.stats.values()) == 20
    assert all(s["background"] > 0 for s in on.stats.values())
    assert all(abs(on.sizes()[k] - 60 * v / 100) <= 2 for k, v in on.options.split.items())
    assert not [i for i in on.labels if i.startswith("bg")]  # no label copy for background


def test_classification_has_no_background(tmp_path):
    spec = balanced(40) | {f"bg{k}": [] for k in range(10)}
    pdir, project = build(tmp_path, spec, task="classification")
    p = plan(pdir, project, background=True)
    assert sum(p.sizes().values()) == 40 and p.type == "classification"
    assert sum(p.stats["train"]["objects"].values()) == p.sizes()["train"]


def test_detection_objects_counted_and_multi_class(tmp_path):
    spec = {f"i{k:03d}": ["a", "a", "b"] if k % 2 else ["b"] for k in range(40)}
    pdir, project = build(tmp_path, spec)
    p = plan(pdir, project)
    assert sum(s["objects"]["a"] for s in p.stats.values()) == 40
    assert sum(s["objects"]["b"] for s in p.stats.values()) == 40


def test_tasks_filter_and_excluded_images(tmp_path):
    ids = list(balanced(40))
    pdir, project = build(tmp_path, balanced(40), tasks=[ids[:20], ids[20:]])
    Manifest(pdir).set_status(ids[:5], "excluded")
    assert sum(plan(pdir, project).sizes().values()) == 35
    second = plan(pdir, project, tasks=[2])
    assert set(ids[20:]) == {i for v in second.parts.values() for i in v}
    Manifest(pdir).set_status(ids[20:], "excluded")
    with pytest.raises(ValueError, match="no images"):
        plan(pdir, project, tasks=[2])


def test_zero_parts(tmp_path):
    pdir, project = build(tmp_path, balanced())
    p = plan(pdir, project, split={"train": 80, "val": 0, "test": 20})
    assert p.sizes() == {"train": 80, "val": 0, "test": 20}
    assert not [w for w in p.warnings if w.get("part") == "val"]


def test_kfold(tmp_path):
    pdir, project = build(tmp_path, balanced())
    p = plan(pdir, project, kfold=5, split={"train": 0, "val": 0, "test": 20})
    assert p.folds is not None and len(p.folds) == 5 and set(p.parts) == {"test"}
    assert p.sizes() == {"test": 20, "folds": [16] * 5}
    seen = [i for f in p.folds for i in f]
    assert len(seen) == len(set(seen)) == 80 and not set(seen) & set(p.parts["test"])
    assert set(seen) | set(p.parts["test"]) == set(balanced())
    for fold in p.stats["folds"]:
        assert set(fold["objects"]) == {"a", "b", "c", "rare"}
    assert p.preview()["sizes"]["folds"] == [16] * 5
    no_test = plan(pdir, project, kfold=4, split={"train": 0, "val": 0, "test": 0})
    assert no_test.parts == {"test": []} and sum(no_test.sizes()["folds"]) == 100
    small_dir, small = build(tmp_path / "y", {"i1": ["a"], "i2": ["a"]})
    with pytest.raises(ValueError, match="folds"):
        plan(small_dir, small, kfold=5)


def test_warnings(tmp_path):
    spec = {f"i{k:02d}": ["a"] for k in range(12)} | {"r1": ["rare"]}
    videos = {f"i{k:02d}": "v" for k in range(10)}
    pdir, project = build(tmp_path, spec, videos=videos)
    codes = {(w["code"], w.get("part")) for w in plan(pdir, project).warnings}
    assert ("few_images", "test") in codes
    assert ("class_missing", "val") in codes or ("class_missing", "test") in codes
    assert ("group_too_big", "val") in codes and ("group_too_big", "test") in codes
    assert any(c == "ratio_drift" for c, _ in codes)
    clean_dir, clean = build(tmp_path / "z", balanced())
    assert [w["code"] for w in plan(clean_dir, clean).warnings] == ["no_curation"]


def test_store_versions_immutable_numbering_and_incomplete(tmp_path):
    pdir, project = build(tmp_path, balanced(30))
    store = DatasetStore(pdir)
    assert store.versions() == []
    first = store.create(plan(pdir, project), " first ")
    second = store.create(plan(pdir, project, seed=3))
    assert (first["version"], second["version"], first["note"]) == (1, 2, "first")
    assert [v["version"] for v in store.versions()] == [2, 1]
    stored = json.loads((pdir / "datasets" / "v1.json").read_text(encoding="utf-8"))
    assert stored["fingerprint"] and stored["options"]["seed"] == 42 and stored["type"] == "detection"
    # immutability: editing the annotations later leaves the stored copy alone
    annotations = load_annotations(pdir)
    annotations.images["i000"] = [Label("rare", bbox=[0, 0, 1, 1])]
    save_annotations(pdir, annotations)
    assert store.read(1)["labels"]["i000"][0]["label"] == "a"
    assert "labels" not in store.get(1) and sum(store.get(1)["sizes"].values()) == 30
    # numbers are never reused, even after deleting the newest
    store.delete(2)
    assert store.create(plan(pdir, project))["version"] == 3
    store.delete(1)
    assert [v["version"] for v in store.versions()] == [3]
    with pytest.raises(KeyError):
        store.get(1)
    with pytest.raises(KeyError):
        store.delete(1)
    # incomplete: an image of the version leaves the project
    assert store.get(3)["incomplete"] is False
    Manifest(pdir).delete(["i000"])
    assert store.get(3)["incomplete"] is True and store.versions()[0]["incomplete"] is True


def test_image_ids_parts_and_folds(tmp_path):
    pdir, project = build(tmp_path, balanced())
    store = DatasetStore(pdir)
    plain = store.create(plan(pdir, project))["version"]
    assert len(store.image_ids(plain, "val")) == 20 and len(store.image_ids(plain)) == 100
    with pytest.raises(ValueError):
        store.image_ids(plain, fold=0)
    with pytest.raises(ValueError):
        store.image_ids(plain, "other")
    kf = store.create(plan(pdir, project, kfold=4, split={"train": 0, "val": 0, "test": 20}))["version"]
    assert len(store.image_ids(kf, "test")) == 20
    val, train = store.image_ids(kf, "val", 1), store.image_ids(kf, "train", 1)
    assert len(val) == 20 and len(train) == 60 and not set(val) & set(train)
    assert store.image_ids(kf, fold=1) == val
    with pytest.raises(ValueError):
        store.image_ids(kf, "train")
    with pytest.raises(ValueError):
        store.image_ids(kf, "val", 4)
