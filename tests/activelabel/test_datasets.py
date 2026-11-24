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


def build(tmp_path, spec, task="detection", videos=None, tasks=None, fps=None):
    """A project whose images are ``spec``: ``{image id: [class names]}`` (empty = background).

    ``videos`` maps image ids to a video id (frames are numbered in spec order; with ``fps`` they get
    a time too).
    """
    store = ProjectStore(tmp_path / "projects")
    project = store.create("P", "", task, ["a", "b", "c", "rare"])
    pdir = store.dir(project.id)
    videos = videos or {}
    records = []
    counters: dict[str, int] = {}
    for i in spec:
        video = videos.get(i, "")
        if video:
            counters[video] = counters.get(video, -1) + 1
        frame = counters.get(video) if video else None
        time = frame / fps if fps and frame is not None else None
        records.append(
            ImageRecord(i, f"{i}.png", f"images/{i}.png", 8, 8, 1, "t", video=video, frame=frame, time=time)
        )
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
    codes = {(w["code"], w.get("part")) for w in plan(pdir, project, videos="whole").warnings}
    assert ("few_images", "test") in codes
    assert ("class_missing", "val") in codes or ("class_missing", "test") in codes
    assert ("group_too_big", "test") in codes  # only where the big group moved the proportions
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


def video_case(tmp_path, frames=50, fps=2, **extra):
    """One video of ``frames`` frames plus two single images: the case that used to give 98/2/0."""
    spec = {f"f{k:02d}": ["a"] for k in range(frames)} | {"x1": ["a"], "x2": ["b"]}
    videos = {f"f{k:02d}": "v1" for k in range(frames)}
    return build(tmp_path, spec, videos=videos, fps=fps, **extra)


def segments_of(p, part):
    (video,) = p.videos
    return [s for s in video["segments"] if s["part"] == part]


def test_big_video_is_cut_into_spread_segments(tmp_path):
    pdir, project = video_case(tmp_path)
    whole = plan(pdir, project, videos="whole")
    assert whole.sizes()["val"] < 3
    p = plan(pdir, project)
    sizes = p.sizes()
    assert sizes["val"] >= 5 and sizes["test"] >= 2 and sizes["train"] > sizes["val"]
    (video,) = p.videos
    assert video["split"] and video["frames"] == 50 and len(video["segments"]) == 10
    assert video["name"] == "v1" and {s["part"] for s in video["segments"]} == {"train", "val", "test"}
    for part in ("val", "test"):
        found = segments_of(p, part)
        assert found
        for before, after in zip(found, found[1:], strict=False):
            assert after["first_frame"] - before["last_frame"] > 5  # far apart: whole segments between
    info = [w for w in p.warnings if w["code"] == "video_split"]
    assert len(info) == 1 and info[0]["segments"] == 10 and info[0]["parts"] == ["val", "test"]
    assert info[0]["gap"] == video["gap"] > 0
    assert p.preview()["videos"] == p.videos


def test_gap_frames_are_left_out_and_counted(tmp_path):
    pdir, project = video_case(tmp_path)  # 2 fps: a gap of 1 s = 2 frames
    p = plan(pdir, project, gap_seconds=1.0)
    (video,) = p.videos
    where = part_of(p)
    frames = {k: f"f{k:02d}" for k in range(50)}
    protected = [n for n, i in frames.items() if where.get(i) in ("val", "test")]
    expected = {
        n
        for n, i in frames.items()
        if i not in where or where[i] == "train"
        if any(abs(n - m) <= 2 for m in protected)
    }
    left_out = {n for n, i in frames.items() if i not in where}
    assert left_out and left_out <= expected
    assert len(left_out) == video["gap"] == p.stats["gap"]
    assert not [i for i in p.labels if i not in where]
    assert sum(p.sizes().values()) == 52 - video["gap"]
    for n, i in frames.items():  # no train frame left within the gap of a val/test frame
        if where.get(i) == "train":
            assert all(abs(n - m) > 2 for m in protected)
    none = plan(pdir, project, gap_seconds=0)
    assert none.videos[0]["gap"] == 0 and sum(none.sizes().values()) == 52


def test_whole_mode_keeps_the_video_together(tmp_path):
    pdir, project = video_case(tmp_path)
    p = plan(pdir, project, videos="whole")
    where = part_of(p)
    assert len({where[f"f{k:02d}"] for k in range(50)}) == 1
    assert p.videos[0]["split"] is False and p.videos[0]["segments"] == [] and p.videos[0]["gap"] == 0
    assert not [w for w in p.warnings if w["code"] == "video_split"] and "gap" not in p.stats
    assert any(w["code"] == "ratio_drift" for w in p.warnings)


def test_auto_keeps_videos_whole_when_proportions_are_reached(tmp_path):
    videos = {f"i{k:03d}": "v1" for k in range(0, 8)} | {f"i{k:03d}": "v2" for k in range(50, 58)}
    pdir, project = build(tmp_path, balanced(60), videos=videos, fps=5)
    p = plan(pdir, project)
    assert [v["split"] for v in p.videos] == [False, False]
    assert plan(pdir, project, videos="whole").parts == p.parts


def test_split_is_deterministic(tmp_path):
    pdir, project = video_case(tmp_path)
    one, two = plan(pdir, project, seed=3), plan(pdir, project, seed=3)
    assert one.parts == two.parts and one.videos == two.videos and one.fingerprint() == two.fingerprint()
    assert one.options.as_dict()["videos"] == "auto" and one.options.as_dict()["gap_seconds"] == 0.5


def test_near_duplicates_inside_a_video_do_not_glue_segments(tmp_path):
    pdir, project = video_case(tmp_path)
    chain = {f"f{k:02d}": f"f{k - 1:02d}" for k in range(1, 50)}  # every frame looks like the previous
    chain["x1"] = "f10"  # a link to another image still groups
    write_json(curation_path(pdir), {"duplicate_of": chain})
    p = plan(pdir, project)
    assert p.videos[0]["split"] and p.sizes()["val"] >= 5 and p.sizes()["test"] >= 2
    where = part_of(p)
    if "f10" in where:
        assert where["x1"] == where["f10"]
    whole = plan(pdir, project, videos="whole")
    assert len({v for i, v in part_of(whole).items() if i.startswith("f") or i == "x1"}) == 1


def test_kfold_with_a_split_video(tmp_path):
    pdir, project = video_case(tmp_path)
    p = plan(pdir, project, kfold=3, split={"train": 0, "val": 0, "test": 10}, gap_seconds=0)
    assert p.folds is not None and all(len(f) >= 5 for f in p.folds) and p.parts["test"]
    (video,) = p.videos
    assert video["split"]
    for n in range(3):
        found = [s for s in video["segments"] if s.get("fold") == n]
        assert len(found) >= 2
        for before, after in zip(found, found[1:], strict=False):
            assert after["first_frame"] - before["last_frame"] > 1  # a neighbour belongs to another part
    info = next(w for w in p.warnings if w["code"] == "video_split")
    assert info["parts"] == ["test", "fold"]
    # only one video and five folds: whole cannot even be built, auto cuts the video
    pdir2, project2 = video_case(tmp_path / "k", frames=40)
    options = {"kfold": 5, "split": {"train": 0, "val": 0, "test": 0}}
    assert plan(pdir2, project2, **options).folds is not None
    with pytest.raises(ValueError, match="folds"):
        plan(pdir2, project2, videos="whole", **options)


def test_video_options_validation():
    for bad in (
        {"videos": "x"},
        {"gap_seconds": -1},
        {"gap_seconds": "1"},
        {"gap_seconds": True},
        {"gap_seconds": 1e9},
    ):
        with pytest.raises(ValueError):
            parse_options(bad, [1])
    ok = parse_options({"videos": "whole", "gap_seconds": 0}, [1])
    assert (ok.videos, ok.gap_seconds) == ("whole", 0.0)
    assert parse_options({}, [1]).videos == "auto" and parse_options({}, [1]).gap_seconds == 0.5


def test_summary_keeps_the_video_report(tmp_path):
    pdir, project = video_case(tmp_path)
    store = DatasetStore(pdir)
    made = store.create(plan(pdir, project))
    assert made["videos"][0]["split"] and store.versions()[0]["videos"] == made["videos"]
    assert any(w["code"] == "video_split" for w in made["warnings"])


def test_cut_video_is_reported_when_every_segment_is_glued_to_another_image(tmp_path):
    frames = {f"f{k:02d}": ["a"] for k in range(50)}
    glue = {f"y{k}": ["a"] for k in range(10)}
    pdir, project = build(tmp_path, frames | glue, videos=dict.fromkeys(frames, "v1"), fps=2)
    links = {f"y{k}": f"f{5 * k:02d}" for k in range(10)}  # each 5-frame segment meets a still image
    write_json(curation_path(pdir), {"duplicate_of": links})
    p = plan(pdir, project)
    (video,) = p.videos
    where = part_of(p)
    assert len({where[i] for i in frames if i in where}) > 1  # the video really is in several parts
    assert video["split"] and len(video["segments"]) == 10
    assert [w for w in p.warnings if w["code"] == "video_split"]
    assert p.stats["gap"] == video["gap"]
