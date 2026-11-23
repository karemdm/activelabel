import pytest
from sample_images import make_image
from sync_fakes import SETTINGS, FakeRemote, box, make_project, progress

from activelabel.annotate import diff as diffing
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, load_annotations, save_annotations
from activelabel.sync import engine, links

CAT = [box("cat")]
CAT2 = [box("cat", 0.5, 0.5)]
CAT3 = [box("cat", 0.2, 0.6)]


class Env:
    pass


def plan(env, **kw):
    return engine.make_plan(env.store, env.project.id, "cvat", SETTINGS, env.remote, progress(), **kw)


def apply(env, plan_id, resolutions=None, default=None):
    return engine.apply_plan(
        env.store,
        env.project.id,
        "cvat",
        SETTINGS,
        env.remote,
        plan_id,
        resolutions or {},
        default,
        progress(),
    )


def set_local(env, index, labels):
    annotations = load_annotations(env.dir) or AnnotationSet(env.project.task, ["cat", "dog"], {}, {})
    annotations.images[env.ids[index]] = labels
    save_annotations(env.dir, annotations)


def local(env, index):
    annotations = load_annotations(env.dir)
    return annotations.images.get(env.ids[index], []) if annotations else []


def set_remote(env, name, labels):
    env.remote.projects[env.rid]["items"][env.remote.key_of(env.rid, name)].labels = labels


def build(tmp_path, count, labels):
    e = Env()
    e.store, e.project, e.ids = make_project(tmp_path, count, labels)
    e.dir = e.store.dir(e.project.id)
    e.remote = FakeRemote()
    return e


@pytest.fixture
def env(tmp_path):
    """Three images, annotated alike on both sides and synchronised once."""
    e = build(tmp_path, 3, {0: CAT, 1: CAT, 2: CAT})
    e.info = e.remote.seed("Pets remote", [(f"img{i}.png", CAT) for i in range(3)])
    first = plan(e, remote_id=e.info.id)
    assert (first["pull"], first["push"], first["add"], first["unchanged"]) == (0, 0, 0, 3)
    apply(e, first["plan_id"])
    e.rid = e.info.id
    return e


def test_first_sync_creates_link_with_bases(env):
    link = links.get_link(env.dir, "cvat")
    assert link["remote_id"] == env.rid and link["remote_name"] == "Pets remote"
    assert set(link["items"]) == set(env.ids)
    assert link["items"][env.ids[0]]["base"] == links.label_hash(CAT)
    assert link["items"][env.ids[0]]["key"] == env.remote.key_of(env.rid, "img0.png")


def test_nothing_changed(env):
    p = plan(env)
    assert (p["pull"], p["push"], p["add"], p["conflicts"], p["unchanged"]) == (0, 0, 0, [], 3)
    assert p["remote_only"] == {"count": 0, "examples": []}
    assert p["diff_pull"]["images_changed"] == 0 and p["diff_push"]["images_changed"] == 0


def test_pull_changed_remote(env):
    set_remote(env, "img1.png", CAT + [box("cat", 0.6, 0.6)])
    p = plan(env)
    assert (p["pull"], p["push"], p["add"]) == (1, 0, 0)
    assert p["diff_pull"]["images_changed"] == 1 and p["diff_pull"]["added"] == 1
    result = apply(env, p["plan_id"])
    assert result["pulled"] == 1 and result["pushed"] == 0 and result["backup"].endswith(".bak.json")
    assert (env.dir / result["backup"]).exists()
    assert len(local(env, 1)) == 2
    assert plan(env)["unchanged"] == 3  # the base was updated: the next comparison is clean


def test_push_changed_local(env):
    set_local(env, 2, CAT2)
    p = plan(env)
    assert (p["pull"], p["push"]) == (0, 1)
    assert p["diff_push"]["images_changed"] == 1 and p["diff_push"]["added"] == 1
    result = apply(env, p["plan_id"])
    assert result["pushed"] == 1
    assert env.remote.labels(env.rid, "img2.png")[0].bbox == CAT2[0].bbox
    assert ("write", [env.remote.key_of(env.rid, "img2.png")]) in env.remote.calls
    assert plan(env)["unchanged"] == 3


def test_both_changed_to_the_same_is_nothing(env):
    set_local(env, 0, CAT2)
    set_remote(env, "img0.png", [box("cat", 0.50001, 0.5)])
    p = plan(env)
    assert (p["pull"], p["push"], p["conflicts"], p["unchanged"]) == (0, 0, [], 3)
    apply(env, p["plan_id"])
    assert links.get_link(env.dir, "cvat")["items"][env.ids[0]]["base"] == links.label_hash(CAT2)


def make_conflicts(env):
    for i in range(3):
        set_local(env, i, CAT2)
        set_remote(env, f"img{i}.png", CAT3)


def test_conflicts_listed_with_summary(env):
    make_conflicts(env)
    p = plan(env)
    assert len(p["conflicts"]) == 3 and p["pull"] == 0 and p["push"] == 0
    first = next(c for c in p["conflicts"] if c["image_id"] == env.ids[0])
    assert first["name"] == "img0.png"
    assert first["kind"] == "labels"
    assert first["local"] == {"labels": 1, "per_class": {"cat": 1}, "removed": False}
    assert first["remote"] == {"labels": 1, "per_class": {"cat": 1}, "removed": False}


def test_apply_resolutions_and_default(env):
    make_conflicts(env)
    p = plan(env)
    result = apply(env, p["plan_id"], {env.ids[0]: "remote"}, "local")
    assert (result["pulled"], result["pushed"], result["conflicts_left"]) == (1, 2, 0)
    assert local(env, 0)[0].bbox == CAT3[0].bbox  # the remote won
    assert env.remote.labels(env.rid, "img1.png")[0].bbox == CAT2[0].bbox  # ActiveLabel won
    assert env.remote.labels(env.rid, "img2.png")[0].bbox == CAT2[0].bbox


def test_unresolved_conflicts_are_left_alone(env):
    make_conflicts(env)
    p = plan(env)
    result = apply(env, p["plan_id"], {env.ids[1]: "local"})
    assert (result["pulled"], result["pushed"], result["conflicts_left"]) == (0, 1, 2)
    assert local(env, 0)[0].bbox == CAT2[0].bbox
    assert len(plan(env)["conflicts"]) == 2  # still conflicts next time


def test_invalid_choice_is_refused(env):
    make_conflicts(env)
    p = plan(env)
    with pytest.raises(engine.SyncError):
        apply(env, p["plan_id"], {env.ids[0]: "both"})
    with pytest.raises(engine.SyncError):
        apply(env, p["plan_id"], default="maybe")


def test_add_images_not_in_remote(env, tmp_path):
    folder = tmp_path / "more"
    make_image(folder / "new.png", seed=99, fmt="PNG")
    ingest_folder(Manifest(env.dir), folder, folder)
    new_id = next(r.id for r in Manifest(env.dir).all() if r.name == "new.png")
    annotations = load_annotations(env.dir)
    annotations.images[new_id] = [box("dog")]
    save_annotations(env.dir, annotations)
    p = plan(env)
    assert p["add"] == 1 and p["diff_push"]["added"] == 1
    result = apply(env, p["plan_id"])
    assert result["added"] == 1
    assert env.remote.labels(env.rid, "new.png")[0].label == "dog"
    item = links.get_link(env.dir, "cvat")["items"][new_id]
    assert item["key"] == env.remote.key_of(env.rid, "new.png")
    assert plan(env)["add"] == 0


def test_remote_only_is_reported_not_imported(env):
    env.remote.put(env.rid, "extra1.png", CAT)
    env.remote.put(env.rid, "extra2.png", CAT)
    p = plan(env)
    assert p["remote_only"] == {"count": 2, "examples": ["extra1.png", "extra2.png"]}
    assert (p["pull"], p["push"], p["add"]) == (0, 0, 0)
    apply(env, p["plan_id"])
    assert len(Manifest(env.dir).all()) == 3


def test_excluded_images_are_ignored(env):
    Manifest(env.dir).set_status([env.ids[0]], "excluded", "test")
    set_local(env, 0, CAT2)
    set_remote(env, "img0.png", CAT3)
    p = plan(env)
    assert (p["pull"], p["push"], p["add"], p["conflicts"], p["unchanged"]) == (0, 0, 0, [], 3)
    # a local removal is not sent to a flat remote (no deleted frames): the image counts as unchanged
    assert p["remove_remote"] == 0 and p["remote_only"]["count"] == 0


def test_excluded_image_missing_remotely_is_not_added(env):
    # linked images whose remote item disappeared are now removed locally (remove_local), not re-added;
    # the excluded one is removed on both sides (unchanged)
    env.remote.projects[env.rid]["items"].clear()
    Manifest(env.dir).set_status([env.ids[0]], "excluded", "test")
    p = plan(env)
    assert (p["add"], p["remove_local"], p["unchanged"]) == (0, 2, 1)


def test_unlinked_excluded_image_is_not_added(tmp_path):
    e = build(tmp_path, 3, {})
    e.rid = e.remote.seed("Pets remote").id
    Manifest(e.dir).set_status([e.ids[0]], "excluded", "test")
    assert plan(e, remote_id=e.rid)["add"] == 2


def test_refuses_when_remote_changed_after_plan(env):
    set_local(env, 0, CAT2)
    p = plan(env)
    assert p["push"] == 1
    set_remote(env, "img0.png", CAT3)
    before = list(env.remote.calls)
    with pytest.raises(engine.RemoteChanged, match="Synchronise again"):
        apply(env, p["plan_id"])
    assert env.remote.calls == before and local(env, 0)[0].bbox == CAT2[0].bbox


def test_refuses_when_remote_item_vanished(env):
    set_remote(env, "img1.png", CAT2)
    p = plan(env)
    assert p["pull"] == 1
    env.remote.projects[env.rid]["items"].clear()
    with pytest.raises(engine.RemoteChanged):
        apply(env, p["plan_id"])


def test_refuses_when_local_changed_after_plan(env):
    set_remote(env, "img1.png", CAT2)
    p = plan(env)
    set_local(env, 1, CAT3)
    with pytest.raises(engine.RemoteChanged, match="annotations changed"):
        apply(env, p["plan_id"])
    assert local(env, 1)[0].bbox == CAT3[0].bbox


def test_unknown_plan(env):
    with pytest.raises(engine.SyncError):
        apply(env, "abcdef12")
    assert engine.load_plan(env.dir, "../../x") is None


def test_plan_is_saved_and_consumed(env):
    set_remote(env, "img1.png", CAT2)
    p = plan(env)
    saved = engine.load_plan(env.dir, p["plan_id"])
    assert saved["remote_labels"] and saved["decisions"][env.ids[1]]["action"] == "pull"
    apply(env, p["plan_id"])
    assert engine.load_plan(env.dir, p["plan_id"]) is None


def test_classes_both_ways(env):
    set_remote(env, "img1.png", [box("bird")])
    p = plan(env)
    assert p["diff_pull"]["unknown_classes"] == ["bird"]
    result = apply(env, p["plan_id"])
    assert result["new_classes"] == ["bird"]
    assert "bird" in env.store.get(env.project.id).classes
    # a local class the remote project does not know is added to it when something is pushed
    env.store.update(env.project.id, classes=[*env.store.get(env.project.id).classes, "fish"])
    set_local(env, 2, [box("fish")])
    apply(env, plan(env)["plan_id"])
    assert "fish" in env.remote.projects[env.rid]["classes"]


def test_remote_class_case_is_matched(env):
    set_remote(env, "img1.png", [box("CAT", 0.4)])
    p = plan(env)
    assert p["diff_pull"]["unknown_classes"] == []
    apply(env, p["plan_id"])
    assert local(env, 1)[0].label == "cat"


def test_unknown_base_empty_side_loses(tmp_path):
    e = build(tmp_path, 3, {0: CAT, 2: CAT2})
    info = e.remote.seed("R", [("img0.png", []), ("img1.png", CAT), ("img2.png", CAT3)])
    p = plan(e, remote_id=info.id)
    assert (p["pull"], p["push"], len(p["conflicts"])) == (1, 1, 1)
    assert p["conflicts"][0]["name"] == "img2.png"


def test_create_remote_on_apply_only(tmp_path):
    e = build(tmp_path, 2, {0: CAT})
    with pytest.raises(engine.SyncError, match="no remote project"):
        plan(e)
    p = plan(e, create=True)
    assert p["create"] is True and p["add"] == 2 and e.remote.projects == {}
    assert apply(e, p["plan_id"])["added"] == 2
    (rid,) = e.remote.projects
    assert e.remote.projects[rid]["name"] == f"ActiveLabel {e.project.id} - Pets"
    assert links.get_link(e.dir, "cvat")["remote_id"] == rid
    e.rid = rid
    assert plan(e)["unchanged"] == 2


def test_plan_uses_name_lookup_when_not_linked(tmp_path):
    e = build(tmp_path, 2, {0: CAT})
    info = e.remote.seed(f"ActiveLabel {e.project.id} - Pets", [("img0.png", CAT)])
    p = plan(e)
    assert p["remote"]["id"] == info.id and p["add"] == 1 and p["create"] is False


def test_pull_diff_matches_annotation_diff(env):
    set_remote(env, "img1.png", CAT2)
    p = plan(env)
    assert p["diff_pull"] == diffing.diff(env.dir, [env.ids[1]], {env.ids[1]: CAT2}, env.project)
