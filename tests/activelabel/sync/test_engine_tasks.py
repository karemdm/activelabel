"""Synchronisation with a CVAT-like remote that has tasks, jobs and deleted frames (docs/TASKS.md)."""

import copy
import tempfile
from pathlib import Path

import pytest
from make_video import make_video
from sync_fakes import SETTINGS, FakeRemote, box, make_project, progress

from activelabel.annotate.base import ToolError
from activelabel.images.manifest import Manifest
from activelabel.images.video import VideoStore
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.storage import write_json
from activelabel.sync import engine, links
from activelabel.tasks_store import ImageTask, TaskStore

CAT = [box("cat")]
CAT2 = [box("cat", 0.5, 0.5)]
CAT3 = [box("cat", 0.2, 0.6)]


class Env:
    pass


def plan(env, **kw):
    return engine.make_plan(env.store, env.project.id, "cvat", SETTINGS, env.remote, progress(), **kw)


def apply(env, plan_id, resolutions=None, default=None, delete_tasks=False):
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
        delete_tasks=delete_tasks,
    )


def set_local(env, index, labels):
    annotations = load_annotations(env.dir) or AnnotationSet(env.project.task, ["cat", "dog"], {}, {})
    annotations.images[env.ids[index]] = labels
    save_annotations(env.dir, annotations)


def local(env, index):
    annotations = load_annotations(env.dir)
    return annotations.images.get(env.ids[index], []) if annotations else []


def set_remote(env, index, labels):
    env.remote.projects[env.rid]["items"][env.key(index)].labels = labels


def build(tmp_path):
    """8 images in two local tasks: Alpha (img0-4, 2 per job) and Beta (img5-7, one job), all labelled."""
    e = Env()
    e.store, e.project, e.ids = make_project(tmp_path, 8, dict.fromkeys(range(8), CAT))
    e.dir = e.store.dir(e.project.id)
    alpha = ImageTask(1, "Alpha", "2026-01-01T00:00:00", 2, e.ids[:5])
    beta = ImageTask(2, "Beta", "2026-01-01T00:00:01", None, e.ids[5:])
    write_json(e.dir / "tasks.json", {"next_id": 3, "tasks": [alpha.as_dict(), beta.as_dict()]})
    e.remote = FakeRemote("cvat")
    e.rid = ""
    e.key = lambda index: links.get_link(e.dir, "cvat")["items"][e.ids[index]]["key"]
    return e


@pytest.fixture
def env(tmp_path):
    return build(tmp_path)


@pytest.fixture
def synced(env):
    """The project sent whole to CVAT (both tasks created) and synchronised once."""
    first = plan(env, create=True)
    assert len(first["tasks_create"]) == 2
    apply(env, first["plan_id"])
    link = links.get_link(env.dir, "cvat")
    env.rid = link["remote_id"]
    env.tids = {int(t): m["remote_task"] for t, m in link["tasks"].items()}
    env.remote.calls.clear()
    env.remote.item_calls.clear()
    return env


def frame_of(env, index):
    return int(env.key(index).partition(":")[2])


def task_of(env, index):
    return env.key(index).partition(":")[0]


# -- sending whole tasks ----------------------------------------------------------
def test_project_scope_creates_whole_tasks(env):
    p = plan(env, create=True)
    assert p["create"] and p["scope"] == {"scope": "project", "task": None, "job": None}
    assert p["tasks_create"] == [
        {"task": 1, "name": "Alpha", "job_size": 2, "images": 5},
        {"task": 2, "name": "Beta", "job_size": None, "images": 3},
    ]
    assert p["add"] == 0 and p["diff_push"]["images_changed"] == 8
    assert (p["images_new"], p["labels_new"]) == (8, 8)  # the annotations that go with the new tasks
    result = apply(env, p["plan_id"])
    assert result["tasks_created"] == 2
    assert (result["images_new"], result["labels_new"]) == (8, 8)  # counted as the plan promised
    adds = [c for c in env.remote.calls if c[0] == "add_task"]
    assert adds == [
        ("add_task", "Alpha", 2, [f"img{i}.png" for i in range(5)]),
        ("add_task", "Beta", None, [f"img{i}.png" for i in range(5, 8)]),
    ]
    link = links.get_link(env.dir, "cvat")
    rid = link["remote_id"]
    assert set(link["tasks"]) == {"1", "2"} and set(link["items"]) == set(env.ids)
    alpha = link["tasks"]["1"]
    assert alpha["name"] == "Alpha" and sorted(alpha["jobs"]) == ["1", "2", "3"]
    tasks = {t.name: t for t in env.remote.tasks(SETTINGS, rid)}
    assert alpha["remote_task"] == tasks["Alpha"].id and tasks["Alpha"].job_size == 2
    assert [(j.start, j.stop) for j in tasks["Alpha"].jobs] == [(0, 1), (2, 3), (4, 4)]
    assert link["items"][env.ids[3]]["key"] == f"{tasks['Alpha'].id}:3"
    assert len(env.remote.labels(rid, "img0.png")) == 1
    env.rid = rid
    again = plan(env)
    assert (again["unchanged"], again["tasks_create"], again["tasks_new"], again["tasks_delete"]) == (
        8,
        [],
        [],
        [],
    )
    assert again["pull"] == again["push"] == again["add"] == 0 and again["conflicts"] == []


def test_excluded_image_of_a_new_task_is_uploaded_then_hidden(env):
    Manifest(env.dir).set_status([env.ids[1]], "excluded", "blurry")
    p = plan(env, create=True)
    assert p["remove_remote"] == 0  # created whole: the frame is hidden right after the upload
    assert p["images_new"] == 7  # the excluded image is sent but is not an annotation to review
    assert apply(env, p["plan_id"])["images_new"] == 7
    link = links.get_link(env.dir, "cvat")
    env.rid = link["remote_id"]
    assert link["items"][env.ids[1]]["removed"] is True
    assert "removed" not in link["items"][env.ids[0]]
    tid = link["tasks"]["1"]["remote_task"]
    assert env.remote.deleted_frames(env.rid, tid) == [1]
    assert f"{tid}:1" in env.remote.projects[env.rid]["items"]  # it was uploaded
    assert ("set_frames", tid, [1], []) in env.remote.calls
    again = plan(env)
    assert (again["unchanged"], again["remove_remote"], again["restore_remote"]) == (8, 0, 0)


def test_task_scope_creates_only_its_own_task(env):
    p = plan(env, create=True, scope="task", task_id=2)
    assert [t["task"] for t in p["tasks_create"]] == [2]
    apply(env, p["plan_id"])
    assert [c[1] for c in env.remote.calls if c[0] == "add_task"] == ["Beta"]
    assert set(links.get_link(env.dir, "cvat")["tasks"]) == {"2"}


# -- labels -----------------------------------------------------------------------
def test_labels_pull_push_and_conflict(synced):
    env = synced
    set_remote(env, 0, CAT2)  # edited there
    set_local(env, 1, CAT2)  # edited here
    set_local(env, 2, CAT2)  # both, differently
    set_remote(env, 2, CAT3)
    set_local(env, 3, CAT2)
    set_remote(env, 3, CAT3)
    p = plan(env)
    assert (p["pull"], p["push"], len(p["conflicts"])) == (1, 1, 2)
    assert {c["kind"] for c in p["conflicts"]} == {"labels"}
    assert p["remote_only"]["count"] == 0
    result = apply(env, p["plan_id"], {env.ids[2]: "local", env.ids[3]: "remote"})
    assert (result["pulled"], result["pushed"], result["conflicts_left"]) == (2, 2, 0)
    assert local(env, 0)[0].bbox == CAT2[0].bbox
    assert env.remote.labels(env.rid, "img1.png")[0].bbox == CAT2[0].bbox
    assert env.remote.labels(env.rid, "img2.png")[0].bbox == CAT2[0].bbox  # local won
    assert local(env, 3)[0].bbox == CAT3[0].bbox  # remote won
    assert plan(env)["unchanged"] == 8


def test_unresolved_conflict_is_left_alone(synced):
    env = synced
    set_local(env, 2, CAT2)
    set_remote(env, 2, CAT3)
    result = apply(env, plan(env)["plan_id"])
    assert result["conflicts_left"] == 1
    assert len(plan(env)["conflicts"]) == 1


# -- images removed and restored --------------------------------------------------
def test_image_excluded_here_is_removed_there_and_restored(synced):
    env = synced
    tid = task_of(env, 0)
    Manifest(env.dir).set_status([env.ids[0]], "excluded", "test")
    p = plan(env)
    assert (p["remove_remote"], p["remove_local"], p["conflicts"]) == (1, 0, [])
    result = apply(env, p["plan_id"])
    assert result["removed_remote"] == 1
    assert ("set_frames", tid, [0], []) in env.remote.calls
    assert env.remote.deleted_frames(env.rid, tid) == [0]
    assert links.get_link(env.dir, "cvat")["items"][env.ids[0]]["removed"] is True
    assert plan(env)["unchanged"] == 8
    Manifest(env.dir).set_status([env.ids[0]], "active")
    p = plan(env)
    assert (p["restore_remote"], p["remove_remote"]) == (1, 0)
    assert apply(env, p["plan_id"])["restored_remote"] == 1
    assert env.remote.deleted_frames(env.rid, tid) == []
    assert "removed" not in links.get_link(env.dir, "cvat")["items"][env.ids[0]]
    assert plan(env)["unchanged"] == 8


def test_frame_deleted_there_excludes_here_and_restores(synced):
    env = synced
    tid = task_of(env, 1)
    env.remote.set_frames(SETTINGS, tid, [frame_of(env, 1)], [])
    p = plan(env)
    assert (p["remove_local"], p["remove_remote"], p["conflicts"]) == (1, 0, [])
    result = apply(env, p["plan_id"])
    assert result["removed_local"] == 1
    record = Manifest(env.dir).load()[env.ids[1]]
    assert record.status == "excluded" and record.excluded_reason == "deleted in cvat"
    assert plan(env)["unchanged"] == 8
    env.remote.set_frames(SETTINGS, tid, [], [frame_of(env, 1)])
    p = plan(env)
    assert p["restore_local"] == 1
    assert apply(env, p["plan_id"])["restored_local"] == 1
    assert Manifest(env.dir).load()[env.ids[1]].status == "active"
    assert plan(env)["unchanged"] == 8


def test_removed_here_edited_there_is_a_conflict(synced):
    env = synced
    tid = task_of(env, 2)
    Manifest(env.dir).set_status([env.ids[2], env.ids[3]], "excluded", "test")
    set_remote(env, 2, CAT3)
    set_remote(env, 3, CAT3)
    p = plan(env)
    assert len(p["conflicts"]) == 2 and {c["kind"] for c in p["conflicts"]} == {"removed_local"}
    one = next(c for c in p["conflicts"] if c["image_id"] == env.ids[2])
    assert one["local"]["removed"] is True and one["remote"]["removed"] is False
    result = apply(env, p["plan_id"], {env.ids[2]: "remote", env.ids[3]: "local"})
    assert (result["restored_local"], result["removed_remote"]) == (1, 1)
    assert Manifest(env.dir).load()[env.ids[2]].status == "active"
    assert local(env, 2)[0].bbox == CAT3[0].bbox  # restored with the labels from the tool
    assert env.remote.deleted_frames(env.rid, tid) == [frame_of(env, 3)]
    assert plan(env)["unchanged"] == 8


def test_edited_here_removed_there_is_a_conflict(synced):
    env = synced
    tid = task_of(env, 2)
    set_local(env, 2, CAT2)
    set_local(env, 3, CAT2)
    env.remote.set_frames(SETTINGS, tid, [frame_of(env, 2), frame_of(env, 3)], [])
    p = plan(env)
    assert len(p["conflicts"]) == 2 and {c["kind"] for c in p["conflicts"]} == {"removed_remote"}
    one = next(c for c in p["conflicts"] if c["image_id"] == env.ids[2])
    assert one["local"]["removed"] is False and one["remote"]["removed"] is True
    result = apply(env, p["plan_id"], {env.ids[2]: "local", env.ids[3]: "remote"})
    assert (result["restored_remote"], result["removed_local"]) == (1, 1)
    assert env.remote.deleted_frames(env.rid, tid) == [frame_of(env, 3)]  # img2 came back
    assert env.remote.labels(env.rid, "img2.png")[0].bbox == CAT2[0].bbox
    assert Manifest(env.dir).load()[env.ids[3]].excluded_reason == "deleted in cvat"
    assert plan(env)["unchanged"] == 8


# -- scopes -----------------------------------------------------------------------
def test_task_scope_only_touches_that_task(synced):
    env = synced
    set_remote(env, 0, CAT2)  # Alpha
    set_remote(env, 6, CAT2)  # Beta
    p = plan(env, scope="task", task_id=1)
    assert p["scope"] == {"scope": "task", "task": 1, "job": None}
    assert (p["pull"], p["unchanged"]) == (1, 4)
    assert env.remote.item_calls[-1] == [env.tids[1]]
    apply(env, p["plan_id"])
    assert env.remote.item_calls[-1] == [env.tids[1]]
    assert local(env, 0)[0].bbox == CAT2[0].bbox and local(env, 6)[0].bbox == CAT[0].bbox
    assert plan(env)["pull"] == 1  # Beta is still waiting


def test_job_scope_only_touches_that_job(synced):
    env = synced
    set_remote(env, 0, CAT2)  # Alpha job 1
    set_remote(env, 2, CAT2)  # Alpha job 2 (img2, img3)
    set_remote(env, 6, CAT2)  # Beta
    p = plan(env, scope="job", task_id=1, job=2)
    assert p["scope"] == {"scope": "job", "task": 1, "job": 2}
    assert (p["pull"], p["unchanged"]) == (1, 1)
    assert env.remote.item_calls[-1] == [env.tids[1]]
    apply(env, p["plan_id"])
    assert local(env, 2)[0].bbox == CAT2[0].bbox
    assert local(env, 0)[0].bbox == CAT[0].bbox and local(env, 6)[0].bbox == CAT[0].bbox
    assert plan(env)["pull"] == 2


def test_scope_errors(synced):
    with pytest.raises(engine.SyncError, match="does not exist"):
        plan(synced, scope="task", task_id=99)
    with pytest.raises(engine.SyncError, match="job does not exist"):
        plan(synced, scope="job", task_id=1, job=9)
    with pytest.raises(engine.SyncError, match="scope"):
        plan(synced, scope="images")


# -- tasks added and deleted ------------------------------------------------------
def test_task_added_remotely_is_imported(synced):
    env = synced
    new = env.remote.new_task(
        env.rid, "From CVAT", [("c0.png", CAT), ("c1.png", CAT2), ("c2.png", [])], job_size=2
    )
    p = plan(env)
    assert p["tasks_new"] == [{"remote_task": new, "name": "From CVAT", "job_size": 2, "images": 3}]
    assert p["remote_only"]["count"] == 0 and p["tasks_create"] == []
    assert plan(env, scope="task", task_id=1)["tasks_new"] == []
    assert plan(env, scope="job", task_id=1, job=1)["tasks_new"] == []
    result = apply(env, p["plan_id"])
    assert result["tasks_imported"] == 1
    tasks = TaskStore(env.dir).load()
    task = tasks[-1]
    assert len(tasks) == 3 and task.name == "From CVAT" and task.job_size == 2 and len(task.images) == 3
    records = Manifest(env.dir).load()
    assert [records[i].name for i in task.images] == ["c0.png", "c1.png", "c2.png"]
    annotations = load_annotations(env.dir)
    assert annotations.images[task.images[0]][0].bbox == CAT[0].bbox
    assert annotations.images[task.images[1]][0].bbox == CAT2[0].bbox
    assert not annotations.images.get(task.images[2])
    link = links.get_link(env.dir, "cvat")
    assert link["tasks"][str(task.id)]["remote_task"] == new
    assert link["items"][task.images[1]]["key"] == f"{new}:1"
    assert ("frame", new, 0) in env.remote.calls
    again = plan(env)
    assert (again["unchanged"], again["tasks_new"], again["pull"], again["push"]) == (11, [], 0, 0)


def _remote_video_task(env, count=3, **mark):
    new = env.remote.new_task(env.rid, "Clip", [(f"clip_frame_{i:06d}.png", CAT) for i in range(count)])
    env.remote.mark_video(new, **mark)
    return new


def test_remote_video_task_is_imported_with_its_video(synced, monkeypatch):
    env = synced
    made = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: made.append(real(*a, **k)) or made[-1])
    _remote_video_task(env, 3, start=4, step=5, fps=10.0, frames=30)
    p = plan(env)
    result = apply(env, p["plan_id"])
    assert result["tasks_imported"] == 1 and result["warnings"] == []
    task = TaskStore(env.dir).load()[-1]
    videos = VideoStore(env.dir).list()
    assert len(videos) == 1 and videos[0].frames == 30
    assert task.source == {"video": videos[0].id, "start": 4, "stop": 14, "step": 5, "fps": videos[0].fps}
    records = Manifest(env.dir).load()
    got = [(records[i].video, records[i].frame, records[i].time) for i in task.images]
    fps = videos[0].fps
    assert got == [(videos[0].id, f, f / fps) for f in (4, 9, 14)]
    assert VideoStore(env.dir).file(videos[0]).exists()
    assert made and not any(Path(m).exists() for m in made)  # the temporary folder is gone


@pytest.mark.parametrize("how", ["none", "raises", "too_short"])
def test_remote_video_task_falls_back_to_images(synced, how, monkeypatch):
    env = synced
    new = _remote_video_task(env, 3, start=0, step=1, frames=2 if how == "too_short" else None)
    if how == "none":
        env.remote.videos.clear()
    if how == "raises":

        def boom(*a, **k):
            raise ToolError("backup refused")

        monkeypatch.setattr(env.remote, "video_source", boom)
    result = apply(env, plan(env)["plan_id"])
    assert result["tasks_imported"] == 1
    task = TaskStore(env.dir).load()[-1]
    assert task.source is None and len(task.images) == 3
    assert VideoStore(env.dir).list() == []
    assert all(r.video == "" for r in (Manifest(env.dir).load()[i] for i in task.images))
    assert bool(result["warnings"]) == (how != "none") and new


def delete_local_task(env, task_id):
    ids = TaskStore(env.dir).delete(task_id)
    Manifest(env.dir).delete(ids)


def test_deleted_local_task_keeps_remote_task_by_default(synced):
    env = synced
    beta = env.tids[2]
    delete_local_task(env, 2)
    p = plan(env)
    assert p["tasks_delete"] == [{"task": 2, "remote_task": beta, "name": "Beta"}]
    result = apply(env, p["plan_id"])
    assert result["tasks_deleted"] == 0
    assert beta in {t.id for t in env.remote.tasks(SETTINGS, env.rid)}
    assert ("delete_task", beta) not in env.remote.calls
    assert not any(c[0] == "set_frames" and c[1] == beta for c in env.remote.calls)  # left exactly as is
    link = links.get_link(env.dir, "cvat")
    assert link["tasks"]["2"]["kept_remote"] is True  # the decision is remembered
    assert all(env.ids[i] not in link["items"] for i in (5, 6, 7))  # the link counts only what is here
    again = plan(env)  # not offered again, nor imported back, nor counted as only in CVAT
    assert (again["tasks_delete"], again["tasks_new"], again["remove_remote"]) == ([], [], 0)
    assert again["remote_only"]["count"] == 0 and again["unchanged"] == 5


def test_deleted_local_task_is_deleted_remotely_on_request(synced):
    env = synced
    beta = env.tids[2]
    delete_local_task(env, 2)
    p = plan(env)
    assert len(p["tasks_delete"]) == 1
    result = apply(env, p["plan_id"], delete_tasks=True)
    assert result["tasks_deleted"] == 1
    assert ("delete_task", beta) in env.remote.calls
    assert not any(c[0] == "set_frames" and c[1] == beta for c in env.remote.calls)
    assert [t.name for t in env.remote.tasks(SETTINGS, env.rid)] == ["Alpha"]
    again = plan(env)
    assert (again["tasks_delete"], again["tasks_new"], again["unchanged"]) == ([], [], 5)


def test_deleted_frame_of_a_remote_task_keeps_its_position(synced):
    env = synced
    new = env.remote.new_task(
        env.rid, "Gaps", [("g0.png", CAT), ("g1.png", CAT2), ("g2.png", []), ("g3.png", [])], job_size=2
    )
    env.remote.set_frames(SETTINGS, new, [1], [])  # deleted in the tool before the import
    p = plan(env)
    assert [t["remote_task"] for t in p["tasks_new"]] == [new]
    apply(env, p["plan_id"])
    task = TaskStore(env.dir).load()[-1]
    records = Manifest(env.dir).load()
    assert [records[i].name for i in task.images] == [
        "g0.png",
        "g1.png",
        "g2.png",
        "g3.png",
    ]  # same positions
    assert records[task.images[1]].status == "excluded"
    assert records[task.images[1]].excluded_reason == "deleted in cvat"
    assert links.get_link(env.dir, "cvat")["items"][task.images[1]].get("removed") is True
    again = plan(env)
    assert (
        again["tasks_new"],
        again["pull"],
        again["push"],
        again["remove_remote"],
        again["restore_remote"],
    ) == ([], 0, 0, 0, 0)


def test_video_of_a_task_without_its_video_falls_back_to_images(tmp_path):
    plain = ImageTask(1, "Images", "2026-01-01T00:00:00+00:00")
    gone = ImageTask(2, "Clip", "2026-01-01T00:00:00+00:00", source={"video": "0" * 16, "stop": 9, "step": 1})
    assert engine._video_of(tmp_path, plain) is None
    assert engine._video_of(tmp_path, gone) is None


def test_task_deleted_in_the_tool_excludes_its_images_here(synced):
    env = synced
    beta = env.tids[2]
    images = TaskStore(env.dir).get(2).images
    env.remote.delete_task(SETTINGS, beta)
    p = plan(env)
    assert p["remove_local"] == len(images) and p["tasks_create"] == []
    name = TaskStore(env.dir).get(2).name
    assert p["tasks_removed_remote"] == [{"task": 2, "name": name, "images": len(images)}]
    apply(env, p["plan_id"])
    records = Manifest(env.dir).load()
    assert all(records[i].status == "excluded" for i in images)
    again = plan(env)  # not sent again, nothing left to do
    assert (again["tasks_create"], again["remove_local"], again["push"], again["add"]) == ([], 0, 0, 0)


def test_tracked_frames_are_read_again_after_a_push(synced, monkeypatch):
    """Writing a tracked object makes CVAT renumber the track and re-interpolate its neighbours."""

    env = synced
    tracked = [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3], track=7)]
    for index in (0, 1):  # two frames of the same track, in sync on both sides
        env.remote.projects[env.rid]["items"][env.key(index)].labels = copy.deepcopy(tracked)
    apply(env, plan(env)["plan_id"])
    annotations = load_annotations(env.dir)
    annotations.images[env.ids[0]] = [Label("cat", bbox=[0.2, 0.1, 0.3, 0.3], track=7)]  # edited here
    save_annotations(env.dir, AnnotationSet(annotations.task, annotations.classes, annotations.images, {}))
    real_write = env.remote.write

    def cvat_like_write(settings, rid, project, updates):
        real_write(settings, rid, project, updates)
        for index in (0, 1):  # new track id; frame 1 re-interpolated towards the new keyframe
            item = env.remote.projects[rid]["items"][env.key(index)]
            item.labels = [Label("cat", bbox=[0.2 if index == 0 else 0.15, 0.1, 0.3, 0.3], track=8)]

    monkeypatch.setattr(env.remote, "write", cvat_like_write)
    p = plan(env)
    assert p["push"] == 1
    apply(env, p["plan_id"])
    after = load_annotations(env.dir).images
    assert after[env.ids[0]][0].track == 8 and after[env.ids[1]][0].bbox[0] == 0.15
    again = plan(env)
    assert (again["pull"], again["push"], len(again["conflicts"])) == (0, 0, 0)


# -- direction ------------------------------------------------------------------------
def test_pull_direction_makes_the_project_like_the_tool(synced):
    env = synced
    set_local(env, 0, CAT2)  # changed here only
    set_remote(env, 1, CAT3)  # changed there only
    set_local(env, 2, CAT2)  # changed on both sides
    set_remote(env, 2, CAT3)
    Manifest(env.dir).set_status([env.ids[3]], "excluded", "x")  # removed here
    p = plan(env, direction="pull")
    assert p["direction"] == "pull" and p["conflicts"] == [] and p["push"] == 0
    assert (p["pull"], p["restore_local"]) == (3, 1)
    apply(env, p["plan_id"])
    assert local(env, 0) == CAT and local(env, 1) == CAT3 and local(env, 2) == CAT3
    assert Manifest(env.dir).load()[env.ids[3]].status == "active"
    again = plan(env)
    assert (again["pull"], again["push"], again["remove_remote"], len(again["conflicts"])) == (0, 0, 0, 0)


def test_push_direction_makes_the_tool_like_the_project(synced):
    env = synced
    set_remote(env, 0, CAT3)  # changed there only
    set_local(env, 2, CAT2)  # changed on both sides
    set_remote(env, 2, CAT3)
    Manifest(env.dir).set_status([env.ids[3]], "excluded", "x")  # removed here
    p = plan(env, direction="push")
    assert p["conflicts"] == [] and p["pull"] == 0 and (p["push"], p["remove_remote"]) == (2, 1)
    apply(env, p["plan_id"])
    assert env.remote.labels(env.rid, "img0.png") == CAT and env.remote.labels(env.rid, "img2.png") == CAT2
    again = plan(env)
    assert (again["pull"], again["push"], again["remove_local"], len(again["conflicts"])) == (0, 0, 0, 0)


def test_one_way_leaves_what_exists_only_on_the_other_side(env):
    first = plan(env, create=True, scope="task", task_id=1)  # only Alpha is sent; Beta stays here only
    apply(env, first["plan_id"])
    rid = links.get_link(env.dir, "cvat")["remote_id"]
    new = env.remote.new_task(rid, "Only there", [("z0.png", CAT)])
    pulled = plan(env, direction="pull")
    assert pulled["tasks_create"] == [] and [t["name"] for t in pulled["only_here"]["tasks"]] == ["Beta"]
    assert [t["remote_task"] for t in pulled["tasks_new"]] == [new]
    pushed = plan(env, direction="push")
    assert pushed["tasks_new"] == [] and [t["name"] for t in pushed["tasks_create"]] == ["Beta"]


def test_unknown_direction(synced):
    with pytest.raises(engine.SyncError, match="direction"):
        plan(synced, direction="sideways")


def test_video_is_sent_under_its_original_name(tmp_path):
    store = VideoStore(tmp_path)
    store.folder.mkdir(parents=True, exist_ok=True)
    tmp = make_video(store.folder / "upload.mp4", frames=30)
    info = store.add("abbey_road.mp4", ".mp4", tmp, "a" * 16, tmp.stat().st_size)
    task = ImageTask(
        1, "Clip", "2026-01-01T00:00:00", source={"video": info.id, "start": 0, "stop": 29, "step": 3}
    )
    path, selection = engine._video_of(tmp_path, task)
    assert path.name == f"{'a' * 16}.mp4" and selection["name"] == "abbey_road.mp4" and selection["step"] == 3


# -- safety: deleting tasks, empty / open tasks, stale plans, scores, deleted projects ---------------
def test_task_with_a_kept_image_is_not_deleted(synced):
    env = synced
    beta = env.tids[2]
    delete_local_task(env, 2)
    set_remote(env, 5, CAT2)  # edited in the tool while deleted here: a conflict
    p = plan(env)
    assert len(p["conflicts"]) == 1 and len(p["tasks_delete"]) == 1
    result = apply(env, p["plan_id"], default="remote", delete_tasks=True)
    assert result["tasks_deleted"] == 0 and ("delete_task", beta) not in env.remote.calls
    assert beta in {t.id for t in env.remote.tasks(SETTINGS, env.rid)}
    assert any("not deleted" in w for w in result["warnings"])
    assert "2" in links.get_link(env.dir, "cvat")["tasks"]


def test_task_with_frames_added_after_the_plan_is_refused(synced):
    env = synced
    beta = env.tids[2]
    delete_local_task(env, 2)
    p = plan(env)
    env.remote.put(env.rid, "late.png", [], key=f"{beta}:3")  # someone adds an image in the tool
    with pytest.raises(engine.RemoteChanged, match="added after"):
        apply(env, p["plan_id"], delete_tasks=True)
    assert not any(c[0] in ("delete_task", "set_frames") for c in env.remote.calls)
    assert beta in {t.id for t in env.remote.tasks(SETTINGS, env.rid)}


def _set_tasks(env, *tasks):
    write_json(env.dir / "tasks.json", {"next_id": 9, "tasks": [t.as_dict() for t in tasks]})


def test_open_and_empty_tasks_are_not_created(env):
    alpha = ImageTask(1, "Alpha", "2026-01-01T00:00:00", 2, env.ids[:5])
    beta = ImageTask(2, "Beta", "2026-01-01T00:00:01", None, env.ids[5:], open=True)
    empty = ImageTask(3, "Empty", "2026-01-01T00:00:02", None, [])
    _set_tasks(env, alpha, beta, empty)
    p = plan(env, create=True)
    assert [t["task"] for t in p["tasks_create"]] == [1]
    assert {(s["task"], s["reason"]) for s in p["skipped_tasks"]} == {(2, "open"), (3, "no_images")}
    assert p["add"] == 0  # the images of the open task are not sent one by one either
    apply(env, p["plan_id"])
    assert [c[1] for c in env.remote.calls if c[0] == "add_task"] == ["Alpha"]
    assert not any(c[0] == "add" for c in env.remote.calls)


def test_two_plans_cannot_both_be_applied(synced):
    env = synced
    set_local(env, 0, CAT2)
    first, second = plan(env), plan(env)
    apply(env, first["plan_id"])
    env.remote.calls.clear()
    with pytest.raises(engine.RemoteChanged, match="link"):
        apply(env, second["plan_id"])
    assert env.remote.calls == []  # nothing was written


def test_untouched_model_label_keeps_its_score_on_pull(synced):
    env = synced
    kept, moved = box("cat", 0.1, 0.1, 0.2, 0.2), box("dog", 0.5, 0.5, 0.2, 0.2)
    kept.score, kept.model = 0.9, "m1"
    moved.score, moved.model = 0.8, "m1"
    set_local(env, 0, [kept, moved])
    apply(env, plan(env)["plan_id"])  # pushed (the fake keeps scores; CVAT does not)
    set_remote(env, 0, [box("cat", 0.1, 0.1, 0.2, 0.2), box("dog", 0.6, 0.6, 0.2, 0.2)])
    result = apply(env, plan(env)["plan_id"])
    assert result["pulled"] == 1
    cat, dog = sorted(local(env, 0), key=lambda lb: lb.label)
    assert (cat.score, cat.model) == (0.9, "m1")  # untouched: still a pre-annotation
    assert (dog.score, dog.model) == (None, None)  # edited by a person: reviewed


def test_remote_frames_with_the_same_base_name_or_the_same_content_are_all_linked(synced):
    env = synced
    new = env.remote.new_task(
        env.rid, "Clash", [("a/1.png", CAT), ("b/1.png", CAT2), ("c/dup.png", [])], job_size=None
    )
    existing = Manifest(env.dir).load()[env.ids[0]]
    env.remote.frame_bytes[(new, 2)] = (env.dir / existing.file).read_bytes()  # same image as one we have
    result = apply(env, plan(env)["plan_id"])
    assert result["tasks_imported"] == 1
    task = TaskStore(env.dir).load()[-1]
    records = Manifest(env.dir).load()
    assert [records[i].name for i in task.images] == [
        "a/1.png",
        "b/1.png",
    ]  # the duplicate is not stored twice
    annotations = load_annotations(env.dir)
    assert annotations.images[task.images[0]][0].bbox == CAT[0].bbox  # each frame got its own labels
    assert annotations.images[task.images[1]][0].bbox == CAT2[0].bbox
    items = links.get_link(env.dir, "cvat")["items"]
    assert items[task.images[0]]["key"] == f"{new}:0" and items[task.images[1]]["key"] == f"{new}:1"
    assert items[env.ids[0]]["key"] != f"{new}:2"  # that image was already linked to its own item
    again = plan(env)
    assert (
        again["remote_only"]["count"] == 1 and again["tasks_new"] == []
    )  # only the duplicate frame stays apart


# -- task names (#393) ---------------------------------------------------------------
def remote_name(env, local_id):
    return next(t.name for t in env.remote.tasks(SETTINGS, env.rid) if t.id == env.tids[local_id])


def test_a_task_renamed_here_is_renamed_in_the_tool(synced):
    env = synced
    TaskStore(env.dir).rename(1, "Alpha two")
    p = plan(env)
    assert p["tasks_rename"] == [
        {"task": 1, "remote_task": env.tids[1], "old": "Alpha", "new": "Alpha two", "direction": "push"}
    ]
    result = apply(env, p["plan_id"])
    assert result["tasks_renamed"] == 1 and result["warnings"] == []
    assert ("rename_task", env.tids[1], "Alpha two") in env.remote.calls
    assert remote_name(env, 1) == "Alpha two" and remote_name(env, 2) == "Beta"
    assert links.get_link(env.dir, "cvat")["tasks"]["1"]["name"] == "Alpha two"
    assert plan(env)["tasks_rename"] == []  # nothing left to send


def test_a_task_renamed_in_the_tool_is_renamed_here(synced):
    env = synced
    env.remote.task_meta[env.rid][env.tids[2]]["name"] = "Beta in CVAT"
    p = plan(env)
    assert [(r["task"], r["new"], r["direction"]) for r in p["tasks_rename"]] == [(2, "Beta in CVAT", "pull")]
    assert apply(env, p["plan_id"])["tasks_renamed"] == 1
    assert [t.name for t in TaskStore(env.dir).load()] == ["Alpha", "Beta in CVAT"]
    assert links.get_link(env.dir, "cvat")["tasks"]["2"]["name"] == "Beta in CVAT"
    assert not any(c[0] == "rename_task" for c in env.remote.calls)
    assert plan(env)["tasks_rename"] == []


def test_a_task_renamed_on_both_sides_keeps_both_names_and_warns(synced):
    env = synced
    TaskStore(env.dir).rename(1, "Here")
    env.remote.task_meta[env.rid][env.tids[1]]["name"] = "There"
    p = plan(env)
    assert p["tasks_rename"] == [] and p["tasks_rename_conflicts"][0]["local"] == "Here"
    result = apply(env, p["plan_id"])
    assert result["tasks_renamed"] == 0 and len(result["warnings"]) == 1
    assert TaskStore(env.dir).get(1).name == "Here" and remote_name(env, 1) == "There"
    assert plan(env)["tasks_rename_conflicts"]  # still reported until one side is changed


def test_direction_decides_a_task_name_difference(synced):
    env = synced
    TaskStore(env.dir).rename(1, "Here")
    env.remote.task_meta[env.rid][env.tids[2]]["name"] = "There"
    p = plan(env, direction="push")
    assert [(r["task"], r["direction"]) for r in p["tasks_rename"]] == [(1, "push"), (2, "push")]
    p = plan(env, direction="pull")
    assert [(r["task"], r["new"], r["direction"]) for r in p["tasks_rename"]] == [
        (1, "Alpha", "pull"),
        (2, "There", "pull"),
    ]


def test_a_task_rename_that_fails_is_a_warning(synced, monkeypatch):
    env = synced
    TaskStore(env.dir).rename(1, "Alpha two")

    def refuse(settings, task_id, name):
        raise ToolError("CVAT refused the name.")

    monkeypatch.setattr(env.remote, "rename_task", refuse)
    result = apply(env, plan(env)["plan_id"])
    assert result["tasks_renamed"] == 0
    assert "Alpha two" in result["warnings"][0] and "CVAT refused" in result["warnings"][0]
    assert links.get_link(env.dir, "cvat")["tasks"]["1"]["name"] == "Alpha"  # still to send next time
    assert plan(env)["tasks_rename"][0]["new"] == "Alpha two"


# -- project name (#587) ------------------------------------------------------------
def remote_project_name(env):
    return env.remote.projects[env.rid]["name"]


def test_a_project_renamed_here_is_renamed_in_the_tool(synced):
    env = synced
    env.store.update(env.project.id, name="Renamed here")
    p = plan(env)
    assert p["project_rename"] == {
        "old": f"ActiveLabel {env.project.id} - {env.project.name}",
        "new": f"ActiveLabel {env.project.id} - Renamed here",
        "direction": "push",
    }
    result = apply(env, p["plan_id"])
    assert result["project_renamed"] is True and result["warnings"] == []
    assert remote_project_name(env) == f"ActiveLabel {env.project.id} - Renamed here"
    assert links.get_link(env.dir, "cvat")["remote_name"] == remote_project_name(env)
    again = plan(env)
    assert again["project_rename"] is None and again["project_rename_conflict"] is None


def test_a_project_renamed_in_the_tool_is_renamed_here_without_the_prefix(synced):
    env = synced
    env.remote.projects[env.rid]["name"] = f"ActiveLabel {env.project.id} - Street"
    p = plan(env)
    assert (p["project_rename"]["new"], p["project_rename"]["direction"]) == ("Street", "pull")
    result = apply(env, p["plan_id"])
    assert result["project_renamed"] is True
    assert env.store.get(env.project.id).name == "Street"
    assert not any(c[0] == "rename_project" for c in env.remote.calls)
    assert plan(env)["project_rename"] is None


def test_a_project_renamed_on_both_sides_keeps_both_names_and_warns(synced):
    env = synced
    env.store.update(env.project.id, name="Here")
    env.remote.projects[env.rid]["name"] = "There"
    p = plan(env)
    assert p["project_rename"] is None and p["project_rename_conflict"]["remote"] == "There"
    result = apply(env, p["plan_id"])
    assert result["project_renamed"] is False and len(result["warnings"]) == 1
    assert env.store.get(env.project.id).name == "Here" and remote_project_name(env) == "There"
    assert plan(env)["project_rename_conflict"]  # still reported until one side is changed


def test_direction_decides_a_project_name_difference(synced):
    env = synced
    env.store.update(env.project.id, name="Here")
    env.remote.projects[env.rid]["name"] = "There"
    assert plan(env, direction="push")["project_rename"]["direction"] == "push"
    pulled = plan(env, direction="pull")["project_rename"]
    assert (pulled["new"], pulled["direction"]) == ("There", "pull")


def test_a_project_imported_from_the_tool_keeps_the_plain_name_there(synced):
    env = synced
    link = links.get_link(env.dir, "cvat")
    link["remote_name"] = "Street"  # imported: the remote project never had the ActiveLabel prefix
    links.save_link(env.dir, "cvat", link)
    env.remote.projects[env.rid]["name"] = "Street"
    env.store.update(env.project.id, name="Street")
    assert plan(env)["project_rename"] is None
    env.store.update(env.project.id, name="Roads")
    p = plan(env)
    assert (p["project_rename"]["old"], p["project_rename"]["new"]) == ("Street", "Roads")
    apply(env, p["plan_id"])
    assert remote_project_name(env) == "Roads"


def test_a_name_typed_at_the_import_stays_here_and_is_not_offered_to_the_tool(synced):
    """#695: the remote project keeps its own name; the typed one is not a rename made here."""
    env = synced
    links.remove_link(env.dir, "cvat")
    env.remote.projects[env.rid]["name"] = "Street"
    env.store.update(env.project.id, name="meu_dataset_round_1")
    info = env.remote.info(env.rid)
    engine.link_from_items(env.store, env.project.id, "cvat", info, env.remote.items(SETTINGS, env.rid, None))
    link = links.get_link(env.dir, "cvat")
    assert (link["remote_name"], link["local_name"]) == ("Street", "meu_dataset_round_1")
    p = plan(env)
    assert p["project_rename"] is None and p["project_rename_conflict"] is None
    apply(env, p["plan_id"])
    assert remote_project_name(env) == "Street"
    assert links.get_link(env.dir, "cvat")["local_name"] == "meu_dataset_round_1"
    env.store.update(env.project.id, name="Roads")  # a later rename here is offered as usual
    again = plan(env)["project_rename"]
    assert (again["old"], again["new"], again["direction"]) == ("Street", "Roads", "push")
    env.remote.projects[env.rid]["name"] = "Avenue"  # and both sides renamed is a conflict
    assert plan(env)["project_rename_conflict"] == {"local": "Roads", "remote": "Avenue"}


def test_a_project_rename_that_fails_is_a_warning_and_stays_to_send(synced, monkeypatch):
    env = synced
    env.store.update(env.project.id, name="Renamed here")

    def refuse(settings, project_id, name):
        raise ToolError("CVAT refused the name.")

    monkeypatch.setattr(env.remote, "rename_project", refuse)
    result = apply(env, plan(env)["plan_id"])
    assert result["project_renamed"] is False and "CVAT refused" in result["warnings"][0]
    assert plan(env)["project_rename"]["direction"] == "push"


def test_comparing_one_task_does_not_swallow_a_project_rename(synced):
    env = synced
    env.remote.projects[env.rid]["name"] = "Changed in the tool"
    p = plan(env, scope="task", task_id=1)
    assert p["project_rename"] is None
    apply(env, p["plan_id"])
    assert plan(env)["project_rename"]["direction"] == "pull"


def test_a_task_kept_in_the_tool_stays_there_when_the_project_changes_later(synced):
    env = synced
    beta = env.tids[2]
    delete_local_task(env, 2)
    apply(env, plan(env)["plan_id"])
    set_local(env, 0, CAT2)
    p = plan(env)
    assert (p["push"], p["tasks_delete"], p["tasks_new"]) == (1, [], [])
    apply(env, p["plan_id"])
    assert beta in {t.id for t in env.remote.tasks(SETTINGS, env.rid)}
    assert not any(c[0] == "delete_task" for c in env.remote.calls)
