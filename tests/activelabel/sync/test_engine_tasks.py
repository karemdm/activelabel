"""Synchronisation with a CVAT-like remote that has tasks, jobs and deleted frames (docs/TASKS.md)."""

import pytest
from sync_fakes import SETTINGS, FakeRemote, box, make_project, progress

from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, load_annotations, save_annotations
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
    e.remote = FakeRemote("cvat", tasks=True)
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
    result = apply(env, p["plan_id"])
    assert result["tasks_created"] == 2
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
    apply(env, p["plan_id"])
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
    assert "2" in links.get_link(env.dir, "cvat")["tasks"]
    again = plan(env)  # still proposed until the person agrees; its images are not counted twice
    assert len(again["tasks_delete"]) == 1 and again["tasks_new"] == [] and again["remove_remote"] == 0


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
    from activelabel.tasks_store import ImageTask

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
    apply(env, p["plan_id"])
    records = Manifest(env.dir).load()
    assert all(records[i].status == "excluded" for i in images)
    again = plan(env)  # not sent again, nothing left to do
    assert (again["tasks_create"], again["remove_local"], again["push"], again["add"]) == ([], 0, 0, 0)


def test_tracked_frames_are_read_again_after_a_push(synced, monkeypatch):
    """Writing a tracked object makes CVAT renumber the track and re-interpolate its neighbours."""
    import copy

    from activelabel.labels import AnnotationSet, Label, save_annotations

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
