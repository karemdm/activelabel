import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.tasks_store import ImageTask, TaskClosed, TaskNotFound, TaskStore, check_job_size, jobs


def _task(n: int, job_size: int | None) -> ImageTask:
    return ImageTask(1, "T", "now", job_size, [f"i{k}" for k in range(n)])


def test_jobs_math():
    hundred = jobs(_task(100, 10))
    assert [(j.n, j.first, j.last) for j in hundred] == [(k, k * 10 - 9, k * 10) for k in range(1, 11)]
    assert [len(j.image_ids) for j in jobs(_task(25, 10))] == [10, 10, 5]
    last = jobs(_task(25, 10))[-1]
    assert (last.first, last.last, last.image_ids[0]) == (21, 25, "i20")
    only = jobs(_task(7, None))
    assert len(only) == 1 and (only[0].first, only[0].last) == (1, 7)
    assert jobs(_task(0, 10)) == []


def test_job_size_validation():
    assert check_job_size(None) is None and check_job_size(5) == 5
    for bad in (0, -1, 2.5, "3", True):
        with pytest.raises(ValueError):
            check_job_size(bad)


def test_create_append_close_rename_delete(tmp_path):
    store = TaskStore(tmp_path)
    assert store.load() == []
    first = store.create("  A ", 10, ["a", "b"], open=True)
    assert (first.id, first.name, first.open) == (1, "A", True)
    store.append(1, ["c", "a"])  # a is already there
    assert store.get(1).images == ["a", "b", "c"]
    second = store.create("B")  # creating a task closes the open one
    assert second.id == 2 and store.get(1).open is False
    with pytest.raises(TaskClosed):
        store.append(1, ["d"])
    assert store.rename(2, "Field").name == "Field"
    with pytest.raises(ValueError):
        store.rename(2, " ")
    with pytest.raises(ValueError):
        store.create("X", 0)
    assert store.delete(1) == ["a", "b", "c"]
    with pytest.raises(TaskNotFound):
        store.get(1)
    assert store.create("C").id == 3  # ids are never reused
    assert TaskStore(tmp_path).next_id() == 4  # persisted


def test_remove_images_drops_empty_tasks(tmp_path):
    store = TaskStore(tmp_path)
    store.create("A", None, ["a", "b"])
    store.create("B", None, ["c"])
    task_of_c = store.task_of("c")
    assert task_of_c is not None and task_of_c.id == 2 and store.task_of("zzz") is None
    assert store.remove_images(["a", "c", "nope"]) == 2
    assert [(t.id, t.images) for t in store.load()] == [(1, ["b"])]


def test_migration_creates_task_one(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.add([ImageRecord(f"id{k}", f"{k}.jpg", f"images/id{k}.jpg", 8, 8, 1, "t") for k in (3, 1, 2)])
    store = TaskStore(tmp_path)
    tasks = store.load()
    assert len(tasks) == 1
    assert (tasks[0].id, tasks[0].name, tasks[0].job_size) == (1, "Task 1", None)
    assert tasks[0].images == ["id3", "id1", "id2"]
    assert (tmp_path / "tasks.json").exists() and store.next_id() == 2


def test_no_images_no_file(tmp_path):
    assert TaskStore(tmp_path).load() == []
    assert not (tmp_path / "tasks.json").exists()


def test_concurrent_creates_get_distinct_ids(tmp_path):
    store = TaskStore(tmp_path)
    threads = [threading.Thread(target=lambda: TaskStore(tmp_path).create("T")) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(t.id for t in store.load()) == list(range(1, 9))


def test_orphans_are_healed_into_one_closed_task(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.add([ImageRecord("a", "a.jpg", "images/a.jpg", 8, 8, 1, "t")])
    store = TaskStore(tmp_path)
    assert [t.id for t in store.load()] == [1]
    manifest.add([ImageRecord(k, f"{k}.jpg", f"images/{k}.jpg", 8, 8, 1, "t") for k in ("c", "b")])
    tasks = store.load()
    assert [(t.id, t.name, t.images, t.open) for t in tasks] == [
        (1, "Task 1", ["a"], False),
        (2, "Task 2", ["c", "b"], False),
    ]
    assert len(store.load()) == 2  # saved, not repeated


def test_the_automatic_name_comes_from_the_tasks_that_exist(tmp_path):
    store = TaskStore(tmp_path)
    assert store.create().name == "Task 1"
    gone = store.create(open=True)  # an upload that added nothing is removed again
    assert gone.name == "Task 2"
    store.delete(gone.id)
    assert store.create().name == "Task 2"  # the number did not advance, the id did
    assert store.get(3).id == 3
    store.create("Field")
    store.create("task 9")  # a name of the same shape counts, whatever its case
    assert store.create().name == "Task 10"


def _age(tmp_path, task_id, seconds):
    """Make a task look as if its last upload chunk came ``seconds`` ago."""
    data = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")
    for task in data["tasks"]:
        if task["id"] == task_id:
            task["touched"] = old
    (tmp_path / "tasks.json").write_text(json.dumps(data), encoding="utf-8")


def test_an_upload_left_open_is_closed_and_marked_incomplete(tmp_path):
    store = TaskStore(tmp_path)
    store.create("A", images=["a"], open=True)
    store.create("B", images=["b"], open=True)  # starting B closed A: that upload was cut off
    assert [(t.open, t.incomplete) for t in store.load()] == [(False, True), (True, False)]
    assert store.close_stale(3600) == []  # B's last chunk came just now: its upload may still be running
    _age(tmp_path, 2, 7200)
    assert [t.id for t in store.close_stale(3600)] == [2]
    assert [(t.open, t.incomplete) for t in store.load()] == [(False, True), (False, True)]
    assert store.close_stale(0) == []  # nothing is open any more


def test_appending_keeps_an_upload_alive_and_closing_it_is_complete(tmp_path):
    store = TaskStore(tmp_path)
    store.create("A", images=["a"], open=True)
    _age(tmp_path, 1, 7200)
    store.append(1, ["b"])  # a new chunk arrived
    assert store.close_stale(3600) == []
    assert store.close(1).incomplete is False
    assert store.close_stale(0) == []


def test_a_task_saved_before_the_marks_existed_still_loads(tmp_path):
    (tmp_path / "tasks.json").write_text(
        json.dumps(
            {
                "next_id": 2,
                "tasks": [
                    {"id": 1, "name": "A", "created": "2026-01-01T00:00:00+00:00", "job_size": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    (task,) = TaskStore(tmp_path).load()
    assert (task.open, task.incomplete) == (False, False)
