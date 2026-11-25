import os
import re

import pytest

from activelabel.backups import (
    KEEP_BACKUPS,
    backup_path,
    list_backups,
    make_backup,
    prune_backups,
    restore_backup,
)
from activelabel.errors import CodedError
from activelabel.labels import AnnotationSet, Label, annotations_path, load_annotations, save_annotations
from activelabel.projects import ProjectStore
from activelabel.tasks import Task


@pytest.fixture
def project(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    created = store.create("P", "", "detection", ["cat"])
    folder = store.dir(created.id)
    labels = {"a": [Label("cat", bbox=[0, 0, 0.5, 0.5])], "b": []}
    save_annotations(folder, AnnotationSet(Task.DETECTION, ["cat"], labels, {"format": "x"}))
    return store, created.id, folder


def test_a_copy_is_named_with_utc_and_reason_and_two_in_a_second_do_not_collide(project):
    _, _, folder = project
    first, second = make_backup(folder, "classes"), make_backup(folder, "classes")
    assert re.fullmatch(r"annotations\.\d{8}T\d{6}Z\.classes\.bak\.json", first)
    assert first != second and (folder / first).exists() and (folder / second).exists()
    assert make_backup(folder / "nowhere") == ""
    with pytest.raises(ValueError, match="reason"):
        make_backup(folder, "because")


def test_list_backups_newest_first_with_counts_reason_and_legacy_names(project):
    _, _, folder = project
    legacy = folder / "annotations.20200101T000000.bak.json"
    legacy.write_bytes(annotations_path(folder).read_bytes())
    os.utime(legacy, (1_600_000_000, 1_600_000_000))
    (folder / "annotations.20990101T000000Z.sync.bak.json").write_bytes(annotations_path(folder).read_bytes())
    (folder / "annotations.20990202T000000Z.bak.json").write_text("{broken", encoding="utf-8")
    (folder / "annotations.notes.bak.json").write_text("{}", encoding="utf-8")  # not a copy of ours
    items = list_backups(folder)
    assert [i["name"] for i in items] == [
        "annotations.20990202T000000Z.bak.json",
        "annotations.20990101T000000Z.sync.bak.json",
        "annotations.20200101T000000.bak.json",
    ]
    assert items[0]["valid"] is False and items[0]["images"] is None
    assert items[1] | {"created": ""} == {
        "name": "annotations.20990101T000000Z.sync.bak.json",
        "created": "",
        "images": 1,
        "labels": 1,
        "reason": "sync",
        "valid": True,
    }
    assert items[1]["created"] == "2099-01-01T00:00:00+00:00"
    assert items[2]["created"] == "2020-09-13T12:26:40+00:00" and items[2]["reason"] == ""


@pytest.mark.parametrize(
    "name",
    [
        "../annotations.json",
        "annotations.json",
        "x/annotations.20990101T000000Z.bak.json",
        "..\a.bak.json",
        "annotations.20990101T000000Z.bak.json\n",
    ],
)
def test_backup_path_rejects_anything_but_a_copy_name(project, name):
    _, _, folder = project
    with pytest.raises(CodedError) as caught:
        backup_path(folder, name)
    assert caught.value.code == "backup_name_invalid"


def test_backup_path_of_a_missing_copy(project):
    _, _, folder = project
    with pytest.raises(FileNotFoundError):
        backup_path(folder, "annotations.20990101T000000Z.bak.json")


def test_restore_brings_the_old_labels_back_after_a_copy_of_the_current_ones(project):
    store, project_id, folder = project
    old = make_backup(folder, "classes")
    save_annotations(folder, AnnotationSet(Task.DETECTION, ["cat"], {"a": []}, {}))
    store.update(project_id, classes=[])  # the class was removed meanwhile
    summary = restore_backup(store, project_id, old)
    assert summary["restored"] == old and summary["images"] == 1 and summary["labels"] == 1
    assert (folder / summary["backup"]).exists() and summary["backup"].endswith(".restore.bak.json")
    restored = load_annotations(folder)
    assert restored is not None and restored.images["a"][0].label == "cat"
    assert store.get(project_id).classes == ["cat"]  # a class the labels use is back in the project
    # the file that was replaced can itself be restored
    again = restore_backup(store, project_id, summary["backup"])
    assert again["labels"] == 0


def test_restore_follows_the_task_of_the_copy(project):
    store, project_id, folder = project
    name = make_backup(folder, "task")
    store.update(project_id, task="classification")
    restore_backup(store, project_id, name)
    assert store.get(project_id).task is Task.DETECTION


def test_restore_refuses_a_copy_that_cannot_be_read(project):
    store, project_id, folder = project
    (folder / "annotations.20990101T000000Z.bak.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(CodedError) as caught:
        restore_backup(store, project_id, "annotations.20990101T000000Z.bak.json")
    assert caught.value.code == "backup_unreadable"
    assert load_annotations(folder) is not None  # untouched


def test_only_the_newest_copies_are_kept(project):
    _, _, folder = project
    old = folder / "annotations.20200101T000000Z.sync.bak.json"
    old.write_bytes(annotations_path(folder).read_bytes())
    names = [make_backup(folder, "classes") for _ in range(KEEP_BACKUPS + 2)]
    left = [item["name"] for item in list_backups(folder)]
    assert len(left) == KEEP_BACKUPS and not old.exists()
    assert left == names[::-1][:KEEP_BACKUPS]  # copies of the same second keep their order
    assert prune_backups(folder, keep=1) == left[1:] and [b["name"] for b in list_backups(folder)] == left[:1]
