import pytest

from activelabel.projects import ProjectNotFound, ProjectStore, clean_classes
from activelabel.tasks import Task


def test_ids_are_sequential_and_never_reused(tmp_path):
    store = ProjectStore(tmp_path)
    first = store.create("Cats", "first", "detection", ["cat"])
    second = store.create("Dogs", "", Task.CLASSIFICATION)
    assert (first.id, second.id) == (1, 2)
    store.delete(second.id)
    assert store.create("Birds", "", "segmentation").id == 3
    assert [p.id for p in store.list()] == [3, 1]


def test_name_is_required_and_unique(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("Cats", "", "detection")
    with pytest.raises(ValueError, match="already exists"):
        store.create(" cats ", "", "detection")
    with pytest.raises(ValueError, match="name"):
        store.create("  ", "", "detection")
    with pytest.raises(ValueError):
        store.create("Other", "", "not-a-task")


def test_update_and_delete(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("Cats", "", "detection", ["cat"])
    changed = store.update(project.id, task="segmentation", classes=["cat", "dog", "Cat"], description="x")
    assert changed.task is Task.SEGMENTATION
    assert changed.classes == ["cat", "dog"]
    assert store.get(project.id).description == "x"
    with pytest.raises(ValueError, match="cannot be changed"):
        store.update(project.id, id=9)
    store.delete(project.id)
    with pytest.raises(ProjectNotFound):
        store.get(project.id)


def test_clean_classes():
    assert clean_classes([" a ", "", "b", "A"]) == ["a", "b"]
