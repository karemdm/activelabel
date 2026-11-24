from ann_helpers import make_manifest

from activelabel.annotations import folder_classes, parse
from activelabel.labels import task_of
from activelabel.tasks import Task


def test_folder_tree(tmp_path):
    manifest = make_manifest(tmp_path, ["cat/1.png", "cat/2.png", "dog/3.png", "4.png"])
    assert folder_classes([r.name for r in manifest.all()]) is None  # 1 of 4 images outside a folder
    manifest = make_manifest(tmp_path / "x", ["cat/1.png", "cat/2.png", "dog/3.png"])
    assert folder_classes([r.name for r in manifest.all()]) == {
        "cat/1.png": "cat",
        "cat/2.png": "cat",
        "dog/3.png": "dog",
    }
    assert folder_classes(["a/1.png", "a/2.png"]) is None
    res = parse([], "folders", manifest.all())
    assert res.classes == ["cat", "dog"] and task_of(res.all_labels()) is Task.CLASSIFICATION
