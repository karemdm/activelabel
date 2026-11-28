from ann_helpers import make_manifest

from activelabel.annotations import folder_classes, parse, split_folders
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


def test_split_folders_are_not_classes():
    names = ["train/cat/1.png", "train/dog/2.png", "val/cat/3.png", "dog/test/4.png", "Cat/Valid/5.png"]
    assert folder_classes(names) == {
        "train/cat/1.png": "cat",
        "train/dog/2.png": "dog",
        "val/cat/3.png": "cat",
        "dog/test/4.png": "dog",
        "Cat/Valid/5.png": "Cat",
    }
    assert split_folders(names) == ["test", "train", "val", "Valid"]
    assert split_folders(["cat/1.png", "dog/2.png"]) == []


def test_only_split_folders_give_no_classes():
    assert folder_classes(["train/1.png", "val/2.png", "test/3.png"]) is None  # no class folder at all
    assert folder_classes(["data/train/1.png", "data/val/2.png"]) is None  # one class ("data") only


def test_accents_spaces_and_a_wrapping_folder():
    names = ["ds/Gato preto/1.png", "ds/cão/2.png", "ds/cão/3.png"]
    assert folder_classes(names) == {names[0]: "Gato preto", names[1]: "cão", names[2]: "cão"}
