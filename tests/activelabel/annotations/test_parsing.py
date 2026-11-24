import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, parse


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


def test_label_studio_json_is_no_longer_recognised(env):
    folder, images, _ = env
    tasks = [{"id": 1, "data": {"image": "/data/upload/1/1a2b3c4d-a.png"}, "annotations": [{"result": []}]}]
    path = write(folder, "export.json", tasks)
    assert detect_format([path]) is None
    with pytest.raises(ValueError, match="Unknown annotation format"):
        parse([path], "labelstudio", images)


def test_matching_case_subfolders_and_ambiguity(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["x/a.png", "y/a.png", "Photo.PNG"])
    recs = {r.name: r for r in manifest.all()}
    data = {
        "images": [
            {"id": 1, "file_name": "x/a.png"},  # exact relative name
            {"id": 2, "file_name": "a.png"},  # ambiguous basename
            {"id": 3, "file_name": "photo.png"},  # case-insensitive
            {"id": 4, "file_name": "x/a.png"},  # duplicate entry
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10]} for i in (1, 2, 3, 4)],
    }
    res = parse([write(tmp_path / "f", "c.json", data)], "coco", manifest.all())
    assert set(res.labels) == {recs["x/a.png"].id, recs["Photo.PNG"].id}
    codes = {p.code: p for p in res.problems}
    assert codes["ambiguous_image_name"].examples == ["a.png"]
    assert "duplicate_image_entry" in codes


def test_detect_none_and_mixed(env):
    folder, _, _ = env
    readme = write(folder, "readme.txt", "hello there")
    coco = write(folder, "c.json", {"images": [], "annotations": []})
    voc = write(folder, "v.xml", "<annotation/>")
    assert detect_format([readme]) is None
    assert detect_format([coco, voc]) is None
    assert detect_format([coco, readme]) == "coco"
    assert detect_format([write(folder, "x.json", {"a": 1})]) is None


def test_an_annotation_named_after_a_repeated_upload_matches_the_first_image(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png"])
    manifest.add_aliases({manifest.all()[0].id: ["copy.png", "a.png"]})
    data = {
        "images": [{"id": 1, "file_name": "copy.png"}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    res = parse([write(tmp_path / "f", "c.json", data)], "coco", manifest.all())
    assert set(res.labels) == {manifest.all()[0].id}
    assert manifest.all()[0].aliases == ["copy.png"]  # a name the image already has is not an alias
