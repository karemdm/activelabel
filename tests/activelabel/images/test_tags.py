import json

import pytest

from activelabel.images.tags import TAGS_FILE, forget_images, image_tags, load_tags, set_tags
from activelabel.storage import DamagedFile


def test_image_tags_cleans_dedupes_and_drops_system_tags():
    assert image_tags([" check ", "", "  ", "blurry", "check", "excluded", "exclude"]) == ["check", "blurry"]
    assert image_tags(("a", "b")) == ["a", "b"]
    assert image_tags([]) == []


def test_load_tags_without_a_file_is_empty(tmp_path):
    assert load_tags(tmp_path) == {}


@pytest.mark.parametrize(
    "content",
    [[], {"images": []}, {"other": 1}, {"images": {"a": "check", "b": [], "c": 3}}],
)
def test_load_tags_of_a_file_with_another_shape_is_empty(tmp_path, content):
    (tmp_path / TAGS_FILE).write_text(json.dumps(content), encoding="utf-8")
    assert load_tags(tmp_path) == {}


def test_load_tags_keeps_only_the_usable_entries(tmp_path):
    (tmp_path / TAGS_FILE).write_text(
        json.dumps({"images": {"a": ["x", 2], "b": [], "c": "no"}}), encoding="utf-8"
    )
    assert load_tags(tmp_path) == {"a": ["x", "2"]}


def test_a_damaged_file_is_reported(tmp_path):
    (tmp_path / TAGS_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(DamagedFile):
        load_tags(tmp_path)


def test_set_tags_stores_cleaned_tags_and_counts_the_images_changed(tmp_path):
    assert set_tags(tmp_path, {"a": ["check", "check", "excluded"], "b": ["blurry"]}) == 2
    assert load_tags(tmp_path) == {"a": ["check"], "b": ["blurry"]}
    assert json.loads((tmp_path / TAGS_FILE).read_text(encoding="utf-8"))["images"] == {
        "a": ["check"],
        "b": ["blurry"],
    }


def test_set_tags_leaves_the_other_images_alone(tmp_path):
    set_tags(tmp_path, {"a": ["x"], "b": ["y"]})
    assert set_tags(tmp_path, {"a": ["z"]}) == 1
    assert load_tags(tmp_path) == {"a": ["z"], "b": ["y"]}


def test_an_empty_list_removes_the_tags(tmp_path):
    set_tags(tmp_path, {"a": ["x"], "b": ["y"]})
    assert set_tags(tmp_path, {"a": []}) == 1
    assert load_tags(tmp_path) == {"b": ["y"]}
    assert set_tags(tmp_path, {"b": ["excluded"]}) == 1  # only system tags: nothing left to keep
    assert load_tags(tmp_path) == {}


def test_unchanged_tags_return_zero_and_write_nothing(tmp_path):
    assert set_tags(tmp_path, {"a": []}) == 0
    assert not (tmp_path / TAGS_FILE).exists()
    set_tags(tmp_path, {"a": ["x"]})
    path = tmp_path / TAGS_FILE
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # a mark to see a rewrite
    before = path.read_bytes()
    assert set_tags(tmp_path, {"a": ["x", "excluded"], "zz": []}) == 0
    assert path.read_bytes() == before


def test_forget_images_removes_only_those(tmp_path):
    set_tags(tmp_path, {"a": ["x"], "b": ["y"], "c": ["z"]})
    forget_images(tmp_path, ["a", "c", "never-had-tags"])
    assert load_tags(tmp_path) == {"b": ["y"]}
    forget_images(tmp_path, {"b"})
    assert load_tags(tmp_path) == {}


def test_forget_images_without_a_file_writes_nothing(tmp_path):
    forget_images(tmp_path, ["a"])
    assert not (tmp_path / TAGS_FILE).exists()
