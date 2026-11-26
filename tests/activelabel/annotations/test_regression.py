import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, parse
from activelabel.annotations.regression import image_key, is_regression_json
from activelabel.labels import task_of
from activelabel.tasks import Task


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


def values(res, recs, name):
    return {lb.label: lb.value for lb in res.labels[recs[name].id]}


def test_csv_with_commas_and_a_named_image_column(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "age,File,height\n34.5,a.png,1.7\n20,b.png,\n")
    assert detect_format([path]) == "regression"
    res = parse([path], "regression", images)
    assert res.format == "regression" and res.files == ["v.csv"]
    assert res.classes == ["age", "height"]
    assert values(res, recs, "a.png") == {"age": 34.5, "height": 1.7}
    assert values(res, recs, "b.png") == {"age": 20.0}  # the empty cell is no value
    assert task_of(res.all_labels()) is Task.REGRESSION


def test_csv_semicolon_and_decimal_comma(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "name;grade;age\na.png;7,5;30\nb.png;8;31,25\n")
    res = parse([path], "regression", images)
    assert res.classes == ["grade", "age"]
    assert values(res, recs, "a.png") == {"grade": 7.5, "age": 30.0}
    assert values(res, recs, "b.png")["age"] == 31.25


def test_csv_quoted_decimal_comma_with_comma_delimiter(env):
    folder, images, recs = env
    path = write(folder, "v.csv", 'image,score\na.png,"3,5"\n')
    res = parse([path], "regression", images)
    assert values(res, recs, "a.png") == {"score": 3.5}


def test_csv_first_column_is_the_image_when_no_name_matches(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "picture,age\nsub/c.png,9\n")
    res = parse([path], "regression", images)
    assert values(res, recs, "sub/c.png") == {"age": 9.0}


def test_bad_values_are_reported_and_skipped(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "image,age,h\na.png,abc,1\nb.png,nan,inf\nsub/c.png,1_0,2\n")
    res = parse([path], "regression", images)
    problem = next(p for p in res.problems if p.code == "value_invalid")
    assert problem.level == "warning" and problem.count == 4 and len(problem.examples) == 4
    assert values(res, recs, "a.png") == {"h": 1.0}
    assert values(res, recs, "b.png") == {}


def test_unmatched_images_and_bad_rows(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "image,age\nzzz.png,3\na.png,5\n,7\nb.png,1,2,3\n")
    res = parse([path], "regression", images)
    assert res.unmatched_count == 1 and res.unmatched == ["zzz.png"]
    assert {p.code for p in res.problems} == {"unmatched_entries", "invalid_annotation"}
    assert values(res, recs, "a.png") == {"age": 5.0}


def test_a_repeated_image_row_is_a_duplicate_entry(env):
    folder, images, recs = env
    path = write(folder, "v.csv", "image,age\na.png,5\na.png,6\n")
    res = parse([path], "regression", images)
    assert values(res, recs, "a.png") == {"age": 5.0}
    assert any(p.code == "duplicate_image_entry" for p in res.problems)


def test_a_csv_without_targets_or_empty_is_invalid(env):
    folder, images, _ = env
    for name, text in (("a.csv", "image\na.png\n"), ("b.csv", "")):
        res = parse([write(folder, name, text)], "regression", images)
        assert [p.code for p in res.problems] == ["invalid_file"] and res.files == []


def test_json_dict_and_list(env):
    folder, images, recs = env
    one = write(folder, "d.json", {"a.png": {"age": 3, "h": None}, "b.png": {"age": "4,5", "h": True}})
    assert detect_format([one]) == "regression"
    res = parse([one], "regression", images)
    assert res.classes == ["age", "h"]
    assert values(res, recs, "a.png") == {"age": 3.0}
    assert values(res, recs, "b.png") == {"age": 4.5}
    assert [p.code for p in res.problems] == ["value_invalid"]
    two = write(folder, "l.json", [{"Image": "a.png", "age": 1}, {"Image": "b.png", "age": 2.5}])
    assert detect_format([two]) == "regression"
    res = parse([two], "regression", images)
    assert values(res, recs, "b.png") == {"age": 2.5}


def test_json_that_is_not_values_is_invalid(env):
    folder, images, _ = env
    path = write(folder, "x.json", {"a.png": 5})
    assert detect_format([path]) is None
    res = parse([path], "regression", images)
    assert [p.code for p in res.problems] == ["invalid_file"]


def test_coco_json_is_still_coco(env):
    folder, _, _ = env
    coco = {"images": [{"id": 1, "file_name": "a.png"}], "annotations": [], "categories": []}
    assert detect_format([write(folder, "c.json", coco)]) == "coco"


def test_the_shapes(env):
    assert image_key(["x", " Path "]) == " Path "
    assert image_key(["x"]) is None
    assert not is_regression_json({})
    assert not is_regression_json([{"image": "a"}])
    assert not is_regression_json([{"age": 1, "h": 2}])
    assert not is_regression_json({"images": {"a": 1}})
    assert not is_regression_json({"a": {"x": [1]}})
    assert not is_regression_json(3)
    assert is_regression_json({"a.png": {}})  # an image without a value still names its image


def test_other_files_are_left_alone(env):
    folder, images, _ = env
    res = parse([write(folder, "n.txt", "x")], "regression", images)
    assert res.files == [] and res.problems == []


def test_a_repeated_csv_column_gives_two_values_of_one_target(env):
    folder, images, recs = env
    res = parse([write(folder, "v.csv", "image,age,age\na.png,1,2\n")], "regression", images)
    assert [lb.value for lb in res.labels[recs["a.png"].id]] == [1.0, 2.0] and res.classes == ["age"]
