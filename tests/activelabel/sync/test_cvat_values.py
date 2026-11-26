import pytest

from activelabel.labels import Label
from activelabel.sync import cvat_values as cv


def test_value_spec_is_a_mutable_text_attribute() -> None:
    spec = cv.value_spec()
    assert spec["name"] == cv.VALUE == "value"
    assert spec["mutable"] is True and spec["input_type"] == "text"


def test_value_specs_match_the_name_ignoring_case() -> None:
    rows = [
        {"id": 1, "attributes": [{"id": 10, "name": "Value"}]},
        {"id": 2, "attributes": [{"id": 20, "name": "VALUE"}]},
        {"id": 3, "attributes": [{"id": 30, "name": "color"}]},
    ]
    assert cv.value_specs(rows) == {1: 10, 2: 20}


def test_value_specs_keep_the_first_spec_of_a_label() -> None:
    rows = [{"id": 1, "attributes": [{"id": 10, "name": "value"}, {"id": 11, "name": "Value"}]}]
    assert cv.value_specs(rows) == {1: 10}


def test_value_specs_ignore_rows_without_attributes_or_id() -> None:
    rows = [
        {"id": 1},
        {"id": 2, "attributes": None},
        {"id": 3, "attributes": [{"name": "value"}]},
        {"id": 4, "attributes": [{"id": 40, "name": "value"}]},
    ]
    assert cv.value_specs(rows) == {4: 40}


@pytest.mark.parametrize(
    ("value", "text"),
    [(31.0, "31"), (2.75, "2.75"), (-4.0, "-4"), (0.0, "0"), (1e20, repr(1e20))],
)
def test_format_value(value: float, text: str) -> None:
    assert cv.format_value(value) == text


@pytest.mark.parametrize(
    ("text", "number"),
    [("2,75", 2.75), (" 3 ", 3.0), ("2.5", 2.5), (5, 5.0), (-1.5, -1.5)],
)
def test_parse_value_reads_numbers(text: object, number: float) -> None:
    assert cv.parse_value(text) == number


@pytest.mark.parametrize("text", ["", None, "abc", "nan", "inf", "-inf", "   "])
def test_parse_value_returns_none_for_what_is_not_a_finite_number(text: object) -> None:
    assert cv.parse_value(text) is None


def test_tag_value_reads_the_value_attribute() -> None:
    tag = {"attributes": [{"spec_id": 5, "value": "x"}, {"spec_id": 7, "value": "2,5"}]}
    assert cv.tag_value(tag, 7) == 2.5


def test_tag_value_is_none_without_spec_or_attribute() -> None:
    tag = {"attributes": [{"spec_id": 7, "value": "2"}]}
    assert cv.tag_value(tag, None) is None
    assert cv.tag_value(tag, 8) is None
    assert cv.tag_value({}, 7) is None
    assert cv.tag_value({"attributes": [{"spec_id": 7, "value": ""}]}, 7) is None


def test_value_attributes_for_a_label_with_value_and_spec() -> None:
    assert cv.value_attributes(Label("age", value=31.0), 9) == [{"spec_id": 9, "value": "31"}]


def test_value_attributes_empty_without_value_or_spec() -> None:
    assert cv.value_attributes(Label("age"), 9) == []
    assert cv.value_attributes(Label("age", value=3.0), None) == []


def test_with_value_replaces_the_value_and_keeps_the_rest() -> None:
    tag = {"id": 4, "label_id": 2, "attributes": [{"spec_id": 1, "value": "a"}, {"spec_id": 9, "value": "1"}]}
    out = cv.with_value(tag, 2.75, 9)
    assert out["id"] == 4 and out["label_id"] == 2
    assert {a["spec_id"]: a["value"] for a in out["attributes"]} == {1: "a", 9: "2.75"}
    assert len(out["attributes"]) == 2
    assert tag["attributes"][1]["value"] == "1"  # the input is not modified


def test_with_value_adds_the_attribute_to_a_tag_without_attributes() -> None:
    assert cv.with_value({"id": 1}, 3.0, 9)["attributes"] == [{"spec_id": 9, "value": "3"}]
