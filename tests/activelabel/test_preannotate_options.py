"""Settings of a pre-annotation: the parameters of each kind of model and the class filter (#86)."""

from __future__ import annotations

import math

import pytest

from activelabel import preannotate_options as po
from activelabel.errors import CodedError


def names(kind: str) -> list[str]:
    return [o["name"] for o in po.schema(kind)]


def test_each_kind_lists_only_the_parameters_that_apply_to_it() -> None:
    assert names("classification") == ["confidence", "margin"]
    assert names("detection") == [
        "confidence",
        "nms_iou",
        "max_detections",
        "class_agnostic_nms",
        "min_size",
    ]
    assert names("keypoints") == names("detection")
    assert names("instance") == [
        "confidence",
        "mask_threshold",
        "nms_iou",
        "max_detections",
        "class_agnostic_nms",
        "min_size",
        "min_region",
        "simplify",
    ]
    assert names("semantic") == ["confidence", "min_region", "simplify"]
    assert names("regression") == ["confidence"]  # a future kind gets the common parameters


def test_the_schema_describes_every_parameter() -> None:
    for kind in po.ALL_KINDS:
        for option in po.schema(kind):
            assert option["level"] in ("basic", "advanced")
            assert option["type"] in ("float", "int", "bool")
            if option["type"] != "bool":
                assert option["min"] <= option["default"] <= option["max"] and option["step"] > 0


def test_the_default_confidence_depends_on_the_kind() -> None:
    assert po.resolve("classification", None)["confidence"] == 0.5
    assert po.resolve("detection", {})["confidence"] == 0.25


def test_resolve_fills_the_defaults_and_keeps_the_values_sent() -> None:
    got = po.resolve("detection", {"nms_iou": 0.7, "max_detections": 20.0, "class_agnostic_nms": True})
    assert got == {
        "confidence": 0.25,
        "nms_iou": 0.7,
        "max_detections": 20,
        "class_agnostic_nms": True,
        "min_size": 0.0,
    }
    assert isinstance(got["max_detections"], int)


@pytest.mark.parametrize(
    ("kind", "given", "code"),
    [
        ("classification", {"nms_iou": 0.5}, "option_unknown"),
        ("detection", {"confidence": 0.0}, "option_range"),
        ("detection", {"confidence": 1.0}, "option_range"),
        ("detection", {"max_detections": 0}, "option_range"),
        ("detection", {"max_detections": 2.5}, "option_not_whole"),
        ("detection", {"nms_iou": "high"}, "option_not_number"),
        ("detection", {"nms_iou": True}, "option_not_number"),
        ("detection", {"nms_iou": math.nan}, "option_not_number"),
        ("detection", {"class_agnostic_nms": 1}, "option_not_bool"),
        ("instance", {"mask_threshold": 0.95}, "option_range"),
    ],
)
def test_resolve_refuses_wrong_values_with_a_code(kind: str, given: dict, code: str) -> None:
    with pytest.raises(CodedError) as error:
        po.resolve(kind, given)
    assert error.value.code == code


def test_a_range_error_names_the_option_and_its_limits() -> None:
    with pytest.raises(CodedError) as error:
        po.resolve("semantic", {"simplify": 0.5})
    assert error.value.params == {"option": "simplify", "min": 0.0, "max": 0.05}


def test_a_trained_model_lists_its_classes_and_which_the_project_lost() -> None:
    card = {"classes": ["cat", "dog", "cat"]}
    assert po.model_classes(card, ["cat"]) == [
        {"name": "cat", "in_project": True},
        {"name": "dog", "in_project": False},
    ]


def test_a_public_model_lists_the_project_classes_its_categories_map_to() -> None:
    card = {"source": "public", "mapping": {"person": "pedestrian", "car": "vehicle", "truck": "vehicle"}}
    assert [c["name"] for c in po.model_classes(card, ["pedestrian", "vehicle"])] == ["pedestrian", "vehicle"]


def test_resolve_classes() -> None:
    available = po.model_classes({"classes": ["a", "b", "c"]}, ["a", "b"])
    assert po.resolve_classes(available, None) == ["a", "b"]
    assert po.resolve_classes(available, ["b", "a"]) == ["a", "b"]  # the model's order
    for chosen, code in (
        (["c"], "class_not_in_model"),
        (["x"], "class_not_in_model"),
        ([], "no_class_selected"),
    ):
        with pytest.raises(CodedError) as error:
            po.resolve_classes(available, chosen)
        assert error.value.code == code


def test_the_range_is_checked_on_the_value_sent_like_the_form_does() -> None:
    """#663: 0.9990004 was rounded to 0.999 first and accepted, while the form refused it."""
    with pytest.raises(CodedError) as info:
        po.resolve("detection", {"confidence": 0.9990004})
    assert info.value.code == "option_range"
    assert (
        po.resolve("detection", {"confidence": 0.1234567})["confidence"] == 0.123457
    )  # still rounded when kept
