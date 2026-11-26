import pytest

from activelabel.training import augmentations as aug
from activelabel.training.augmentations import AugmentationError

KINDS = ("classification", "detection", "instance", "keypoints", "semantic")


def _raises(code, fn, *args):
    with pytest.raises(AugmentationError) as info:
        fn(*args)
    assert info.value.code == code
    return info.value


def _names(entries):
    return [e.name if hasattr(e, "name") else e["name"] for e in entries]


@pytest.mark.parametrize("technique", aug.TECHNIQUES, ids=lambda t: t.name)
def test_defaults_pass_their_own_check(technique):
    for spec in technique.params:
        assert spec.check(spec.default) == spec.default


def test_techniques_per_kind():
    assert "randaugment" in _names(aug.techniques_for("classification"))
    for kind in KINDS[1:]:
        assert "randaugment" not in _names(aug.techniques_for(kind))
    assert "scale_jitter" not in _names(aug.techniques_for("classification"))
    assert "scale_jitter" in _names(aug.techniques_for("detection"))
    assert "vflip" not in _names(aug.techniques_for("keypoints"))
    assert "vflip" in _names(aug.techniques_for("semantic"))


def test_classification_strong_preset():
    assert aug.preset("strong", "classification") == [
        {"name": "random_resized_crop", "params": {"scale_min": 0.4}},
        {"name": "hflip", "params": {"p": 0.5}},
        {"name": "rotation", "params": {"degrees": 15.0}},
        {"name": "trivialaugment", "params": {}},
        {
            "name": "color_jitter",
            "params": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
        },
    ]


@pytest.mark.parametrize("level", aug.PRESETS_NAMES)
def test_presets_validate_and_keypoints_have_no_flip(level):
    assert "hflip" not in _names(aug.preset(level, "keypoints"))
    for kind in KINDS:
        entries = aug.preset(level, kind)
        assert aug.validate(entries, kind) == entries


@pytest.mark.parametrize("level", ["custom", "x"])
def test_preset_rejects_other_levels(level):
    error = _raises("augmentation_level_invalid", aug.preset, level, "detection")
    assert error.params["choices"] == ["none", "light", "standard", "strong"]


def test_validate_orders_and_fills_defaults():
    given = [{"name": "color_jitter"}, {"name": "hflip", "params": {"p": 0.3}}]
    checked = aug.validate(given, "detection")
    assert _names(checked) == ["hflip", "color_jitter"]
    assert checked[0]["params"] == {"p": 0.3}
    assert checked[1]["params"] == {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0}
    assert aug.validate([], "detection") == []


def test_validate_rejects_not_a_list():
    _raises("augmentations_not_list", aug.validate, {"name": "hflip"}, "detection")


@pytest.mark.parametrize(
    "entry",
    [{"name": "hflip", "extra": 1}, {"name": "hflip", "params": [1]}, "hflip", {"params": {}}, {"name": 3}],
)
def test_validate_rejects_bad_entries(entry):
    _raises("augmentation_entry_invalid", aug.validate, [entry], "detection")


def test_validate_rejects_unknown_technique_and_param():
    error = _raises("augmentation_unknown", aug.validate, [{"name": "randaugment"}], "detection")
    assert error.params == {"name": "randaugment", "kind": "detection"}
    _raises("augmentation_unknown", aug.validate, [{"name": "nope"}], "detection")
    entry = {"name": "hflip", "params": {"q": 1}}
    error = _raises("augmentation_param_unknown", aug.validate, [entry], "detection")
    assert error.params == {"name": "hflip", "param": "q"}


def test_validate_rejects_out_of_range_param():
    entry = {"name": "hflip", "params": {"p": 2}}
    error = _raises("param_max", aug.validate, [entry], "detection")
    assert error.params["name"] == "hflip.p"


def test_validate_rejects_repeated_and_conflicting():
    error = _raises(
        "augmentation_repeated", aug.validate, [{"name": "hflip"}, {"name": "hflip"}], "detection"
    )
    assert error.params == {"name": "hflip"}
    both = [{"name": "randaugment"}, {"name": "trivialaugment"}]
    error = _raises("augmentation_conflict", aug.validate, both, "classification")
    assert {error.params["name"], error.params["other"]} == {"randaugment", "trivialaugment"}


def test_scale_jitter_bounds_cannot_cross():
    # scale_min tops out at 1.0 and scale_max starts at 1.0: the ranges keep the zoom bounds in order
    entry = {"name": "scale_jitter", "params": {"scale_min": 1.0, "scale_max": 1.0}}
    assert aug.validate([entry], "detection")[0]["params"] == {"scale_min": 1.0, "scale_max": 1.0}
    bad = {"name": "scale_jitter", "params": {"scale_min": 1.5}}
    _raises("param_max", aug.validate, [bad], "detection")


def test_keypoint_flip_needs_pairs():
    _raises("augmentation_flip_pairs_required", aug.validate, [{"name": "hflip"}], "keypoints")
    entry = {"name": "hflip", "params": {"p": 0.5}}
    _raises("augmentation_flip_pairs_required", aug.validate, [entry], "keypoints")
    assert aug.validate([{"name": "hflip"}], "detection")[0]["params"] == {"p": 0.5}
    entry = {"name": "hflip", "params": {"flip_pairs": []}}
    _raises("augmentation_param_unknown", aug.validate, [entry], "detection")


@pytest.mark.parametrize(
    "pairs", [[[0, 1], [1, 2]], [[-1, 1]], [[True, 1]], [[2, 2]], [0, 1], "x", [[0, 1, 2]], [[0.5, 1]]]
)
def test_bad_flip_pairs(pairs):
    entry = {"name": "hflip", "params": {"flip_pairs": pairs}}
    _raises("augmentation_flip_pairs_invalid", aug.validate, [entry], "keypoints")


def test_good_flip_pairs_are_kept():
    entry = {"name": "hflip", "params": {"flip_pairs": [[0, 1], [2, 3]]}}
    assert aug.validate([entry], "keypoints")[0]["params"] == {"p": 0.5, "flip_pairs": [[0, 1], [2, 3]]}


def test_resolve():
    _raises("augmentation_custom_needs_list", aug.resolve, "custom", None, "classification")
    assert aug.resolve("light", None, "detection") == aug.preset("light", "detection")
    assert aug.resolve("custom", [], "detection") == []
    custom = aug.resolve("custom", [{"name": "blur"}, {"name": "hflip"}], "detection")
    assert _names(custom) == ["hflip", "blur"]
    _raises("augmentation_level_invalid", aug.resolve, "x", None, "detection")


def test_max_flip_index():
    assert aug.max_flip_index([]) == -1
    assert aug.max_flip_index([{"name": "hflip", "params": {"p": 0.5}}]) == -1
    entries = [{"name": "hflip", "params": {"p": 0.5, "flip_pairs": [[0, 1], [4, 2]]}}]
    assert aug.max_flip_index(entries) == 4


def test_catalogue_shape():
    body = aug.catalogue("classification")
    assert list(body) == ["techniques", "presets"]
    assert _names(body["techniques"]) == _names(aug.techniques_for("classification"))
    assert list(body["presets"]) == list(aug.PRESETS_NAMES)
    assert body["presets"]["none"] == []
    keys = {"name", "help", "geometric", "batch", "conflicts", "needs_flip_pairs", "params"}
    assert keys <= set(body["techniques"][0])


def test_needs_flip_pairs_only_for_keypoint_hflip():
    flags = {
        (kind, t["name"]): t["needs_flip_pairs"] for kind in KINDS for t in aug.catalogue(kind)["techniques"]
    }
    assert [k for k, v in flags.items() if v] == [("keypoints", "hflip")]


def test_every_preset_above_none_does_something_for_every_kind():
    """A level the person can pick is never the same as "none" (keypoints "light" was empty, #635)."""
    for kind in ("classification", "detection", "instance", "keypoints", "semantic"):
        for level in ("light", "standard", "strong"):
            assert aug.preset(level, kind), (kind, level)
    assert all(e["name"] != "hflip" for e in aug.preset("light", "keypoints"))
