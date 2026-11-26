import math

from activelabel.training.quality import (
    CODES,
    MIN_VAL_IMAGES,
    _class_scores,
    _loss_gap,
    _number,
    _single_class,
    quality_alerts,
)


def codes(alerts):
    return [a["code"] for a in alerts]


def classifier(matrix, recalls=None, support=10):
    classes = ["a", "b", "c"][: len(matrix)]
    per_class = {c: {"recall": (recalls or {}).get(c, 0.8), "support": support} for c in classes}
    return {
        "metric": "accuracy",
        "best_epoch": 2,
        "val": {"accuracy": 0.8, "confusion_matrix": matrix, "classes": classes, "per_class": per_class},
    }


def test_number_only_accepts_finite_numbers():
    assert _number(1) == 1.0 and _number(0.5) == 0.5
    assert _number(None) is None and _number("1") is None and _number(True) is None
    assert _number(math.nan) is None and _number(math.inf) is None


def test_no_results_or_no_val_means_no_alerts():
    assert quality_alerts(None) == []
    assert quality_alerts({"metric": "accuracy"}) == []
    assert quality_alerts({"metric": "accuracy", "val": None}) == []


def test_a_healthy_run_has_no_alert():
    results = classifier([[9, 1], [1, 9]], {"a": 0.9, "b": 0.9})
    metrics = [{"epoch": 2, "train_loss": 0.3, "val_loss": 0.4}]
    assert quality_alerts(results, metrics, val_images=20) == []


def test_main_metric_zero_missing_or_nan():
    for value in (0, 0.0, None, math.nan):
        results = {"metric": "map50", "val": {"map50": value}}
        assert quality_alerts(results) == [{"code": "metric_zero", "params": {"metric": "map50"}}]
    assert quality_alerts({"metric": "map50", "val": {"map50": 0.2}}) == []
    assert quality_alerts({"metric": "map50", "val": {"other": 0}}) == []  # not in the results: no claim
    assert quality_alerts({"val": {"map50": 0}}) == []


def test_a_classifier_that_predicts_one_class():
    alerts = quality_alerts(classifier([[10, 0], [9, 1]]))
    assert alerts[0] == {"code": "single_class", "params": {"class": "a", "share": 95}}
    assert _single_class(classifier([[10, 0], [5, 5]])["val"]) is None  # 75%
    assert _single_class({"classes": ["a"], "confusion_matrix": [[3]]}) is None  # one class
    assert _single_class({"classes": ["a", "b"], "confusion_matrix": [[0, 0], [0, 0]]}) is None
    assert _single_class({"classes": ["a", "b"], "confusion_matrix": [[1, "x"], [1]]}) is None
    assert _single_class({"classes": ["a", "b"], "confusion_matrix": [[1]]}) is None
    assert _single_class({"classes": ["a", "b"]}) is None


def test_segmentation_with_only_background():
    val = {
        "miou": 0.0,
        "pixel_accuracy": 0.93,
        "background_iou": 0.93,
        "classes": ["background", "x"],
        "confusion_matrix": [[95, 0], [5, 0]],
    }
    assert quality_alerts({"metric": "miou", "val": val}) == [{"code": "only_background", "params": {}}]
    low = {**val, "pixel_accuracy": 0.2}
    assert codes(quality_alerts({"metric": "miou", "val": low})) == ["metric_zero"]


def test_weak_classes_are_named_and_the_background_is_left_out():
    val = {
        "iou": 0.5,
        "background_iou": 0.9,
        "per_class": {
            "background": {"iou": 0.9, "support": 100},
            "car": {"iou": 0.0, "support": 5},
            "dog": {"iou": 0.1, "support": 5},
            "cat": {"iou": 0.7, "support": 5},
            "bird": {"iou": 0.7, "support": 0},  # no samples: not counted
        },
    }
    assert set(_class_scores(val)) == {"car", "dog", "cat"}
    alerts = quality_alerts({"metric": "iou", "val": val})
    assert alerts[0] == {"code": "class_zero", "params": {"class": "car"}}
    assert alerts[1]["code"] == "class_far_below" and alerts[1]["params"]["class"] == "dog"
    assert alerts[1]["params"]["mean"] == 0.267


def test_class_gap_needs_a_decent_mean_and_two_classes_and_names_at_most_five():
    weak = {"per_class": {"a": {"ap50": 0.05, "support": 3}, "b": {"ap50": 0.3, "support": 3}}}
    assert quality_alerts({"metric": "ap", "val": weak}) == []  # mean 0.175 <= 0.2
    assert quality_alerts({"metric": "ap", "val": {"per_class": {"a": {"ap50": 0.0, "support": 3}}}}) == []
    all_zero = {"per_class": {"a": {"ap50": 0.0, "support": 3}, "b": {"ap50": 0.0, "support": 3}}}
    assert quality_alerts({"val": all_zero}) == []  # nothing stands out
    many = {
        "per_class": {f"c{k}": {"ap50": 0.0, "support": 1} for k in range(8)}
        | {"ok": {"ap50": 1.0, "support": 1}}
    }
    assert len(quality_alerts({"val": many})) == 5
    broken = {"per_class": {"a": "x", "b": {"support": 2}, "c": {"ap50": "?", "support": 2}}}
    assert _class_scores(broken) == {} and _class_scores({"per_class": []}) == {}


def test_validation_much_worse_than_training():
    results = {"best_epoch": 2, "val": {"accuracy": 0.7}, "metric": "accuracy"}
    rows = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0},
        {"epoch": 2, "train_loss": 0.2, "val_loss": 0.9},
    ]
    assert _loss_gap(results, rows) == {"train": 0.2, "val": 0.9}
    assert codes(quality_alerts(results, rows)) == ["val_worse_than_train"]
    assert _loss_gap(results, [{"epoch": 2, "train_loss": 0.02, "val_loss": 0.08}]) is None  # tiny difference
    assert _loss_gap(results, [{"epoch": 2, "train_loss": 1.0, "val_loss": 1.5}]) is None  # under 2x
    assert _loss_gap(results, [{"epoch": 2, "train_loss": 0.2}]) is None  # no val loss
    assert _loss_gap(results, [{"epoch": 2, "train_loss": -1, "val_loss": 5}]) is None
    assert _loss_gap(results, []) is None
    assert _loss_gap({}, [{"epoch": 9, "train_loss": 0.1, "val_loss": 0.9}, "x"]) == {
        "train": 0.1,
        "val": 0.9,
    }  # last row


def test_too_few_validation_images_and_classes():
    results = classifier([[1, 0], [0, 1]], support=1)
    found = quality_alerts(results, val_images=2)
    assert codes(found) == ["few_val_images", "few_val_per_class"]
    assert found[0]["params"] == {"n": 2}
    assert found[1]["params"] == {"classes": "a, b", "n": 2}
    assert "few_val_images" not in codes(quality_alerts(results, val_images=MIN_VAL_IMAGES))
    assert "few_val_images" not in codes(quality_alerts(results))  # size unknown
    detection = {"metric": "map50", "val": {"map50": 0.5, "per_class": {"a": {"ap50": 0.5, "support": 1}}}}
    assert quality_alerts(detection) == []  # per-class support is only checked for classifiers


def test_every_code_is_produced_and_listed():
    seen = set(
        codes(
            quality_alerts(
                classifier([[10, 0], [9, 1]], {"a": 0.0, "b": 0.9}, support=1),
                [{"epoch": 2, "train_loss": 0.1, "val_loss": 0.9}],
                3,
            )
        )
    )
    seen |= {"only_background", "metric_zero", "class_far_below"}
    assert seen == set(CODES)
