from curation_images import LIMITS, build

from activelabel.curation.issues import BLURRY, DARK, NEAR_DUPLICATE, flag_issues
from activelabel.curation.quality import measure_images
from activelabel.curation.summary import _histogram, _worth_excluding, suggest_exclusions, summary
from activelabel.images.manifest import EXCLUDED, ImageRecord


def test_summary(tmp_path):
    manifest = build(tmp_path)
    measure_images(manifest)
    flag_issues(manifest, LIMITS)
    data = summary(manifest)
    assert data["issue_ids"]["dark"] == ["dark"]
    assert data["issues"]["dark"] == 1
    hist = data["histograms"]["brightness"]
    assert len(hist["counts"]) == 20 and sum(hist["counts"]) == 7 and len(hist["edges"]) == 21


def test_histogram_of_nothing_and_of_equal_values():
    assert _histogram([]) == {"edges": [], "counts": []}
    flat = _histogram([5.0, 5.0, 5.0])
    assert sum(flat["counts"]) == 3 and flat["counts"][0] == 3 and flat["edges"][:2] == [5.0, 5.05]


def test_suggest_exclusions_is_conservative():
    def record(image_id, *issues):
        return ImageRecord(
            image_id, f"{image_id}.png", f"images/{image_id}.png", 64, 48, 1, "2026", issues=list(issues)
        )

    assert _worth_excluding(record("a", NEAR_DUPLICATE))
    assert _worth_excluding(record("b", BLURRY, DARK))
    assert not _worth_excluding(record("c", BLURRY))  # blur alone is left to the person
    assert not _worth_excluding(record("d", DARK))


def test_suggest_exclusions_skips_excluded_images(tmp_path):
    manifest = build(tmp_path)
    measure_images(manifest)
    flag_issues(manifest, LIMITS)
    manifest.update({"dark": {"issues": [NEAR_DUPLICATE]}})
    assert suggest_exclusions(manifest) == ["dark"]
    manifest.set_status(["dark"], EXCLUDED, "x")
    assert suggest_exclusions(manifest) == []
