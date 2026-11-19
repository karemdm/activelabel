import pytest
from PIL import Image
from sample_images import make_image

from visionloop.curation.issues import (
    Thresholds,
    flag_issues,
    load_duplicate_of,
    load_thresholds,
    save_thresholds,
)
from visionloop.curation.quality import hamming, measure_images
from visionloop.curation.summary import suggest_exclusions, summary
from visionloop.images.manifest import EXCLUDED, ImageRecord, Manifest
from visionloop.jobs import Cancelled

LIMITS = Thresholds(min_side=40, low_contrast=3.0)  # the sample images are small and soft


def add(manifest, image_id, **kwargs):
    path = manifest.project_dir / "images" / f"{image_id}.png"
    make_image(path, **kwargs)
    with Image.open(path) as img:
        width, height = img.size
    record = ImageRecord(
        image_id, f"{image_id}.png", f"images/{image_id}.png", width, height, 1, "2026-01-01"
    )
    manifest.add([record])
    return path


def build(tmp_path):
    manifest = Manifest(tmp_path)
    add(manifest, "ok1", seed=1)
    add(manifest, "ok2", seed=2)
    add(manifest, "ok3", seed=3)
    add(manifest, "dark", seed=4, brightness=10)
    add(manifest, "blur", seed=5, blur=3)
    add(manifest, "tiny", size=(32, 24), seed=6)
    add(manifest, "wide", size=(200, 48), seed=7)
    return manifest


def test_measure_and_issues(tmp_path):
    manifest = build(tmp_path)
    assert measure_images(manifest) == 7
    assert measure_images(manifest) == 0  # already measured
    assert measure_images(manifest, recompute=True) == 7
    record = manifest.get("ok1")
    assert 0 < record.metrics["brightness"] < 255
    assert len(record.phash) == 16
    assert manifest.get("dark").metrics["brightness"] < manifest.get("ok1").metrics["brightness"]
    assert manifest.get("blur").metrics["sharpness"] < manifest.get("ok1").metrics["sharpness"]

    counts = flag_issues(manifest, LIMITS)
    assert "dark" in manifest.get("dark").issues
    assert "blurry" in manifest.get("blur").issues
    assert "small" in manifest.get("tiny").issues
    assert "odd_aspect" in manifest.get("wide").issues
    assert manifest.get("ok1").issues == []
    assert counts["dark"] == 1 and counts["small"] == 1 and set(counts) >= {"grayscale", "near_duplicate"}


def test_grayscale_flag(tmp_path):
    manifest = Manifest(tmp_path)
    path = add(manifest, "color", seed=1)
    gray = manifest.project_dir / "images" / "gray.png"
    Image.open(path).convert("L").save(gray)
    manifest.add([ImageRecord("gray", "g.png", "images/gray.png", 64, 48, 1, "2026")])
    measure_images(manifest)
    flag_issues(manifest, LIMITS)
    assert "grayscale" in manifest.get("gray").issues
    assert "grayscale" not in manifest.get("color").issues


def test_near_duplicates_keep_the_biggest(tmp_path):
    manifest = Manifest(tmp_path)
    big = add(manifest, "big", size=(128, 96), seed=1)
    small = manifest.project_dir / "images" / "small.png"
    Image.open(big).resize((64, 48)).save(small)
    manifest.add([ImageRecord("small", "s.png", "images/small.png", 64, 48, 1, "2026")])
    add(manifest, "other", size=(128, 96), seed=9)
    measure_images(manifest)
    assert hamming(manifest.get("big").phash, manifest.get("small").phash) <= 6
    flag_issues(manifest, LIMITS)
    assert "near_duplicate" in manifest.get("small").issues
    assert "near_duplicate" not in manifest.get("big").issues
    assert "near_duplicate" not in manifest.get("other").issues
    assert load_duplicate_of(tmp_path) == {"small": "big"}
    assert "small" in suggest_exclusions(manifest)


def test_excluded_are_ignored(tmp_path):
    manifest = build(tmp_path)
    measure_images(manifest)
    manifest.set_status(["dark"], EXCLUDED, "x")
    counts = flag_issues(manifest, LIMITS)
    assert counts["dark"] == 0 and manifest.get("dark").issues == []
    assert summary(manifest)["totals"] == {"images": 7, "active": 6, "excluded": 1, "measured": 6}


def test_thresholds_roundtrip(tmp_path):
    assert load_thresholds(tmp_path) == Thresholds()
    save_thresholds(tmp_path, Thresholds(min_side=100, dark=30.0))
    loaded = load_thresholds(tmp_path)
    assert loaded.min_side == 100 and loaded.dark == 30.0 and loaded.bright == Thresholds().bright


def test_summary(tmp_path):
    manifest = build(tmp_path)
    measure_images(manifest)
    flag_issues(manifest, LIMITS)
    data = summary(manifest)
    assert data["issue_ids"]["dark"] == ["dark"]
    assert data["issues"]["dark"] == 1
    hist = data["histograms"]["brightness"]
    assert len(hist["counts"]) == 20 and sum(hist["counts"]) == 7 and len(hist["edges"]) == 21


class Stop:
    def __init__(self, after):
        self.calls, self.after = 0, after

    def update(self, done=None, total=None, message=None):
        self.calls += 1
        if self.calls > self.after:
            raise Cancelled

    def check(self):
        pass


def test_cancel_keeps_measured(tmp_path, monkeypatch):
    import visionloop.curation.quality as quality

    monkeypatch.setattr(quality, "BATCH", 2)
    manifest = build(tmp_path)
    with pytest.raises(Cancelled):
        measure_images(manifest, Stop(2))
    done = [r for r in manifest.all() if r.metrics]
    assert 0 < len(done) < 7


def test_corrupt_file_is_skipped(tmp_path):
    manifest = build(tmp_path)
    (tmp_path / "images" / "tiny.png").write_bytes(b"not an image")
    assert measure_images(manifest) == 6
    assert manifest.get("tiny").metrics == {}
