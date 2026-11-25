import random

import pytest
from curation_images import LIMITS, add, build
from PIL import Image

from activelabel.curation.issues import (
    Thresholds,
    _group_duplicates,
    flag_issues,
    load_duplicate_of,
    load_thresholds,
    save_thresholds,
)
from activelabel.curation.quality import hamming, measure_images
from activelabel.curation.summary import suggest_exclusions, summary
from activelabel.errors import CodedError
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest


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


def test_group_duplicates_matches_a_brute_force_search():
    rng = random.Random(7)
    hashes = [rng.getrandbits(64) for _ in range(150)]
    for k in range(0, 150, 5):  # near copies at distances 0..8 of an earlier hash
        flips = rng.sample(range(64), k % 9)
        hashes[k + 1] = hashes[k] ^ sum(1 << b for b in flips)
    texts = [f"{h:016x}" for h in hashes]
    ids = [f"i{n}" for n in range(len(texts))]
    for limit in (0, 3, 6, 10):
        parent = list(range(len(ids)))

        def find(x, parent=parent):
            while parent[x] != x:
                x = parent[x]
            return x

        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if hamming(texts[a], texts[b]) <= limit:
                    parent[find(a)] = find(b)
        expected: dict[int, set[str]] = {}
        for n, i in enumerate(ids):
            expected.setdefault(find(n), set()).add(i)
        want = sorted(sorted(g) for g in expected.values() if len(g) > 1)
        got = sorted(sorted(g) for g in _group_duplicates(ids, texts, limit))
        assert got == want


def test_a_threshold_out_of_its_limits_is_a_coded_error():
    """The Limits panel showed "dark must be between 0 and 255." in English inside the pt interface."""
    with pytest.raises(CodedError) as caught:
        Thresholds(dark=-5)
    assert caught.value.code == "curation_threshold_range"
    assert caught.value.params == {"threshold": "dark", "min": 0, "max": 255}
    with pytest.raises(CodedError) as caught:
        Thresholds(min_side=10.5)
    assert caught.value.code == "curation_threshold_not_whole"
    with pytest.raises(CodedError) as caught:
        Thresholds(bright="x")  # type: ignore[arg-type]
    assert caught.value.code == "curation_threshold_not_number"
