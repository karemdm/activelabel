import pytest
from curation_images import Stop, build
from PIL import Image
from sample_images import make_image

import activelabel.curation.quality as quality
from activelabel.curation.quality import _measure_record, _save_pending, hamming, measure_file, measure_images
from activelabel.jobs import Cancelled


def test_cancel_keeps_measured(tmp_path, monkeypatch):
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


def test_oversized_image_is_skipped_not_fatal(tmp_path, monkeypatch):
    manifest = build(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # "tiny" (32x24) passes, the others are bombs
    assert measure_images(manifest) == 1
    assert manifest.get("tiny").metrics and not manifest.get("ok1").metrics


def test_manifest_is_written_once_per_save_interval(tmp_path, monkeypatch):
    manifest = build(tmp_path)
    saves = []
    original = manifest.update
    monkeypatch.setattr(manifest, "update", lambda changes: (saves.append(len(changes)), original(changes)))
    monkeypatch.setattr(quality, "BATCH", 2)
    monkeypatch.setattr(quality, "SAVE_EVERY", 4)
    assert measure_images(manifest) == 7
    assert saves == [4, 3]


def test_hamming_counts_the_differing_bits():
    assert hamming("0000000000000000", "0000000000000000") == 0
    assert hamming("0000000000000000", "ffffffffffffffff") == 64
    assert hamming("000000000000000f", "0000000000000001") == 3


def test_measure_file_gives_metrics_and_a_hash(tmp_path):
    path = tmp_path / "a.png"
    make_image(path, seed=1)
    metrics, phash = measure_file(path)
    assert set(metrics) >= {"brightness", "contrast", "sharpness"} and len(phash) == 16


def test_measure_record_returns_none_for_an_unreadable_file(tmp_path):
    manifest = build(tmp_path)
    (tmp_path / "images" / "ok1.png").write_bytes(b"not an image")
    assert _measure_record(manifest, manifest.get("ok1")) == ("ok1", None)
    image_id, values = _measure_record(manifest, manifest.get("ok2"))
    assert image_id == "ok2" and set(values) == {"metrics", "phash"}


def test_save_pending_writes_and_empties(tmp_path):
    manifest = build(tmp_path)
    pending = {"ok1": {"metrics": {"brightness": 1.0}, "phash": "0" * 16}}
    _save_pending(manifest, pending)
    assert pending == {} and manifest.get("ok1").metrics == {"brightness": 1.0}
    _save_pending(manifest, pending)  # nothing waiting: nothing written
