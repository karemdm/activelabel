import pytest

from visionloop.images.manifest import ACTIVE, EXCLUDED, ImageRecord, Manifest


def record(image_id, name="a.jpg"):
    return ImageRecord(image_id, name, f"images/{image_id}.jpg", 10, 8, 100, "2026-01-01T00:00:00+00:00")


def test_add_skips_known_ids(tmp_path):
    manifest = Manifest(tmp_path)
    assert len(manifest.add([record("a"), record("b")])) == 2
    assert [r.id for r in manifest.add([record("a"), record("c")])] == ["c"]
    assert [r.id for r in manifest.all()] == ["a", "b", "c"]


def test_exclude_restore_and_delete(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.add([record("a"), record("b")])
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    assert manifest.set_status(["a", "zzz"], EXCLUDED, "blurry") == 1
    assert manifest.get("a").excluded_reason == "blurry"
    assert [r.id for r in manifest.active()] == ["b"]
    assert manifest.set_status(["a"], ACTIVE) == 1
    assert manifest.get("a").excluded_reason == ""
    assert manifest.delete(["a"]) == 1
    assert not (tmp_path / "images" / "a.jpg").exists()
    with pytest.raises(KeyError):
        manifest.get("a")


def test_update_fields(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.add([record("a")])
    manifest.update({"a": {"metrics": {"brightness": 12.0}, "issues": ["dark"]}, "zzz": {"issues": []}})
    assert manifest.get("a").issues == ["dark"]
    with pytest.raises(ValueError):
        manifest.update({"a": {"file": "x"}})
