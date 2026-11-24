import pytest

from activelabel.images.manifest import ACTIVE, EXCLUDED, ImageRecord, Manifest


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


def test_cached_parse_never_leaks_changes_or_hides_edits(tmp_path):
    import json

    from activelabel.images.manifest import ImageRecord, Manifest

    manifest = Manifest(tmp_path)
    manifest.add(
        [ImageRecord("a", "a.png", "images/a.png", 1, 1, 1, "t", issues=["dark"], metrics={"x": 1.0})]
    )
    first = manifest.load()
    first["a"].issues.append("blurry")  # changing a loaded record must not change what is stored
    first["a"].metrics["x"] = 9.0
    assert manifest.get("a").issues == ["dark"] and manifest.get("a").metrics == {"x": 1.0}
    data = json.loads(manifest.path.read_text(encoding="utf-8"))
    data[0]["name"] = "edited by hand.png"
    manifest.path.write_text(json.dumps(data, indent=3), encoding="utf-8")  # another process changed the file
    assert manifest.get("a").name == "edited by hand.png"
