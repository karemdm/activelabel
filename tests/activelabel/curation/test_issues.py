import io
import random

import pytest
from curation_images import LIMITS, add, build
from PIL import Image
from sample_images import make_image

from activelabel.curation.issues import (
    Thresholds,
    _group_duplicates,
    flag_files,
    flag_issues,
    load_duplicate_of,
    load_thresholds,
    map_links,
    save_thresholds,
)
from activelabel.curation.quality import hamming, measure_images
from activelabel.curation.similarity import SimilarityStore
from activelabel.curation.summary import suggest_exclusions, summary
from activelabel.errors import CodedError
from activelabel.images.ingest import ingest_files
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


def test_files_deleted_or_changed_on_disk_are_flagged_and_kept_by_the_thresholds(tmp_path):
    """#620: Curation finds a deleted file and a replaced one (even of the same size); the flags survive a new
    flagging with other thresholds and go away when the file is back."""
    manifest = Manifest(tmp_path / "p")
    sources = [make_image(tmp_path / "src" / f"{n}.png", seed=n) for n in range(4)]
    ingest_files(manifest, [(p.name, io.BytesIO(p.read_bytes())) for p in sources])
    gone, edited, swapped, fine = sorted(manifest.all(), key=lambda r: r.name)
    files = {r.id: manifest.file_path(r) for r in (gone, edited, swapped)}
    backup = {i: p.read_bytes() for i, p in files.items()}
    files[gone.id].unlink()
    files[edited.id].write_bytes(backup[edited.id] + b"x")
    same_size = bytearray(backup[swapped.id])
    same_size[-20] ^= 0xFF  # one byte flipped: same size, other content
    files[swapped.id].write_bytes(bytes(same_size))
    assert flag_files(manifest) == {"file_missing": 1, "file_changed": 2}
    assert manifest.get(gone.id).issues == ["file_missing"] and manifest.get(fine.id).issues == []
    assert manifest.file_problem(manifest.get(swapped.id)) is None  # the quick check only sees the size
    measure_images(manifest)
    counts = flag_issues(manifest, LIMITS)
    assert counts["file_missing"] == 1 and counts["file_changed"] == 2
    assert "file_changed" in manifest.get(edited.id).issues
    for image_id, data in backup.items():
        files[image_id].write_bytes(data)
    assert flag_files(manifest) == {"file_missing": 0, "file_changed": 0}
    assert all("file_" not in " ".join(r.issues) for r in manifest.all())


def test_the_similarity_map_adds_near_duplicates_the_hash_misses(tmp_path):
    manifest = Manifest(tmp_path)
    add(manifest, "big", size=(128, 96), seed=1)
    add(manifest, "crop", size=(120, 90), seed=2)  # another picture for the hash
    add(manifest, "other", size=(128, 96), seed=9)
    measure_images(manifest)
    assert hamming(manifest.get("big").phash, manifest.get("crop").phash) > LIMITS.phash_distance
    SimilarityStore(tmp_path).update(near_pairs=[["big", "crop", 0.01], ["big", "other", 0.08]])
    flag_issues(manifest, LIMITS)
    assert "near_duplicate" in manifest.get("crop").issues
    assert "near_duplicate" not in manifest.get("other").issues  # farther than embedding_distance
    assert load_duplicate_of(tmp_path) == {"crop": "big"}
    off = Thresholds(min_side=40, low_contrast=3.0, embedding_distance=0)
    flag_issues(manifest, off)
    assert "near_duplicate" not in manifest.get("crop").issues


def test_map_links_reads_the_pairs_within_the_distance(tmp_path):
    assert map_links(tmp_path, 0.05) == []  # no map
    SimilarityStore(tmp_path).update(near_pairs=[["a", "b", 0.01], ["a", "c", 0.06], ["bad"]])
    assert map_links(tmp_path, 0.05) == [("a", "b")]
    assert map_links(tmp_path, 0.1) == [("a", "b"), ("a", "c")]
    assert map_links(tmp_path, 0) == []


def test_group_duplicates_joins_the_extra_links_of_known_ids():
    hashes = ["0000000000000000", "ffffffffffffffff", "0f0f0f0f0f0f0f0f"]
    groups = _group_duplicates(["a", "b", "c"], hashes, 0, [("a", "b"), ("b", "zzz")])
    assert groups == [["a", "b"]]


def test_the_embedding_distance_has_limits():
    with pytest.raises(CodedError):
        Thresholds(embedding_distance=0.5)
    assert Thresholds(embedding_distance=0).embedding_distance == 0.0
