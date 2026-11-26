import io
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sample_images import make_corrupt, make_image

from activelabel.images.ingest import IngestReport, ingest_files, ingest_folder, thumbnail
from activelabel.images.manifest import Manifest
from activelabel.jobs import Cancelled, Job, Progress


def _bytes(tmp_path: Path, name: str, **kw) -> bytes:
    return make_image(tmp_path / "_src" / name, **kw).read_bytes()


def _zip(entries: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


@pytest.fixture
def manifest(tmp_path):
    return Manifest(tmp_path / "project")


def test_images_are_stored_by_content(manifest, tmp_path):
    a, b = _bytes(tmp_path, "a.jpg", seed=1), _bytes(tmp_path, "b.png", seed=2, fmt="PNG")
    report = ingest_files(manifest, [("x/a.JPG", io.BytesIO(a)), ("b.png", io.BytesIO(b))])
    assert report.added == 2 and not report.rejected
    records = manifest.all()
    assert {r.name for r in records} == {"x/a.JPG", "b.png"}
    first = next(r for r in records if r.name == "x/a.JPG")
    assert first.file == f"images/{first.id}.jpg" and len(first.id) == 16
    assert manifest.file_path(first).read_bytes() == a
    assert (first.width, first.height, first.size_bytes) == (64, 48, len(a))
    assert not list((manifest.project_dir / "images").glob(".*"))


def test_duplicates_within_and_across_uploads(manifest, tmp_path):
    a = _bytes(tmp_path, "a.jpg", seed=1)
    report = ingest_files(manifest, [("a.jpg", io.BytesIO(a)), ("copy.jpg", io.BytesIO(a))])
    assert report.added == 1 and report.duplicates == ["copy.jpg"]
    again = ingest_files(manifest, [("other.jpg", io.BytesIO(a))])
    assert again.added == 0 and again.duplicates == ["other.jpg"]
    assert len(manifest.all()) == 1


def test_a_repeated_image_keeps_its_other_names_as_aliases(manifest, tmp_path):
    a = _bytes(tmp_path, "a.jpg", seed=1)
    ingest_files(manifest, [("a.jpg", io.BytesIO(a)), ("copy.jpg", io.BytesIO(a)), ("a.jpg", io.BytesIO(a))])
    ingest_files(manifest, [("other/copy.jpg", io.BytesIO(a)), ("copy.jpg", io.BytesIO(a))])
    (record,) = manifest.all()
    assert record.name == "a.jpg" and record.aliases == ["copy.jpg", "other/copy.jpg"]


def test_corrupt_small_and_unsupported(manifest, tmp_path):
    bad = make_corrupt(tmp_path / "bad.jpg").read_bytes()
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (4, 40)).save(tiny)
    report = ingest_files(
        manifest,
        [
            ("bad.jpg", io.BytesIO(bad)),
            ("tiny.png", io.BytesIO(tiny.read_bytes())),
            ("doc.pdf", io.BytesIO(b"x")),
        ],
    )
    assert report.added == 0 and manifest.all() == []
    assert {(r["name"], r["reason"]) for r in report.rejected} == {
        ("bad.jpg", "corrupt"),
        ("tiny.png", "too small"),
        ("doc.pdf", "unsupported"),
    }
    assert not list((manifest.project_dir / "images").iterdir())


def test_exif_orientation_swaps_size(manifest, tmp_path):
    path = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), "red").save(path, exif=exif)
    ingest_files(manifest, [("rot.jpg", io.BytesIO(path.read_bytes()))])
    record = manifest.all()[0]
    assert (record.width, record.height) == (20, 40)
    with Image.open(thumbnail(manifest, record)) as thumb:
        assert thumb.size == (20, 40)


def test_zip_with_subfolders_and_other_files(manifest, tmp_path):
    a, b = _bytes(tmp_path, "a.jpg", seed=1), _bytes(tmp_path, "b.jpg", seed=2)
    archive = _zip(
        {
            "set/a.jpg": a,
            "set/deep/b.jpg": b,
            "set/labels.json": b"{}",
            "set/notes.txt": b"hi",
            "set/readme.pdf": b"%PDF",
            "__MACOSX/set/._a.jpg": b"junk",
            "set/.hidden.jpg": a,
            "set/.DS_Store": b"x",
            "set/Thumbs.db": b"x",
            "set/dir/": b"",
        }
    )
    report = ingest_files(manifest, [("data.zip", archive)])
    assert report.added == 2
    assert {r.name for r in manifest.all()} == {"set/a.jpg", "set/deep/b.jpg"}
    assert sorted(report.other_files) == ["set/labels.json", "set/notes.txt"]
    assert report.rejected == [{"name": "set/readme.pdf", "reason": "unsupported"}]
    assert report.duplicates == []


def test_zip_slip_is_rejected(manifest, tmp_path):
    a = _bytes(tmp_path, "a.jpg", seed=1)
    archive = _zip({"../evil.jpg": a, "/abs.jpg": a, "ok/a.jpg": a})
    report = ingest_files(manifest, [("s.zip", archive)])
    assert report.added == 1
    assert {r["reason"] for r in report.rejected} == {"unsafe path"} and len(report.rejected) == 2
    assert not (manifest.project_dir.parent / "evil.jpg").exists()


def test_bad_zip_and_size_limit(manifest, monkeypatch, tmp_path):
    report = ingest_files(manifest, [("x.zip", io.BytesIO(b"not a zip"))])
    assert report.rejected == [{"name": "x.zip", "reason": "corrupt"}]
    monkeypatch.setattr("activelabel.images.ingest.MAX_ZIP_BYTES", 10)
    big = _zip({"a.jpg": _bytes(tmp_path, "a.jpg")})
    report = ingest_files(manifest, [("big.zip", big)])
    assert report.added == 0 and "too large" in report.rejected[0]["reason"]


def test_folder_import(manifest, tmp_path):
    root = tmp_path / "root"
    make_image(root / "set" / "a.jpg", seed=1)
    make_image(root / "set" / "sub" / "b.jpg", seed=2)
    (root / "set" / "notes.md").write_text("x")
    report = ingest_folder(manifest, root / "set", root)
    assert report.added == 2
    assert {r.name for r in manifest.all()} == {"a.jpg", "sub/b.jpg"}
    assert report.rejected == [{"name": "notes.md", "reason": "unsupported"}]


def test_folder_import_refused(manifest, tmp_path):
    make_image(tmp_path / "outside" / "a.jpg")
    with pytest.raises(ValueError, match="ACTIVELABEL_IMPORT_ROOT"):
        ingest_folder(manifest, tmp_path / "outside", None)
    (tmp_path / "root").mkdir()
    with pytest.raises(ValueError, match="inside the import root"):
        ingest_folder(manifest, tmp_path / "outside", tmp_path / "root")
    with pytest.raises(ValueError, match="inside the import root"):
        ingest_folder(manifest, tmp_path / "root" / ".." / "outside", tmp_path / "root")
    assert manifest.all() == []


def test_cancellation_keeps_manifest_consistent(manifest, tmp_path):
    files = [(f"{i}.jpg", io.BytesIO(_bytes(tmp_path, f"{i}.jpg", seed=i))) for i in range(6)]
    cancel = threading.Event()
    progress = Progress(Job("j", 1, "ingest"), cancel)
    original = progress.update

    def update(done=None, total=None, message=None):
        if done == 3:
            cancel.set()
        original(done, total, message)

    progress.update = update  # type: ignore[method-assign]
    with pytest.raises(Cancelled):
        ingest_files(manifest, files, progress)
    stored = sorted(p.name for p in (manifest.project_dir / "images").iterdir())
    records = manifest.all()
    assert 0 < len(records) < 6
    assert stored == sorted(Path(r.file).name for r in records)


def test_batches_flush_manifest(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr("activelabel.images.ingest.BATCH_SIZE", 2)
    calls = []
    real = manifest.add

    def counting_add(records):
        calls.append(len(records))
        return real(records)

    monkeypatch.setattr(manifest, "add", counting_add)
    files = [(f"{i}.jpg", io.BytesIO(_bytes(tmp_path, f"{i}.jpg", seed=i))) for i in range(5)]
    assert ingest_files(manifest, files).added == 5
    assert calls == [2, 2, 1]


def test_thumbnail_cached_and_regenerated(manifest, tmp_path):
    data = _bytes(tmp_path, "a.png", size=(600, 300), fmt="PNG")
    ingest_files(manifest, [("a.png", io.BytesIO(data))])
    record = manifest.all()[0]
    path = thumbnail(manifest, record)
    assert path == manifest.project_dir / "thumbs" / f"{record.id}.jpg"
    with Image.open(path) as thumb:
        assert thumb.format == "JPEG" and thumb.mode == "RGB" and thumb.size == (256, 128)
    mtime = path.stat().st_mtime_ns
    assert thumbnail(manifest, record) == path and path.stat().st_mtime_ns == mtime
    path.unlink()
    assert thumbnail(manifest, record).exists()


def test_report_as_dict_caps_lists():
    report = IngestReport(added=1, duplicates=[str(i) for i in range(300)])
    data = report.as_dict()
    assert len(data["duplicates"]) == 200 and data["duplicates_count"] == 300 and data["added"] == 1


def test_annotation_files_sent_with_images_are_listed(tmp_path):
    report = ingest_files(Manifest(tmp_path / "p"), [("labels.json", io.BytesIO(b"{}"))])
    assert report.other_files == ["labels.json"] and report.rejected == []


def test_report_ids_are_exactly_the_images_stored_in_order(tmp_path):
    manifest = Manifest(tmp_path / "p")
    first = _bytes(tmp_path, "a.png", seed=1, fmt="PNG")
    second = _bytes(tmp_path, "b.png", seed=2, fmt="PNG")
    report = ingest_files(
        manifest,
        [("b.png", io.BytesIO(second)), ("a.png", io.BytesIO(first)), ("dup.png", io.BytesIO(first))],
    )
    stored = list(manifest.load())
    assert report.ids == stored and len(report.ids) == 2 and report.duplicates == ["dup.png"]
    assert manifest.get(report.ids[0]).name == "b.png"
    again = ingest_files(manifest, [("a.png", io.BytesIO(first))])
    assert again.ids == []


class _NoSeekable:
    """An upload as Python 3.10 hands it over (SpooledTemporaryFile has no ``seekable``)."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()


def test_zip_upload_without_seekable(manifest, tmp_path):
    data = _zip({"a.png": _bytes(tmp_path, "a.png", seed=5, fmt="PNG")}).getvalue()
    assert not hasattr(_NoSeekable(data), "seekable")
    report = ingest_files(manifest, [("pics.zip", _NoSeekable(data))])
    assert report.added == 1 and not report.rejected


def test_the_content_decides_the_kind_of_a_file(manifest, tmp_path):
    png = _bytes(tmp_path, "p.png", seed=1, fmt="PNG")
    gif = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(gif, format="GIF")
    report = ingest_files(
        manifest, [("fake.jpg", io.BytesIO(gif.getvalue())), ("photo.jpg", io.BytesIO(png))]
    )
    assert report.rejected == [{"name": "fake.jpg", "reason": "unsupported"}] and report.added == 1
    stored = manifest.all()[0]
    assert stored.name == "photo.jpg" and stored.file.endswith(".png")  # stored as what it really is


def test_a_colon_in_a_name_is_fine_but_a_drive_path_is_not(manifest, tmp_path):
    files = [
        (name, io.BytesIO(_bytes(tmp_path, "a.jpg", seed=i)))
        for i, name in enumerate(["a:b.jpg", "C:/x.jpg", "d:y.jpg"])
    ]
    report = ingest_files(manifest, files)
    assert report.added == 2 and [r["name"] for r in report.rejected] == ["C:/x.jpg"]
    assert {r.name for r in manifest.all()} == {"a:b.jpg", "d:y.jpg"}


def test_a_huge_image_is_too_large_not_corrupt(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    report = ingest_files(manifest, [("big.jpg", io.BytesIO(_bytes(tmp_path, "big.jpg", seed=4)))])
    assert report.rejected == [{"name": "big.jpg", "reason": "too large"}]


def test_report_tells_which_image_each_file_became(manifest, tmp_path):
    a, b = _bytes(tmp_path, "a.jpg", seed=1), _bytes(tmp_path, "b.jpg", seed=2)
    report = ingest_files(
        manifest,
        [
            ("a.jpg", io.BytesIO(a)),
            ("copy.jpg", io.BytesIO(a)),
            ("bad.jpg", io.BytesIO(b"nope")),
            ("b.jpg", io.BytesIO(b)),
        ],
    )
    first, _, third, fourth = report.per_file[0], report.per_file[1], report.per_file[2], report.per_file[3]
    assert (
        report.per_file[1] == first and third is None and fourth == report.ids[1]
    )  # a duplicate points to the first
    assert len(report.per_file) == 4 and report.ids == [first, fourth]


def test_an_empty_file_and_an_archive_inside_an_archive_get_their_own_reasons(manifest, tmp_path):
    """A 0-byte file is "empty" (not "corrupt") and a zip in a zip is "nested archive" (#433)."""
    inner = _zip({"a.jpg": _bytes(tmp_path, "a.jpg", seed=3)}).getvalue()
    outer = _zip({"inner.zip": inner})
    report = ingest_files(manifest, [("zero.jpg", io.BytesIO(b"")), ("outer.zip", outer)])
    assert report.added == 0
    assert {(r["name"], r["reason"]) for r in report.rejected} == {
        ("zero.jpg", "empty"),
        ("inner.zip", "nested archive"),
    }


def _encoded(image: Image.Image, fmt: str, **kw) -> io.BytesIO:
    out = io.BytesIO()
    image.save(out, format=fmt, **kw)
    out.seek(0)
    return out


def test_colour_modes_wrong_extensions_and_damaged_files(manifest):
    """#620: CMYK, alpha and palette images are kept and look right; 16-bit grey is refused (it would turn
    white in RGB); a truncated file is corrupt; a PNG named .jpg is stored as .png."""
    cmyk = Image.new("CMYK", (40, 30), (255, 0, 255, 0))  # pure green in CMYK
    rgba = Image.new("RGBA", (40, 30), (200, 10, 10, 128))
    palette = Image.new("RGB", (40, 30), (10, 10, 200)).convert("P")
    grey16 = Image.fromarray(np.linspace(0, 65535, 40 * 30).reshape(30, 40).astype(np.uint16))
    full = _encoded(Image.new("RGB", (80, 60), (90, 90, 90)), "JPEG").getvalue()
    report = ingest_files(
        manifest,
        [
            ("cmyk.jpg", _encoded(cmyk, "JPEG")),
            ("alpha.png", _encoded(rgba, "PNG")),
            ("palette.png", _encoded(palette, "PNG")),
            ("deep.png", _encoded(grey16, "PNG")),
            ("cut.jpg", io.BytesIO(full[: len(full) // 2])),
            ("named.jpg", _encoded(Image.new("RGB", (40, 30), (1, 2, 3)), "PNG")),
        ],
    )
    assert {(r["name"], r["reason"]) for r in report.rejected} == {
        ("deep.png", "high bit depth"),
        ("cut.jpg", "corrupt"),
    }
    records = {r.name: r for r in manifest.all()}
    assert set(records) == {"cmyk.jpg", "alpha.png", "palette.png", "named.jpg"}
    assert records["named.jpg"].file.endswith(".png")
    with Image.open(thumbnail(manifest, records["cmyk.jpg"])) as thumb:
        r, g, b = thumb.convert("RGB").getpixel((20, 15))
        assert g > 200 and r < 60 and b < 60
