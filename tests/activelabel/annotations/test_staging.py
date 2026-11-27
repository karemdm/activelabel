import functools
import io
import os
import shutil
import time
import zipfile

import pytest

from activelabel.annotations import discard_staged, load_staged, pending_report, set_report, stage
from activelabel.annotations.staging import MAX_AGE_SECONDS, REPORT_SUFFIX
from activelabel.errors import CodedError


def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    buffer.seek(0)
    return buffer


def test_stage_zip_and_filtering(tmp_path):
    folder = tmp_path / "p"
    token = stage(
        folder,
        [
            (
                "set.zip",
                zip_bytes(
                    {"ds/data.yaml": "names: [a]", "ds/labels/x.txt": "0 .5 .5 .1 .1", "ds/img.jpg": "zz"}
                ),
            )
        ],
    )
    files = load_staged(folder, token)
    assert [p.name for p in files] == ["data.yaml", "x.txt"]
    discard_staged(folder, token)
    assert not (folder / "staging" / token).exists()
    discard_staged(folder, token)  # already gone: no error


def test_stage_zip_slip_refused(tmp_path):
    folder = tmp_path / "p"
    for evil in ("../evil.txt", "/abs.txt", "a/../../evil.txt"):
        with pytest.raises(ValueError, match="Unsafe"):
            stage(folder, [("bad.zip", zip_bytes({"ok.json": "[]", evil: "x"}))])
    assert not (tmp_path / "evil.txt").exists()
    assert list((folder / "staging").iterdir()) == []  # nothing left behind
    with pytest.raises(ValueError, match="Unsafe"):
        stage(folder, [("../x.json", io.BytesIO(b"[]"))])
    with pytest.raises(ValueError, match="No annotation"):
        stage(folder, [("photo.jpg", io.BytesIO(b"x"))])
    with pytest.raises(ValueError, match="token"):
        load_staged(folder, "../..")


def test_old_staging_is_removed(tmp_path):
    folder = tmp_path / "p"
    old = stage(folder, [("a.json", io.BytesIO(b"[]"))])
    recent = stage(folder, [("b.json", io.BytesIO(b"[]"))])
    past = time.time() - MAX_AGE_SECONDS - 60
    os.utime(folder / "staging" / old, (past, past))
    stage(folder, [("c.json", io.BytesIO(b"[]"))])
    assert not (folder / "staging" / old).exists() and (folder / "staging" / recent).exists()


def test_images_outside_class_folders_are_not_an_annotation_upload(tmp_path):
    folder = tmp_path / "p"
    with pytest.raises(ValueError, match="No annotation"):
        stage(folder, [("fotos.zip", zip_bytes({"a/1.png": "x", "a/2.png": "x"}))])  # one class only
    with pytest.raises(ValueError, match="No annotation"):
        stage(folder, [("fotos.zip", zip_bytes({"1.png": "x", "2.png": "x"}))])
    token = stage(folder, [("set.zip", zip_bytes({"c.json": "[]", "a/1.png": "x", "b/2.png": "x"}))])
    assert [p.name for p in load_staged(folder, token)] == ["c.json"]  # annotation files win over images


def test_an_invalid_staging_token_has_a_code(tmp_path):
    with pytest.raises(CodedError) as info:
        load_staged(tmp_path / "p", "../..")
    assert info.value.code == "staging_token_invalid"


def _stat_of_a_discarded_folder(real_stat, token, path, *args, **kwargs):
    """``Path.stat``, except that the report of ``token`` is discarded (by another request) just before."""
    if path.name == f"{token}{REPORT_SUFFIX}":
        discard_staged(path.parent.parent, token)
    return real_stat(path, *args, **kwargs)


def test_pending_report_skips_a_staging_folder_removed_while_it_reads(tmp_path, monkeypatch):
    old = stage(tmp_path, [("a.json", io.BytesIO(b"{}"))])
    new = stage(tmp_path, [("b.json", io.BytesIO(b"{}"))])
    set_report(tmp_path, old, {"token": old})
    set_report(tmp_path, new, {"token": new})
    stat = functools.partial(_stat_of_a_discarded_folder, type(tmp_path).stat, new)
    monkeypatch.setattr(type(tmp_path), "stat", lambda path, *a, **k: stat(path, *a, **k))
    assert pending_report(tmp_path) == {"token": old}


def test_an_uploaded_file_named_like_the_report_is_kept_and_does_not_clash_with_it(tmp_path):
    token = stage(
        tmp_path,
        [
            (".report.json", io.BytesIO(b"[]")),
            ("set.zip", zip_bytes({"sub/.report.json": "[]", "c.json": "[]"})),
        ],
    )
    set_report(tmp_path, token, {"token": token})
    names = sorted(
        p.relative_to(tmp_path / "staging" / token).as_posix() for p in load_staged(tmp_path, token)
    )
    assert names == [".report.json", "c.json", "sub/.report.json"]
    assert pending_report(tmp_path) == {"token": token}
    discard_staged(tmp_path, token)
    assert pending_report(tmp_path) is None and not list((tmp_path / "staging").iterdir())


def test_a_report_whose_files_expired_is_not_pending(tmp_path):
    token = stage(tmp_path, [("a.json", io.BytesIO(b"{}"))])
    set_report(tmp_path, token, {"token": token})
    shutil.rmtree(tmp_path / "staging" / token)
    assert pending_report(tmp_path) is None
