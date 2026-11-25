import io
import os
import time
import zipfile

import pytest

from activelabel.annotations import discard_staged, load_staged, stage
from activelabel.annotations.staging import MAX_AGE_SECONDS
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
