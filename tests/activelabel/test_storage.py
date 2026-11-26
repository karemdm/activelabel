"""Tests of the JSON storage helpers."""

import json
import os as _os
from dataclasses import dataclass

import pytest

from activelabel import storage


def test_replace_retried_while_windows_reports_file_in_use(tmp_path, monkeypatch):
    real, calls = _os.replace, []

    def flaky(src, dst):
        if not str(dst).endswith("a.json"):  # the work queue may write its own files
            return real(src, dst)
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(storage.os, "replace", flaky)
    monkeypatch.setattr(storage, "RETRY_DELAY", 0)
    storage.write_json(tmp_path / "a.json", {"x": 1})
    assert storage.read_json(tmp_path / "a.json") == {"x": 1} and len(calls) == 3


def test_damaged_json_names_the_file(tmp_path):
    bad = tmp_path / "annotations.json"
    bad.write_text('{"images": ', encoding="utf-8")
    with pytest.raises(ValueError, match="annotations.json is damaged"):
        storage.read_json(bad)
    assert storage.read_json(tmp_path / "missing.json", 5) == 5


@dataclass
class _Box:
    name: str
    size: int = 0


def test_stamped_adds_the_version_first_to_objects_only(tmp_path):
    out = storage.stamped(tmp_path / "tasks.json", {"a": 1})
    assert list(out) == [storage.FORMAT_KEY, "a"] and out[storage.FORMAT_KEY] == storage.FORMAT_VERSION
    assert storage.stamped(tmp_path / "images.json", [1, 2]) == [1, 2]
    assert storage.stamped(tmp_path / "n.json", 7) == 7


def test_stamped_keeps_an_existing_version_and_unversioned_files(tmp_path):
    data = {storage.FORMAT_KEY: 5, "a": 1}
    assert storage.stamped(tmp_path / "tasks.json", data) is data
    for name in storage.UNVERSIONED_FILES:
        plain = {"a": 1}
        assert storage.stamped(tmp_path / name, plain) is plain


def test_write_json_puts_the_version_first_and_skips_unversioned_files(tmp_path):
    storage.write_json(tmp_path / "tasks.json", {"next_id": 1, "tasks": []})
    assert list(json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))) == [
        storage.FORMAT_KEY,
        "next_id",
        "tasks",
    ]
    for name in ("project.json", "links.json", "pulled.json"):
        storage.write_json(tmp_path / name, {"a": 1})
        assert json.loads((tmp_path / name).read_text(encoding="utf-8")) == {"a": 1}


def test_read_json_drops_the_version_of_any_release_and_keeps_unknown_keys(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({storage.FORMAT_KEY: 99, "next_id": 1, "future": [1]}), encoding="utf-8")
    assert storage.read_json(path) == {"next_id": 1, "future": [1]}
    path.write_text("[1, 2]", encoding="utf-8")
    assert storage.read_json(path) == [1, 2]


def test_known_fields_drops_unknown_keys():
    assert storage.known_fields(_Box, {"name": "a", "size": 2, "extra": 1}) == {"name": "a", "size": 2}
