import os as _os

import pytest

from activelabel import storage


def test_replace_retried_while_windows_reports_file_in_use(tmp_path, monkeypatch):
    real, calls = _os.replace, []

    def flaky(src, dst):
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
