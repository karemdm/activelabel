import json

import pytest

from activelabel.annotate import credentials
from activelabel.annotate import credentials as cred
from activelabel.annotate.base import ToolError
from activelabel.errors import CodedError


def test_secret_only_in_keyring(tmp_path, fake_keyring):
    saved = cred.save_settings(tmp_path, "cvat", "https://cvat.example/", "ann", "s3cret-value")
    assert saved.url == "https://cvat.example" and saved.secret == "s3cret-value"
    assert "s3cret-value" not in (tmp_path / "tools.json").read_text()
    assert json.loads((tmp_path / "tools.json").read_text()) == {
        "cvat": {"url": "https://cvat.example", "username": "ann"}
    }
    assert fake_keyring.store == {("activelabel", "cvat:https://cvat.example"): "s3cret-value"}
    view = cred.public_view(cred.load_settings(tmp_path, "cvat"))
    assert "s3cret-value" not in json.dumps(view)
    assert view == {
        "tool": "cvat",
        "configured": True,
        "url": "https://cvat.example",
        "username": "ann",
        "needs": ["url", "username", "secret"],
    }


def test_secret_kept_when_not_given(tmp_path):
    cred.save_settings(tmp_path, "cvat", "http://c", "u", "tok")
    cred.save_settings(tmp_path, "cvat", "http://c", "u", None)
    assert cred.load_settings(tmp_path, "cvat").secret == "tok"


def test_needs_and_configured(tmp_path):
    assert cred.public_view(cred.load_settings(tmp_path, "fiftyone")) == {
        "tool": "fiftyone",
        "configured": True,
        "url": "",
        "username": "",
        "needs": [],
    }
    cred.save_settings(tmp_path, "cvat", "http://c", "u", None)
    view = cred.public_view(cred.load_settings(tmp_path, "cvat"))
    assert view["configured"] is False and view["needs"] == ["url", "username", "secret"]


def test_an_old_workspace_with_settings_of_a_removed_tool_keeps_working(tmp_path, fake_keyring):
    """``tools.json`` may still hold an entry (and the keyring a secret) of a tool that is gone."""
    (tmp_path / "tools.json").write_text(
        '{"labelstudio": {"url": "http://ls", "username": ""}}', encoding="utf-8"
    )
    fake_keyring.store[("activelabel", "labelstudio:http://ls")] = "old-token"
    with pytest.raises(ToolError, match="Unknown annotation tool"):
        cred.load_settings(tmp_path, "labelstudio")
    cred.save_settings(tmp_path, "cvat", "http://c", "u", "pw")  # saving another tool does not break it
    assert cred.load_settings(tmp_path, "cvat").url == "http://c"
    assert fake_keyring.store[("activelabel", "labelstudio:http://ls")] == "old-token"  # never touched


def test_forget(tmp_path, fake_keyring):
    cred.save_settings(tmp_path, "cvat", "http://c", "u", "p")
    assert cred.forget(tmp_path, "cvat") is True
    assert fake_keyring.store == {} and cred.load_settings(tmp_path, "cvat").url == ""
    assert cred.forget(tmp_path, "cvat") is False


def test_tool_urls_must_be_http(tmp_path, monkeypatch):
    for bad in ("file:///etc/passwd", "ftp://host", "javascript:alert(1)", "cvat.local", "http://"):
        with pytest.raises(CodedError) as info:
            credentials.save_settings(tmp_path, "cvat", bad, "u", None)
        assert info.value.code == "url_invalid"
    assert not (tmp_path / "tools.json").exists()
    saved = credentials.save_settings(tmp_path, "cvat", "https://cvat.local:8080/", "u", None)
    assert saved.url == "https://cvat.local:8080"
