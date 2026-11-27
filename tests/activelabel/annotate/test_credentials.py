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
        "format_version": 2,
        "cvat": {"url": "https://cvat.example", "username": "ann"},
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
    for bad in (
        "file:///etc/passwd",
        "ftp://host",
        "javascript:alert(1)",
        "cvat.local",
        "http://",
        "http://127.0.0.1:8113  /x y",
        "http://my host",
    ):
        with pytest.raises(CodedError) as info:
            credentials.save_settings(tmp_path, "cvat", bad, "u", None)
        assert info.value.code == "url_invalid"
    assert not (tmp_path / "tools.json").exists()
    saved = credentials.save_settings(tmp_path, "cvat", "  https://cvat.local:8080/  ", "u", None)
    assert saved.url == "https://cvat.local:8080"  # surrounding spaces and the last slash are dropped


def test_tool_url_with_user_or_password_is_refused_without_echo(tmp_path):
    for bad in ("http://ana:s3cret@cvat.local", "https://ana@cvat.local:8080/"):
        with pytest.raises(CodedError) as info:
            credentials.save_settings(tmp_path, "cvat", bad, "u", None)
        assert info.value.code == "url_has_credentials"
        assert "s3cret" not in str(info.value) and "s3cret" not in str(info.value.params)
    assert not (tmp_path / "tools.json").exists()


class _ForbiddenKeyring:
    def __getattr__(self, name):
        raise AssertionError("the keyring was used")


def test_server_url_is_read_without_the_keyring(tmp_path, fake_keyring):
    assert cred.server_url(tmp_path, "cvat") == ""  # nothing saved
    cred.save_settings(tmp_path, "cvat", "https://cvat.example///", "ann", "tok")
    fake_keyring.store.clear()  # a keyring with no secret still gives the address
    assert cred.server_url(tmp_path, "cvat") == "https://cvat.example"
    assert cred.server_url(tmp_path, "other") == ""


def test_server_url_never_touches_the_keyring(tmp_path, monkeypatch):
    cred.save_settings(tmp_path, "cvat", "http://c/", "u", None)

    monkeypatch.setattr(credentials, "keyring", _ForbiddenKeyring())
    assert cred.server_url(tmp_path, "cvat") == "http://c"


def test_server_url_of_a_damaged_entry_is_empty(tmp_path):
    (tmp_path / "tools.json").write_text('{"cvat": "oops"}', encoding="utf-8")
    assert cred.server_url(tmp_path, "cvat") == ""


def test_a_new_address_keeps_the_saved_secret_when_the_field_is_left_empty(tmp_path, fake_keyring):
    """#634: the keyring key holds the address, so a new address used to lose the secret."""
    cred.save_settings(tmp_path, "cvat", "http://127.0.0.1:8199", "u", "tok")
    saved = cred.save_settings(tmp_path, "cvat", "http://127.0.0.1:8132/", "u", None)
    assert saved.url == "http://127.0.0.1:8132" and saved.secret == "tok"
    assert fake_keyring.store == {
        ("activelabel", "cvat:http://127.0.0.1:8132"): "tok"
    }  # the old entry is gone


def test_a_new_address_with_a_new_secret_takes_the_new_one(tmp_path, fake_keyring):
    cred.save_settings(tmp_path, "cvat", "http://a", "u", "old")
    assert cred.save_settings(tmp_path, "cvat", "http://b", "u", "new").secret == "new"
    assert (
        fake_keyring.store[("activelabel", "cvat:http://a")] == "old"
    )  # only a moved secret leaves its old key


def test_a_secret_already_saved_for_the_new_address_is_kept(tmp_path, fake_keyring):
    cred.save_settings(tmp_path, "cvat", "http://b", "u", "for-b")
    cred.save_settings(tmp_path, "cvat", "http://a", "u", "for-a")
    assert cred.save_settings(tmp_path, "cvat", "http://b", "u", None).secret == "for-b"


def test_a_keyring_that_cannot_delete_the_old_entry_does_not_fail_the_save(
    tmp_path, fake_keyring, monkeypatch
):
    cred.save_settings(tmp_path, "cvat", "http://a", "u", "tok")
    monkeypatch.setattr(fake_keyring, "delete_password", refuse)
    assert cred.save_settings(tmp_path, "cvat", "http://b", "u", None).secret == "tok"


def refuse(service, key):
    raise RuntimeError("keyring locked")
