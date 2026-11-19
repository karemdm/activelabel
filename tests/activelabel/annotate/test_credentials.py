import json

from activelabel.annotate import credentials as cred


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
    cred.save_settings(tmp_path, "labelstudio", "http://ls", "", "tok")
    cred.save_settings(tmp_path, "labelstudio", "http://ls", "", None)
    assert cred.load_settings(tmp_path, "labelstudio").secret == "tok"


def test_needs_and_configured(tmp_path):
    assert cred.public_view(cred.load_settings(tmp_path, "fiftyone")) == {
        "tool": "fiftyone",
        "configured": True,
        "url": "",
        "username": "",
        "needs": [],
    }
    cred.save_settings(tmp_path, "labelstudio", "http://ls", "", None)
    view = cred.public_view(cred.load_settings(tmp_path, "labelstudio"))
    assert view["configured"] is False and view["needs"] == ["url", "secret"]


def test_forget(tmp_path, fake_keyring):
    cred.save_settings(tmp_path, "cvat", "http://c", "u", "p")
    assert cred.forget(tmp_path, "cvat") is True
    assert fake_keyring.store == {} and cred.load_settings(tmp_path, "cvat").url == ""
    assert cred.forget(tmp_path, "cvat") is False
