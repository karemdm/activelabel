import logging
import os
from importlib import metadata
from pathlib import Path

import pytest

from activelabel import cli, fiftyone_plugin
from activelabel.annotate.fiftyone_tool import view_name
from activelabel.constants import SERVER_URL_ENV
from activelabel.fiftyone_plugin.activelabel_review import VIEW_PREFIX

PLUGIN = fiftyone_plugin.PLUGINS_DIR / "activelabel_annotate"


class FakeConfig:
    plugins_dir = "the-persons-own"


class FakeFiftyOne:
    config = FakeConfig()


class FakeDataset:
    def __init__(self, info):
        self.info = info
        self.saved = 0

    def save(self):
        self.saved += 1


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(fiftyone_plugin, "fo", FakeFiftyOne())
    monkeypatch.setattr(FakeFiftyOne, "config", FakeConfig())
    monkeypatch.delenv("FIFTYONE_PLUGINS_DIR", raising=False)
    monkeypatch.delenv(SERVER_URL_ENV, raising=False)


def test_the_annotate_plugin_has_its_manifest_and_its_script():
    manifest = (PLUGIN / "fiftyone.yml").read_text(encoding="utf-8")
    assert 'name: "@activelabel/annotate"' in manifest
    assert 'js_bundle: "dist/index.umd.js"' in manifest
    script = (PLUGIN / "dist" / "index.umd.js").read_text(encoding="utf-8")
    assert "open_in_annotate" in script
    assert "activelabel_editable" in script


def test_the_plugin_files_are_in_the_package_data():
    pyproject = (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    assert "fiftyone_plugin/*/fiftyone.yml" in pyproject
    assert "fiftyone_plugin/*/dist/*.js" in pyproject


def test_the_plugin_folder_is_for_this_process_only(isolated):
    fiftyone_plugin.use_plugins()
    assert os.environ["FIFTYONE_PLUGINS_DIR"] == str(fiftyone_plugin.PLUGINS_DIR)
    assert FakeFiftyOne.config.plugins_dir == str(fiftyone_plugin.PLUGINS_DIR)


def test_the_installed_fiftyone_is_the_one_the_plugin_was_checked_with():
    """An upgrade of FiftyOne must be checked in a browser (images open in Annotate), then pinned here."""
    pytest.importorskip("fiftyone")
    assert metadata.version("fiftyone") == fiftyone_plugin.VERIFIED_FIFTYONE


def test_another_fiftyone_version_is_logged(isolated, monkeypatch, caplog):
    monkeypatch.setattr(fiftyone_plugin, "installed_fiftyone", lambda: "9.0.0")
    with caplog.at_level(logging.WARNING, logger=fiftyone_plugin.__name__):
        fiftyone_plugin.use_plugins()
    assert "9.0.0" in caplog.text and fiftyone_plugin.VERIFIED_FIFTYONE in caplog.text


def test_the_verified_version_logs_nothing(isolated, monkeypatch, caplog):
    monkeypatch.setattr(fiftyone_plugin, "installed_fiftyone", lambda: fiftyone_plugin.VERIFIED_FIFTYONE)
    with caplog.at_level(logging.WARNING, logger=fiftyone_plugin.__name__):
        fiftyone_plugin.use_plugins()
    assert caplog.text == ""


def test_no_fiftyone_installed_reads_as_none(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(fiftyone_plugin.metadata, "version", missing)
    assert fiftyone_plugin.installed_fiftyone() is None


def test_the_server_address_is_recorded_on_the_dataset(isolated, monkeypatch):
    monkeypatch.setenv(SERVER_URL_ENV, "http://127.0.0.1:8167")
    dataset = FakeDataset({"project_id": 4})
    fiftyone_plugin.record_server(dataset)
    assert dataset.info == {"project_id": 4, fiftyone_plugin.URL_KEY: "http://127.0.0.1:8167"}
    assert dataset.saved == 1
    fiftyone_plugin.record_server(dataset)  # the same address: nothing to save again
    assert dataset.saved == 1


def test_without_an_announced_server_the_dataset_is_left_alone(isolated):
    dataset = FakeDataset({"project_id": 4})
    fiftyone_plugin.record_server(dataset)
    assert dataset.info == {"project_id": 4}
    assert dataset.saved == 0


def test_the_serve_command_announces_its_address(monkeypatch):
    monkeypatch.setenv(SERVER_URL_ENV, "")  # restored by monkeypatch at the end
    monkeypatch.setattr(cli, "create_app", lambda: "the-app")
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: None)
    cli.main(["serve", "--host", "0.0.0.0", "--port", "8167"])
    assert os.environ[SERVER_URL_ENV] == "http://127.0.0.1:8167"


def test_the_session_view_name_matches_the_one_the_plugin_reads():
    assert view_name("abc123") == f"{VIEW_PREFIX}abc123"


def test_the_review_plugin_has_its_manifest_and_its_operator():
    plugin = fiftyone_plugin.PLUGINS_DIR / "activelabel_review"
    assert (plugin / "__init__.py").is_file()
    assert "send_to_activelabel" in (plugin / "fiftyone.yml").read_text(encoding="utf-8")
