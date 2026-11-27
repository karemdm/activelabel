import logging
import os
from importlib import metadata
from pathlib import Path

import pytest

from activelabel import fiftyone_plugin

PLUGIN = fiftyone_plugin.PLUGINS_DIR / "activelabel_annotate"


class FakeConfig:
    plugins_dir = "the-persons-own"


class FakeFiftyOne:
    config = FakeConfig()


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(fiftyone_plugin, "fo", FakeFiftyOne())
    monkeypatch.setattr(FakeFiftyOne, "config", FakeConfig())
    monkeypatch.delenv("FIFTYONE_PLUGINS_DIR", raising=False)


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
