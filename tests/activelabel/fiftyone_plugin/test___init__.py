import os
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
