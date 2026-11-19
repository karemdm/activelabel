from pathlib import Path

from visionloop.config import Settings


def test_from_env(tmp_path):
    settings = Settings.from_env({"VISIONLOOP_WORKSPACE": str(tmp_path), "VISIONLOOP_LANG": "pt"})
    assert settings.workspace == tmp_path and settings.language == "pt"
    assert settings.projects_dir == tmp_path / "projects"
    assert settings.import_root is None
    assert (
        Settings.from_env({"VISIONLOOP_IMPORT_ROOT": str(tmp_path)}).import_root == Path(tmp_path).resolve()
    )
