"""The version check of the built wheel (tests/package/check_version.py)."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("check_version", Path(__file__).with_name("check_version.py"))
assert _SPEC is not None and _SPEC.loader is not None
check_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_version)


def _wheel(folder: Path, version: str | None) -> Path:
    """Write a minimal wheel whose METADATA has the given version.

    Args:
        folder: where to write it.
        version: the ``Version`` field, or ``None`` for a METADATA without it.

    Returns:
        The path of the wheel.
    """
    path = folder / "activelabel-x-py3-none-any.whl"
    field = f"Version: {version}\n" if version is not None else ""
    metadata = f"Metadata-Version: 2.4\nName: activelabel\n{field}"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("activelabel-x.dist-info/METADATA", metadata)
    return path


@pytest.mark.parametrize(
    ("found", "expected", "no_tag_yet", "ok"),
    [
        ("0.1.0", None, False, True),
        ("0.1.0", "0.1.0", False, True),
        ("0.1.0", "0.2.0", False, False),
        ("0.0", None, False, False),
        ("0.0", None, True, True),
        ("0.0", "0.1.0", True, False),
        ("0.0.0+unknown", None, False, False),
        ("0.0.0+unknown", None, True, True),  # no tag yet: the only-version scheme falls back
        ("0.0.0+unknown", "0.1.0", True, False),
    ],
)
def test_problem(found, expected, no_tag_yet, ok):
    assert (check_version.problem(found, expected, no_tag_yet) is None) is ok


def test_main_reads_the_wheel(tmp_path, capsys):
    wheel = _wheel(tmp_path, "0.1.0")
    assert check_version.main([str(wheel), "0.1.0"]) == 0
    assert "wheel version: 0.1.0" in capsys.readouterr().out


def test_main_refuses_an_untagged_build_unless_told_there_is_no_tag(tmp_path):
    wheel = _wheel(tmp_path, "0.0")
    assert check_version.main([str(wheel)]) == 1
    assert check_version.main([str(wheel), "--no-tag-yet"]) == 0


def test_a_wheel_without_version_stops(tmp_path):
    with pytest.raises(SystemExit, match="no Version field"):
        check_version.wheel_version(_wheel(tmp_path, None))


def test_a_file_without_metadata_stops(tmp_path):
    path = tmp_path / "empty.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.txt", "")
    with pytest.raises(SystemExit, match="no METADATA"):
        check_version.wheel_version(path)
