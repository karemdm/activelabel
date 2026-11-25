"""The version comes from the installed package metadata, with a fallback for an uninstalled source folder."""

import importlib
import importlib.metadata

import pytest

import activelabel


def _not_installed(name: str) -> str:
    """Stand-in for ``importlib.metadata.version`` when the package is not installed.

    Args:
        name: distribution name.

    Returns:
        Never returns.

    Raises:
        PackageNotFoundError: always.
    """
    raise importlib.metadata.PackageNotFoundError(name)


def _reload_with(monkeypatch: pytest.MonkeyPatch, fake: object) -> str:
    """Reload the package with a replaced ``importlib.metadata.version`` and restore it afterwards.

    Args:
        monkeypatch: pytest fixture.
        fake: the replacement function.

    Returns:
        The ``__version__`` the reloaded package computed.
    """
    monkeypatch.setattr(importlib.metadata, "version", fake)
    try:
        return importlib.reload(activelabel).__version__
    finally:
        monkeypatch.undo()
        importlib.reload(activelabel)


def test_the_package_has_a_version_string():
    assert isinstance(activelabel.__version__, str)
    assert activelabel.__version__


def test_the_version_comes_from_the_metadata(monkeypatch):
    assert _reload_with(monkeypatch, lambda name: "9.8.7") == "9.8.7"


def test_the_version_falls_back_when_the_package_is_not_installed(monkeypatch):
    assert _reload_with(monkeypatch, _not_installed) == activelabel.FALLBACK_VERSION
