"""Every module of the package imports.

Many modules import lazily inside functions, so a broken import there only shows up at run time.
Importing each module here catches a moved or renamed module before a user does.
"""

import importlib
import pkgutil

import pytest

import activelabel


def _module_names() -> list[str]:
    # the training workers import torch inside their functions, so they import without it too
    return sorted(info.name for info in pkgutil.walk_packages(activelabel.__path__, prefix="activelabel."))


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name):
    importlib.import_module(name)


def test_the_walk_finds_the_package():
    names = _module_names()
    assert len(names) > 50
    assert "activelabel.web.app" in names
