"""Every module of the package imports.

Importing each module catches a moved, renamed or circular import before a user does. The training workers
run in the managed training environment and need torch: they are skipped where it is absent.
"""

import importlib
import importlib.util
import pkgutil

import pytest

import activelabel


def _module_names() -> list[str]:
    return sorted(info.name for info in pkgutil.walk_packages(activelabel.__path__, prefix="activelabel."))


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name):
    if name.startswith("activelabel.training.workers.") and importlib.util.find_spec("torch") is None:
        # the workers run in the managed training environment and import torch at the top
        pytest.skip("torch is not installed in the app environment (the training workers need it)")
    importlib.import_module(name)


def test_the_walk_finds_the_package():
    names = _module_names()
    assert len(names) > 50
    assert "activelabel.web.app" in names
