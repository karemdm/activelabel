"""Load a training worker script as a module (the workers need torch: skipped where it is absent)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

WORKERS = Path(__file__).resolve().parents[2] / "src" / "activelabel" / "training" / "workers"


def load_worker(name: str) -> ModuleType:
    """The worker ``name`` (a file of ``training/workers``) imported under a private module name.

    Args:
        name: file name without ``.py``.

    Returns:
        The module; the test is skipped when torch is not installed.
    """
    pytest.importorskip("torch", reason="the training workers import torch, absent from the app environment")
    spec = importlib.util.spec_from_file_location(f"{name}_worker_under_test", WORKERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
