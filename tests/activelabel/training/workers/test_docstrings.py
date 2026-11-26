"""The workers' module docstrings describe their imports as they are (imports at the top, issue #306/#556)."""

from __future__ import annotations

from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[4] / "src" / "activelabel" / "training" / "workers"


@pytest.mark.parametrize("name", ["classification", "detection", "segmentation"])
def test_module_docstring_does_not_claim_imports_inside_functions(name: str) -> None:
    """The torch workers import torch at the top of the module, so their docstring must not say otherwise."""
    text = (WORKERS / f"{name}.py").read_text(encoding="utf-8")
    docstring = text.split('"""')[1]
    assert "inside the functions" not in docstring.replace("\n", " ")
    assert "import torch\n" in text
