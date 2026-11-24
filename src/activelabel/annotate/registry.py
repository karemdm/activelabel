"""Annotation tools by name."""

from __future__ import annotations

import importlib
from typing import Any

from activelabel.annotate.base import AnnotationTool, ToolError

OVERRIDES: dict[str, AnnotationTool] = {}  # tests register fakes here
_LAZY = {
    "labelstudio": ("activelabel.annotate.labelstudio_tool", "LabelStudioTool"),
}


def get_tool(name: str) -> AnnotationTool:
    """The tool called ``name``.

    Raises:
        ToolError: unknown or not available tool.
    """
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name == "fiftyone":
        from activelabel.annotate.fiftyone_tool import FiftyOneTool

        return FiftyOneTool()
    if name not in _LAZY:
        raise ToolError(f"The tool '{name}' is not available.")
    module, cls = _LAZY[name]
    try:
        factory: Any = getattr(importlib.import_module(module), cls)
        tool: AnnotationTool = factory()
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"The tool '{name}' is not available.") from exc
    return tool
