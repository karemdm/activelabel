"""Annotation tools by name."""

from __future__ import annotations

import importlib
from typing import Any

from activelabel.annotate.base import TOOLS, AnnotationTool, ToolError

OVERRIDES: dict[str, AnnotationTool] = {}  # tests register fakes here
_LAZY = {
    "cvat": ("activelabel.annotate.cvat_tool", "CvatTool"),
    "labelstudio": ("activelabel.annotate.labelstudio_tool", "LabelStudioTool"),
}


def get_tool(name: str) -> AnnotationTool:
    """The tool called ``name``.

    Raises:
        ToolError: unknown or not available tool.
    """
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name not in TOOLS:
        raise ToolError(f"Unknown annotation tool: {name}")
    if name == "fiftyone":
        from activelabel.annotate.fiftyone_tool import FiftyOneTool

        return FiftyOneTool()
    module, cls = _LAZY[name]
    try:
        factory: Any = getattr(importlib.import_module(module), cls)
        tool: AnnotationTool = factory()
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"The tool '{name}' is not available.") from exc
    return tool
