"""Annotation tools by name."""

from __future__ import annotations

from activelabel.annotate.base import AnnotationTool, ToolError
from activelabel.annotate.fiftyone_tool import FiftyOneTool

OVERRIDES: dict[str, AnnotationTool] = {}  # tests register fakes here


def get_tool(name: str) -> AnnotationTool:
    """The tool called ``name``.

    Raises:
        ToolError: unknown or not available tool.
    """
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name == "fiftyone":
        return FiftyOneTool()
    raise ToolError(f"The tool '{name}' is not available.")
