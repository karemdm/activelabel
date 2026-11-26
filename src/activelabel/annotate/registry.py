"""Annotation tools by name."""

from __future__ import annotations

from activelabel.annotate.base import AnnotationTool, ToolError
from activelabel.annotate.fiftyone_tool import FiftyOneTool
from activelabel.optional import FIFTYONE, FIFTYONE_INSTALL

OVERRIDES: dict[str, AnnotationTool] = {}  # tests register fakes here


def get_tool(name: str) -> AnnotationTool:
    """The tool called ``name``.

    Args:
        name: name of the tool (``fiftyone``, or a fake registered in :data:`OVERRIDES`).

    Returns:
        The tool.

    Raises:
        ToolError: unknown or not available tool (``fiftyone_not_installed`` without FiftyOne).
    """
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name == "fiftyone" and not FIFTYONE:
        raise ToolError(
            f"FiftyOne is not installed. Add it with: {FIFTYONE_INSTALL}",
            "fiftyone_not_installed",
            install=FIFTYONE_INSTALL,
        )
    if name == "fiftyone":
        return FiftyOneTool()
    raise ToolError(f"The tool '{name}' is not available.")
