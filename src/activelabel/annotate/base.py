"""Shared contract of the annotation tools (see docs/ANNOTATION.md).

A tool takes the project's FiftyOne dataset (already refreshed from the project, with the
current ground truth in ``ground_truth``) and a list of image ids, lets people edit the
annotations somewhere, and gives them back as internal labels. The session bookkeeping, the
diff and the apply step are the same for every tool and live in :mod:`activelabel.annotate.sessions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from activelabel.labels import Label
from activelabel.projects import Project

TOOLS = ("fiftyone", "cvat", "labelstudio")
GROUND_TRUTH = "ground_truth"


class ToolError(RuntimeError):
    """A readable problem with a tool (bad login, server not reachable, task deleted...)."""


@dataclass
class ToolSettings:
    """How to reach a tool. ``secret`` (password or API token) comes from the keyring only."""

    tool: str
    url: str = ""
    username: str = ""
    secret: str = field(default="", repr=False)


@dataclass
class Sent:
    """What :meth:`AnnotationTool.send` returns: links to open, and the tool's own reference."""

    links: list[dict[str, str]]  # [{"label": "...", "url": "..."}]
    ref: dict[str, Any]  # e.g. {"anno_key": "..."} or {"view": "..."}; stored in the session


class AnnotationTool(Protocol):
    """One annotation tool. Implementations must not print secrets or write them to files."""

    name: str

    def check(self, settings: ToolSettings) -> str:
        """Test the connection and login; return a short readable message. Raises ToolError."""
        ...

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Make the images (``dataset`` samples whose ``image_id`` is in ``image_ids``) editable in the tool.

        The current ground truth is pre-loaded. Raises ToolError.
        """
        ...

    def pull(
        self,
        dataset: Any,
        image_ids: list[str],
        project: Project,
        settings: ToolSettings,
        ref: dict[str, Any],
    ) -> dict[str, list[Label]]:
        """Read back the annotations of those images as internal labels: ``{image_id: [Label, ...]}``.

        Every image of the session must be present (an empty list = no annotations). Raises ToolError.
        """
        ...

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any], delete_remote: bool) -> None:
        """Forget the session in the tool (and delete remote tasks if asked). Never raises if gone."""
        ...
