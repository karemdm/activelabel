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

TOOLS = ("fiftyone", "cvat")
GROUND_TRUTH = "ground_truth"


class ToolError(RuntimeError):
    """A readable problem with a tool (bad login, server not reachable, task deleted...).

    The message is English (API and logs). An error the interface should show in its own language also
    carries a stable ``code`` and the ``params`` its translated text needs (``error.<code>`` in the
    i18n files); one without a code is shown with its message.

    Attributes:
        code: stable identifier such as ``cvat_unreachable``, or an empty text.
        params: values for the translated text, e.g. ``{"status": 500}``.
    """

    def __init__(self, message: str, code: str = "", **params: Any) -> None:
        super().__init__(message)
        self.code = code
        self.params = params


def need_dataset(dataset: Any) -> None:
    """Refuse to go on without the project's FiftyOne dataset.

    Args:
        dataset: the project's FiftyOne dataset, or None when it was never created.

    Raises:
        ToolError: ``dataset`` is None (the project was never opened in FiftyOne).
    """
    if dataset is None:
        raise ToolError("Open the project in FiftyOne first.")


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
        """Test the connection and login.

        Args:
            settings: URL, user and secret of the tool.

        Returns:
            A short readable message.

        Raises:
            ToolError: the tool cannot be reached or the login is refused.
        """
        ...

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Make the images (``dataset`` samples whose ``image_id`` is in ``image_ids``) editable in the tool.

        The current ground truth is pre-loaded.

        Args:
            dataset: the project's FiftyOne dataset.
            image_ids: ids of the images of the session.
            project: the project (task and classes).
            settings: URL, user and secret of the tool.
            session_id: id of the annotation session.

        Returns:
            Links to open the tool and the tool's own reference of the session.

        Raises:
            ToolError: the images cannot be sent to the tool.
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

        Every image of the session must be present (an empty list = no annotations).

        Args:
            dataset: the project's FiftyOne dataset.
            image_ids: ids of the images of the session.
            project: the project (task and classes).
            settings: URL, user and secret of the tool.
            ref: the tool's reference of the session (``Sent.ref``).

        Returns:
            The labels of each image, keyed by image id.

        Raises:
            ToolError: the annotations cannot be read from the tool.
        """
        ...

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any]) -> None:
        """Forget the session in the tool. Does not raise when it is already gone.

        Args:
            dataset: the project's FiftyOne dataset, or None.
            settings: URL, user and secret of the tool.
            ref: the tool's reference of the session (``Sent.ref``).
        """
        ...
