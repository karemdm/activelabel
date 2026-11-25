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
    """What :meth:`AnnotationTool.send` returns: links to open, and the tool's own reference.

    Attributes:
        links: links to open the tool (``{"label", "url"}``).
        ref: the tool's own reference of the session (e.g. ``{"view": "..."}``), stored in the session.
        kept: how many of the images still hold edits made in the tool in an earlier session (not brought
            back yet), which the tool shows instead of the project's labels.
    """

    links: list[dict[str, str]]
    ref: dict[str, Any]
    kept: int = 0


class Pulled(dict[str, list[Label]]):
    """What a tool's pull may return: the labels by image id, plus what the tool knows about them.

    A plain ``dict`` is accepted as well: then every image counts as edited in the tool.

    Attributes:
        edited: ids of the images whose labels were changed in the tool since ActiveLabel wrote them; the
            other images keep the project's labels (a project edit made meanwhile is not undone).
        written: hash (:func:`activelabel.sync.links.label_hash`) of the project labels ActiveLabel wrote
            into the tool, by image id: an edited image whose project labels no longer have this hash was
            edited on both sides.
    """

    def __init__(
        self,
        labels: dict[str, list[Label]],
        edited: set[str],
        written: dict[str, str] | None = None,
    ) -> None:
        """Wrap the labels of a pull.

        Args:
            labels: the labels of each image, keyed by image id.
            edited: the images changed in the tool since ActiveLabel wrote them.
            written: hash of the project labels written into the tool, by image id.
        """
        super().__init__(labels)
        self.edited = set(edited)
        self.written = dict(written or {})


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

        Every image of the session must be present (an empty list = no annotations). A tool that can tell
        which images were edited in it returns a :class:`Pulled`.

        Args:
            dataset: the project's FiftyOne dataset.
            image_ids: ids of the images of the session.
            project: the project (task and classes).
            settings: URL, user and secret of the tool.
            ref: the tool's reference of the session (``Sent.ref``).

        Returns:
            The labels of each image, keyed by image id (a :class:`Pulled` or a plain dict).

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
