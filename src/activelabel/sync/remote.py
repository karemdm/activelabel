"""Contract of a remote project in CVAT, as seen by the synchronisation (docs/SYNC.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from activelabel.annotate.base import ToolSettings
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.tasks import Task


@dataclass
class RemoteInfo:
    """A remote project: its id, name, web address and number of images."""

    id: str
    name: str
    url: str
    images: int = 0


@dataclass
class RemoteItem:
    """One image of the remote project with its current labels (relative coordinates 0..1).

    ``not_converted`` counts what the tool has on that image but the labels do not show, by kind
    (CVAT: shape types such as ``ellipse`` or ``mask``, ``tracks`` that are not read,
    ``rotated_rectangle`` read as its enclosing box).
    """

    key: str  # "<task_id>:<frame>" in CVAT
    file_name: str
    width: int
    height: int
    labels: list[Label] = field(default_factory=list)
    not_converted: dict[str, int] = field(default_factory=dict)


@dataclass
class RemoteJob:
    """One job of a remote task: frames ``start``..``stop`` (0-based, inclusive) and its progress."""

    id: str
    n: int  # 1-based position in the task
    start: int
    stop: int
    url: str = ""
    state: str = ""  # CVAT: new, in progress, completed, rejected
    stage: str = ""  # CVAT: annotation, validation, acceptance
    assignee: str = ""


@dataclass
class RemoteTask:
    """One remote task (a package of images) with its jobs (docs/TASKS.md)."""

    id: str
    name: str
    size: int = 0
    job_size: int | None = None  # CVAT segment_size (None: one job)
    url: str = ""
    jobs: list[RemoteJob] = field(default_factory=list)


class RemoteProject(Protocol):
    """A remote project in one tool. Implementations raise annotate.base.ToolError with readable messages."""

    tool: str

    def check(self, settings: ToolSettings) -> str:
        """Log in to the server; return ``Connected to <url> as <user>`` (ToolError when it fails).

        Args:
            settings: Connection settings of the tool (server address and credentials).

        Returns:
            The text ``Connected to <url> as <user>``.
        """
        ...

    def audit(self, settings: ToolSettings) -> dict[str, Any]:
        """Read-only inventory of the tasks per project, the loose ones grouped by name pattern (CVAT).

        Args:
            settings: Connection settings of the tool (server address and credentials).

        Returns:
            The inventory, as a dictionary ready to be shown by the API.
        """
        ...

    def find(self, settings: ToolSettings, name: str) -> RemoteInfo | None:
        """The remote project with exactly this name, or None.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            name: Exact name of the remote project.

        Returns:
            The remote project, or None when none has this name.
        """
        ...

    def get(self, settings: ToolSettings, remote_id: str) -> RemoteInfo:
        """One remote project (ToolError when it no longer exists).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.

        Returns:
            The remote project.
        """
        ...

    def create(self, settings: ToolSettings, project: Project) -> RemoteInfo:
        """Create an empty remote project for this project (its task and classes).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            project: The local project the remote one is created for.

        Returns:
            The new remote project.
        """
        ...

    def items(
        self, settings: ToolSettings, remote_id: str, task: Task, task_ids: list[str] | None = None
    ) -> list[RemoteItem]:
        """Every image of the remote project (or of these remote tasks) with its current labels.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.
            task: The local task whose classes the remote labels are read against.
            task_ids: Ids of the remote tasks to read; None reads the whole project.

        Returns:
            One item per image, with its labels in relative coordinates.
        """
        ...

    def set_frames(self, settings: ToolSettings, task_id: str, delete: list[int], restore: list[int]) -> None:
        """Mark frames of a remote task deleted / not deleted.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            task_id: Id of the remote task.
            delete: Frame numbers (0-based) to mark as deleted.
            restore: Frame numbers (0-based) to mark as not deleted.
        """
        ...

    def delete_task(self, settings: ToolSettings, task_id: str) -> None:
        """Delete a remote task.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            task_id: Id of the remote task.
        """
        ...

    def frames(self, settings: ToolSettings, task_id: str) -> list[tuple[str, bool]]:
        """``(file name, deleted)`` of every frame of a remote task, in order.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            task_id: Id of the remote task.

        Returns:
            One ``(file name, deleted)`` pair per frame, in frame order.
        """
        ...

    def frame(self, settings: ToolSettings, task_id: str, frame: int) -> bytes:
        """The image of one frame of a remote task.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            task_id: Id of the remote task.
            frame: Frame number (0-based).

        Returns:
            The bytes of the image.
        """
        ...

    def video_source(
        self, settings: ToolSettings, task_id: str, dest_dir: Path
    ) -> tuple[Path, dict[str, int]] | None:
        """The original video of a remote video task, downloaded into ``dest_dir``.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            task_id: Id of the remote task.
            dest_dir: Folder the video is downloaded into.

        Returns:
            ``(file, {"start", "stop", "step"})`` (the frames of the task are ``start + i * step``
            of the video), or None when the task is not a video task or the tool cannot give the video.
        """
        ...

    def tasks(self, settings: ToolSettings, remote_id: str) -> list[RemoteTask]:
        """The remote tasks of the project with their jobs, oldest first (empty when the tool has none).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.

        Returns:
            The remote tasks, oldest first.
        """
        ...

    def add_task(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        name: str,
        job_size: int | None,
        images: list[tuple[ImageRecord, Path, list[Label]]],
        video: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[RemoteTask, dict[str, str]]:
        """Create one remote task with these images in this order (jobs of ``job_size``) and their labels.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project the task is created in.
            project: The local project the images belong to.
            name: Name of the new remote task.
            job_size: Frames per job, or None for a single job.
            images: ``(record, file, labels)`` of every image, in the order of the task.
            video: ``(file, {"start", "stop", "step"})`` of the original video when the task is made
                from a video, else None.

        Returns:
            The task and ``{image_id: key}``.
        """
        ...

    def add(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> dict[str, str]:
        """Upload images with their labels; return ``{image_id: key}``.

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.
            project: The local project the images belong to.
            images: ``(record, file, labels)`` of every image to upload.

        Returns:
            The remote item key of every uploaded image, by image id.
        """
        ...

    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Replace the labels of these remote items (``{key: labels}``).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.
            project: The local project the labels belong to.
            updates: The new labels of every remote item, by item key.
        """
        ...

    def classes(self, settings: ToolSettings, remote_id: str) -> dict[str, str]:
        """The classes of the remote project by their id in the tool (an id stays when a class is renamed).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.

        Returns:
            ``{class id: name}``.
        """
        ...

    def ensure_classes(self, settings: ToolSettings, remote_id: str, classes: list[str]) -> None:
        """Make sure the remote project knows these classes (adds the missing ones).

        Args:
            settings: Connection settings of the tool (server address and credentials).
            remote_id: Id of the remote project.
            classes: Names of the classes the remote project must know.
        """
        ...
