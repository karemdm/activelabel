"""Contract of a remote project in CVAT or Label Studio, as seen by the synchronisation (docs/SYNC.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
    """One image of the remote project with its current labels (relative coordinates 0..1)."""

    key: str  # e.g. "<task_id>:<frame>" in CVAT, the task id in Label Studio
    file_name: str
    width: int
    height: int
    labels: list[Label] = field(default_factory=list)


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

    def find(self, settings: ToolSettings, name: str) -> RemoteInfo | None:
        """The remote project with exactly this name, or None."""
        ...

    def get(self, settings: ToolSettings, remote_id: str) -> RemoteInfo:
        """One remote project (ToolError when it no longer exists)."""
        ...

    def create(self, settings: ToolSettings, project: Project) -> RemoteInfo:
        """Create an empty remote project for this project (its task and classes)."""
        ...

    def items(
        self, settings: ToolSettings, remote_id: str, task: Task, task_ids: list[str] | None = None
    ) -> list[RemoteItem]:
        """Every image of the remote project (or of these remote tasks) with its current labels."""
        ...

    supports_tasks: bool  # tasks/jobs, deleted frames (CVAT); False: one flat list of items

    def set_frames(self, settings: ToolSettings, task_id: str, delete: list[int], restore: list[int]) -> None:
        """Mark frames of a remote task deleted / not deleted (only when ``supports_tasks``)."""
        ...

    def delete_task(self, settings: ToolSettings, task_id: str) -> None:
        """Delete a remote task (only when ``supports_tasks``)."""
        ...

    def frame(self, settings: ToolSettings, task_id: str, frame: int) -> bytes:
        """The image of one frame of a remote task (only when ``supports_tasks``)."""
        ...

    def tasks(self, settings: ToolSettings, remote_id: str) -> list[RemoteTask]:
        """The remote tasks of the project with their jobs, oldest first (empty when the tool has none)."""
        ...

    def add_task(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        name: str,
        job_size: int | None,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> tuple[RemoteTask, dict[str, str]]:
        """Create one remote task with these images in this order (jobs of ``job_size``) and their labels.

        Returns the task and ``{image_id: key}``.
        """
        ...

    def add(
        self,
        settings: ToolSettings,
        remote_id: str,
        project: Project,
        images: list[tuple[ImageRecord, Path, list[Label]]],
    ) -> dict[str, str]:
        """Upload images with their labels; return ``{image_id: key}``."""
        ...

    def write(
        self, settings: ToolSettings, remote_id: str, project: Project, updates: dict[str, list[Label]]
    ) -> None:
        """Replace the labels of these remote items (``{key: labels}``)."""
        ...

    def ensure_classes(self, settings: ToolSettings, remote_id: str, classes: list[str]) -> None:
        """Make sure the remote project knows these classes (adds the missing ones)."""
        ...
