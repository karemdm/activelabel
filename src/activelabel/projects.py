"""Projects: name, description, automatic id, task and classes.

Each project is a folder ``<workspace>/projects/<id>/`` with ``project.json``. Ids are
never reused, even after a project is deleted.
"""

from __future__ import annotations

import builtins
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from activelabel.storage import now, read_json, write_json
from activelabel.tasks import Task

MAX_NAME = 100
MAX_DESCRIPTION = 2000


class ProjectNotFound(KeyError):
    """No project with this id."""


@dataclass
class Project:
    """A project.

    Attributes:
        id: automatic, sequential, never reused.
        name: unique (ignoring case).
        description: free text.
        task: classification, detection, segmentation or keypoints.
        classes: class names, in order (the order is the class index in trained models).
        created: ISO time (UTC).
        updated: ISO time (UTC).
    """

    id: int
    name: str
    description: str
    task: Task
    classes: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        data = asdict(self)
        data["task"] = self.task.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """Inverse of :meth:`as_dict`."""
        return cls(**{**data, "task": Task(data["task"])})


def clean_classes(classes: list[str] | tuple[str, ...]) -> list[str]:
    """Strip names, drop empty ones and repeated ones (keeps the first order)."""
    seen: dict[str, None] = {}
    for name in classes:
        name = name.strip()
        if name and name.lower() not in {s.lower() for s in seen}:
            seen[name] = None
    return list(seen)


class ProjectStore:
    """Projects on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()

    def dir(self, project_id: int) -> Path:
        """Folder of a project."""
        return self.root / str(project_id)

    def _file(self, project_id: int) -> Path:
        return self.dir(project_id) / "project.json"

    def list(self) -> list[Project]:
        """Every project, newest first."""
        if not self.root.exists():
            return []
        projects = []
        for folder in self.root.iterdir():
            data = read_json(folder / "project.json")
            if folder.is_dir() and folder.name.isdigit() and data:
                projects.append(Project.from_dict(data))
        return sorted(projects, key=lambda p: p.id, reverse=True)

    def get(self, project_id: int) -> Project:
        """One project.

        Raises:
            ProjectNotFound: when it does not exist.
        """
        data = read_json(self._file(project_id))
        if not data:
            raise ProjectNotFound(project_id)
        return Project.from_dict(data)

    def _check(self, name: str, description: str, own_id: int | None = None) -> tuple[str, str]:
        name, description = name.strip(), description.strip()
        if not name:
            raise ValueError("Give the project a name.")
        if len(name) > MAX_NAME:
            raise ValueError(f"The name is too long (max {MAX_NAME} characters).")
        if len(description) > MAX_DESCRIPTION:
            raise ValueError(f"The description is too long (max {MAX_DESCRIPTION} characters).")
        for other in self.list():
            if other.id != own_id and other.name.lower() == name.lower():
                raise ValueError(f"A project named '{other.name}' already exists.")
        return name, description

    def create(
        self,
        name: str,
        description: str,
        task: Task | str,
        classes: builtins.list[str] | tuple[str, ...] = (),
    ) -> Project:
        """Create a project with the next free id.

        Raises:
            ValueError: empty, too long or repeated name; unknown task.
        """
        with self._lock:
            name, description = self._check(name, description)
            counter = self.root / "next_id.json"
            next_id = int(read_json(counter, 1))
            existing = [p.id for p in self.list()]
            project_id = max([next_id, *(i + 1 for i in existing)])
            stamp = now()
            project = Project(project_id, name, description, Task(task), clean_classes(classes), stamp, stamp)
            write_json(self._file(project_id), project.as_dict())
            write_json(counter, project_id + 1)
            return project

    def update(self, project_id: int, **changes: Any) -> Project:
        """Change name, description, task or classes.

        Raises:
            ProjectNotFound: when it does not exist.
            ValueError: invalid values or an unknown field.
        """
        allowed = {"name", "description", "task", "classes"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"These fields cannot be changed: {', '.join(sorted(unknown))}")
        with self._lock:
            project = self.get(project_id)
            name, description = self._check(
                changes.get("name", project.name), changes.get("description", project.description), project_id
            )
            project.name, project.description = name, description
            if "task" in changes:
                project.task = Task(changes["task"])
            if "classes" in changes:
                project.classes = clean_classes(changes["classes"])
            project.updated = now()
            write_json(self._file(project_id), project.as_dict())
            return project

    def delete(self, project_id: int) -> None:
        """Delete a project and all its files (images, annotations).

        Raises:
            ProjectNotFound: when it does not exist.
        """
        with self._lock:
            self.get(project_id)
            shutil.rmtree(self.dir(project_id))
