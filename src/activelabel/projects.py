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

from activelabel.errors import CodedError
from activelabel.storage import DamagedFile, known_fields, now, read_json, write_json
from activelabel.tasks import Task
from activelabel.textcheck import has_control_characters

MAX_NAME = 100
MAX_DESCRIPTION = 2000
MAX_CLASS_NAME = 100
MAX_CLASSES = 1000


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
        """JSON-ready form.

        Returns:
            The project as the dict stored in ``project.json`` (the task as its value).
        """
        data = asdict(self)
        data["task"] = self.task.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """Inverse of :meth:`as_dict`.

        Args:
            data: the content of ``project.json`` (fields a newer version added are left out).

        Returns:
            The project.
        """
        return cls(**{**known_fields(cls, data), "task": Task(data["task"])})


def check_classes(classes: list[str] | tuple[str, ...]) -> None:
    """Refuse class lists that are too long or have a name that is too long (or has control characters).

    Args:
        classes: the class names as typed (they are cleaned before the check).

    Raises:
        CodedError: ``class_name_too_long``, ``class_name_invalid`` or ``too_many_classes``.
    """
    names = clean_classes(classes)
    if len(names) > MAX_CLASSES:
        raise CodedError(
            "too_many_classes", f"Too many classes (max {MAX_CLASSES}).", max=MAX_CLASSES, count=len(names)
        )
    for name in names:
        if len(name) > MAX_CLASS_NAME:
            raise CodedError(
                "class_name_too_long",
                f"The class name '{name[:30]}...' is too long (max {MAX_CLASS_NAME} characters).",
                max=MAX_CLASS_NAME,
                name=name[:30],
            )
        if has_control_characters(name):
            raise CodedError(
                "class_name_invalid", "Class names cannot contain control characters.", name=name[:30]
            )


def clean_classes(classes: list[str] | tuple[str, ...]) -> list[str]:
    """Strip names, drop empty ones and repeated ones (keeps the first order).

    Args:
        classes: the class names as typed.

    Returns:
        The names without spaces around, empty ones, or repeats (ignoring case); the first spelling stays.
    """
    seen: dict[str, str] = {}  # lower case -> first spelling
    for name in classes:
        name = name.strip()
        if name:
            seen.setdefault(name.lower(), name)
    return list(seen.values())


class ProjectStore:
    """Projects on disk."""

    def __init__(self, root: Path) -> None:
        """Open the projects of a workspace.

        Args:
            root: the ``projects`` folder (created when the first project is).
        """
        self.root = root
        self._lock = threading.RLock()

    def dir(self, project_id: int) -> Path:
        """Folder of a project.

        Args:
            project_id: the project.

        Returns:
            Its folder (it may not exist).
        """
        return self.root / str(project_id)

    def _file(self, project_id: int) -> Path:
        """The ``project.json`` of a project.

        Args:
            project_id: the project.

        Returns:
            Its path.
        """
        return self.dir(project_id) / "project.json"

    def _folders(self) -> dict[int, Path]:
        """Project folders by id (a folder counts even when its project.json is damaged).

        Returns:
            Folder per id; empty when there is no projects folder yet.
        """
        if not self.root.exists():
            return {}
        return {int(f.name): f for f in self.root.iterdir() if f.is_dir() and f.name.isdigit()}

    def list(self) -> list[Project]:
        """Every readable project, newest first (see :meth:`damaged` for the others).

        Returns:
            The projects, highest id first.
        """
        projects = []
        for folder in self._folders().values():
            try:
                data = read_json(folder / "project.json")
            except DamagedFile:
                continue
            if data:
                projects.append(Project.from_dict(data))
        return sorted(projects, key=lambda p: p.id, reverse=True)

    def damaged(self) -> dict[int, str]:
        """Projects whose ``project.json`` cannot be read: ``{id: readable message}``.

        Returns:
            Message per project id.
        """
        out: dict[int, str] = {}
        for project_id, folder in self._folders().items():
            try:
                read_json(folder / "project.json")
            except DamagedFile as exc:
                out[project_id] = str(exc)
        return out

    def get(self, project_id: int) -> Project:
        """One project.

        Args:
            project_id: the project.

        Returns:
            The project.

        Raises:
            ProjectNotFound: when it does not exist.
        """
        data = read_json(self._file(project_id))
        if not data:
            raise ProjectNotFound(project_id)
        return Project.from_dict(data)

    def _check(self, name: str, description: str, own_id: int | None = None) -> tuple[str, str]:
        """Validate a name and a description.

        Args:
            name: the project name as typed.
            description: the description as typed.
            own_id: the project being changed, which may keep its own name; None for a new project.

        Returns:
            The name and the description, stripped.

        Raises:
            CodedError: empty, too long or repeated name, too long description, control characters.
        """
        name, description = name.strip(), description.strip()
        if not name:
            raise CodedError("project_name_empty", "Give the project a name.")
        if len(name) > MAX_NAME:
            raise CodedError(
                "project_name_too_long", f"The name is too long (max {MAX_NAME} characters).", max=MAX_NAME
            )
        if len(description) > MAX_DESCRIPTION:
            raise CodedError(
                "description_too_long",
                f"The description is too long (max {MAX_DESCRIPTION} characters).",
                max=MAX_DESCRIPTION,
            )
        if has_control_characters(name) or has_control_characters(description, allow_newlines=True):
            raise CodedError(
                "control_characters", "The name and description cannot contain control characters."
            )
        for other in self.list():
            if other.id != own_id and other.name.lower() == name.lower():
                raise CodedError(
                    "project_name_taken", f"A project named '{other.name}' already exists.", name=other.name
                )
        return name, description

    def create(
        self,
        name: str,
        description: str,
        task: Task | str,
        classes: builtins.list[str] | tuple[str, ...] = (),
    ) -> Project:
        """Create a project with the next free id.

        Args:
            name: unique, ignoring case.
            description: free text.
            task: the task, or its value (``detection``...).
            classes: the class names, in order.

        Returns:
            The new project.

        Raises:
            CodedError: empty, too long or repeated name, or a bad class list.
            ValueError: unknown task.
        """
        with self._lock:
            name, description = self._check(name, description)
            check_classes(classes)
            counter = self.root / "next_id.json"
            next_id = int(read_json(counter, 1))
            project_id = max([next_id, *(i + 1 for i in self._folders())])  # damaged folders keep their id
            stamp = now()
            project = Project(project_id, name, description, Task(task), clean_classes(classes), stamp, stamp)
            write_json(self._file(project_id), project.as_dict())
            write_json(counter, project_id + 1)
            return project

    def update(self, project_id: int, **changes: Any) -> Project:
        """Change name, description, task or classes.

        Args:
            project_id: the project.
            **changes: any of ``name``, ``description``, ``task``, ``classes``.

        Returns:
            The project after the change.

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
                check_classes(changes["classes"])
                project.classes = clean_classes(changes["classes"])
            project.updated = now()
            write_json(self._file(project_id), project.as_dict())
            return project

    def delete(self, project_id: int) -> None:
        """Delete a project and all its files (images, annotations).

        Args:
            project_id: the project.

        Raises:
            ProjectNotFound: when it does not exist.
        """
        with self._lock:
            self.get(project_id)
            shutil.rmtree(self.dir(project_id))
