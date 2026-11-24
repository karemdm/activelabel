"""Tasks of a project: packages of images split into jobs (see docs/TASKS.md).

Stored in ``<project>/tasks.json``. Jobs are not stored: they are derived from the task image
order and its ``job_size``. Named ``tasks_store`` to avoid confusion with :mod:`activelabel.tasks`
(the annotation type) and :mod:`activelabel.jobs` (background processing).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from activelabel.images.manifest import Manifest
from activelabel.storage import now, read_json, write_json
from activelabel.textcheck import has_control_characters

MAX_NAME = 100
MAX_JOB_SIZE = 100_000

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per file, shared by every store object of the same project."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


class TaskNotFound(KeyError):
    """No task with this id."""


class TaskClosed(Exception):
    """Images cannot be added: the task is closed."""


@dataclass
class TaskJob:
    """A fixed-size chunk of a task.

    Attributes:
        n: job number, 1-based.
        first: 1-based position in the task of its first image.
        last: 1-based position in the task of its last image.
        image_ids: ids of its images, in task order.
    """

    n: int
    first: int
    last: int
    image_ids: list[str]


@dataclass
class ImageTask:
    """A package of images added together.

    Attributes:
        id: automatic, sequential, never reused.
        name: free text.
        created: ISO time (UTC).
        job_size: images per job, or None for a single job with every image.
        images: image ids in upload order.
        open: True while a chunked upload may still append images.
        source: for a task made from a video, ``{"video", "start", "stop", "step", "fps"}``.
    """

    id: int
    name: str
    created: str
    job_size: int | None = None
    images: list[str] = field(default_factory=list)
    open: bool = False
    source: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def check_job_size(job_size: Any) -> int | None:
    """Validate "images per job".

    Raises:
        ValueError: not a whole number from 1 to ``MAX_JOB_SIZE`` (and not None).
    """
    if job_size is None:
        return None
    if isinstance(job_size, bool) or not isinstance(job_size, int) or not 1 <= job_size <= MAX_JOB_SIZE:
        raise ValueError(f"Images per job must be a whole number from 1 to {MAX_JOB_SIZE} (or empty).")
    return job_size


def jobs(task: ImageTask) -> list[TaskJob]:
    """Jobs of a task: chunks of ``job_size`` images (one job with everything when it is None)."""
    total = len(task.images)
    if not total:
        return []
    size = task.job_size or total
    return [
        TaskJob(k + 1, start + 1, min(start + size, total), task.images[start : start + size])
        for k, start in enumerate(range(0, total, size))
    ]


def clean_name(name: str) -> str:
    """Strip a task name.

    Raises:
        ValueError: empty or too long.
    """
    name = name.strip()
    if not name:
        raise ValueError("Give the task a name.")
    if len(name) > MAX_NAME:
        raise ValueError(f"The name is too long (max {MAX_NAME} characters).")
    if has_control_characters(name):
        raise ValueError("The name cannot contain control characters.")
    return name


class TaskStore:
    """Tasks of one project, saved in ``<project>/tasks.json``."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.path = project_dir / "tasks.json"
        self._lock = _lock_for(self.path)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the project's task lock (re-entrant): ingest and task registration are then atomic."""
        with self._lock:
            yield

    # -- persistence ------------------------------------------------------------
    def _read(self, registering: Iterable[str] = ()) -> tuple[int, list[ImageTask]]:
        """Tasks as saved, migrating a project that has images but no tasks.json.

        Images in no task are put in a new closed task, except ``registering`` (ids the caller is
        about to place itself).
        """
        data = read_json(self.path)
        if data is None:
            records = Manifest(self.project_dir).all()
            if not records:
                return 1, []
            tasks = [ImageTask(1, "Task 1", now(), None, [r.id for r in records])]
            self._write(2, tasks)
            return 2, tasks
        next_id, tasks = int(data["next_id"]), [ImageTask(**t) for t in data["tasks"]]
        placed = {i for t in tasks for i in t.images} | set(registering)
        orphans = [i for i in Manifest(self.project_dir).load() if i not in placed]
        if orphans:  # images added by code that does not know about tasks
            tasks.append(ImageTask(next_id, f"Task {next_id}", now(), None, orphans))
            next_id += 1
            self._write(next_id, tasks)
        return next_id, tasks

    def _write(self, next_id: int, tasks: list[ImageTask]) -> None:
        write_json(self.path, {"next_id": next_id, "tasks": [t.as_dict() for t in tasks]})

    def load(self) -> list[ImageTask]:
        """Every task, in creation order (runs the migration on first read)."""
        with self._lock:
            return self._read()[1]

    def ensure(self) -> None:
        """Make sure tasks.json exists (migrating when there are images), before new images are stored."""
        with self._lock:
            self._read()
            if not self.path.exists():
                self._write(1, [])

    def next_id(self) -> int:
        """The id the next task will get (never reused)."""
        with self._lock:
            return self._read()[0]

    def get(self, task_id: int) -> ImageTask:
        """One task.

        Raises:
            TaskNotFound: unknown id.
        """
        for task in self.load():
            if task.id == task_id:
                return task
        raise TaskNotFound(task_id)

    def task_of(self, image_id: str) -> ImageTask | None:
        """The task that holds an image, or None."""
        return next((t for t in self.load() if image_id in t.images), None)

    # -- changes ----------------------------------------------------------------
    def create(
        self,
        name: str | None = None,
        job_size: int | None = None,
        images: list[str] | None = None,
        *,
        open: bool = False,
        source: dict[str, Any] | None = None,
    ) -> ImageTask:
        """Create a task with the next free id (default name "Task <id>"); other open tasks are closed.

        Raises:
            ValueError: empty or too long name; invalid ``job_size``.
        """
        name = clean_name(name) if name is not None else None
        job_size = check_job_size(job_size)
        with self._lock:
            next_id, tasks = self._read(images or ())
            for other in tasks:
                other.open = False
            task = ImageTask(
                next_id, name or f"Task {next_id}", now(), job_size, list(images or []), open, source
            )
            tasks.append(task)
            self._write(next_id + 1, tasks)
            return task

    def _change(
        self, task_id: int, registering: Iterable[str] = ()
    ) -> tuple[int, list[ImageTask], ImageTask]:
        next_id, tasks = self._read(registering)
        for task in tasks:
            if task.id == task_id:
                return next_id, tasks, task
        raise TaskNotFound(task_id)

    def append(self, task_id: int, image_ids: list[str]) -> ImageTask:
        """Add images at the end of an open task.

        Raises:
            TaskNotFound: unknown id.
            TaskClosed: the task is closed.
        """
        with self._lock:
            next_id, tasks, task = self._change(task_id, image_ids)
            if not task.open:
                raise TaskClosed(task_id)
            known = set(task.images)
            task.images.extend(i for i in dict.fromkeys(image_ids) if i not in known)
            self._write(next_id, tasks)
            return task

    def set_source(self, task_id: int, source: dict[str, Any]) -> ImageTask:
        """Record the video a task was made from (``{"video", "start", "stop", "step", "fps"}``)."""
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            task.source = source
            self._write(next_id, tasks)
            return task

    def close(self, task_id: int) -> ImageTask:
        """Fix the images of a task (no more appends)."""
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            task.open = False
            self._write(next_id, tasks)
            return task

    def rename(self, task_id: int, name: str) -> ImageTask:
        """Change the name of a task.

        Raises:
            TaskNotFound: unknown id.
            ValueError: empty or too long name.
        """
        name = clean_name(name)
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            task.name = name
            self._write(next_id, tasks)
            return task

    def delete(self, task_id: int) -> list[str]:
        """Remove a task. Returns its image ids (the caller deletes the images).

        Raises:
            TaskNotFound: unknown id.
        """
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            self._write(next_id, [t for t in tasks if t.id != task_id])
            return list(task.images)

    def remove_images(self, image_ids: list[str]) -> int:
        """Take images out of their tasks; a task left empty is removed. Returns how many were removed."""
        gone = set(image_ids)
        with self._lock:
            next_id, tasks = self._read()
            removed = 0
            kept: list[ImageTask] = []
            for task in tasks:
                before = len(task.images)
                task.images = [i for i in task.images if i not in gone]
                removed += before - len(task.images)
                if task.images or before == 0:
                    kept.append(task)
            if removed:
                self._write(next_id, kept)
            return removed
