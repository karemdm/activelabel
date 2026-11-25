"""Tasks of a project: packages of images split into jobs (see docs/TASKS.md).

Stored in ``<project>/tasks.json``. Jobs are not stored: they are derived from the task image
order and its ``job_size``. Named ``tasks_store`` to avoid confusion with :mod:`activelabel.tasks`
(the annotation type) and :mod:`activelabel.jobs` (background processing).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.images.manifest import Manifest
from activelabel.storage import now, read_json, write_json
from activelabel.textcheck import has_control_characters

MAX_NAME = 100
MAX_JOB_SIZE = 100_000
STALE_UPLOAD_SECONDS = 1800  # an open task with no new chunk for this long was cut off
_DEFAULT_NAME = re.compile(r"task (\d+)", re.IGNORECASE)

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per file, shared by every store object of the same project.

    Args:
        path: the ``tasks.json`` of the project.

    Returns:
        The same lock for the same file every time.
    """
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
        incomplete: True when the upload was cut off (the page was closed, the connection lost): the task
            was closed by the app, not by the browser, so it may lack images.
        touched: ISO time (UTC) of the last chunk of an open task; empty once it is closed.
        source: for a task made from a video, ``{"video", "start", "stop", "step", "fps"}``.
    """

    id: int
    name: str
    created: str
    job_size: int | None = None
    images: list[str] = field(default_factory=list)
    open: bool = False
    source: dict[str, Any] | None = None
    incomplete: bool = False
    touched: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            The task as the dict stored in ``tasks.json``.
        """
        return asdict(self)


def _last_touch(task: ImageTask) -> datetime:
    """When the last chunk of a task came (its creation for a task saved without the mark).

    Args:
        task: the task.

    Returns:
        The time of its last chunk.
    """
    return datetime.fromisoformat(task.touched or task.created)


def check_job_size(job_size: Any) -> int | None:
    """Validate "images per job".

    Args:
        job_size: the value as received (a number, or None for a single job).

    Returns:
        ``job_size`` unchanged when valid.

    Raises:
        CodedError: ``job_size_invalid``: not a whole number from 1 to ``MAX_JOB_SIZE`` (and not None).
    """
    if job_size is None:
        return None
    if isinstance(job_size, bool) or not isinstance(job_size, int) or not 1 <= job_size <= MAX_JOB_SIZE:
        raise CodedError(
            "job_size_invalid",
            f"Images per job must be a whole number from 1 to {MAX_JOB_SIZE} (or empty).",
            max=MAX_JOB_SIZE,
        )
    return job_size


def jobs(task: ImageTask) -> list[TaskJob]:
    """Jobs of a task: chunks of ``job_size`` images (one job with everything when it is None).

    Args:
        task: the task.

    Returns:
        Its jobs in order; none when the task has no images.
    """
    total = len(task.images)
    if not total:
        return []
    size = task.job_size or total
    return [
        TaskJob(k + 1, start + 1, min(start + size, total), task.images[start : start + size])
        for k, start in enumerate(range(0, total, size))
    ]


def default_name(tasks: Iterable[ImageTask]) -> str:
    """The automatic name of a new task: ``Task <n>``, one more than the highest number the tasks have.

    The number comes from the tasks that exist, not from the ids: an upload that added nothing (its task
    is removed again) or a deleted task does not use up a number.

    Args:
        tasks: the tasks the project has.

    Returns:
        ``Task 1`` when no task is named like that, else the next number.
    """
    numbers = [int(m[1]) for t in tasks if (m := _DEFAULT_NAME.fullmatch(t.name))]
    return f"Task {max(numbers, default=0) + 1}"


def clean_name(name: str) -> str:
    """Strip a task name.

    Args:
        name: the name as typed.

    Returns:
        The name without spaces around.

    Raises:
        CodedError: ``task_name_empty``, ``task_name_too_long`` or ``control_characters``.
    """
    name = name.strip()
    if not name:
        raise CodedError("task_name_empty", "Give the task a name.")
    if len(name) > MAX_NAME:
        raise CodedError(
            "task_name_too_long", f"The name is too long (max {MAX_NAME} characters).", max=MAX_NAME
        )
    if has_control_characters(name):
        raise CodedError("control_characters", "The name cannot contain control characters.")
    return name


class TaskStore:
    """Tasks of one project, saved in ``<project>/tasks.json``."""

    def __init__(self, project_dir: Path) -> None:
        """Open the tasks of a project.

        Args:
            project_dir: the project's folder.
        """
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

        Args:
            registering: image ids the caller is about to place in a task itself.

        Returns:
            The id the next task gets, and the tasks.
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
            tasks.append(ImageTask(next_id, default_name(tasks), now(), None, orphans))
            next_id += 1
            self._write(next_id, tasks)
        return next_id, tasks

    def _write(self, next_id: int, tasks: list[ImageTask]) -> None:
        """Save the tasks.

        Args:
            next_id: the id the next task gets.
            tasks: every task of the project.
        """
        write_json(self.path, {"next_id": next_id, "tasks": [t.as_dict() for t in tasks]})

    def load(self) -> list[ImageTask]:
        """Every task, in creation order (runs the migration on first read).

        A reader never waits for a long upload: while another thread holds the lock (ingest holds it for
        a whole upload) the tasks are read as last saved (the file is replaced atomically). Images not
        registered yet are then not shown, and the migration of orphans waits for a quiet moment.

        Returns:
            The tasks.
        """
        if not self._lock.acquire(blocking=False):
            data = read_json(self.path)
            return [ImageTask(**t) for t in data["tasks"]] if data else []
        try:
            return self._read()[1]
        finally:
            self._lock.release()

    def ensure(self) -> None:
        """Make sure tasks.json exists (migrating when there are images), before new images are stored."""
        with self._lock:
            self._read()
            if not self.path.exists():
                self._write(1, [])

    def next_id(self) -> int:
        """The id the next task will get (never reused).

        Returns:
            The id.
        """
        with self._lock:
            return self._read()[0]

    def get(self, task_id: int) -> ImageTask:
        """One task.

        Args:
            task_id: the task.

        Returns:
            The task.

        Raises:
            TaskNotFound: unknown id.
        """
        for task in self.load():
            if task.id == task_id:
                return task
        raise TaskNotFound(task_id)

    def task_of(self, image_id: str) -> ImageTask | None:
        """The task that holds an image, or None.

        Args:
            image_id: the image.

        Returns:
            The task, or None when no task holds it.
        """
        return next((t for t in self.load() if image_id in t.images), None)

    def check_name_free(self, name: str, own_id: int | None = None) -> None:
        """Refuse a name another task of the project already has (case does not matter).

        Args:
            name: the name wanted.
            own_id: the task being renamed, which may keep its own name; None for a new task.

        Raises:
            CodedError: ``task_name_taken``.
        """
        wanted = name.strip().lower()
        for task in self.load():
            if task.id != own_id and task.name.lower() == wanted:
                raise CodedError(
                    "task_name_taken", f"A task named '{task.name}' already exists.", name=task.name
                )

    def _unique(self, name: str, tasks: list[ImageTask]) -> str:
        """``name``, or ``name (2)``, ``name (3)``... when another task has it (tasks made by imports).

        Args:
            name: the name wanted.
            tasks: the tasks that already exist.

        Returns:
            A name no task in ``tasks`` has.
        """
        taken = {t.name.lower() for t in tasks}
        candidate, n = name, 1
        while candidate.lower() in taken:
            n += 1
            candidate = f"{name} ({n})"
        return candidate

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

        A name another task already has gets a number added (``Name (2)``): callers that must refuse
        a repeated name call :meth:`check_name_free` first (the API does, before storing images).

        Args:
            name: the name; None for "Task <id>".
            job_size: images per job, or None for a single job.
            images: ids of the images in the task, in order.
            open: True while a chunked upload may still append images.
            source: for a task made from a video, where it came from.

        Returns:
            The new task.

        Raises:
            CodedError: empty or too long name; invalid ``job_size``.
        """
        name = clean_name(name) if name is not None else None
        job_size = check_job_size(job_size)
        with self._lock:
            next_id, tasks = self._read(images or ())
            for other in tasks:  # an upload still open when another task starts was cut off
                other.incomplete = other.incomplete or other.open
                other.open = False
                other.touched = ""
            name = self._unique(name, tasks) if name else default_name(tasks)
            task = ImageTask(next_id, name, now(), job_size, list(images or []), open, source, touched=now())
            tasks.append(task)
            self._write(next_id + 1, tasks)
            return task

    def _change(
        self, task_id: int, registering: Iterable[str] = ()
    ) -> tuple[int, list[ImageTask], ImageTask]:
        """Read the tasks and pick one, for a change that is then written back.

        Args:
            task_id: the task to change.
            registering: image ids the caller is about to place itself (see :meth:`_read`).

        Returns:
            The next id, every task, and the task to change.

        Raises:
            TaskNotFound: unknown id.
        """
        next_id, tasks = self._read(registering)
        for task in tasks:
            if task.id == task_id:
                return next_id, tasks, task
        raise TaskNotFound(task_id)

    def append(self, task_id: int, image_ids: list[str]) -> ImageTask:
        """Add images at the end of an open task.

        Args:
            task_id: the task.
            image_ids: the images to add (ones already in it are skipped).

        Returns:
            The task after the change.

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
            task.touched = now()
            self._write(next_id, tasks)
            return task

    def set_source(self, task_id: int, source: dict[str, Any]) -> ImageTask:
        """Record the video a task was made from.

        Args:
            task_id: the task.
            source: ``{"video", "start", "stop", "step", "fps"}``.

        Returns:
            The task after the change.

        Raises:
            TaskNotFound: unknown id.
        """
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            task.source = source
            self._write(next_id, tasks)
            return task

    def close(self, task_id: int) -> ImageTask:
        """Fix the images of a task (no more appends).

        Args:
            task_id: the task.

        Returns:
            The task after the change.

        Raises:
            TaskNotFound: unknown id.
        """
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            task.open = False
            task.touched = ""
            self._write(next_id, tasks)
            return task

    def close_stale(self, max_age: float) -> list[ImageTask]:
        """Close the open tasks whose last chunk is older than ``max_age`` seconds and mark them incomplete.

        A chunked upload that stopped (the page was closed, the connection lost, the service restarted)
        would leave its task open for good; the task list closes them when it is read. Never waits: while
        another thread holds the lock (an upload is being stored) nothing is changed.

        Args:
            max_age: seconds without a new chunk after which an open task counts as cut off; 0 closes every
                open task.

        Returns:
            The tasks that were closed.
        """
        if not self._lock.acquire(blocking=False):
            return []
        try:
            next_id, tasks = self._read()
            limit = datetime.now(timezone.utc) - timedelta(seconds=max_age)
            stale = [t for t in tasks if t.open and _last_touch(t) <= limit]
            for task in stale:
                task.open, task.incomplete, task.touched = False, True, ""
            if stale:
                self._write(next_id, tasks)
            return stale
        finally:
            self._lock.release()

    def rename(self, task_id: int, name: str) -> ImageTask:
        """Change the name of a task.

        Args:
            task_id: the task.
            name: the new name.

        Returns:
            The task after the change.

        Raises:
            TaskNotFound: unknown id.
            CodedError: empty, too long or repeated name.
        """
        name = clean_name(name)
        with self._lock:
            self.check_name_free(name, task_id)
            next_id, tasks, task = self._change(task_id)
            task.name = name
            self._write(next_id, tasks)
            return task

    def delete(self, task_id: int) -> list[str]:
        """Remove a task.

        Args:
            task_id: the task.

        Returns:
            Its image ids (the caller deletes the images).

        Raises:
            TaskNotFound: unknown id.
        """
        with self._lock:
            next_id, tasks, task = self._change(task_id)
            self._write(next_id, [t for t in tasks if t.id != task_id])
            return list(task.images)

    def remove_images(self, image_ids: list[str]) -> int:
        """Take images out of their tasks; a task left empty is removed.

        Args:
            image_ids: the images to take out.

        Returns:
            How many images were removed.
        """
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
