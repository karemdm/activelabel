"""Background jobs with progress and cancellation.

A job function receives a :class:`Progress`. It reports with ``progress.update(...)`` and
calls ``progress.check()`` between steps: when someone cancels the job, ``check`` raises
:class:`Cancelled` and the job stops at a safe point.
"""

from __future__ import annotations

import builtins
import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from activelabel.constants import CANCELLED, DONE, FAILED, FINISHED, QUEUED, RUNNING
from activelabel.storage import now
from activelabel.textcheck import message_of

LOG = logging.getLogger(__name__)


def database_failure(exc: BaseException) -> bool:
    """Whether an exception says the FiftyOne database did not answer (or could not be started).

    Args:
        exc: the exception raised.

    Returns:
        True for FiftyOne's ``ServiceListenTimeout``, pymongo's ``ServerSelectionTimeoutError`` and any
        error whose text names ``mongod``.
    """
    name = type(exc).__name__
    return name in ("ServiceListenTimeout", "ServerSelectionTimeoutError") or "mongod" in message_of(exc)


def readable_error(exc: BaseException) -> str:
    """Message shown to people: no class name (it stays in the log, with the traceback).

    FiftyOne database failures get a hint, with the technical detail at the end.

    Args:
        exc: the exception that ended the job.

    Returns:
        The text to show; a generic sentence when the exception carries no message.
    """
    name = type(exc).__name__
    detail = f"{name}: {exc}"
    message = message_of(exc)
    if database_failure(exc):
        return (
            "The FiftyOne database did not respond. Try again; if it keeps failing, restart ActiveLabel "
            f"(another program using FiftyOne may be holding it). Details: {detail}"
        )
    return message or "The activity failed (no details were given)."


class Cancelled(Exception):
    """The job was cancelled by the user."""


@dataclass
class Job:
    """State of one job, as the interface shows it."""

    id: str
    project_id: int
    kind: str
    status: str = QUEUED
    done: int = 0
    total: int = 0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # what the job is about (e.g. the run number)
    error: str = ""
    error_code: str = ""  # stable code of a failure the interface translates (see web/errors.py)
    error_params: dict[str, Any] = field(default_factory=dict)  # values for that translated text
    created: str = field(default_factory=now)
    started: str = ""
    finished: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            All the fields of the job as a dict.
        """
        return asdict(self)


class Progress:
    """Handle a job function uses to report progress and to notice cancellation."""

    def __init__(self, job: Job, cancel: threading.Event) -> None:
        """Bind the handle to one job.

        Args:
            job: the job whose numbers and message are updated.
            cancel: set by whoever cancels the job; ``check`` raises when it is set.
        """
        self._job = job
        self._cancel = cancel

    def update(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """Report progress (any argument may be left out). Also checks for cancellation.

        Args:
            done: steps finished so far.
            total: steps in all.
            message: what the job is doing, for the interface.

        Raises:
            Cancelled: when cancellation was requested.
        """
        if total is not None:
            self._job.total = total
        if done is not None:
            self._job.done = done
        if message is not None:
            self._job.message = message
        self.check()

    def check(self) -> None:
        """Stop here if the job was cancelled.

        Raises:
            Cancelled: when cancellation was requested.
        """
        if self._cancel.is_set():
            raise Cancelled


JobFunction = Callable[[Progress], dict[str, Any] | None]


class JobManager:
    """Runs jobs one at a time, in submission order, in a background thread."""

    def __init__(self, workers: int = 1, keep: int = 50) -> None:
        """Start the worker threads.

        Args:
            workers: how many jobs may run at the same time.
            keep: finished jobs kept in the list; the oldest finished ones are dropped past it.
        """
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="activelabel-job")
        self._jobs: dict[str, Job] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(
        self, project_id: int, kind: str, function: JobFunction, params: dict[str, Any] | None = None
    ) -> Job:
        """Queue a job and return its state (updated in place while it runs).

        Args:
            project_id: project the job belongs to.
            kind: what the job does (``import``, ``analyze``...).
            function: the work; receives a :class:`Progress` and may return a result dict.
            params: say what the job is about (for example ``{"number": 3}``) and are listed with it.

        Returns:
            The job, in the queued state.
        """
        job = Job(uuid.uuid4().hex[:12], project_id, kind, params=dict(params or {}))
        cancel = threading.Event()
        with self._lock:
            self._jobs[job.id] = job
            self._cancel[job.id] = cancel
            self._trim()
        self._pool.submit(self._run, job, cancel, function)
        return job

    def _run(self, job: Job, cancel: threading.Event, function: JobFunction) -> None:
        """Run one job in a worker thread and record how it ended.

        Args:
            job: the job to update.
            cancel: set when the job is cancelled; a job cancelled while queued never starts.
            function: the work of the job.
        """
        if cancel.is_set():
            job.status, job.finished = CANCELLED, now()
            return
        job.status, job.started = RUNNING, now()
        try:
            job.result = function(Progress(job, cancel)) or {}
            job.status = DONE
        except Cancelled:
            job.status = CANCELLED
        except Exception as exc:  # noqa: BLE001 - shown to the user, the service keeps running
            LOG.error("Job %s (%s) failed: %s", job.id, job.kind, traceback.format_exc(limit=5))
            job.error = readable_error(exc)
            code, params = getattr(exc, "code", ""), getattr(exc, "params", {})
            job.error_code = code if isinstance(code, str) else ""
            job.error_params = dict(params) if isinstance(params, dict) else {}
            job.result = {}
            job.status = FAILED  # last: a reader that sees FAILED finds the error and the traceback
        job.finished = now()

    def cancel(self, job_id: str) -> Job:
        """Ask a job to stop (queued jobs never start; running ones stop at their next check).

        Args:
            job_id: id of the job.

        Returns:
            The job (already finished jobs are returned unchanged).

        Raises:
            KeyError: unknown job.
        """
        with self._lock:
            job = self._jobs[job_id]
            event = self._cancel[job_id]
        if job.status not in FINISHED:
            event.set()
            if job.status == QUEUED:
                job.message = "Cancelling…"
        return job

    def get(self, job_id: str) -> Job:
        """One job.

        Args:
            job_id: id of the job.

        Returns:
            The job.

        Raises:
            KeyError: unknown job.
        """
        return self._jobs[job_id]

    def list(self, project_id: int | None = None) -> list[Job]:
        """Jobs, newest first (optionally of one project).

        Args:
            project_id: only the jobs of this project; None for all.

        Returns:
            The jobs, newest first.
        """
        with self._lock:  # submit and clear change the dict from other threads
            jobs = [j for j in self._jobs.values() if project_id is None or j.project_id == project_id]
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def active(self, project_id: int) -> builtins.list[Job]:
        """Queued or running jobs of a project.

        Args:
            project_id: the project.

        Returns:
            The jobs that have not finished.
        """
        return [j for j in self.list(project_id) if j.status not in FINISHED]

    def clear(self, project_id: int) -> int:
        """Remove the finished jobs of a project (queued and running ones stay).

        Args:
            project_id: the project.

        Returns:
            How many jobs were removed.
        """
        with self._lock:
            finished = [
                j.id for j in self._jobs.values() if j.project_id == project_id and j.status in FINISHED
            ]
            for job_id in finished:
                self._jobs.pop(job_id, None)
                self._cancel.pop(job_id, None)
        return len(finished)

    def _trim(self) -> None:
        """Drop the oldest finished jobs when there are more than ``keep`` jobs."""
        finished = [j for j in sorted(self._jobs.values(), key=lambda j: j.created) if j.status in FINISHED]
        for job in finished[: max(0, len(self._jobs) - self._keep)]:
            self._jobs.pop(job.id, None)
            self._cancel.pop(job.id, None)

    def shutdown(self) -> None:
        """Cancel everything and stop the worker."""
        with self._lock:
            events = list(self._cancel.values())
        for event in events:
            event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
