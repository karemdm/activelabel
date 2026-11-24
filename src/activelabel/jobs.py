"""Background jobs with progress and cancellation.

A job function receives a :class:`Progress`. It reports with ``progress.update(...)`` and
calls ``progress.check()`` between steps: when someone cancels the job, ``check`` raises
:class:`Cancelled` and the job stops at a safe point.
"""

from __future__ import annotations

import builtins
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from activelabel.storage import now
from activelabel.textcheck import message_of

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)


def readable_error(exc: BaseException) -> str:
    """Message shown to people: no class name (it stays in the traceback of the job result).

    FiftyOne database failures get a hint, with the technical detail at the end.
    """
    name = type(exc).__name__
    detail = f"{name}: {exc}"
    message = message_of(exc)
    if name in ("ServiceListenTimeout", "ServerSelectionTimeoutError") or "mongod" in message:
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
    error: str = ""
    created: str = field(default_factory=now)
    started: str = ""
    finished: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


class Progress:
    """Handle a job function uses to report progress and to notice cancellation."""

    def __init__(self, job: Job, cancel: threading.Event) -> None:
        self._job = job
        self._cancel = cancel

    def update(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """Report progress (any argument may be left out). Also checks for cancellation."""
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
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="activelabel-job")
        self._jobs: dict[str, Job] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, project_id: int, kind: str, function: JobFunction) -> Job:
        """Queue a job and return its state (updated in place while it runs)."""
        job = Job(uuid.uuid4().hex[:12], project_id, kind)
        cancel = threading.Event()
        with self._lock:
            self._jobs[job.id] = job
            self._cancel[job.id] = cancel
            self._trim()
        self._pool.submit(self._run, job, cancel, function)
        return job

    def _run(self, job: Job, cancel: threading.Event, function: JobFunction) -> None:
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
            job.error = readable_error(exc)
            job.result = {"traceback": traceback.format_exc(limit=5)}
            job.status = FAILED  # last: a reader that sees FAILED finds the error and the traceback
        job.finished = now()

    def cancel(self, job_id: str) -> Job:
        """Ask a job to stop (queued jobs never start; running ones stop at their next check).

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
        """One job. Raises KeyError when unknown."""
        return self._jobs[job_id]

    def list(self, project_id: int | None = None) -> list[Job]:
        """Jobs, newest first (optionally of one project)."""
        with self._lock:  # submit and clear change the dict from other threads
            jobs = [j for j in self._jobs.values() if project_id is None or j.project_id == project_id]
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def active(self, project_id: int) -> builtins.list[Job]:
        """Queued or running jobs of a project."""
        return [j for j in self.list(project_id) if j.status not in FINISHED]

    def clear(self, project_id: int) -> int:
        """Remove the finished jobs of a project (queued and running ones stay).

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
