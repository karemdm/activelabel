"""Global scheduler of the work of the app: one lane per device, a limit per lane, first come first served.

Heavy work (training runs with their grid and fold children, pre-annotations, curation analyses, ONNX
exports, video extractions) runs on a *device lane*: ``cpu``, ``cuda:0``, ``cuda:1``... or ``mps``. Every
lane runs at most its limit of jobs at a time (by default one per GPU and one for the CPU, see
:class:`Limits`); the others wait in the order they were asked for, whatever project they belong to. The
light activities (import, sync, sending annotations...) share the ``light`` lane, one at a time.

The waiting and running jobs are written to ``<workspace>/queue.json`` on every change, so a restart
knows what was waiting (it is queued again, in the same order) and what was running (it was
*interrupted*: a training run can be resumed, docs/TRAINING.md). See docs/DESIGN.md, "Work queue".
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from activelabel.constants import QUEUED, RUNNING
from activelabel.storage import DamagedFile, now, read_json, write_json

LOG = logging.getLogger(__name__)

#: Lane of the activities that do not compute much (import, sync, annotation transfer...).
LIGHT = "light"
#: Lane of the work done by the processor.
CPU = "cpu"


class Queued(Protocol):
    """What the scheduler reads of a job (:class:`activelabel.jobs.Job` has it all)."""

    id: str
    project_id: int
    kind: str
    device: str
    params: dict[str, Any]
    status: str
    done: int
    total: int
    message: str
    created: str
    started: str


@dataclass(frozen=True)
class Limits:
    """How many jobs may run at the same time on each lane.

    Attributes:
        cpu: jobs on the ``cpu`` lane.
        gpu: jobs on each GPU lane (``cuda:N``, ``mps``).
        light: light activities at the same time.
    """

    cpu: int = 1
    gpu: int = 1
    light: int = 1

    def of(self, lane: str) -> int:
        """The limit of one lane.

        Args:
            lane: ``light``, ``cpu`` or a GPU (``cuda:N``, ``mps``).

        Returns:
            How many jobs the lane may run at the same time (at least 1).
        """
        if lane == LIGHT:
            return max(1, self.light)
        if lane == CPU:
            return max(1, self.cpu)
        return max(1, self.gpu)


def device_lane(device: str, gpus: int = 0, mps: bool = False) -> str:
    """The lane of a device choice; ``auto`` is the device the worker will pick.

    Args:
        device: ``auto``, ``cpu``, ``cuda:N`` or ``mps`` (empty means ``auto``).
        gpus: NVIDIA GPUs of the computer (``auto`` picks the first one).
        mps: whether the computer has an Apple GPU (``auto`` picks it when there is no NVIDIA GPU).

    Returns:
        ``cuda:0``, ``mps`` or ``cpu`` for ``auto``; the device itself otherwise.
    """
    if device in ("", "auto"):
        return "cuda:0" if gpus > 0 else "mps" if mps else CPU
    return device


def seconds_between(start: str, end: str) -> float:
    """Seconds between two :func:`activelabel.storage.now` timestamps.

    Args:
        start: the earlier timestamp.
        end: the later timestamp.

    Returns:
        The seconds (0 when a timestamp cannot be read).
    """
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def remaining_seconds(job: Queued, at: str) -> float | None:
    """Estimated seconds left of a running job, from its progress so far.

    Args:
        job: the running job.
        at: the current timestamp.

    Returns:
        The estimate, or None while the job has not reported any finished step out of a known total.
    """
    if job.status != RUNNING or job.total <= 0 or job.done <= 0 or not job.started:
        return None
    elapsed = seconds_between(job.started, at)
    return round(elapsed / job.done * max(0, job.total - job.done), 1)


class Scheduler:
    """Global queue: decides when each job starts, keeps the order and writes it to disk."""

    def __init__(self, path: Path | None = None, limits: Limits | None = None) -> None:
        """Create an empty queue.

        Args:
            path: ``queue.json`` to keep the queue in across restarts; None keeps it in memory only.
            limits: jobs per lane at the same time (one per lane by default).
        """
        self.path = path
        self.limits = limits or Limits()
        self._lock = threading.Lock()
        self._waiting: list[tuple[Queued, Callable[[], None]]] = []
        self._running: list[Queued] = []
        self._closed = False
        self._interrupted: list[dict[str, Any]] = []
        self._listeners: list[Callable[[], None]] = []
        self._saved: list[dict[str, Any]] | None = None

    # -- queue ------------------------------------------------------------

    def add(self, job: Queued, start: Callable[[], None]) -> None:
        """Put a job at the end of its lane; it starts as soon as the lane has room.

        Args:
            job: the queued job (its ``device`` is the lane).
            start: runs the job (called in a new thread) and must call :meth:`finished` at the end.
        """
        with self._lock:
            if self._closed:
                return
            self._waiting.append((job, start))
        self._dispatch()

    def discard(self, job: Queued) -> bool:
        """Take a waiting job out of the queue (it will never start).

        Args:
            job: the job.

        Returns:
            True when the job was waiting; False when it had already started (or was never queued).
        """
        with self._lock:
            index = next((i for i, (j, _) in enumerate(self._waiting) if j.id == job.id), None)
            if index is not None:
                self._waiting.pop(index)
        if index is None:
            return False
        self._dispatch()
        return True

    def finished(self, job: Queued) -> None:
        """A running job ended: free its place and start the next ones.

        Args:
            job: the job that ended (done, failed or cancelled).
        """
        with self._lock:
            self._running = [j for j in self._running if j.id != job.id]
        self._dispatch()

    def _dispatch(self) -> None:
        """Start every waiting job whose lane has room, oldest first, then save and tell the listeners."""
        starts: list[Callable[[], None]] = []
        with self._lock:
            if not self._closed:
                used: dict[str, int] = {}
                for job in self._running:
                    used[job.device] = used.get(job.device, 0) + 1
                blocked: set[str] = set()
                for job, start in list(self._waiting):
                    lane = job.device
                    if lane in blocked or used.get(lane, 0) >= self.limits.of(lane):
                        blocked.add(lane)  # first come first served inside a lane
                        continue
                    self._waiting.remove((job, start))
                    self._running.append(job)
                    used[lane] = used.get(lane, 0) + 1
                    starts.append(start)
            self._save()
            listeners = list(self._listeners)
        for start in starts:
            threading.Thread(target=start, name="activelabel-job", daemon=True).start()
        for listener in listeners:
            try:
                listener()
            except Exception:  # noqa: BLE001 - a listener must not stop the queue
                LOG.exception("A queue listener failed")

    def listen(self, listener: Callable[[], None]) -> None:
        """Call a function after every change of the queue (for example to write queue positions).

        Args:
            listener: called without arguments, outside the queue's lock.
        """
        with self._lock:
            self._listeners.append(listener)

    def close(self) -> None:
        """Stop starting jobs and stop writing ``queue.json`` (what it holds is kept for the restart)."""
        with self._lock:
            self._closed = True

    # -- reading ----------------------------------------------------------

    def position(self, job_id: str) -> int | None:
        """Place of a waiting job in its lane.

        Args:
            job_id: id of the job.

        Returns:
            1 for the next job of the lane, 2 for the one after it...; None when the job is not waiting.
        """
        with self._lock:
            waiting = [j for j, _ in self._waiting]
        for job in waiting:
            if job.id == job_id:
                return [j.id for j in waiting if j.device == job.device].index(job_id) + 1
        return None

    def snapshot(self, project_id: int | None = None) -> dict[str, Any]:
        """The queue as the interface shows it.

        Args:
            project_id: only the jobs of this project (the lanes still count every project); None for all.

        Returns:
            ``{limits, lanes, items, interrupted}``: see docs/DESIGN.md, "Work queue".
        """
        at = now()
        with self._lock:
            running = list(self._running)
            waiting = [j for j, _ in self._waiting]
            interrupted = [dict(e) for e in self._interrupted]
        lanes = sorted({j.device for j in (*running, *waiting)} | {CPU})
        items: list[dict[str, Any]] = []
        for job in running:
            items.append(self._item(job, RUNNING, None, remaining_seconds(job, at), None))
        for job in waiting:
            same = [j for j in waiting if j.device == job.device]
            ahead = [j for j in running if j.device == job.device] + same[: same.index(job)]
            items.append(self._item(job, QUEUED, same.index(job) + 1, None, self._wait(job, ahead, at)))
        if project_id is not None:
            items = [i for i in items if i["project_id"] == project_id]
            interrupted = [e for e in interrupted if e.get("project_id") == project_id]
        return {
            "limits": {
                "cpu": self.limits.of(CPU),
                "gpu": self.limits.of("cuda:0"),
                "light": self.limits.of(LIGHT),
            },
            "lanes": [
                {
                    "device": lane,
                    "limit": self.limits.of(lane),
                    "running": sum(1 for j in running if j.device == lane),
                    "queued": sum(1 for j in waiting if j.device == lane),
                }
                for lane in lanes
            ],
            "items": items,
            "interrupted": interrupted,
        }

    @staticmethod
    def _item(
        job: Queued, status: str, position: int | None, eta: float | None, wait: float | None
    ) -> dict[str, Any]:
        """One line of :meth:`snapshot`.

        Args:
            job: the job.
            status: ``running`` or ``queued`` as the scheduler sees it.
            position: place in the lane (waiting jobs only).
            eta: estimated seconds until it ends (running jobs only, when known).
            wait: estimated seconds until it starts (waiting jobs only, when known).

        Returns:
            The JSON-ready line.
        """
        return {
            "id": job.id,
            "project_id": job.project_id,
            "kind": job.kind,
            "device": job.device,
            "params": dict(job.params),
            "status": status,
            "position": position,
            "done": job.done,
            "total": job.total,
            "message": job.message,
            "created": job.created,
            "started": job.started,
            "eta_seconds": eta,
            "wait_seconds": wait,
        }

    def _wait(self, job: Queued, ahead: list[Queued], at: str) -> float | None:
        """Estimated seconds before a waiting job starts.

        Only known on a lane that runs one job at a time and when every job ahead is running with a
        known estimate (a waiting job has no estimate of its own).

        Args:
            job: the waiting job.
            ahead: the jobs of its lane that run or wait before it.
            at: the current timestamp.

        Returns:
            The estimate, or None when it is not known.
        """
        if self.limits.of(job.device) != 1:
            return None
        estimates = [remaining_seconds(j, at) for j in ahead]
        if any(e is None for e in estimates):
            return None
        return round(sum(e for e in estimates if e is not None), 1)

    # -- persistence ------------------------------------------------------

    @staticmethod
    def _entry(job: Queued, status: str) -> dict[str, Any]:
        """What ``queue.json`` keeps of a job.

        Args:
            job: the job.
            status: ``queued`` or ``running``.

        Returns:
            The JSON-ready entry.
        """
        return {
            "id": job.id,
            "project_id": job.project_id,
            "kind": job.kind,
            "device": job.device,
            "params": dict(job.params),
            "status": status,
            "created": job.created,
            "started": job.started,
        }

    def _save(self) -> None:
        """Write ``queue.json`` (caller holds the lock); a failure is logged, the queue goes on."""
        if self.path is None or self._closed:
            return
        entries = [self._entry(j, RUNNING) for j in self._running]
        entries += [self._entry(j, QUEUED) for j, _ in self._waiting]
        try:
            write_json(self.path, {"jobs": entries})  # write_json adds format_version
        except OSError:
            LOG.exception("Could not write %s", self.path)

    def saved(self) -> list[dict[str, Any]]:
        """The jobs ``queue.json`` held when the app started, oldest first (read once).

        The ones that were running are recorded as interrupted (:meth:`snapshot` lists them).

        Returns:
            The entries (``status`` is ``queued`` or ``running``); empty without a file or when it is damaged.
        """
        with self._lock:
            if self._saved is not None:
                return list(self._saved)
        try:
            data = read_json(self.path, {}) if self.path is not None else {}
        except DamagedFile:
            LOG.exception("The saved queue could not be read; it starts empty")
            data = {}
        jobs = data.get("jobs") if isinstance(data, dict) else None
        entries = [e for e in jobs if isinstance(e, dict)] if isinstance(jobs, list) else []
        with self._lock:
            self._saved = entries
            self._interrupted = [
                {**e, "status": "interrupted"} for e in entries if e.get("status") == RUNNING
            ]
        return list(entries)
