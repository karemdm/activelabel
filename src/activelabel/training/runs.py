"""Training runs on disk: ``<project>/runs/r<N>/`` (see docs/TRAINING.md).

Files of a run: ``config.yaml``, ``run.json`` (status, times, children), ``metrics.jsonl`` (one
line per epoch), ``results.json``. Numbers are never reused (a counter file).
"""

from __future__ import annotations

import builtins
import json
import shutil
import threading
from pathlib import Path
from typing import Any

from activelabel.storage import now, read_json, write_json
from activelabel.training.config import RunConfig

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
STATUSES = (QUEUED, RUNNING, DONE, FAILED, CANCELLED)

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """One lock per folder, shared by every store object of the same project."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


def read_jsonl(file: Path) -> list[dict[str, Any]]:
    """Lines of a JSON-lines file (none when it is missing; a half-written last line is skipped)."""
    if not file.exists():
        return []
    out = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class RunNotFound(KeyError):
    """No run with this number."""


class RunBusy(ValueError):
    """The run is running and cannot be deleted."""


class RunStore:
    """Runs of one project."""

    def __init__(self, project_dir: Path) -> None:
        self.dir = project_dir / "runs"
        self._lock = _lock_for(self.dir)

    def path(self, number: int) -> Path:
        """Folder of a run."""
        return self.dir / f"r{number}"

    def _numbers(self) -> list[int]:
        if not self.dir.exists():
            return []
        return sorted(int(p.name[1:]) for p in self.dir.glob("r*") if p.is_dir() and p.name[1:].isdigit())

    def _require(self, number: int) -> Path:
        folder = self.path(number)
        if not (folder / "run.json").exists():
            raise RunNotFound(number)
        return folder

    def create(self, config: RunConfig) -> dict[str, Any]:
        """Store a new run as queued, with the next number. Returns its ``run.json``."""
        with self._lock:
            counter_file = self.dir / "counter.json"
            counter = int((read_json(counter_file, {}) or {}).get("next", 1))
            number = max([counter, *[n + 1 for n in self._numbers()]])
            folder = self.path(number)
            folder.mkdir(parents=True)
            (folder / "config.yaml").write_text(config.to_yaml(), encoding="utf-8", newline="\n")
            run = {"number": number, "status": QUEUED, "created": now(), "children": []}
            write_json(folder / "run.json", run)
            write_json(counter_file, {"next": number + 1})
        return run

    def get(self, number: int) -> dict[str, Any]:
        """The ``run.json`` of a run.

        Raises:
            RunNotFound: unknown number.
        """
        return dict(read_json(self._require(number) / "run.json"))

    def config(self, number: int) -> RunConfig:
        """The saved configuration of a run.

        Raises:
            RunNotFound: unknown number.
        """
        return RunConfig.from_yaml((self._require(number) / "config.yaml").read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """``run.json`` of every run, newest first."""
        out = []
        for n in reversed(self._numbers()):
            data = read_json(self.path(n) / "run.json")
            if data:
                out.append(data)
        return out

    def update(self, number: int, **fields: Any) -> dict[str, Any]:
        """Change fields of ``run.json`` atomically (``number`` and ``created`` stay).

        Raises:
            RunNotFound: unknown number.
            ValueError: unknown status.
        """
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"Unknown status '{fields['status']}'.")
        with self._lock:
            run = self.get(number)
            run.update({k: v for k, v in fields.items() if k not in ("number", "created")})
            write_json(self.path(number) / "run.json", run)
        return run

    def update_child(
        self, number: int, index: int, top: dict[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Change fields of one child of a grid or K-fold run (and ``top`` fields of the run itself).

        Raises:
            RunNotFound: unknown number.
            IndexError: no child at ``index``.
        """
        with self._lock:
            run = self.get(number)
            children = list(run.get("children") or [])
            children[index] = {**children[index], **fields}
            run["children"] = children
            run.update({k: v for k, v in (top or {}).items() if k not in ("number", "created")})
            write_json(self.path(number) / "run.json", run)
        return run

    def child_path(self, number: int, child: str) -> Path:
        """Folder of a child (``c1``, ``c2``...) of a grid or K-fold run.

        Raises:
            RunNotFound: unknown run or child.
        """
        folder = self._require(number) / child
        if not (child.startswith("c") and child[1:].isdigit() and (folder / "config.yaml").exists()):
            raise RunNotFound(f"{number}/{child}")
        return folder

    def child_config(self, number: int, child: str) -> RunConfig:
        """Configuration of one child (single mode, one fold)."""
        text = (self.child_path(number, child) / "config.yaml").read_text(encoding="utf-8")
        return RunConfig.from_yaml(text)

    def metrics(self, number: int, child: str | None = None) -> builtins.list[dict[str, Any]]:
        """Lines of ``metrics.jsonl`` (empty before training; a half-written last line is skipped)."""
        folder = self.child_path(number, child) if child else self._require(number)
        return read_jsonl(folder / "metrics.jsonl")

    def results(self, number: int, child: str | None = None) -> dict[str, Any] | None:
        """``results.json`` of a finished run (or child), or ``None``."""
        folder = self.child_path(number, child) if child else self._require(number)
        return read_json(folder / "results.json")

    def delete(self, number: int) -> None:
        """Remove a run and its files (its number is not reused).

        Raises:
            RunNotFound: unknown number.
            RunBusy: the run is running.
        """
        with self._lock:
            if self.get(number)["status"] == RUNNING:
                raise RunBusy(number)
            shutil.rmtree(self.path(number))
