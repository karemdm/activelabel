"""Saved models: ``<project>/models/m<N>/`` (see docs/TRAINING.md, step 6).

A model is the best weights of a finished training run, with the run's configuration and results
and a card (``model.json``); a public pretrained model (training.public_models) is only a card, with
``source: "public"`` and the mapping of its categories to the project's classes. Numbers are never
reused. One model of the project can be marked *current* (``models/current.json``); exports (ONNX) are
files next to the weights.
"""

from __future__ import annotations

import builtins
import shutil
import threading
from pathlib import Path
from typing import Any

from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.storage import now, read_json, write_json
from activelabel.training.catalog import get_model
from activelabel.training.runs import DONE, RunStore

FORMATS = ("onnx",)
EXPORT_FILES = {"onnx": "model.onnx"}
WEIGHTS = "best.pt"

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path.resolve(), threading.RLock())


class ModelNotFound(KeyError):
    """No model with this number."""


class ModelError(ValueError):
    """A readable reason why a model cannot be saved or changed."""


class ModelStore:
    """Saved models of one project."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.dir = project_dir / "models"
        self._lock = _lock_for(self.dir)

    def path(self, number: int) -> Path:
        """Folder of a model."""
        return self.dir / f"m{number}"

    def _numbers(self) -> builtins.list[int]:
        if not self.dir.exists():
            return []
        return sorted(int(p.name[1:]) for p in self.dir.glob("m*") if p.is_dir() and p.name[1:].isdigit())

    def _require(self, number: int) -> Path:
        folder = self.path(number)
        if not (folder / "model.json").exists():
            raise ModelNotFound(number)
        return folder

    def current(self) -> int | None:
        """Number of the current model, or ``None``."""
        value = (read_json(self.dir / "current.json", {}) or {}).get("model")
        return value if isinstance(value, int) and (self.path(value) / "model.json").exists() else None

    def save_run(self, run_number: int, note: str = "") -> dict[str, Any]:
        """Save the best weights of a finished run as the next model.

        The run's ``last.pt`` is removed (only the best weights matter from now on). The first model
        of a project becomes the current one.

        Raises:
            RunNotFound: unknown run.
            ModelError: the run is not finished, has no weights, or is already saved.
        """
        runs = RunStore(self.project_dir)
        run = runs.get(run_number)
        if run["status"] != DONE:
            raise ModelError("Only a finished training run can be saved as a model.")
        run_dir = runs.path(run_number)
        if not (run_dir / WEIGHTS).exists():
            raise ModelError("This training run has no weights to save.")
        config = runs.config(run_number)
        results = runs.results(run_number) or {}
        spec = get_model(config.model)
        metric = results.get("metric") or config.metric or (spec.default_metric() if spec else "")
        classes = run.get("classes")  # recorded when the run started (older runs: read the version)
        if not classes:
            try:
                classes = list(DatasetStore(self.project_dir).read(config.dataset)["classes"])
            except (DatasetNotFound, KeyError):
                raise ModelError(
                    "The data split version of this run was deleted and the run has no list of classes."
                ) from None
        classes = list(classes)
        with self._lock:
            for existing in self.list():
                if existing.get("run") == run_number:
                    raise ModelError(f"This training run is already saved as model m{existing['number']}.")
            counter_file = self.dir / "counter.json"
            counter = int((read_json(counter_file, {}) or {}).get("next", 1))
            number = max([counter, *[n + 1 for n in self._numbers()]])
            folder = self.path(number)
            folder.mkdir(parents=True)
            shutil.copy2(run_dir / WEIGHTS, folder / WEIGHTS)
            shutil.copy2(run_dir / "config.yaml", folder / "config.yaml")
            write_json(folder / "results.json", results)
            card: dict[str, Any] = {
                "number": number,
                "created": now(),
                "note": str(note or "").strip(),
                "run": run_number,
                "child": run.get("best_child"),
                "model": config.model,
                "task_kind": spec.task_kind if spec else "",
                "dataset": config.dataset,
                "dataset_fingerprint": config.dataset_fingerprint,
                "start": config.start,
                "mode": config.mode,
                "metric": metric,
                "value": (run.get("best") or {}).get("value"),
                "best_fold_value": (run.get("best") or {}).get("best_fold_value"),
                "classes": classes,
                "device": run.get("device"),
                "packages": run.get("packages"),
                "size": (folder / WEIGHTS).stat().st_size,
                "exports": {},
            }
            write_json(folder / "model.json", card)
            write_json(counter_file, {"next": number + 1})
            if self.current() is None:
                write_json(self.dir / "current.json", {"model": number})
        (run_dir / "last.pt").unlink(missing_ok=True)
        return self.get(number)

    def add_public(
        self, public_id: str, mapping: dict[str, str], project_classes: list[str], note: str = ""
    ) -> dict[str, Any]:
        """Add a public pretrained model (no local training) as the next model.

        Args:
            public_id: id in training.public_models.
            mapping: public category → project class; categories not in it are ignored.
            project_classes: the project's classes (every target of the mapping must be one).
            note: free text.

        Raises:
            ModelError: unknown public model, an empty mapping, or a category or class that does not
                exist.
        """
        from activelabel.training.public_models import get_public

        model = get_public(public_id)
        if model is None:
            raise ModelError("This public model does not exist.")
        mapping = {str(k): str(v) for k, v in (mapping or {}).items() if v}
        if not mapping:
            raise ModelError("Map at least one of the model's categories to a class of the project.")
        unknown = sorted(set(mapping) - set(model.classes()))
        if unknown:
            raise ModelError(f"The model has no category named {', '.join(unknown)}.")
        missing = sorted(set(mapping.values()) - set(project_classes))
        if missing:
            raise ModelError(f"The project has no class named {', '.join(missing)}.")
        with self._lock:
            counter_file = self.dir / "counter.json"
            counter = int((read_json(counter_file, {}) or {}).get("next", 1))
            number = max([counter, *[n + 1 for n in self._numbers()]])
            folder = self.path(number)
            folder.mkdir(parents=True)
            card: dict[str, Any] = {
                "number": number,
                "created": now(),
                "note": str(note or "").strip()[:500],
                "source": "public",
                "public": model.id,
                "run": None,
                "model": model.builder,
                "label": model.label,
                "task_kind": model.task_kind,
                "mapping": mapping,
                "classes": [c for c in project_classes if c in set(mapping.values())],
                "metric": "",
                "value": None,
                "size": None,
                "exports": {},
            }
            write_json(folder / "model.json", card)
            write_json(counter_file, {"next": number + 1})
            if self.current() is None:
                write_json(self.dir / "current.json", {"model": number})
        return self.get(number)

    def get(self, number: int) -> dict[str, Any]:
        """The card of a model, with ``current``.

        Raises:
            ModelNotFound: unknown number.
        """
        card = dict(read_json(self._require(number) / "model.json"))
        card["current"] = self.current() == number
        return card

    def results(self, number: int) -> dict[str, Any]:
        """``results.json`` of a model (the run's results when it was saved)."""
        return dict(read_json(self._require(number) / "results.json", {}) or {})

    def list(self) -> builtins.list[dict[str, Any]]:
        """Cards of every model, newest first."""
        current = self.current()
        out = []
        for n in reversed(self._numbers()):
            card = read_json(self.path(n) / "model.json")
            if card:
                out.append({**card, "current": current == n})
        return out

    def update(self, number: int, **fields: Any) -> dict[str, Any]:
        """Change fields of ``model.json`` (``number``, ``created`` and ``run`` stay).

        Raises:
            ModelNotFound: unknown number.
        """
        with self._lock:
            folder = self._require(number)
            card = dict(read_json(folder / "model.json"))
            card.update({k: v for k, v in fields.items() if k not in ("number", "created", "run", "current")})
            write_json(folder / "model.json", card)
        return self.get(number)

    def set_export(self, number: int, fmt: str, **fields: Any) -> dict[str, Any]:
        """Change the state of one export (``status``, ``error``, ``size``…).

        Raises:
            ModelNotFound: unknown number.
        """
        with self._lock:
            folder = self._require(number)
            card = dict(read_json(folder / "model.json"))
            exports = dict(card.get("exports") or {})
            exports[fmt] = {**(exports.get(fmt) or {}), **fields}
            card["exports"] = exports
            write_json(folder / "model.json", card)
        return self.get(number)

    def set_current(self, number: int | None) -> None:
        """Mark a model as the current one (``None``: no current model).

        Raises:
            ModelNotFound: unknown number.
        """
        with self._lock:
            if number is not None:
                self._require(number)
            write_json(self.dir / "current.json", {"model": number})

    def file(self, number: int, name: str) -> Path:
        """Path of the weights (``best.pt``) or of an export file of a model.

        Raises:
            ModelNotFound: unknown model, or the file does not exist.
        """
        if name not in (WEIGHTS, *EXPORT_FILES.values()):
            raise ModelNotFound(f"{number}/{name}")
        path = self._require(number) / name
        if not path.exists():
            raise ModelNotFound(f"{number}/{name}")
        return path

    def delete(self, number: int) -> None:
        """Remove a model and its files (its number is not reused; a current model stops being current).

        Raises:
            ModelNotFound: unknown number.
            ModelError: an export of the model is running.
            OSError: a file of the model is in use (nothing was removed).
        """
        with self._lock:
            card = self.get(number)
            if any(
                (e or {}).get("status") in ("queued", "running") for e in (card.get("exports") or {}).values()
            ):
                raise ModelError("An export of this model is running: wait for it to end.")
            # renamed first: if a file is in use the rename fails and the model stays as it was
            trash = self.dir / f".deleting-m{number}"
            shutil.rmtree(trash, ignore_errors=True)
            self.path(number).rename(trash)
            shutil.rmtree(trash, ignore_errors=True)
            if card["current"]:
                write_json(self.dir / "current.json", {"model": None})
