"""Run configuration: what is needed to repeat a training run (see docs/TRAINING.md).

A configuration never holds a device or a path: those are decided when the run starts.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any

import yaml

from activelabel.training.catalog import METRICS, STARTS, ModelSpec, get_model

MAX_GRID = 50
MODES = ("single", "grid")
KEYS = ("dataset", "dataset_fingerprint", "fold", "model", "start", "mode", "metric", "params", "grid")


class ConfigError(ValueError):
    """The configuration is not valid (the message is readable)."""


@dataclass
class RunConfig:
    """Options of a run.

    Attributes:
        dataset: dataset version number.
        dataset_fingerprint: fingerprint of that version when the run was created.
        fold: ``None`` (version without folds), ``"all"`` or a fold index (K-fold versions).
        model: catalogue id.
        start: ``transfer``, ``finetune`` or ``scratch``.
        mode: ``single`` or ``grid``.
        params: hyperparameters set explicitly (the rest take the catalogue default).
        grid: hyperparameter name -> list of values (grid mode); ``start`` may be a grid key.
        metric: selection metric ("" = the default of the model's kind of task).
    """

    dataset: int
    model: str
    fold: int | str | None = None
    start: str = "transfer"
    mode: str = "single"
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    metric: str = ""
    dataset_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in a stable key order (params and grid sorted by name)."""
        return {
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "fold": self.fold,
            "model": self.model,
            "start": self.start,
            "mode": self.mode,
            "metric": self.metric,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "grid": {k: list(self.grid[k]) for k in sorted(self.grid)},
        }

    @classmethod
    def from_dict(cls, data: Any) -> RunConfig:
        """Build from a dict (API body or YAML).

        Raises:
            ConfigError: not a mapping, unknown key (for example a device or a path), or a missing
                dataset or model.
        """
        if not isinstance(data, dict):
            raise ConfigError("The configuration must be a mapping of options.")
        unknown = set(data) - set(KEYS)
        if unknown:
            raise ConfigError(f"Unknown option(s): {', '.join(sorted(map(str, unknown)))}.")
        for key in ("dataset", "model"):
            if data.get(key) in (None, ""):
                raise ConfigError(f"Choose the {key}.")
        params, grid = data.get("params") or {}, data.get("grid") or {}
        if not isinstance(params, dict) or not isinstance(grid, dict):
            raise ConfigError("'params' and 'grid' must be mappings.")
        if any(not isinstance(v, list) for v in grid.values()):
            raise ConfigError("Every grid entry must be a list of values.")
        dataset = data["dataset"]
        if isinstance(dataset, bool) or not isinstance(dataset, int):
            raise ConfigError("The dataset version must be a number.")
        return cls(
            dataset=dataset,
            model=str(data["model"]),
            fold=data.get("fold"),
            start=data.get("start") or "transfer",
            mode=data.get("mode") or "single",
            params=dict(params),
            grid={str(k): list(v) for k, v in grid.items()},
            metric=data.get("metric") or "",
            dataset_fingerprint=data.get("dataset_fingerprint") or "",
        )

    def to_yaml(self) -> str:
        """The ``config.yaml`` text."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> RunConfig:
        """Inverse of :meth:`to_yaml`.

        Raises:
            ConfigError: invalid YAML or content.
        """
        try:
            return cls.from_dict(yaml.safe_load(text))
        except yaml.YAMLError:
            raise ConfigError("The configuration is not valid YAML.") from None

    def grid_size(self) -> int:
        """How many runs the grid makes (1 without a grid)."""
        return math.prod(len(v) for v in self.grid.values()) if self.grid else 1

    def effective_params(self, model: ModelSpec | None = None) -> dict[str, Any]:
        """Every parameter of the model (catalogue default unless set here), ``start`` included."""
        spec = model or get_model(self.model)
        if spec is None:
            raise ConfigError(f"Unknown model '{self.model}'.")
        values = {p.name: p.default for p in spec.param_specs()}
        values.update(self.params)
        values["start"] = self.start
        return values

    def is_composite(self) -> bool:
        """True for a grid search or a run over every fold: it trains several children."""
        return self.mode == "grid" or self.fold == "all"

    def children(self, folds: int) -> list[RunConfig]:
        """Single-mode, single-fold configurations of the trainings this run makes.

        Ordered by grid combination, then fold (``folds`` children per combination when the fold is
        ``"all"``; one otherwise).
        """
        fold_list: list[int | str | None] = list(range(folds)) if self.fold == "all" else [self.fold]
        return [replace(c, fold=f, mode="single", grid={}) for c in self.expand() for f in fold_list]

    def expand(self) -> list[RunConfig]:
        """One single-mode configuration per grid combination, in a deterministic order.

        Grid names are taken sorted; the first name changes slowest (``itertools.product``).
        Without a grid it returns this configuration alone.
        """
        if not self.grid:
            return [self]
        names = sorted(self.grid)
        out = []
        for combo in itertools.product(*(self.grid[n] for n in names)):
            values = dict(zip(names, combo, strict=True))
            start = values.pop("start", self.start)
            out.append(
                RunConfig(
                    dataset=self.dataset,
                    model=self.model,
                    fold=self.fold,
                    start=start,
                    mode="single",
                    params={**self.params, **values},
                    metric=self.metric,
                    dataset_fingerprint=self.dataset_fingerprint,
                )
            )
        return out


def validate(config: RunConfig, project_type: str, folds: int = 0) -> RunConfig:
    """Check a configuration against the catalogue and the dataset version.

    Args:
        config: the configuration.
        project_type: ``classification``, ``detection``, ``segmentation`` or ``keypoints``.
        folds: number of folds of the dataset version (0 when it has none).

    Returns:
        A normalised copy: default metric filled in, fold ``"all"`` for a K-fold version.

    Raises:
        ConfigError: with a readable message.
    """
    model = get_model(config.model)
    if model is None:
        raise ConfigError(f"Unknown model '{config.model}'.")
    if project_type not in model.types:
        raise ConfigError(f"The model '{config.model}' is not for {project_type} projects.")
    if config.mode not in MODES:
        raise ConfigError("The mode must be 'single' or 'grid'.")
    if config.start not in STARTS:
        raise ConfigError(f"The start must be one of: {', '.join(STARTS)}.")
    if config.start != "scratch" and not model.supports_pretrained:
        raise ConfigError(f"The model '{config.model}' has no pretrained weights: start from scratch.")

    fold = config.fold
    if folds:
        if fold is None:
            fold = "all"
        elif fold != "all" and (isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < folds):
            raise ConfigError(f"The fold must be 'all' or a number from 0 to {folds - 1}.")
    elif fold is not None:
        raise ConfigError("This dataset version has no folds: leave the fold empty.")

    metrics, default = METRICS[model.task_kind]
    metric = config.metric or default
    if metric not in metrics:
        raise ConfigError(f"The selection metric must be one of: {', '.join(metrics)}.")

    params = _check_values(model, config.params)
    grid: dict[str, list[Any]] = {}
    for name, values in config.grid.items():
        if name in config.params:
            raise ConfigError(f"'{name}' is both a fixed value and a grid list: use only one.")
        if not values:
            raise ConfigError(f"The grid list of '{name}' is empty.")
        checked = [_check_one(model, name, v) for v in values]
        if len({repr(v) for v in checked}) != len(checked):
            raise ConfigError(f"The grid list of '{name}' repeats a value.")
        grid[name] = checked
    if config.mode == "single" and grid:
        raise ConfigError("A single run has no grid: use the grid mode or remove the lists.")
    if config.mode == "grid" and not grid:
        raise ConfigError("Grid mode needs at least one list of values.")
    size = math.prod(len(v) for v in grid.values()) if grid else 1
    total = size * (folds if fold == "all" else 1)
    if total > MAX_GRID:
        what = "The grid makes" if fold != "all" else "The grid and the folds make"
        raise ConfigError(f"{what} {total} trainings; at most {MAX_GRID} are allowed.")

    return RunConfig(
        dataset=config.dataset,
        model=config.model,
        fold=fold,
        start=config.start,
        mode=config.mode,
        params=params,
        grid=grid,
        metric=metric,
        dataset_fingerprint=config.dataset_fingerprint,
    )


def _check_one(model: ModelSpec, name: str, value: Any) -> Any:
    spec = model.param(name)
    if spec is None:
        raise ConfigError(f"'{name}' is not a hyperparameter of {model.id}.")
    try:
        return spec.check(value)
    except ValueError as error:
        raise ConfigError(str(error)) from None


def _check_values(model: ModelSpec, values: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, value in values.items():
        if name == "start":
            raise ConfigError("Set the start with the 'start' option, not inside 'params'.")
        out[name] = _check_one(model, name, value)
    return out
