"""Which FiftyOne dataset belongs to a project: a name recorded in ``<project>/fiftyone.json``.

Kept apart from :mod:`activelabel.fiftyone_bridge` so reading the record does not load FiftyOne.
"""

from __future__ import annotations

from pathlib import Path

from activelabel.storage import read_json, write_json

RECORD_FILE = "fiftyone.json"


def recorded_name(project_dir: Path) -> str | None:
    """Dataset name recorded for a project, or None when it has none yet."""
    data = read_json(project_dir / RECORD_FILE)
    name = data.get("dataset") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def record_name(project_dir: Path, name: str) -> None:
    """Remember the dataset name of a project."""
    write_json(project_dir / RECORD_FILE, {"dataset": name})
