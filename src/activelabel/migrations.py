"""Versions of the stored data and the migrations that run when the server starts.

The data a workspace holds has a format version. JSON objects carry it in ``format_version``
(:data:`~activelabel.storage.FORMAT_KEY`); a file without it is version 0. The files the code before the field
read with strict code (:data:`~activelabel.storage.UNVERSIONED_FILES`), the lists (``images.json``,
``videos.json``) and the counters have the version of the workspace, kept in ``<workspace>/format.json``
(version 0 when that file does not exist: the formats of these files have not changed since).

A :class:`Migration` takes the files it names from the version before it to its own. :func:`migrate` runs the
ones a workspace still needs, when the server starts:

* nothing is written when no pending step changes a file (version 1 only added the field, which the code
  reads as absent: an old workspace opens as it is);
* the files the steps touch are copied to ``<workspace>/format-backups/`` first;
* a step may only **add** data (new keys, at any depth); one that removes or changes a value is refused
  before anything is written, so the release before it still reads every migrated file;
* it is idempotent: a file that already says the step's version is skipped, and the workspace version moves
  on after each step, so running it again (or after an interruption) changes nothing more. A step's
  ``change`` must itself be idempotent, for the unversioned files of an interrupted step;
* a workspace newer than this code is left alone and read as it is (the newer fields are ignored).
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.storage import FORMAT_KEY, FORMAT_VERSION, UNVERSIONED_FILES, DamagedFile, now, write_json

log = logging.getLogger(__name__)

WORKSPACE_FILE = "format.json"
BACKUP_DIR = "format-backups"


@dataclass(frozen=True)
class Migration:
    """One step of the stored format.

    Attributes:
        version: The version the step produces (the one before it is ``version - 1``).
        summary: What changed, in a sentence (written to the log).
        patterns: Files the step changes, as globs relative to the workspace (``projects/*/tasks.json``).
        change: The new content of a file from its current one (only additions); None when the step only
            declares a version and rewrites nothing.
    """

    version: int
    summary: str
    patterns: tuple[str, ...] = ()
    change: Callable[[Any], Any] | None = None


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "JSON objects carry format_version; a file without it is version 0 and is read the same way.",
    ),
    Migration(
        2,
        "Image tags in <project>/tags.json and tags on labels in annotations.json (new optional keys: "
        "nothing is rewritten, older code ignores them).",
    ),
)


class MigrationError(CodedError):
    """A migration step would remove or change stored data (only additions are allowed)."""

    def __init__(self, version: int, file: str, where: str) -> None:
        """Keep the step, the file and the place of the refused change.

        Args:
            version: The version of the refused step.
            file: The file, relative to the workspace.
            where: Path inside the file of the value removed or changed (``tasks > 0 > name``).
        """
        super().__init__(
            "format_migration_refused",
            f"The migration to format {version} would remove or change '{where}' in {file}; "
            "nothing was changed.",
            version=version,
            file=file,
            where=where,
        )


def workspace_version(workspace: Path) -> int:
    """The format version of a workspace (``format_version`` of ``format.json``; 0 when it has none).

    Args:
        workspace: The workspace folder.

    Returns:
        The version the last migration left, or 0.

    Raises:
        DamagedFile: ``format.json`` cannot be read.
    """
    path = workspace / WORKSPACE_FILE
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DamagedFile(
            f"The file {WORKSPACE_FILE} is damaged and cannot be read ({exc}): {path}", path.name
        ) from exc
    return file_version(data, 0)


def file_version(data: Any, default: int) -> int:
    """The format version of a stored value.

    Args:
        data: The content of a file, as parsed.
        default: The version when the file does not say its own (the workspace's).

    Returns:
        ``format_version`` of an object that has it, else ``default``.
    """
    if isinstance(data, dict) and isinstance(data.get(FORMAT_KEY), int):
        return int(data[FORMAT_KEY])
    return default


def removed_or_changed(old: Any, new: Any, where: str = "") -> str | None:
    """The first place where ``new`` lost or changed a value of ``old`` (``format_version`` aside).

    Args:
        old: The content before the step.
        new: The content after it.
        where: The path of ``old`` inside the file, for the answer.

    Returns:
        The path (``tasks > 0 > name``) of a value removed or changed; None when ``new`` only adds.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in old.items():
            inner = f"{where} > {key}" if where else str(key)
            if key == FORMAT_KEY:
                continue
            if key not in new:
                return inner
            found = removed_or_changed(value, new[key], inner)
            if found:
                return found
        return None
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return where or "(whole file)"
        for index, (before, after) in enumerate(zip(old, new, strict=True)):
            found = removed_or_changed(before, after, f"{where} > {index}" if where else str(index))
            if found:
                return found
        return None
    return None if old == new and type(old) is type(new) else (where or "(whole file)")


def _files(workspace: Path, patterns: Iterable[str]) -> list[Path]:
    """The files of the workspace that match any of the patterns, sorted, without the backups.

    Args:
        workspace: The workspace folder.
        patterns: Globs relative to the workspace.

    Returns:
        The matching files.
    """
    found = {p for pattern in patterns for p in workspace.glob(pattern) if p.is_file()}
    return sorted(p for p in found if BACKUP_DIR not in p.relative_to(workspace).parts)


def _back_up(workspace: Path, files: list[Path], start: int, target: int) -> Path:
    """Copy the files a migration will touch, keeping their place in the workspace.

    Args:
        workspace: The workspace folder.
        files: The files to copy.
        start: The version of the workspace before the migration.
        target: The version after it.

    Returns:
        The folder of the copy (``format-backups/v<start>-to-v<target>-<time>``).
    """
    stamp = now().replace(":", "").replace("+", "_")
    folder = workspace / BACKUP_DIR / f"v{start}-to-v{target}-{stamp}"
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = workspace / BACKUP_DIR / f"v{start}-to-v{target}-{stamp}-{suffix}"
    for path in files:
        target_path = folder / path.relative_to(workspace)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)
    marker = workspace / WORKSPACE_FILE
    if marker.exists():
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker, folder / WORKSPACE_FILE)
    return folder


def _run_step(workspace: Path, step: Migration, start: int) -> int:
    """Apply one step to every file it names that is still older than it.

    Every new content is computed and checked before the first file is written.

    Args:
        workspace: The workspace folder.
        step: The step (its ``change`` is not None).
        start: The version of the workspace before the step (the version of an unversioned file).

    Returns:
        How many files were rewritten.

    Raises:
        MigrationError: the step would remove or change a value.
    """
    assert step.change is not None
    writes: list[tuple[Path, Any, bool]] = []
    for path in _files(workspace, step.patterns):
        text = path.read_text(encoding="utf-8")
        old = json.loads(text)
        if file_version(old, start) >= step.version:
            continue
        new = step.change(copy.deepcopy(old))
        where = removed_or_changed(old, new)
        if where is not None:
            raise MigrationError(step.version, path.relative_to(workspace).as_posix(), where)
        if isinstance(new, dict) and path.name not in UNVERSIONED_FILES:
            new = {FORMAT_KEY: step.version, **{k: v for k, v in new.items() if k != FORMAT_KEY}}
        writes.append((path, new, "\n" in text.strip()))
    for path, new, readable in writes:
        write_json(path, new, indent=1 if readable else None)
    return len(writes)


def migrate(workspace: Path, migrations: tuple[Migration, ...] = MIGRATIONS) -> Path | None:
    """Bring the stored data of a workspace to the newest format this code knows.

    Args:
        workspace: The workspace folder (it may not exist yet).
        migrations: The steps, oldest first (injectable for tests).

    Returns:
        The folder of the backup made before the first change; None when nothing had to change.

    Raises:
        MigrationError: a step would remove or change data; the files of that step are untouched, the
            steps before it stay applied and the backup holds the files as they were.
    """
    if not workspace.is_dir():
        return None
    start = workspace_version(workspace)
    newest = max((m.version for m in migrations), default=FORMAT_VERSION)
    if start > newest:
        log.warning(
            "The workspace has format %s, newer than this version (%s): it is read as is.", start, newest
        )
        return None
    steps = [m for m in migrations if m.version > start and m.change is not None]
    if not steps:
        return None
    backup = _back_up(workspace, _files(workspace, (p for m in steps for p in m.patterns)), start, newest)
    log.info("Workspace format %s -> %s; copy of the files before it: %s", start, newest, backup)
    version = start
    for step in steps:
        changed = _run_step(workspace, step, version)
        version = step.version
        _mark(workspace, version)
        log.info("Format %s: %s (%s files)", step.version, step.summary, changed)
    if version < newest:  # the last steps only declare a version
        _mark(workspace, newest)
    return backup


def _mark(workspace: Path, version: int) -> None:
    """Record the format version of the workspace in ``format.json``.

    Args:
        workspace: The workspace folder.
        version: The version its data now has.
    """
    write_json(workspace / WORKSPACE_FILE, {FORMAT_KEY: version, "migrated": now()})
