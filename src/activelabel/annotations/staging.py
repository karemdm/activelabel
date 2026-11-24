"""Temporary storage of uploaded annotation files until the user decides (``<project>/staging/<token>/``)."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO

STAGING = "staging"
MAX_AGE_SECONDS = 24 * 3600
MAX_TOTAL_BYTES = 1024**3
ALLOWED = {".json", ".xml", ".txt", ".yaml", ".yml", ".names"}
_TOKEN = re.compile(r"^[0-9a-f]{16}$")


def _safe_rel(name: str) -> PurePosixPath:
    """Relative path of an upload name.

    Raises:
        ValueError: absolute, drive or ``..`` paths.
    """
    n = name.replace("\\", "/")
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if n.startswith("/") or (len(n) > 1 and n[1] == ":") or ".." in parts or not parts:
        raise ValueError(f"Unsafe file path in upload: {name}")
    return PurePosixPath(*parts)


def _copy(src: IO[bytes], dest: Path, budget: list[int]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while chunk := src.read(1 << 20):
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise ValueError("The annotation files are too large.")
            out.write(chunk)


def _check_token(token: str) -> None:
    if not _TOKEN.match(token):
        raise ValueError("Invalid staging token.")


def cleanup(project_dir: Path) -> None:
    """Remove staging folders (and their target files) not touched for 24 hours."""
    root = project_dir / STAGING
    if not root.exists():
        return
    limit = time.time() - MAX_AGE_SECONDS
    for child in root.iterdir():
        try:
            expired = child.stat().st_mtime < limit
        except OSError:  # gone meanwhile
            continue
        if expired:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def _extract_zip(handle: BinaryIO, target: Path, budget: list[int]) -> int:
    saved = 0
    with tempfile.TemporaryFile() as tmp:
        shutil.copyfileobj(handle, tmp)
        try:
            archive = zipfile.ZipFile(tmp)
        except zipfile.BadZipFile as exc:
            raise ValueError("The zip file is not valid.") from exc
        with archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            rels = [_safe_rel(i.filename) for i in infos]  # refuse the whole zip on any unsafe path
            for info, rel in zip(infos, rels, strict=True):
                if "__MACOSX" in rel.parts or rel.name.startswith("._") or rel.suffix.lower() not in ALLOWED:
                    continue
                with archive.open(info) as src:
                    _copy(src, target.joinpath(*rel.parts), budget)
                saved += 1
    return saved


def stage(project_dir: Path, files: list[tuple[str, BinaryIO]]) -> str:
    """Save uploaded annotation files (or zips of them) and return a token.

    Only json, xml, txt, yaml and names files are kept (other files are ignored). Staging folders older
    than 24 hours are removed first.

    Raises:
        ValueError: unsafe path (zip slip), invalid zip, too large, or no annotation file in the upload.
    """
    root = project_dir / STAGING
    cleanup(project_dir)
    token = secrets.token_hex(8)
    target = root / token
    target.mkdir(parents=True)
    budget = [MAX_TOTAL_BYTES]
    try:
        saved = 0
        for name, handle in files:
            rel = _safe_rel(name)
            if rel.suffix.lower() == ".zip":
                saved += _extract_zip(handle, target, budget)
            elif rel.suffix.lower() in ALLOWED:
                _copy(handle, target.joinpath(*rel.parts), budget)
                saved += 1
        if not saved:
            raise ValueError("No annotation files were found in the upload.")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return token


def load_staged(project_dir: Path, token: str) -> list[Path]:
    """Files of a staging folder, sorted.

    Raises:
        ValueError: invalid token.
        FileNotFoundError: the staging does not exist (discarded or expired).
    """
    _check_token(token)
    folder = project_dir / STAGING / token
    if not folder.is_dir():
        raise FileNotFoundError(f"Staging {token} not found.")
    return sorted(p for p in folder.rglob("*") if p.is_file())


def _target_file(project_dir: Path, token: str) -> Path:
    _check_token(token)
    return project_dir / STAGING / f"{token}.target.json"


def set_target(project_dir: Path, token: str, target: dict[str, Any]) -> None:
    """Remember the video task (``{"task": id, "class_name": ...}``) the staged files were checked against."""
    _target_file(project_dir, token).write_text(json.dumps(target), encoding="utf-8")


def get_target(project_dir: Path, token: str) -> dict[str, Any] | None:
    """The video task remembered by :func:`set_target`, or None."""
    path = _target_file(project_dir, token)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def discard_staged(project_dir: Path, token: str) -> None:
    """Delete a staging folder (no error when it is already gone)."""
    _check_token(token)
    shutil.rmtree(project_dir / STAGING / token, ignore_errors=True)
    _target_file(project_dir, token).unlink(missing_ok=True)
