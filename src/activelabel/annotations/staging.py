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

from activelabel.annotations.folders import folder_classes
from activelabel.errors import CodedError

STAGING = "staging"
#: ``staging/<token>.report.json``: the report of a check, beside the folder (no upload can clash).
REPORT_SUFFIX = ".report.json"
MAX_AGE_SECONDS = 24 * 3600
MAX_TOTAL_BYTES = 1024**3
ALLOWED = {".json", ".xml", ".txt", ".yaml", ".yml", ".names", ".csv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_TOKEN = re.compile(r"^[0-9a-f]{16}$")
_NOT_IN_FILE_NAMES = re.compile('[<>:"|?*]')


def _safe_rel(name: str) -> PurePosixPath:
    """Relative path of an upload name.

    Args:
        name: file name as uploaded; a backslash is read as ``/``.

    Returns:
        The relative path without empty or ``.`` parts.

    Raises:
        CodedError: absolute, drive or ``..`` paths (``annotation_path_unsafe``).
    """
    n = name.replace("\\", "/")
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if n.startswith("/") or (len(n) > 1 and n[1] == ":") or ".." in parts or not parts:
        raise CodedError("annotation_path_unsafe", f"Unsafe file path in upload: {name}", name=name[:100])
    return PurePosixPath(*parts)


def _copy(src: IO[bytes], dest: Path, budget: list[int]) -> None:
    """Copy a stream to a file, counting the bytes against a budget.

    Args:
        src: readable binary stream.
        dest: file to write; its folders are created.
        budget: one-item list with the bytes still allowed; reduced by what is copied.

    Raises:
        CodedError: the budget is used up (``annotation_too_large``).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while chunk := src.read(1 << 20):
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise CodedError("annotation_too_large", "The annotation files are too large.")
            out.write(chunk)


def _check_token(token: str) -> None:
    """Check the shape of a staging token.

    Args:
        token: the token to check.

    Raises:
        CodedError: the token is not 16 lowercase hex digits (``staging_token_invalid``).
    """
    if not _TOKEN.match(token):
        raise CodedError("staging_token_invalid", "Invalid staging token.")


def cleanup(project_dir: Path) -> None:
    """Remove staging folders (and their target files) not touched for 24 hours.

    Args:
        project_dir: the project's folder.
    """
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


def _extract_zip(handle: BinaryIO, target: Path, budget: list[int], images: list[PurePosixPath]) -> int:
    """Extract the annotation files of a zip; the paths of its images are appended to ``images``.

    Args:
        handle: the uploaded zip.
        target: staging folder that receives the annotation files.
        budget: one-item list with the bytes still allowed; reduced by what is saved.
        images: list that receives the paths of the images found in the zip.

    Returns:
        How many annotation files were saved.

    Raises:
        CodedError: invalid zip (``annotation_zip_invalid``), unsafe path or too large.
    """
    saved = 0
    with tempfile.TemporaryFile() as tmp:
        shutil.copyfileobj(handle, tmp)
        try:
            archive = zipfile.ZipFile(tmp)
        except zipfile.BadZipFile as exc:
            raise CodedError("annotation_zip_invalid", "The zip file is not valid.") from exc
        with archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            rels = [_safe_rel(i.filename) for i in infos]  # refuse the whole zip on any unsafe path
            for info, rel in zip(infos, rels, strict=True):
                if "__MACOSX" in rel.parts or rel.name.startswith("._"):
                    continue
                if rel.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(rel)
                if rel.suffix.lower() not in ALLOWED:
                    continue
                with archive.open(info) as src:
                    _copy(src, target.joinpath(*rel.parts), budget)
                saved += 1
    return saved


def _touch_all(target: Path, rels: list[PurePosixPath]) -> None:
    """Create an empty file per image path under ``target`` (the class folders are what counts).

    Args:
        target: staging folder.
        rels: relative paths of the images.
    """
    for rel in rels:
        placeholder = target.joinpath(*rel.parts)
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch()


def stage_folders(project_dir: Path, names: list[str]) -> str:
    """Stage the names of images already in the project, to read their class folders (#568).

    Nothing is uploaded again: an empty file per name is kept, as :func:`stage` does for an upload of
    images in class folders, so the same check and apply steps follow (characters a file name cannot hold
    on every system are replaced by ``_``). Staging folders older than 24 hours are removed first.

    Args:
        project_dir: the project's folder.
        names: ``ImageRecord.name`` values (``cat/a.jpg``).

    Returns:
        The token of the staging folder.

    Raises:
        CodedError: an unsafe name (``annotation_path_unsafe``).
    """
    cleanup(project_dir)
    # a colon is fine in an image name, not in a file name on every system
    rels = [_safe_rel(_NOT_IN_FILE_NAMES.sub("_", name)) for name in names]
    token = secrets.token_hex(8)
    target = project_dir / STAGING / token
    target.mkdir(parents=True)
    try:
        _touch_all(target, rels)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return token


def stage(project_dir: Path, files: list[tuple[str, BinaryIO]]) -> str:
    """Save uploaded annotation files (or zips of them) and return a token.

    Only json, csv, xml, txt, yaml and names files are kept (other files are ignored). An upload with
    images and no annotation file (a zip or folder with one sub-folder per class) keeps an empty file
    per image (only when they sit in at least two class folders), so that ``detect_format`` can see
    the class folders. Staging folders older than 24 hours are removed first.

    Args:
        project_dir: the project's folder.
        files: ``(name, stream)`` pairs as sent by the browser; names may contain sub-folders.

    Returns:
        The token of the staging folder.

    Raises:
        CodedError: unsafe path (zip slip), invalid zip, too large, or no annotation file or image in
            the upload (``annotation_files_missing``).
    """
    root = project_dir / STAGING
    cleanup(project_dir)
    token = secrets.token_hex(8)
    target = root / token
    target.mkdir(parents=True)
    budget = [MAX_TOTAL_BYTES]
    try:
        saved = 0
        images: list[PurePosixPath] = []
        for name, handle in files:
            rel = _safe_rel(name)
            if rel.suffix.lower() == ".zip":
                saved += _extract_zip(handle, target, budget, images)
            elif rel.suffix.lower() in ALLOWED:
                _copy(handle, target.joinpath(*rel.parts), budget)
                saved += 1
            elif rel.suffix.lower() in IMAGE_SUFFIXES:
                images.append(rel)
        if not saved and folder_classes([str(rel) for rel in images]):
            _touch_all(target, images)
            saved = len(images)
        if not saved:
            raise CodedError("annotation_files_missing", "No annotation files were found in the upload.")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return token


def load_staged(project_dir: Path, token: str) -> list[Path]:
    """Files of a staging folder, sorted.

    Args:
        project_dir: the project's folder.
        token: the staging token.

    Returns:
        The files of the staging folder, sorted (an uploaded file named like the report is kept).

    Raises:
        CodedError: invalid token.
        FileNotFoundError: the staging does not exist (discarded or expired).
    """
    _check_token(token)
    folder = project_dir / STAGING / token
    if not folder.is_dir():
        raise FileNotFoundError(f"Staging {token} not found.")
    return sorted(p for p in folder.rglob("*") if p.is_file())


def _target_file(project_dir: Path, token: str) -> Path:
    """Path of the file that remembers the video task of a staging.

    Args:
        project_dir: the project's folder.
        token: the staging token.

    Returns:
        The path ``staging/<token>.target.json``.
    """
    _check_token(token)
    return project_dir / STAGING / f"{token}.target.json"


def set_target(project_dir: Path, token: str, target: dict[str, Any]) -> None:
    """Remember the video task (``{"task": id, "class_name": ...}``) the staged files were checked against.

    Args:
        project_dir: the project's folder.
        token: the staging token.
        target: the video task as ``{"task": id, "class_name": ...}``.
    """
    _target_file(project_dir, token).write_text(json.dumps(target), encoding="utf-8")


def get_target(project_dir: Path, token: str) -> dict[str, Any] | None:
    """The video task remembered by :func:`set_target`, or None.

    Args:
        project_dir: the project's folder.
        token: the staging token.

    Returns:
        The remembered video task, or None when none was set or the file is unreadable.
    """
    path = _target_file(project_dir, token)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def set_report(project_dir: Path, token: str, report: dict[str, Any]) -> None:
    """Keep the report of a check beside its staging folder, so it can be shown again after a reload.

    It lives as long as the staged files: until they are applied, discarded or expire. It is kept outside
    the folder, so a file with any name in an upload cannot replace it or be taken for it.

    Args:
        project_dir: the project's folder.
        token: the staging token the report belongs to.
        report: the report, as the check returns it (JSON-serialisable, with its ``token``).
    """
    _check_token(token)
    (project_dir / STAGING / f"{token}{REPORT_SUFFIX}").write_text(json.dumps(report), encoding="utf-8")


def _mtime(path: Path) -> float:
    """When a file was last changed, or -1 when it is gone (a staging folder removed meanwhile).

    Args:
        path: the file.

    Returns:
        Its modification time in seconds, or -1.0.
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def pending_report(project_dir: Path) -> dict[str, Any] | None:
    """The report of the latest check whose files were neither applied nor discarded, or None.

    Args:
        project_dir: the project's folder.

    Returns:
        The report kept by :func:`set_report` in the newest staging folder that has one; None when there
        is none (or it is unreadable).
    """
    root = project_dir / STAGING
    found = sorted(root.glob(f"*{REPORT_SUFFIX}"), key=_mtime) if root.is_dir() else []
    for path in reversed(found):
        if not path.with_name(path.name.removesuffix(REPORT_SUFFIX)).is_dir():  # its files are gone (expired)
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict):
            return report
    return None


def discard_staged(project_dir: Path, token: str) -> None:
    """Delete a staging folder (with its report) and its target (no error when it is already gone).

    Args:
        project_dir: the project's folder.
        token: the staging token.
    """
    _check_token(token)
    shutil.rmtree(project_dir / STAGING / token, ignore_errors=True)
    _target_file(project_dir, token).unlink(missing_ok=True)
    (project_dir / STAGING / f"{token}{REPORT_SUFFIX}").unlink(missing_ok=True)
