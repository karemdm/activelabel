"""Bring images into a project: uploads, zip archives, folders on the machine, thumbnails.

Every image is read fully with Pillow before it is stored; unreadable files are rejected.
Images are stored by content (id = first 16 hex digits of the SHA-256), without re-encoding.
Files are streamed to disk, so memory stays low whatever the size of the upload.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
import uuid
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO

from PIL import Image, ImageOps

from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.jobs import Cancelled, Progress
from activelabel.storage import now

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
ANNOTATION_EXTENSIONS = frozenset({".json", ".txt", ".xml", ".yaml", ".yml"})
IGNORED_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})

MIN_SIDE = 8
BATCH_SIZE = 200
MAX_ZIP_BYTES = 20 * 1024**3
MAX_ZIP_ENTRIES = 100_000
MAX_EXAMPLES = 200
CHUNK = 1024 * 1024
ORIENTATION_TAG = 0x0112


@dataclass
class IngestReport:
    """What an import did.

    Attributes:
        added: images stored.
        duplicates: names of images already in the project (or repeated in the upload).
        rejected: ``{"name": ..., "reason": ...}`` for files that were not stored
            (``corrupt``, ``too small``, ``unsupported``, ``unsafe path``, ...).
        other_files: names of annotation files (json, txt, xml, yaml) sent with the images or inside archives.
        ids: ids of the images stored by this import, in order (not part of :meth:`as_dict`).
    """

    added: int = 0
    duplicates: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form: counts plus at most 200 examples of each list."""
        return {
            "added": self.added,
            "duplicates": self.duplicates[:MAX_EXAMPLES],
            "duplicates_count": len(self.duplicates),
            "rejected": self.rejected[:MAX_EXAMPLES],
            "rejected_count": len(self.rejected),
            "other_files": self.other_files[:MAX_EXAMPLES],
            "other_files_count": len(self.other_files),
        }


class _ZipTooLarge(Exception):
    """The archive is larger than allowed once expanded."""


def _clean_name(name: str) -> str | None:
    """Normalised relative name, or None when it is absolute or climbs out with ``..``."""
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def _is_hidden(name: str) -> bool:
    parts = name.split("/")
    return any(p.startswith(".") or p == "__MACOSX" for p in parts) or parts[-1].lower() in IGNORED_NAMES


def _read_image(path: Path) -> tuple[int, int]:
    """Check a file with Pillow and return its displayed (EXIF-oriented) width and height.

    Raises:
        ValueError: the file is not a readable image (the message is the rejection reason).
    """
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            orientation = image.getexif().get(ORIENTATION_TAG, 1)
    except Exception as exc:  # noqa: BLE001 - Pillow raises many kinds of errors for bad files
        raise ValueError("corrupt") from exc
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    if min(width, height) < MIN_SIDE:
        raise ValueError("too small")
    return width, height


class _Ingester:
    """State of one import: ids seen, pending manifest records, report, progress."""

    def __init__(self, manifest: Manifest, progress: Progress | None) -> None:
        self.manifest = manifest
        self.progress = progress
        self.report = IngestReport()
        self.known: set[str] = set(manifest.load())
        self.pending: list[ImageRecord] = []
        self.done = 0
        self.total = 0
        self.images_dir = manifest.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def tick(self, message: str) -> None:
        """Report progress; raises Cancelled when the user cancelled."""
        if self.progress is not None:
            self.progress.update(self.done, self.total, message)

    def flush(self) -> None:
        """Write pending records to the manifest in one go."""
        if self.pending:
            batch, self.pending = self.pending, []
            self.manifest.add(batch)

    def reject(self, name: str, reason: str) -> None:
        """Record a file that was not stored."""
        self.report.rejected.append({"name": name, "reason": reason})

    def handle(self, name: str, stream: BinaryIO) -> None:
        """Process one uploaded or on-disk file."""
        self.tick(name)
        self.done += 1
        clean = _clean_name(name)
        if clean is None:
            self.reject(name, "unsafe path")
            return
        if _is_hidden(clean):
            return
        ext = PurePosixPath(clean).suffix.lower()
        if ext == ".zip":
            self._expand_zip(clean, stream)
        elif ext in IMAGE_EXTENSIONS:
            self._store_image(clean, stream, ext)
        elif ext in ANNOTATION_EXTENSIONS:
            self.report.other_files.append(clean)  # annotation files sent with the images: step 2
        else:
            self.reject(clean, "unsupported")

    def _store_image(
        self,
        name: str,
        stream: IO[bytes],
        ext: str,
        limit: int | None = None,
        salt: bytes = b"",
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Stream a file to disk, validate it and register it. Returns the bytes read.

        ``salt`` is hashed after the content (video frames: video, frame and task, so identical
        frames still get their own id); ``extra`` are more fields of the image record.
        """
        tmp = self.images_dir / f".{uuid.uuid4().hex}.part"
        size = 0
        try:
            digest = hashlib.sha256()
            with tmp.open("wb") as out:
                while chunk := stream.read(CHUNK):
                    size += len(chunk)
                    if limit is not None and size > limit:
                        raise _ZipTooLarge
                    digest.update(chunk)
                    out.write(chunk)
            digest.update(salt)
            image_id = digest.hexdigest()[:16]
            if image_id in self.known:
                self.report.duplicates.append(name)
                return size
            try:
                width, height = _read_image(tmp)
            except ValueError as exc:
                self.reject(name, str(exc))
                return size
            relative = f"images/{image_id}{ext}"
            os.replace(tmp, self.manifest.project_dir / relative)
            self.known.add(image_id)
            self.pending.append(
                ImageRecord(image_id, name, relative, width, height, size, now(), **(extra or {}))
            )
            self.report.added += 1
            self.report.ids.append(image_id)
            if len(self.pending) >= BATCH_SIZE:
                self.flush()
            return size
        finally:
            tmp.unlink(missing_ok=True)

    def _expand_zip(self, archive_name: str, stream: BinaryIO) -> None:
        with contextlib.ExitStack() as stack:
            if not hasattr(stream, "seekable"):
                # Python 3.10: an upload (SpooledTemporaryFile) has no seekable(), which zipfile needs
                # to open the entries; copy it to a real temporary file first
                raw: Any = stack.enter_context(tempfile.TemporaryFile())  # its type differs per OS
                copy: BinaryIO = raw
                shutil.copyfileobj(stream, copy, 1 << 20)
                copy.seek(0)
                stream = copy
            self._read_zip(archive_name, stream)

    def _read_zip(self, archive_name: str, stream: BinaryIO) -> None:
        try:
            archive = zipfile.ZipFile(stream)
        except zipfile.BadZipFile:
            self.reject(archive_name, "corrupt")
            return
        with archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                self.reject(archive_name, f"too many entries (more than {MAX_ZIP_ENTRIES})")
                return
            if sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
                self.reject(archive_name, "archive is too large when expanded")
                return
            self.total += len(infos)
            remaining = MAX_ZIP_BYTES
            for info in infos:
                self.tick(info.filename)
                self.done += 1
                if info.is_dir():
                    continue
                try:
                    remaining -= self._zip_entry(archive, info, remaining)
                except _ZipTooLarge:
                    self.reject(archive_name, "archive is too large when expanded")
                    return

    def _zip_entry(self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, remaining: int) -> int:
        """Process one archive entry; returns the real number of bytes it used."""
        clean = _clean_name(info.filename)
        if clean is None:
            self.reject(info.filename, "unsafe path")
            return 0
        if _is_hidden(clean):
            return 0
        ext = PurePosixPath(clean).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            try:
                with archive.open(info) as entry:
                    return self._store_image(clean, entry, ext, limit=remaining)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError):
                self.reject(clean, "corrupt")
                return 0
        if ext in ANNOTATION_EXTENSIONS:
            self.report.other_files.append(clean)
        else:
            self.reject(clean, "unsupported")
        return 0

    def finish(self) -> IngestReport:
        """Write what is pending and report the end."""
        self.flush()
        self.tick("done")
        return self.report


def ingest_files(
    manifest: Manifest,
    files: Iterable[tuple[str, BinaryIO]],
    progress: Progress | None = None,
) -> IngestReport:
    """Store uploaded images (and the images of uploaded zip archives) in a project.

    Args:
        manifest: images of the project.
        files: ``(name, stream)`` pairs as sent by the browser; names may contain sub-folders.
        progress: optional progress/cancellation handle.

    Returns:
        What was added, skipped as duplicate, rejected, and annotation files found in zips.

    Raises:
        Cancelled: the user cancelled; images stored so far stay in the manifest.
    """
    ingester = _Ingester(manifest, progress)
    try:
        for name, stream in files:
            ingester.total += 1
            ingester.handle(name, stream)
    finally:
        ingester.flush()
    return ingester.finish()


def ingest_frames(
    manifest: Manifest,
    frames: Iterable[tuple[str, Path, str, int, float]],
    progress: Progress | None = None,
    salt: str = "",
) -> list[str]:
    """Store JPEG frames taken from a video, in order, every one of them (docs/VIDEO.md).

    Unlike :func:`ingest_files`, identical frames are not skipped: the id hashes the content plus
    the video id, the frame index and ``salt`` (the task), so every selected frame is stored.

    Args:
        manifest: images of the project.
        frames: ``(name, jpeg path, video id, frame index, seconds)`` in the order wanted.
        progress: optional progress/cancellation handle.
        salt: more text mixed into the ids (the id of the task being created).

    Returns:
        Ids of the images stored, in order.

    Raises:
        ValueError: a frame could not be stored (unreadable or already present).
        Cancelled: the user cancelled; the frames stored so far are removed again.
    """
    ingester = _Ingester(manifest, progress)
    items = list(frames)
    ingester.total = len(items)
    try:
        for name, path, video, index, seconds in items:
            ingester.tick(name)
            ingester.done += 1
            before = len(ingester.report.ids)
            with path.open("rb") as stream:
                ingester._store_image(
                    name,
                    stream,
                    ".jpg",
                    salt=f"{video}:{index}:{salt}".encode(),
                    extra={"video": video, "frame": index, "time": seconds},
                )
            if len(ingester.report.ids) == before:
                reason = ingester.report.rejected[-1]["reason"] if ingester.report.rejected else "duplicate"
                raise ValueError(f"Frame {index} could not be stored ({reason}).")
    except BaseException:
        ingester.flush()
        manifest.delete(ingester.report.ids)
        raise
    ingester.flush()
    return ingester.report.ids


def _walk(folder: Path) -> Iterator[Path]:
    for current, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        for name in sorted(names):
            yield Path(current) / name


def ingest_folder(
    manifest: Manifest,
    folder: Path,
    import_root: Path | None,
    progress: Progress | None = None,
) -> IngestReport:
    """Import every image below a folder of this machine (names are relative to the folder).

    Args:
        manifest: images of the project.
        folder: folder to import, inside ``import_root``.
        import_root: allowed root (``ACTIVELABEL_IMPORT_ROOT``); None means importing by path is off.
        progress: optional progress/cancellation handle.

    Returns:
        The import report.

    Raises:
        ValueError: importing by path is off, or the folder is not a folder inside the root.
        Cancelled: the user cancelled; images stored so far stay in the manifest.
    """
    if import_root is None:
        raise ValueError("Importing by path is off: set ACTIVELABEL_IMPORT_ROOT.")
    root = import_root.resolve()
    target = Path(folder).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"The folder must be inside the import root ({root}).")
    if not target.is_dir():
        raise ValueError(f"Not a folder: {target}")
    paths = list(_walk(target))
    ingester = _Ingester(manifest, progress)
    ingester.total = len(paths)
    try:
        for path in paths:
            name = path.relative_to(target).as_posix()
            if path.is_symlink() and root not in path.resolve().parents:
                ingester.done += 1
                ingester.reject(name, "unsafe path")
                continue
            with path.open("rb") as stream:
                ingester.handle(name, stream)
    finally:
        ingester.flush()
    return ingester.finish()


def thumbnail(manifest: Manifest, record: ImageRecord, size: int = 256) -> Path:
    """Path of the JPEG thumbnail of an image, created on first request and then cached.

    Args:
        manifest: images of the project.
        record: the image.
        size: longest side in pixels.

    Returns:
        ``<project>/thumbs/<id>.jpg`` (EXIF-oriented, RGB, quality 80).
    """
    target = manifest.project_dir / "thumbs" / f"{record.id}.jpg"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{uuid.uuid4().hex}.part")
    try:
        with Image.open(manifest.file_path(record)) as image:
            rgb = (ImageOps.exif_transpose(image) or image).convert("RGB")
        rgb.thumbnail((size, size))
        rgb.save(tmp, format="JPEG", quality=80)
        os.replace(tmp, target)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return target


__all__ = ["Cancelled", "IngestReport", "ingest_files", "ingest_folder", "ingest_frames", "thumbnail"]
