"""Bring images into a project: uploads, zip archives, folders on the machine, thumbnails.

Every image is read fully with Pillow before it is stored; unreadable files are rejected.
Images are stored by content (id = first 16 hex digits of the SHA-256), without re-encoding.
Files are streamed to disk, so memory stays low whatever the size of the upload.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
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
ANNOTATION_EXTENSIONS = frozenset({".json", ".txt", ".xml", ".yaml", ".yml", ".csv"})
IGNORED_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
#: Text files that come with a dataset but hold no annotations (compared with the lower-case stem).
NOT_ANNOTATION_STEMS = frozenset(
    {"readme", "license", "licence", "copying", "notice", "authors", "changelog"}
)
HEAD_BYTES = 8192  # how much of a text file is read to tell YOLO lines from prose
YOLO_MIN_FIELDS = 5  # class, x, y, w, h

MIN_SIDE = 8
BATCH_SIZE = 200
MAX_ZIP_BYTES = 20 * 1024**3
MAX_ZIP_ENTRIES = 100_000
MAX_EXAMPLES = 200
CHUNK = 1024 * 1024
ORIENTATION_TAG = 0x0112
# Pillow format -> extension the file is stored with (the content decides, not the name)
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "MPO": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WEBP": ".webp",
}
DRIVE_PATH = re.compile(r"[A-Za-z]:/")


@dataclass
class IngestReport:
    """What an import did.

    Attributes:
        added: images stored.
        duplicates: names of images already in the project (or repeated in the upload).
        rejected: ``{"name": ..., "reason": ...}`` for files that were not stored
            (``empty``, ``corrupt``, ``too small``, ``too large``, ``unsupported``, ``nested archive``,
            ``unsafe path``, ...).
        other_files: names of annotation files (json, txt, xml, yaml) sent with the images or inside archives.
        ids: ids of the images stored by this import, in order (not part of :meth:`as_dict`).
        per_file: for every file handled at the top level (not the entries of a zip), in order: the id of
            its image (the existing image for a duplicate), or None when no image came of it (not part of
            :meth:`as_dict`).
    """

    added: int = 0
    duplicates: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    per_file: list[str | None] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form: counts plus at most 200 examples of each list.

        Returns:
            The counts, and the first examples of duplicates, rejected files and annotation files.
        """
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


#: Pillow modes of images with more than 8 bits per channel that it cannot turn into RGB faithfully (#620);
#: 16-bit colour PNGs open as 8-bit RGB and are fine. CMYK, palette and alpha images convert well.
HIGH_DEPTH_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N", "I", "F"})


def _clean_name(name: str) -> str | None:
    """Normalised relative name, or None when it is absolute (``/``, ``C:/``) or climbs out with ``..``.

    A colon elsewhere in the name (``a:b.png``) is fine: files are stored under their id.

    Args:
        name: the name as sent (either kind of slash).

    Returns:
        The name with ``/`` separators and without empty or ``.`` parts, or None when it is unsafe.
    """
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or DRIVE_PATH.match(name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def _looks_like_yolo(head: bytes) -> bool:
    """Whether the start of a text file has a YOLO line (a class id and four or more numbers).

    Args:
        head: the first bytes of the file.

    Returns:
        True when at least one line is ``<integer> <number> <number> <number> <number> ...``.
    """
    for line in head.decode("utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < YOLO_MIN_FIELDS or not fields[0].isdigit():
            continue
        try:
            for field_ in fields[1:]:
                float(field_)
        except ValueError:
            continue
        return True
    return False


def _is_annotation_file(name: str, stream: IO[bytes]) -> bool:
    """Whether a file sent with images is an annotation file (to be imported in step 2).

    A ``.txt`` counts only when it has YOLO lines, and README, LICENSE and the like never count.

    Args:
        name: the cleaned relative name (``/`` separators).
        stream: its content (a ``.txt`` is read for its first bytes).

    Returns:
        True for a json, xml, yaml or csv file and for a ``.txt`` with YOLO lines.
    """
    path = PurePosixPath(name)
    if path.stem.lower() in NOT_ANNOTATION_STEMS:
        return False
    if path.suffix.lower() == ".txt":
        return _looks_like_yolo(stream.read(HEAD_BYTES))
    return True


def _is_hidden(name: str) -> bool:
    """Whether a file is one the system adds and people did not (``.hidden``, ``__MACOSX``, ``Thumbs.db``).

    Args:
        name: the cleaned relative name (``/`` separators).

    Returns:
        True when the file is to be skipped without a word.
    """
    parts = name.split("/")
    return any(p.startswith(".") or p == "__MACOSX" for p in parts) or parts[-1].lower() in IGNORED_NAMES


def _read_image(path: Path) -> tuple[int, int, str]:
    """Check a file with Pillow and return its displayed (EXIF-oriented) width and height and its extension.

    The extension comes from the content (a PNG sent as ``photo.jpg`` is stored as ``.png``).

    Args:
        path: the file to check.

    Returns:
        The width, the height and the extension (with the dot) of the real format.

    Raises:
        ValueError: the file is not a readable image of a supported kind (the message is the rejection
            reason: ``corrupt``, ``too large``, ``unsupported``, ``high bit depth``, ``too small``).
    """
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            orientation = image.getexif().get(ORIENTATION_TAG, 1)
            extension = FORMAT_EXTENSIONS.get(image.format or "")
            mode = image.mode
    except Image.DecompressionBombError as exc:
        raise ValueError("too large") from exc
    except Exception as exc:  # noqa: BLE001 - Pillow raises many kinds of errors for bad files
        raise ValueError("corrupt") from exc
    if extension is None:
        raise ValueError("unsupported")
    if mode in HIGH_DEPTH_MODES:  # 16-bit grey or 32-bit: turned into RGB as is, it would become all white
        raise ValueError("high bit depth")
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    if min(width, height) < MIN_SIDE:
        raise ValueError("too small")
    return width, height, extension


class _Ingester:
    """State of one import: ids seen, pending manifest records, report, progress."""

    def __init__(self, manifest: Manifest, progress: Progress | None) -> None:
        """Start an import.

        Args:
            manifest: images of the project; its ids are the ones already known.
            progress: optional progress/cancellation handle.
        """
        self.manifest = manifest
        self.progress = progress
        self.report = IngestReport()
        self.known: set[str] = set(manifest.load())
        self.pending: list[ImageRecord] = []
        self.aliases: dict[str, list[str]] = {}  # names of repeated files, by the id of the image they repeat
        self.last_id: str | None = None  # image of the file just stored (also when it was a duplicate)
        self.done = 0
        self.total = 0
        self.images_dir = manifest.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def tick(self, message: str) -> None:
        """Report progress.

        Args:
            message: what is being done (the name of the file).

        Raises:
            Cancelled: the user cancelled.
        """
        if self.progress is not None:
            self.progress.update(self.done, self.total, message)

    def flush(self) -> None:
        """Write the pending records, and the names of repeated files, to the manifest in one go."""
        if self.pending:
            batch, self.pending = self.pending, []
            self.manifest.add(batch)
        if self.aliases:
            names, self.aliases = self.aliases, {}
            self.manifest.add_aliases(names)

    def reject(self, name: str, reason: str) -> None:
        """Record a file that was not stored.

        Args:
            name: the file.
            reason: ``corrupt``, ``too small``, ``unsupported``...
        """
        self.report.rejected.append({"name": name, "reason": reason})

    def handle(self, name: str, stream: BinaryIO) -> None:
        """Process one uploaded or on-disk file (and note which image it became in ``report.per_file``).

        Args:
            name: its name, with sub-folders.
            stream: its content.
        """
        self.last_id = None
        try:
            self._handle(name, stream)
        finally:
            self.report.per_file.append(self.last_id)

    def _handle(self, name: str, stream: BinaryIO) -> None:
        """Store an image, expand a zip, or note an annotation file, according to the name.

        Args:
            name: the file name, with sub-folders.
            stream: its content.
        """
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
            self.last_id = None  # an archive holds many images
        elif ext in IMAGE_EXTENSIONS:
            self._store_image(clean, stream)
        elif ext in ANNOTATION_EXTENSIONS:
            if _is_annotation_file(clean, stream):
                self.report.other_files.append(clean)  # annotation files sent with the images: step 2
        else:
            self.reject(clean, "unsupported")

    def _store_image(
        self,
        name: str,
        stream: IO[bytes],
        limit: int | None = None,
        salt: bytes = b"",
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Stream a file to disk, validate it and register it (stored under its real kind's extension).

        Args:
            name: the name the image had.
            stream: its content.
            limit: most bytes to read; None for no limit (a zip entry is limited by what is left of the
                archive's allowance).
            salt: hashed after the content (video frames: video, frame and task, so identical frames
                still get their own id).
            extra: more fields of the image record.

        Returns:
            The bytes read.

        Raises:
            _ZipTooLarge: more than ``limit`` bytes.
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
            if size == 0:
                self.reject(name, "empty")
                return 0
            digest.update(salt)
            image_id = digest.hexdigest()[:16]
            self.last_id = image_id
            if image_id in self.known:
                self.report.duplicates.append(name)
                self.aliases.setdefault(image_id, []).append(name)
                return size
            try:
                width, height, ext = _read_image(tmp)
            except ValueError as exc:
                self.last_id = None
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
        """Store the images of an uploaded archive.

        Args:
            archive_name: the name of the archive.
            stream: its content (copied to a temporary file when it cannot seek).
        """
        with contextlib.ExitStack() as stack:
            if not hasattr(stream, "seekable"):
                # a stream without seekable() cannot be opened by zipfile: copy it to a real
                # temporary file first
                raw: Any = stack.enter_context(tempfile.TemporaryFile())  # its type differs per OS
                copy: BinaryIO = raw
                shutil.copyfileobj(stream, copy, 1 << 20)
                copy.seek(0)
                stream = copy
            self._read_zip(archive_name, stream)

    def _read_zip(self, archive_name: str, stream: BinaryIO) -> None:
        """Go through the entries of an archive, refusing one with too many or too large entries.

        Args:
            archive_name: the name of the archive.
            stream: its content, seekable.
        """
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
        """Process one archive entry.

        Args:
            archive: the open archive.
            info: the entry.
            remaining: bytes the archive may still use once expanded.

        Returns:
            The real number of bytes the entry used.

        Raises:
            _ZipTooLarge: the entry is larger than ``remaining``.
        """
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
                    return self._store_image(clean, entry, limit=remaining)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError):
                self.reject(clean, "corrupt")
                return 0
        if ext in ANNOTATION_EXTENSIONS:
            with archive.open(info) as entry:
                if _is_annotation_file(clean, entry):
                    self.report.other_files.append(clean)
        elif ext == ".zip":
            self.reject(clean, "nested archive")
        else:
            self.reject(clean, "unsupported")
        return 0

    def finish(self) -> IngestReport:
        """Write what is pending and report the end.

        Returns:
            The report of the import.
        """
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
    """Every file below a folder, in name order, without hidden folders and ``__MACOSX``.

    Args:
        folder: where to start.

    Yields:
        The path of each file.
    """
    for current, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        for name in sorted(names):
            yield Path(current) / name


def check_folder(folder: Path, import_root: Path | None) -> Path:
    """The resolved folder to import, after checking it may be imported.

    Args:
        folder: folder asked for.
        import_root: allowed root (``ACTIVELABEL_IMPORT_ROOT``); None means importing by path is off.

    Returns:
        The folder, resolved.

    Raises:
        ValueError: importing by path is off, or the folder is not a folder inside the root.
    """
    if import_root is None:
        raise ValueError("Importing by path is off: set ACTIVELABEL_IMPORT_ROOT.")
    root = import_root.resolve()
    target = Path(folder).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"The folder must be inside the import root ({root}).")
    if not target.is_dir():
        raise ValueError(f"Not a folder: {target}")
    return target


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
        ValueError: see :func:`check_folder`.
        Cancelled: the user cancelled; images stored so far stay in the manifest.
    """
    target = check_folder(folder, import_root)
    assert import_root is not None  # check_folder refused otherwise
    root = import_root.resolve()
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


__all__ = [
    "Cancelled",
    "IngestReport",
    "check_folder",
    "ingest_files",
    "ingest_folder",
    "ingest_frames",
    "thumbnail",
]
