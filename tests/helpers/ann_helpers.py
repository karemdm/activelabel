"""Helpers for annotation tests: a project with small images and files written from text."""

import hashlib
import json
from pathlib import Path

from sample_images import make_image

from visionloop.images.manifest import ImageRecord, Manifest
from visionloop.projects import ProjectStore


def make_manifest(project_dir: Path, names: list[str], size=(100, 50)) -> Manifest:
    """Create images (``size`` pixels) with these names and a manifest listing them."""
    manifest = Manifest(project_dir)
    records = []
    for i, name in enumerate(names):
        image_id = hashlib.sha256(f"{i}{name}".encode()).hexdigest()[:16]
        file = f"images/{image_id}.png"
        make_image(project_dir / file, size=size, seed=i, fmt="PNG")
        records.append(ImageRecord(image_id, name, file, size[0], size[1], 1, "2026-01-01T00:00:00+00:00"))
    manifest.add(records)
    return manifest


def make_project(tmp_path: Path, names: list[str], task="detection", classes=(), size=(100, 50)):
    """Return ``(store, project, manifest)`` with images named ``names``."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Test", "", task, list(classes))
    return store, project, make_manifest(store.dir(project.id), names, size)


def write(folder: Path, name: str, content) -> Path:
    """Write text (or JSON for dict/list) to ``folder/name``."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content) if isinstance(content, (dict, list)) else content, encoding="utf-8")
    return path
