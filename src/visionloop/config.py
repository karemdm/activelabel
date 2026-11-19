"""Service settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Settings of the service.

    Attributes:
        workspace: ``VISIONLOOP_WORKSPACE``: projects, images and annotations live here.
        language: ``VISIONLOOP_LANG``: default interface language (``en`` or ``pt``).
        import_root: ``VISIONLOOP_IMPORT_ROOT``: folders on this machine that may be imported
            by path (besides uploads from the browser). Empty: importing by path is off.
    """

    workspace: Path = Path.home() / ".visionloop"
    language: str = "en"
    import_root: Path | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from the environment (``environ`` is injectable for tests)."""
        env = os.environ if environ is None else environ
        default = cls()
        root = env.get("VISIONLOOP_IMPORT_ROOT", "")
        return cls(
            workspace=Path(env.get("VISIONLOOP_WORKSPACE", str(default.workspace))).expanduser(),
            language=env.get("VISIONLOOP_LANG", default.language),
            import_root=Path(root).expanduser().resolve() if root else None,
        )

    @property
    def projects_dir(self) -> Path:
        """Folder with one sub-folder per project."""
        return self.workspace / "projects"
