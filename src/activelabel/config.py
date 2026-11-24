"""Service settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Settings of the service.

    Attributes:
        workspace: ``ACTIVELABEL_WORKSPACE``: projects, images and annotations live here (empty or unset:
            ``~/.activelabel``).
        language: ``ACTIVELABEL_LANG``: default interface language (``en`` or ``pt``).
        import_root: ``ACTIVELABEL_IMPORT_ROOT``: folders on this machine that may be imported
            by path (besides uploads from the browser). Empty: importing by path is off.
    """

    workspace: Path = Path.home() / ".activelabel"
    language: str = "en"
    import_root: Path | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from the environment.

        Args:
            environ: the variables to read; None reads ``os.environ`` (injectable for tests).

        Returns:
            The settings, with defaults for what is unset or empty.
        """
        env = os.environ if environ is None else environ
        default = cls()
        root = env.get("ACTIVELABEL_IMPORT_ROOT", "").strip()
        workspace = env.get("ACTIVELABEL_WORKSPACE", "").strip()  # empty means not set, not "."
        return cls(
            workspace=Path(workspace).expanduser() if workspace else default.workspace,
            language=env.get("ACTIVELABEL_LANG", "").strip() or default.language,
            import_root=Path(root).expanduser().resolve() if root else None,
        )

    @property
    def projects_dir(self) -> Path:
        """Folder with one sub-folder per project.

        Returns:
            ``<workspace>/projects``.
        """
        return self.workspace / "projects"
