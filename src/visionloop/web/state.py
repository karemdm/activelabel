"""Objects shared by every request: settings, projects and jobs."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from visionloop.config import Settings
from visionloop.images.manifest import Manifest
from visionloop.jobs import JobManager
from visionloop.projects import Project, ProjectNotFound, ProjectStore


class AppState:
    """Service state, kept in ``app.state.visionloop``."""

    def __init__(self, settings: Settings, jobs: JobManager | None = None) -> None:
        self.settings = settings
        self.store = ProjectStore(settings.projects_dir)
        self.jobs = jobs or JobManager()

    def project(self, project_id: int) -> Project:
        """A project, or HTTP 404 with a readable message."""
        try:
            return self.store.get(project_id)
        except ProjectNotFound:
            raise HTTPException(404, "This project does not exist (it may have been deleted).") from None

    def manifest(self, project_id: int) -> Manifest:
        """Images of an existing project (HTTP 404 when the project does not exist)."""
        self.project(project_id)
        return Manifest(self.store.dir(project_id))

    def busy(self, project_id: int) -> None:
        """HTTP 409 when the project has a job queued or running."""
        if self.jobs.active(project_id):
            raise HTTPException(409, "Wait for the running task to finish, or cancel it, and try again.")


def get_state(request: Request) -> Any:
    """FastAPI dependency: the service state."""
    return request.app.state.visionloop
