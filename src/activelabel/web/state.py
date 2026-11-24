"""Objects shared by every request: settings, projects and jobs."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from activelabel.config import Settings
from activelabel.images.manifest import Manifest
from activelabel.jobs import JobManager
from activelabel.projects import Project, ProjectNotFound, ProjectStore
from activelabel.training.runner import TrainingRunner
from activelabel.web.errors import ApiError


class AppState:
    """Service state, kept in ``app.state.activelabel``."""

    def __init__(self, settings: Settings, jobs: JobManager | None = None) -> None:
        """Build the settings-dependent services: stores, job manager and training runner.

        Args:
            settings: The settings of the application.
            jobs: The job manager (a new one when omitted).
        """
        self.settings = settings
        self.store = ProjectStore(settings.projects_dir)
        self.jobs = jobs or JobManager()
        self.runner = TrainingRunner(
            settings.workspace,
            self.store.dir,
            lambda: [p.id for p in self.store.list()],
            project=self.store.get,
        )

    def project(self, project_id: int) -> Project:
        """A project, or HTTP 404 with a readable message.

        Args:
            project_id: Id of the project.

        Returns:
            The project.
        """
        try:
            return self.store.get(project_id)
        except ProjectNotFound:
            raise ApiError(
                404, "project_not_found", "This project does not exist (it may have been deleted)."
            ) from None

    def manifest(self, project_id: int) -> Manifest:
        """Images of an existing project (HTTP 404 when the project does not exist).

        Args:
            project_id: Id of the project.

        Returns:
            The image manifest of the project.
        """
        self.project(project_id)
        return Manifest(self.store.dir(project_id))

    def manifest_with_images(self, project_id: int) -> Manifest:
        """Images of a project that has at least one active image (HTTP 400 otherwise).

        Args:
            project_id: Id of the project.

        Returns:
            The image manifest of the project.
        """
        manifest = self.manifest(project_id)
        if not manifest.active():
            raise ApiError(400, "no_images", "Add images to the project first (step 1).")
        return manifest

    def busy(self, project_id: int, training: bool = False) -> None:
        """HTTP 409 when the project has a job queued or running.

        With ``training`` a training run, ONNX export or pre-annotation of the project counts too
        (use it before deleting the project or its images).

        Args:
            project_id: Id of the project.
            training: Also count training runs, exports and pre-annotations.
        """
        if self.jobs.active(project_id):
            raise ApiError(
                409,
                "busy",
                "An activity of this project is still running: wait for it or cancel it, then try again.",
            )
        if training and self.runner.has_active(project_id):
            raise ApiError(
                409,
                "busy_training",
                "Wait for the training, export or pre-annotation of this project to finish, or cancel it.",
            )


def get_state(request: Request) -> Any:
    """FastAPI dependency: the service state.

    Args:
        request: The request.

    Returns:
        The state of the application.
    """
    return request.app.state.activelabel
