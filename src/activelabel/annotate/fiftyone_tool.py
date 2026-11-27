"""The FiftyOne App itself as an annotation tool (no login)."""

from __future__ import annotations

from typing import Any

from activelabel import fiftyone_bridge
from activelabel.annotate.base import Pulled, Sent, ToolError, ToolSettings, need_dataset
from activelabel.annotate.fo_labels import labels_and_notes
from activelabel.labels import Label
from activelabel.optional import F
from activelabel.projects import Project


def view_name(session_id: str) -> str:
    """Name of the saved view of a session.

    Args:
        session_id: id of the annotation session.

    Returns:
        The saved view name.
    """
    return f"annotate - session {session_id}"


class FiftyOneTool:
    """Edit ``ground_truth`` in the FiftyOne App on a saved view of the session images."""

    name = "fiftyone"

    def check(self, settings: ToolSettings) -> str:
        """No login needed.

        Args:
            settings: not used.

        Returns:
            A message saying that no login is needed.
        """
        return "FiftyOne needs no login."

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Save a view with the images and open the App on it.

        Args:
            dataset: the project's FiftyOne dataset.
            image_ids: ids of the images of the session.
            project: the project (its task tells how labels are read).
            settings: not used.
            session_id: id of the annotation session (names the saved view).

        Returns:
            The link to the App on the view, the view name as reference and how many images still hold
            edits of an earlier session (the dataset was just refreshed, so only those differ from what
            ActiveLabel wrote).

        Raises:
            ToolError: no dataset, none of the images is in it, or the App could not be started.
        """
        need_dataset(dataset)
        name = view_name(session_id)
        view = dataset.match(F("image_id").is_in(list(image_ids)))
        if len(view) == 0:
            raise ToolError("None of these images is in the FiftyOne dataset.")
        dataset.save_view(name, view, overwrite=True)
        try:
            url = fiftyone_bridge.launch_app(dataset, name)
        except OSError as exc:
            raise ToolError(str(exc)) from exc
        kept = len(fiftyone_bridge.edited_images(dataset, image_ids, project.task))
        return Sent([{"label": "FiftyOne", "url": url}], {"view": name}, kept=kept)

    def pull(
        self,
        dataset: Any,
        image_ids: list[str],
        project: Project,
        settings: ToolSettings,
        ref: dict[str, Any],
    ) -> Pulled:
        """Read ``ground_truth`` of the session images, and which of them were edited in FiftyOne.

        Args:
            dataset: the project's FiftyOne dataset.
            image_ids: ids of the images of the session.
            project: the project (its task tells how labels are read).
            settings: not used.
            ref: not used.

        Returns:
            The labels of each image, keyed by image id (an empty list when it has none), with the images
            edited in FiftyOne since ActiveLabel wrote them, the hash of the project labels written into
            each sample, the shapes that could not be read as they are (``problems``) and the tags of each
            sample (``tags``).

        Raises:
            ToolError: no dataset.
        """
        need_dataset(dataset)
        dataset.reload()
        wanted = set(image_ids)
        out: dict[str, list[Label]] = {i: [] for i in image_ids}
        problems: dict[str, dict[str, int]] = {}
        view = dataset.match(F("image_id").is_in(list(wanted)))
        ids = view.values("image_id")
        values = fiftyone_bridge.truth_values(view, project.task, fiftyone_bridge.fields_of(dataset))
        for image_id, value in zip(ids, values, strict=True):
            if image_id in wanted:
                out[image_id], notes = labels_and_notes(value, project.task)
                if notes:
                    problems[image_id] = notes
        edited = fiftyone_bridge.edited_images(dataset, image_ids, project.task)
        written = fiftyone_bridge.written_labels(dataset, image_ids)
        return Pulled(out, edited, written, problems, fiftyone_bridge.sample_tags(dataset, image_ids))

    def applied(
        self, dataset: Any, image_ids: list[str], project: Project, labels: dict[str, list[Label]]
    ) -> None:
        """Mark the session images as equal to the project, so they no longer count as edited.

        Args:
            dataset: the project's FiftyOne dataset, or None.
            image_ids: ids of the images of the applied session.
            project: the project (its task tells how labels are read).
            labels: the project labels after the apply, by image id.
        """
        if dataset is not None:
            fiftyone_bridge.mark_applied(dataset, image_ids, project.task, labels)

    def discarded(
        self, dataset: Any, image_ids: list[str], project: Project, labels: dict[str, list[Label]]
    ) -> None:
        """Put the session images back to the project's labels in FiftyOne, undoing the edits made there.

        Args:
            dataset: the project's FiftyOne dataset, or None.
            image_ids: ids of the images of the discarded session.
            project: the project (its task tells how labels are written).
            labels: the project labels, by image id.
        """
        if dataset is not None:
            fiftyone_bridge.restore_project_labels(dataset, image_ids, project.task, labels)

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any]) -> None:
        """Delete the saved view.

        Args:
            dataset: the project's FiftyOne dataset, or None.
            settings: not used.
            ref: the reference returned by :meth:`send` (``view`` is the saved view to delete).
        """
        name = ref.get("view")
        if dataset is not None and name and dataset.has_saved_view(name):
            dataset.delete_saved_view(name)
