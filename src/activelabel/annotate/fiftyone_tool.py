"""The FiftyOne App itself as an annotation tool (no login)."""

from __future__ import annotations

from typing import Any

from activelabel.annotate.base import GROUND_TRUTH, Sent, ToolError, ToolSettings, need_dataset
from activelabel.annotate.fo_labels import labels_from_fo
from activelabel.labels import Label
from activelabel.projects import Project


def view_name(session_id: str) -> str:
    """Name of the saved view of a session."""
    return f"annotate - session {session_id}"


class FiftyOneTool:
    """Edit ``ground_truth`` in the FiftyOne App on a saved view of the session images."""

    name = "fiftyone"

    def check(self, settings: ToolSettings) -> str:
        """No login needed."""
        return "FiftyOne needs no login."

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Save a view with the images and open the App on it."""
        from fiftyone import ViewField as F

        from activelabel import fiftyone_bridge

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
        return Sent([{"label": "FiftyOne", "url": url}], {"view": name})

    def pull(
        self,
        dataset: Any,
        image_ids: list[str],
        project: Project,
        settings: ToolSettings,
        ref: dict[str, Any],
    ) -> dict[str, list[Label]]:
        """Read ``ground_truth`` of the session images."""
        from fiftyone import ViewField as F

        need_dataset(dataset)
        dataset.reload()
        wanted = set(image_ids)
        out: dict[str, list[Label]] = {i: [] for i in image_ids}
        view = dataset.match(F("image_id").is_in(list(wanted)))
        ids, values = view.values("image_id"), view.values(GROUND_TRUTH)
        for image_id, value in zip(ids, values, strict=True):
            if image_id in wanted:
                out[image_id] = labels_from_fo(value, project.task)
        return out

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any]) -> None:
        """Delete the saved view."""
        name = ref.get("view")
        if dataset is not None and name and dataset.has_saved_view(name):
            dataset.delete_saved_view(name)
