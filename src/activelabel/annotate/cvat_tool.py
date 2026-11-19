"""CVAT adapter of the annotation tools, through FiftyOne's CVAT backend.

``send`` creates CVAT tasks with ``dataset.annotate`` (current ground truth pre-loaded, one task
per ~50 images); ``pull`` downloads them with ``dataset.load_annotations`` into a temporary field,
converts that field to internal labels and removes the field; ``close`` cleans the remote tasks
(optional) and the local annotation run. The password is passed to FiftyOne only as a call
argument: it is never logged, stored or put in an exception message.
"""

from __future__ import annotations

import math
import re
from typing import Any

from activelabel.annotate.base import GROUND_TRUTH, Sent, ToolError, ToolSettings
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.tasks import Task

TASK_SIZE = 50  # images per CVAT task (CVAT Online throttles very large uploads)
MISSING_CORNER = (0.0, 0.0)  # where missing keypoints are placed in CVAT (points cannot be empty)

# Label type of the CVAT schema per task (FiftyOne's names).
LABEL_TYPES = {
    Task.CLASSIFICATION: "classification",  # CVAT tags, one per image
    Task.DETECTION: "detections",
    Task.SEGMENTATION: "polylines",  # filled polylines are CVAT polygons
    Task.KEYPOINTS: "keypoints",
}


def anno_key_of(session_id: str) -> str:
    """Annotation run key of a session: letters, digits and underscores only."""
    return "al_" + re.sub(r"\W", "_", session_id)


def pull_field(anno_key: str) -> str:
    """Temporary field that receives the annotations of a session."""
    return f"_al_pull_{anno_key}"


def label_schema(project: Project) -> dict[str, Any]:
    """CVAT label schema of ``ground_truth``: type per task, project classes, everything editable."""
    if not project.classes:
        raise ToolError("The project has no classes yet: add at least one class before sending to CVAT.")
    return {
        GROUND_TRUTH: {
            "type": LABEL_TYPES[project.task],
            "classes": list(project.classes),
            "allow_additions": True,
            "allow_deletions": True,
            "allow_label_edits": True,
            "allow_spatial_edits": True,
        }
    }


def _is_missing(point: Any) -> bool:
    return any(isinstance(c, float) and math.isnan(c) for c in point)


def _keypoints(value: Any) -> list[Any]:
    return list(getattr(value, "keypoints", None) or [])


def _convert_keypoints(view: Any, to_corner: bool) -> None:
    """Missing points (NaN) <-> top-left corner, on the samples of ``view`` (saved)."""
    for sample in view.iter_samples(autosave=True):
        for kp in _keypoints(sample[GROUND_TRUTH]):
            pts = list(kp.points or [])
            if to_corner:
                kp.points = [MISSING_CORNER if _is_missing(p) else tuple(p) for p in pts]
            else:
                kp.points = [(math.nan, math.nan) if tuple(p) == MISSING_CORNER else tuple(p) for p in pts]


def _labels_from_fo(value: Any, task: Task) -> list[Label]:
    from activelabel.annotate.fo_labels import labels_from_fo

    return labels_from_fo(value, task)


def explain(err: Exception, url: str = "") -> ToolError:
    """Turn CVAT / FiftyOne / network errors into a readable :class:`ToolError`."""
    if isinstance(err, ToolError):
        return err
    text = str(err)
    low = text.lower()
    name = type(err).__name__.lower()
    where = f" at {url}" if url else ""
    if "auth/login" in low or "401" in low or "unauthorized" in low or "403" in low:
        return ToolError(f"CVAT login failed{where}: check the username and password.")
    if "429" in low or "too many requests" in low:
        return ToolError("CVAT asked to slow down (too many requests): wait a minute and try again.")
    if "maximum number of" in low or "upgrade" in low:
        return ToolError(f"CVAT plan limit reached: {text.splitlines()[0] if text else 'see your plan'}")
    if "connection" in name or "timeout" in name or "max retries" in low or "name or service" in low:
        return ToolError(f"Cannot reach CVAT{where}: check the address and your network.")
    if "404" in low or "not found" in low or "invalidurl" in name or "missingschema" in name:
        return ToolError(f"CVAT answered 'not found'{where}: check the address (or the task was deleted).")
    return ToolError(f"CVAT error: {text.splitlines()[0] if text else type(err).__name__}")


def _login(settings: ToolSettings) -> Any:
    """Authenticated FiftyOne CVAT client (call ``close()`` on it)."""
    from fiftyone.utils.cvat import CVATAnnotationAPI

    if not settings.url or not settings.username or not settings.secret:
        raise ToolError("CVAT needs the server address, the username and the password.")
    return CVATAnnotationAPI(
        "cvat", settings.url.rstrip("/"), username=settings.username, password=settings.secret
    )


def _credentials(settings: ToolSettings) -> dict[str, str]:
    if not settings.url or not settings.username or not settings.secret:
        raise ToolError("CVAT needs the server address, the username and the password.")
    return {"url": settings.url.rstrip("/"), "username": settings.username, "password": settings.secret}


class CvatTool:
    """CVAT through FiftyOne's ``annotate`` / ``load_annotations``."""

    name = "cvat"

    def check(self, settings: ToolSettings) -> str:
        """Log in to the server; return ``Connected to <url> as <user>``."""
        try:
            api = _login(settings)
        except Exception as err:
            raise explain(err, settings.url) from None
        try:
            api.close()
        except Exception:  # noqa: S110 - closing a checked session is best effort
            pass
        return f"Connected to {settings.url.rstrip('/')} as {settings.username}"

    def send(
        self, dataset: Any, image_ids: list[str], project: Project, settings: ToolSettings, session_id: str
    ) -> Sent:
        """Create the CVAT tasks of the images with the current ground truth pre-loaded."""
        creds = _credentials(settings)
        anno_key = anno_key_of(session_id)
        schema = label_schema(project)
        view = dataset.select_by("image_id", image_ids)
        keypoints = project.task is Task.KEYPOINTS
        if keypoints:
            _convert_keypoints(view, to_corner=True)
        try:
            results = view.annotate(
                anno_key,
                backend="cvat",
                label_schema=schema,
                project_name=f"ActiveLabel {project.id} - {project.name}",
                task_name=f"ActiveLabel {project.id} {anno_key}",
                task_size=TASK_SIZE,
                launch_editor=False,
                **creds,
            )
        except Exception as err:
            if anno_key in dataset.list_annotation_runs():
                dataset.delete_annotation_run(anno_key)  # free the key for a new attempt
            raise explain(err, creds["url"]) from None
        finally:
            if keypoints:
                _convert_keypoints(view, to_corner=False)
        task_ids = list(getattr(results, "task_ids", None) or [])
        links = [
            {"label": f"CVAT task {i + 1}", "url": f"{creds['url']}/tasks/{tid}"}
            for i, tid in enumerate(task_ids)
        ]
        return Sent(links=links, ref={"anno_key": anno_key})

    def pull(
        self,
        dataset: Any,
        image_ids: list[str],
        project: Project,
        settings: ToolSettings,
        ref: dict[str, Any],
    ) -> dict[str, list[Label]]:
        """Download the annotations into a temporary field and convert them to internal labels."""
        creds = _credentials(settings)
        anno_key = str(ref.get("anno_key") or "")
        if not anno_key:
            raise ToolError("This session has no CVAT annotation run.")
        field = pull_field(anno_key)
        try:
            dataset.load_annotations(anno_key, dest_field=field, unexpected="keep", cleanup=False, **creds)
            view = dataset.select_by("image_id", image_ids)
            out: dict[str, list[Label]] = {i: [] for i in image_ids}
            for image_id, value in zip(view.values("image_id"), view.values(field), strict=True):
                if image_id not in out or value is None:
                    continue
                if project.task is Task.KEYPOINTS:
                    for kp in _keypoints(value):
                        kp.points = [
                            (math.nan, math.nan) if tuple(p) == MISSING_CORNER else tuple(p)
                            for p in kp.points or []
                        ]
                out[image_id] = _labels_from_fo(value, project.task)
        except Exception as err:
            raise explain(err, creds["url"]) from None
        finally:
            try:
                if field in dataset.get_field_schema():
                    dataset.delete_sample_field(field)
            except Exception:  # noqa: S110 - the temporary field is harmless if it stays
                pass
        return out

    def close(self, dataset: Any, settings: ToolSettings, ref: dict[str, Any], delete_remote: bool) -> None:
        """Delete the remote tasks (when asked) and forget the annotation run; ignore missing ones."""
        anno_key = str(ref.get("anno_key") or "")
        if not anno_key:
            return
        if anno_key not in dataset.list_annotation_runs():
            return
        if delete_remote:
            try:
                dataset.load_annotation_results(anno_key, **_credentials(settings)).cleanup()
            except ToolError:
                raise
            except Exception as err:
                if "not found" not in str(err).lower() and "404" not in str(err):
                    raise explain(err, settings.url) from None
        try:
            dataset.delete_annotation_run(anno_key)
        except Exception as err:
            if "not found" not in str(err).lower() and "no annotation run" not in str(err).lower():
                raise ToolError(f"Could not forget the annotation run: {err}") from None
