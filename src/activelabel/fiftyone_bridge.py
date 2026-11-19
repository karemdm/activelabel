"""Show a project in the FiftyOne App: images, quality measures, issues and ground truth.

Every project has one persistent dataset, ``activelabel-<project id>``. Samples are keyed by
the image id (field ``image_id``). Segmentation polygons become ``Polylines`` with
``filled=True`` (simpler than rasterised masks, and they keep their exact shape).
"""

from __future__ import annotations

from typing import Any

import fiftyone as fo
from fiftyone import ViewField as F

from activelabel.curation.issues import ISSUES, NEAR_DUPLICATE, load_duplicate_of
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.tasks import Task

GROUND_TRUTH = "ground_truth"
EXCLUDED_TAG = "excluded"
VIEW_ALL_ISSUES = "curation - quality issues"
VIEW_EXCLUDED = "curation - excluded"
VIEW_DUPLICATES = "curation - near duplicates"

_session: Any = None


def dataset_name(project_id: int) -> str:
    """Name of the FiftyOne dataset of a project."""
    return f"activelabel-{project_id}"


def _label_type(task: Task) -> type:
    return {
        Task.CLASSIFICATION: fo.Classification,
        Task.DETECTION: fo.Detections,
        Task.SEGMENTATION: fo.Polylines,
        Task.KEYPOINTS: fo.Keypoints,
    }[task]


def _declare_fields(dataset: fo.Dataset, task: Task) -> None:
    """Make sure every field exists with the right type (also when there are no samples)."""
    scalars: dict[str, Any] = {
        "image_id": fo.StringField,
        "name": fo.StringField,
        "status": fo.StringField,
        "brightness": fo.FloatField,
        "contrast": fo.FloatField,
        "sharpness": fo.FloatField,
        "duplicate_of": fo.StringField,
        "duplicate_group": fo.StringField,
    }
    for name, kind in scalars.items():
        dataset.add_sample_field(name, kind)
    dataset.add_sample_field("image_size", fo.ListField, subfield=fo.IntField)
    dataset.add_sample_field("quality_issues", fo.ListField, subfield=fo.StringField)
    wanted = _label_type(task)
    if GROUND_TRUTH in dataset.get_field_schema():
        current = dataset.get_field(GROUND_TRUTH)
        if getattr(current, "document_type", None) is not wanted:  # the task of the project changed
            dataset.delete_sample_field(GROUND_TRUTH)
    dataset.add_sample_field(GROUND_TRUTH, fo.EmbeddedDocumentField, embedded_doc_type=wanted)


def _ground_truth(task: Task, labels: list[Label]) -> Any:
    """FiftyOne label of one image for the task (None when it has no labels)."""
    if not labels:
        return None
    if task is Task.CLASSIFICATION:
        return fo.Classification(label=labels[0].label)
    if task is Task.DETECTION:
        return fo.Detections(
            detections=[fo.Detection(label=lb.label, bounding_box=lb.bbox) for lb in labels if lb.bbox]
        )
    if task is Task.SEGMENTATION:
        polylines = [
            fo.Polyline(
                label=lb.label, points=[[tuple(p) for p in ring] for ring in lb.polygons], filled=True
            )
            for lb in labels
            if lb.polygons
        ]
        return fo.Polylines(polylines=polylines)
    return fo.Keypoints(
        keypoints=[
            fo.Keypoint(
                label=lb.label,
                points=[(p[0], p[1]) if p[2] > 0 else (float("nan"), float("nan")) for p in lb.keypoints],
            )
            for lb in labels
            if lb.keypoints
        ]
    )


def _apply(
    sample: fo.Sample,
    record: ImageRecord,
    manifest: Manifest,
    task: Task,
    annotations: AnnotationSet | None,
    duplicate_of: dict[str, str],
    kept: set[str],
) -> None:
    sample.filepath = str(manifest.file_path(record).resolve())
    sample["image_id"] = record.id
    sample["name"] = record.name
    sample["status"] = record.status
    for metric in ("brightness", "contrast", "sharpness"):
        sample[metric] = record.metrics.get(metric)
    sample["image_size"] = [record.width, record.height]
    sample["quality_issues"] = list(record.issues)
    sample["duplicate_of"] = duplicate_of.get(record.id)
    sample["duplicate_group"] = duplicate_of.get(record.id) or (record.id if record.id in kept else None)
    tags = [t for t in sample.tags if t != EXCLUDED_TAG]
    sample.tags = [*tags, EXCLUDED_TAG] if record.status == EXCLUDED else tags
    sample[GROUND_TRUTH] = _ground_truth(task, annotations.images.get(record.id, []) if annotations else [])


def _save_views(dataset: fo.Dataset) -> None:
    views: dict[str, fo.DatasetView] = {
        VIEW_ALL_ISSUES: dataset.match(F("quality_issues").length() > 0),
        VIEW_EXCLUDED: dataset.match_tags(EXCLUDED_TAG),
        VIEW_DUPLICATES: dataset.match(F("duplicate_group") != None).sort_by("duplicate_group"),  # noqa: E711
    }
    for issue in ISSUES:
        if issue != NEAR_DUPLICATE:
            views[f"curation - {issue.replace('_', ' ')}"] = dataset.match(
                F("quality_issues").contains(issue)
            )
    for name, view in views.items():
        dataset.save_view(name, view, overwrite=True)


def sync_dataset(project: Project, manifest: Manifest, annotations: AnnotationSet | None) -> fo.Dataset:
    """Create or refresh the persistent FiftyOne dataset of a project, in place.

    Samples are matched by ``image_id``: re-running updates them, adds new images and removes
    the samples of deleted images. Excluded images stay, tagged ``excluded``. The saved views
    ``curation - quality issues``, ``curation - excluded``, ``curation - near duplicates`` and
    one ``curation - <issue>`` per issue (spaces instead of underscores) are refreshed too.

    Args:
        project: the project (name of the dataset, task, classes).
        manifest: images, measures and issues.
        annotations: ground truth, or None when there is none yet.

    Returns:
        The dataset.
    """
    name = dataset_name(project.id)
    dataset = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name, persistent=True)
    dataset.persistent = True
    _declare_fields(dataset, project.task)

    records = {r.id: r for r in manifest.all()}
    duplicate_of = load_duplicate_of(manifest.project_dir)
    kept = set(duplicate_of.values())

    known: dict[str, str] = {}
    stale: list[str] = []
    for sample_id, image_id in zip(dataset.values("id"), dataset.values("image_id"), strict=True):
        if image_id in records and image_id not in known:
            known[image_id] = sample_id
        else:
            stale.append(sample_id)
    if stale:
        dataset.delete_samples(stale)

    new: list[fo.Sample] = []
    for record in records.values():
        if record.id not in known:
            sample = fo.Sample(filepath=str(manifest.file_path(record).resolve()))
            _apply(sample, record, manifest, project.task, annotations, duplicate_of, kept)
            new.append(sample)
    if new:
        dataset.add_samples(new)
    for sample in dataset.select(list(known.values())).iter_samples(autosave=True):
        _apply(sample, records[sample["image_id"]], manifest, project.task, annotations, duplicate_of, kept)

    dataset.classes = {GROUND_TRUTH: list(project.classes)}
    dataset.info = {**dataset.info, "project_id": project.id, "project_name": project.name}
    dataset.save()
    _save_views(dataset)
    return dataset


def resolve_view(dataset: fo.Dataset, requested: str | None) -> str | None:
    """Saved view to open for a request.

    Accepts a saved view name, an issue name (``blurry``, ``near_duplicate``, with or without the
    ``curation - `` prefix) or None. Unknown names fall back to the view of all issues.
    """
    if not requested:
        return None
    saved = set(dataset.list_saved_views())
    if requested in saved:
        return requested
    issue = requested.removeprefix("curation - ").strip()
    candidates = [VIEW_DUPLICATES] if issue == NEAR_DUPLICATE else [f"curation - {issue.replace('_', ' ')}"]
    for name in candidates:
        if name in saved:
            return name
    return VIEW_ALL_ISSUES if VIEW_ALL_ISSUES in saved else None


def launch_app(dataset: fo.Dataset, view: str | None = None, port: int = 5151) -> str:
    """Start the FiftyOne App in the background (no browser opened) and return its URL.

    A session that is already running on the port is reused: it just switches to the dataset
    (and saved view, when given).
    """
    global _session
    view = resolve_view(dataset, view)
    target: Any = dataset.load_saved_view(view) if view else dataset
    if _session is not None and getattr(_session, "port", None) == port:
        try:
            _session.dataset = dataset
            if view:
                _session.view = target
            return str(_session.url)
        except Exception:  # noqa: BLE001 - the session was closed: start a new one
            _session = None
    _session = fo.launch_app(target, port=port, remote=True)
    return str(_session.url)


def pull_exclusions(dataset: fo.Dataset, manifest: Manifest, tag: str = "exclude") -> int:
    """Exclude the images the user tagged in the App, then remove the tag.

    Args:
        dataset: the project's dataset.
        manifest: images of the project.
        tag: the tag marking images to exclude.

    Returns:
        How many images were excluded (images already excluded do not count).
    """
    tagged = dataset.match_tags(tag)
    ids = [i for i in tagged.values("image_id") if i]
    changed = manifest.set_status(ids, EXCLUDED, "tagged in FiftyOne") if ids else 0
    if ids:
        tagged.tag_samples(EXCLUDED_TAG)
        tagged.untag_samples(tag)
    return changed
