"""HTTP API used by the interface (see docs/DESIGN.md)."""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from activelabel import annotations as ann
from activelabel.curation import issues, quality
from activelabel.curation import summary as curation_summary
from activelabel.images.ingest import ingest_files, ingest_folder, thumbnail
from activelabel.images.manifest import ACTIVE, EXCLUDED, ImageRecord, Manifest
from activelabel.importing.common import read_origin
from activelabel.jobs import Job, Progress
from activelabel.labels import AnnotationSet, annotations_path, load_annotations, save_annotations
from activelabel.projects import Project
from activelabel.tasks import Task
from activelabel.web.state import AppState, get_state

router = APIRouter(prefix="/api")
STATE = Depends(get_state)
MAX_PAGE = 500
UPLOADS = Annotated[list[UploadFile], File()]


# -- bodies -------------------------------------------------------------------
class NewProject(BaseModel):
    """Body of POST /projects."""

    name: str
    description: str = ""
    task: Task
    classes: list[str] = Field(default_factory=list)


class ProjectChanges(BaseModel):
    """Body of PATCH /projects/{id}; only the given fields change."""

    name: str | None = None
    description: str | None = None
    task: Task | None = None
    classes: list[str] | None = None
    annotations: Literal["convert", "discard"] | None = (
        None  # what to do with the ground truth on a task change
    )
    classes_action: Literal["remove_annotations"] | None = None  # when removed classes are still in use


class DeleteProject(BaseModel):
    """Body of DELETE /projects/{id}: the project name, typed again."""

    confirm_name: str


class FolderImport(BaseModel):
    """Body of POST /projects/{id}/images/folder."""

    path: str


class ImageIds(BaseModel):
    """Body of the exclude / restore / delete calls."""

    ids: list[str]
    reason: str = ""
    confirm: bool = False


# -- helpers ------------------------------------------------------------------
def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc).strip("'\""))


def project_overview(state: AppState, project: Project) -> dict[str, Any]:
    """Project with counts and the status of each step (what the stepper shows)."""
    project_dir = state.store.dir(project.id)
    records = Manifest(project_dir).all()
    active = [r for r in records if r.status == ACTIVE]
    annotations = load_annotations(project_dir)
    ids = {r.id for r in active}
    # an image listed in the annotations without objects is annotated too (a background image)
    labelled = [i for i in (annotations.images if annotations else {}) if i in ids]
    measured = [r for r in active if r.metrics]
    with_issues = [r for r in active if r.issues]
    if not measured:
        curation = "not_analyzed"
    elif len(measured) < len(active):
        curation = "outdated"  # images added after the last analysis
    else:
        curation = "analyzed"
    return {
        **project.as_dict(),
        "origin": read_origin(project_dir),  # where an imported project came from (None otherwise)
        "counts": {
            "images": len(records),
            "active": len(active),
            "excluded": len(records) - len(active),
            "annotated": len(labelled),
            "labels": sum(len(annotations.images[i]) for i in labelled) if annotations else 0,
            "with_issues": len(with_issues),
            "analyzed": len(measured),
        },
        "steps": {
            "images": "done" if active else "empty",
            "annotations": "loaded" if labelled else "none",
            "curation": curation,
        },
    }


def _image_item(record: ImageRecord, labels: int) -> dict[str, Any]:
    item = record.as_dict()
    item.pop("file", None)
    item["labels"] = labels
    return item


# -- projects -----------------------------------------------------------------
@router.get("/projects")
def list_projects(state: AppState = STATE) -> list[dict[str, Any]]:
    """Every project with its counts, newest first."""
    return [project_overview(state, p) for p in state.store.list()]


@router.post("/projects")
def create_project(body: NewProject, state: AppState = STATE) -> dict[str, Any]:
    """Create a project (the id is automatic)."""
    try:
        project = state.store.create(body.name, body.description, body.task, body.classes)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return project_overview(state, project)


@router.get("/projects/{project_id}")
def get_project(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """One project with counts and step status."""
    return project_overview(state, state.project(project_id))


def _counts(annotations: AnnotationSet) -> tuple[int, int]:
    """Images with labels and labels of a ground truth."""
    return sum(1 for v in annotations.images.values() if v), sum(len(v) for v in annotations.images.values())


def _classes_used(annotations: AnnotationSet, keep: set[str]) -> dict[str, dict[str, Any]]:
    """Classes of the ground truth that are not in ``keep`` (lower case): name -> labels and image ids."""
    used: dict[str, dict[str, Any]] = {}
    for image_id, labels in annotations.images.items():
        for lb in labels:
            if lb.label.lower() not in keep:
                entry = used.setdefault(lb.label, {"labels": 0, "ids": set()})
                entry["labels"] += 1
                entry["ids"].add(image_id)
    return used


def _backup_annotations(project_dir: Path) -> None:
    """Copy annotations.json to annotations.<timestamp>.bak.json."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copyfile(annotations_path(project_dir), project_dir / f"annotations.{stamp}.bak.json")


@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: ProjectChanges, state: AppState = STATE) -> dict[str, Any]:
    """Change name, description, task or classes.

    Changing the task of a project whose ground truth is of another task needs ``annotations``:
    ``convert`` (when possible) or ``discard`` (a backup copy is kept). Without it the answer is
    HTTP 409 with ``detail`` describing the options, and nothing changes.

    Removing classes that the ground truth still uses needs ``classes_action: "remove_annotations"``
    (a backup copy is kept); otherwise HTTP 409 ``classes_in_use``. Adding or reordering classes never
    needs a decision. The task decision is checked first.
    """
    project = state.project(project_id)
    changes = body.model_dump(exclude_none=True, exclude={"annotations", "classes_action"})
    project_dir = state.store.dir(project_id)
    ground_truth = load_annotations(project_dir) if "task" in changes else None
    result: dict[str, Any] | None = None
    converted: AnnotationSet | None = None
    if ground_truth and ground_truth.images and ground_truth.task is not body.task and body.task is not None:
        images, labels = _counts(ground_truth)
        old, new = ground_truth.task, body.task
        possible = ann.conversion_possible(old, new)
        if body.annotations is None:
            raise HTTPException(
                409,
                {
                    "code": "task_change_annotations",
                    "message": (
                        f"This project has annotations for {old.value.capitalize()} ({images} images). "
                        f"Changing the task to {new.value.capitalize()} needs a decision about them."
                    ),
                    "from_task": old.value,
                    "to_task": new.value,
                    "images": images,
                    "labels": labels,
                    "options": ["convert", "discard"] if possible else ["discard"],
                },
            )
        if body.annotations == "convert":
            converted = ann.convert_annotations(ground_truth, new)
            if converted is None:
                raise HTTPException(400, f"Annotations for {old.value} cannot be converted to {new.value}.")
            result = dict(zip(("images", "labels"), _counts(converted), strict=True))
        else:
            result = {"images": images, "labels": labels}
        result = {"action": body.annotations, **result}
    classes_result: dict[str, int] | None = None
    cleaned: AnnotationSet | None = None
    if body.classes is not None and not (result is not None and converted is None):  # not after a discard
        current = converted or load_annotations(project_dir)
        keep = {c.lower() for c in body.classes}
        used = _classes_used(current, keep) if current else {}
        if used:
            if body.classes_action is None:
                names = ", ".join(used)
                n_labels = sum(u["labels"] for u in used.values())
                n_images = len({i for u in used.values() for i in u["ids"]})
                raise HTTPException(
                    409,
                    {
                        "code": "classes_in_use",
                        "message": (
                            f"The classes {names} are used by {n_labels} annotations in {n_images} images."
                            if len(used) > 1
                            else f"The class {names} is used by {n_labels} annotations in {n_images} images."
                        ),
                        "classes": [
                            {"name": n, "labels": u["labels"], "images": len(u["ids"])}
                            for n, u in used.items()
                        ],
                        "options": ["remove_annotations"],
                    },
                )
            assert current is not None
            kept_images = {k: [lb for lb in v if lb.label.lower() in keep] for k, v in current.images.items()}
            removed = sum(len(v) for v in current.images.values()) - sum(len(v) for v in kept_images.values())
            touched = sum(1 for k, v in kept_images.items() if len(v) != len(current.images[k]))
            cleaned = AnnotationSet(current.task, current.classes, kept_images, current.source)
            classes_result = {"removed_labels": removed, "images": touched}
    try:
        project = state.store.update(project_id, **changes)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    if converted is not None:
        save_annotations(project_dir, cleaned or converted)
    elif result is not None:
        _backup_annotations(project_dir)
        annotations_path(project_dir).unlink()
    elif cleaned is not None:
        _backup_annotations(project_dir)
        save_annotations(project_dir, cleaned)
    overview = project_overview(state, project)
    if result is not None:
        overview["annotations_result"] = result
    if classes_result is not None:
        overview["classes_result"] = classes_result
    return overview


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, body: DeleteProject, state: AppState = STATE) -> dict[str, Any]:
    """Delete a project with all its files. The name must be typed again."""
    project = state.project(project_id)
    if body.confirm_name.strip() != project.name:
        raise HTTPException(400, "The name does not match; nothing was deleted.")
    state.busy(project_id)
    state.store.delete(project_id)
    return {"deleted": project_id}


# -- images -------------------------------------------------------------------
@router.post("/projects/{project_id}/images")
def upload_images(project_id: int, files: UPLOADS, state: AppState = STATE) -> dict[str, Any]:
    """Add uploaded images and .zip archives (the interface sends them in chunks)."""
    manifest = state.manifest(project_id)
    report = ingest_files(manifest, [(f.filename or "image", f.file) for f in files])
    return report.as_dict()


@router.post("/projects/{project_id}/images/folder")
def import_folder(project_id: int, body: FolderImport, state: AppState = STATE) -> dict[str, Any]:
    """Import a folder of this machine (only inside ACTIVELABEL_IMPORT_ROOT), as a job."""
    manifest = state.manifest(project_id)
    root = state.settings.import_root
    if root is None:
        raise HTTPException(400, "Importing by path is off: set ACTIVELABEL_IMPORT_ROOT.")
    folder = Path(body.path).expanduser()

    def run(progress: Progress) -> dict[str, Any]:
        return ingest_folder(manifest, folder, root, progress).as_dict()

    return state.jobs.submit(project_id, "import_folder", run).as_dict()


@router.get("/projects/{project_id}/images")
def list_images(
    project_id: int,
    status: str = Query("active", pattern="^(active|excluded|all)$"),
    issue: str = "",
    ids: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=MAX_PAGE),
    state: AppState = STATE,
) -> dict[str, Any]:
    """A page of images, filtered by status, issue or a comma-separated list of ids."""
    manifest = state.manifest(project_id)
    records = manifest.all()
    if status != "all":
        records = [r for r in records if r.status == status]
    if issue:
        records = [r for r in records if issue in r.issues]
    if ids:
        wanted = set(ids.split(","))
        records = [r for r in records if r.id in wanted]
    annotations = load_annotations(state.store.dir(project_id))
    counts = {k: len(v) for k, v in annotations.images.items()} if annotations else {}
    duplicate_of = issues.load_duplicate_of(state.store.dir(project_id))
    names = {r.id: r.name for r in manifest.all()} if duplicate_of else {}
    page = records[offset : offset + limit]
    items = []
    for record in page:
        item = _image_item(record, counts.get(record.id, 0))
        original = duplicate_of.get(record.id)
        if original and "near_duplicate" in record.issues:
            item["duplicate_of"], item["duplicate_of_name"] = original, names.get(original, "")
        items.append(item)
    return {"total": len(records), "items": items}


def _record(state: AppState, project_id: int, image_id: str) -> tuple[Any, ImageRecord]:
    manifest = state.manifest(project_id)
    try:
        return manifest, manifest.get(image_id)
    except KeyError:
        raise HTTPException(404, "This image is not in the project.") from None


@router.get("/projects/{project_id}/images/{image_id}/thumb")
def image_thumb(project_id: int, image_id: str, state: AppState = STATE) -> FileResponse:
    """Thumbnail (JPEG, made on first request)."""
    manifest, record = _record(state, project_id, image_id)
    return FileResponse(thumbnail(manifest, record), headers={"Cache-Control": "max-age=86400"})


@router.get("/projects/{project_id}/images/{image_id}/file")
def image_file(project_id: int, image_id: str, state: AppState = STATE) -> FileResponse:
    """The original file."""
    manifest, record = _record(state, project_id, image_id)
    return FileResponse(manifest.file_path(record), filename=Path(record.name).name)


@router.post("/projects/{project_id}/images/exclude")
def exclude_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Leave images out of the project (reversible)."""
    return {"changed": state.manifest(project_id).set_status(body.ids, EXCLUDED, body.reason)}


@router.post("/projects/{project_id}/images/restore")
def restore_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Bring excluded images back."""
    return {"changed": state.manifest(project_id).set_status(body.ids, ACTIVE)}


@router.post("/projects/{project_id}/images/delete")
def delete_images(project_id: int, body: ImageIds, state: AppState = STATE) -> dict[str, int]:
    """Delete images and their files for good (also their annotations and thumbnails)."""
    if not body.confirm:
        raise HTTPException(400, "Deleting cannot be undone: confirm it first.")
    state.busy(project_id)
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    deleted = manifest.delete(body.ids)
    for image_id in body.ids:
        (project_dir / "thumbs" / f"{image_id}.jpg").unlink(missing_ok=True)
    annotations = load_annotations(project_dir)
    if annotations and any(i in annotations.images for i in body.ids):
        for image_id in body.ids:
            annotations.images.pop(image_id, None)
        save_annotations(project_dir, annotations)
    return {"deleted": deleted}


# -- jobs -----------------------------------------------------------------------
@router.get("/projects/{project_id}/jobs")
def project_jobs(project_id: int, state: AppState = STATE) -> list[dict[str, Any]]:
    """Running and recent jobs of a project, newest first."""
    state.project(project_id)
    return [j.as_dict() for j in state.jobs.list(project_id)]


@router.delete("/projects/{project_id}/jobs")
def clear_jobs(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Remove the finished jobs of a project from the list (running ones stay)."""
    state.project(project_id)
    return {"removed": state.jobs.clear(project_id)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: AppState = STATE) -> dict[str, Any]:
    """One job, of any project (including imports, which belong to project 0)."""
    try:
        return state.jobs.get(job_id).as_dict()
    except KeyError:
        raise HTTPException(404, "This task no longer exists.") from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: AppState = STATE) -> dict[str, Any]:
    """Ask a job to stop."""
    try:
        job: Job = state.jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "This task no longer exists.") from None
    return job.as_dict()


# -- annotations ----------------------------------------------------------------
class ApplyAnnotations(BaseModel):
    """Body of POST /projects/{id}/annotations/{token}/apply."""

    action: str
    mode: str = "replace"


@router.post("/projects/{project_id}/annotations")
def check_annotations(project_id: int, files: UPLOADS, state: AppState = STATE) -> dict[str, Any]:
    """Keep the uploaded annotation files aside and say what is in them (nothing is imported yet)."""
    project = state.project(project_id)
    manifest = state.manifest_with_images(project_id)
    project_dir = state.store.dir(project_id)
    try:
        token = ann.stage(project_dir, [(f.filename or "annotations", f.file) for f in files])
    except ValueError as exc:
        raise _bad_request(exc) from exc
    staged = ann.load_staged(project_dir, token)
    fmt = ann.detect_format(staged)
    if fmt is None:
        ann.discard_staged(project_dir, token)
        raise HTTPException(
            400,
            "The annotation format was not recognised. Supported: COCO JSON, YOLO txt (with data.yaml or "
            "classes.txt), Pascal VOC XML, CVAT for images XML and Label Studio JSON.",
        )
    result = ann.parse(staged, fmt, manifest.active())
    report = ann.validate(result, project)
    return {**report.as_dict(), "token": token, "has_annotations": load_annotations(project_dir) is not None}


@router.post("/projects/{project_id}/annotations/{token}/apply")
def apply_annotations(
    project_id: int, token: str, body: ApplyAnnotations, state: AppState = STATE
) -> dict[str, Any]:
    """Import (or discard) staged annotation files as the user decided."""
    manifest = state.manifest(project_id)
    state.busy(project_id)
    try:
        return ann.apply(state.store, project_id, manifest, token, body.action, body.mode)
    except (ValueError, FileNotFoundError) as exc:
        raise _bad_request(exc) from exc


@router.delete("/projects/{project_id}/annotations/{token}")
def discard_annotations(project_id: int, token: str, state: AppState = STATE) -> dict[str, bool]:
    """Throw away staged annotation files."""
    state.project(project_id)
    try:
        ann.discard_staged(state.store.dir(project_id), token)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"discarded": True}


@router.get("/projects/{project_id}/annotations")
def annotations_summary(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """What the project ground truth has: task, classes, labels per class."""
    project = state.project(project_id)
    ground_truth = load_annotations(state.store.dir(project_id))
    if ground_truth is None:
        return {"loaded": False}
    active = {r.id for r in state.manifest(project_id).active()}
    per_class: dict[str, int] = {}
    images = 0
    for image_id, labels in ground_truth.images.items():
        if image_id not in active:
            continue
        images += 1
        for label in labels:
            per_class[label.label] = per_class.get(label.label, 0) + 1
    return {
        "loaded": True,
        "task": ground_truth.task.value,
        "task_matches": ground_truth.task is project.task,
        "class_names": ground_truth.classes,
        "images_annotated": images,
        "labels_total": sum(per_class.values()),
        "labels_per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "source": {k: v for k, v in ground_truth.source.items() if k in ("format", "saved", "files")},
    }


# -- curation -------------------------------------------------------------------
class Analyze(BaseModel):
    """Body of POST /projects/{id}/curation/analyze."""

    recompute: bool = False


def _curation(state: AppState, project_id: int) -> dict[str, Any]:
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    return {
        **curation_summary.summary(manifest),
        "thresholds": asdict(issues.load_thresholds(project_dir)),
        "defaults": asdict(issues.Thresholds()),
        "duplicate_of": issues.load_duplicate_of(project_dir),
    }


@router.post("/projects/{project_id}/curation/analyze")
def analyze(project_id: int, body: Analyze | None = None, state: AppState = STATE) -> dict[str, Any]:
    """Measure the images and flag issues, as a job (it can be cancelled)."""
    manifest = state.manifest_with_images(project_id)
    project_dir = state.store.dir(project_id)
    recompute = bool(body and body.recompute)

    def run(progress: Progress) -> dict[str, Any]:
        measured = quality.measure_images(manifest, progress, recompute=recompute)
        progress.update(message="Flagging issues")
        counts = issues.flag_issues(manifest, issues.load_thresholds(project_dir))
        return {"measured": measured, "issues": counts, "step": "curation"}

    return state.jobs.submit(project_id, "analyze", run).as_dict()


@router.delete("/projects/{project_id}/curation")
def reset_curation(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Forget the analysis (measures, issues, hashes, duplicates); thresholds stay."""
    manifest = state.manifest(project_id)
    state.busy(project_id)
    return {"reset": issues.reset_curation(manifest)}


@router.get("/projects/{project_id}/curation")
def curation_overview(project_id: int, state: AppState = STATE) -> dict[str, Any]:
    """Counts per issue, ids per issue, histograms and thresholds."""
    return _curation(state, project_id)


@router.put("/projects/{project_id}/curation/thresholds")
def set_thresholds(project_id: int, body: dict[str, float], state: AppState = STATE) -> dict[str, Any]:
    """Change thresholds and flag the images again (no new measuring)."""
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    current = asdict(issues.load_thresholds(project_dir))
    unknown = set(body) - set(current)
    if unknown:
        raise HTTPException(400, f"Unknown thresholds: {', '.join(sorted(unknown))}")
    try:
        thresholds = issues.Thresholds(**{**current, **body})
    except (TypeError, ValueError) as exc:
        raise _bad_request(exc) from exc
    issues.save_thresholds(project_dir, thresholds)
    issues.flag_issues(manifest, thresholds)
    return _curation(state, project_id)


@router.get("/projects/{project_id}/curation/suggested")
def suggested(project_id: int, state: AppState = STATE) -> dict[str, list[str]]:
    """Images that are probably worth excluding (never excluded automatically)."""
    return {"ids": curation_summary.suggest_exclusions(state.manifest(project_id))}


# -- FiftyOne -------------------------------------------------------------------
class OpenFiftyOne(BaseModel):
    """Body of POST /projects/{id}/fiftyone."""

    view: str | None = None


@router.post("/projects/{project_id}/fiftyone")
def open_fiftyone(
    project_id: int, body: OpenFiftyOne | None = None, state: AppState = STATE
) -> dict[str, Any]:
    """Refresh the project's FiftyOne dataset and open the App (job; returns the URL as its result)."""
    project = state.project(project_id)
    manifest = state.manifest(project_id)
    project_dir = state.store.dir(project_id)
    view = body.view if body else None

    def run(progress: Progress) -> dict[str, Any]:
        from activelabel import fiftyone_bridge

        progress.update(message="Preparing the FiftyOne dataset")
        dataset = fiftyone_bridge.sync_dataset(project, manifest, load_annotations(project_dir))
        progress.update(message="Opening the FiftyOne App")
        return {"url": fiftyone_bridge.launch_app(dataset, view), "view": view or "", "step": "curation"}

    return state.jobs.submit(project_id, "fiftyone", run).as_dict()


@router.post("/projects/{project_id}/fiftyone/pull-exclusions")
def pull_exclusions(project_id: int, state: AppState = STATE) -> dict[str, int]:
    """Exclude the images tagged ``exclude`` in the FiftyOne App."""
    state.project(project_id)
    manifest = state.manifest(project_id)
    import fiftyone as fo

    from activelabel import fiftyone_bridge

    name = fiftyone_bridge.dataset_name(project_id)
    if not fo.dataset_exists(name):
        raise HTTPException(
            400, "Open the project in FiftyOne first, tag images with 'exclude', then try again."
        )
    return {"excluded": fiftyone_bridge.pull_exclusions(fo.load_dataset(name), manifest)}
