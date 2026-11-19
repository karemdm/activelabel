# ActiveLabel — stage 1 design

Stage 1 covers: create a project, choose its task, upload images, load annotations (if
any), check them against the task, and curate the images (corrupt, blurry, dark, low
quality, near duplicates), excluding the ones that should not be used.

Principles (from what went wrong in the previous tool):

- **Steps, not buttons.** A project page is a sequence of steps with a clear status each:
  *1 Images → 2 Annotations (optional) → 3 Curation*. Names say what the person does.
- **Every job shows progress and can be cancelled.** When it ends, it says what changed
  and links to where the result is ("See result" scrolls to it or opens the view).
- **Nothing is lost by mistake.** Excluding an image is reversible; deleting asks first.
- **FiftyOne is a viewer, not a second place to manage data.** The project data lives in
  activelabel (JSON files per project). FiftyOne mirrors it for deep inspection, with saved
  views per issue. Tagging an image `exclude` in FiftyOne can be brought back.
- English and Portuguese, light and dark theme, works at phone width.

## Data on disk

```
<workspace>/projects/<id>/
  project.json        name, description, task, classes, dates
  images.json         one record per image (id = content hash, status, metrics, issues)
  images/<id>.<ext>   original files (never re-encoded)
  thumbs/<id>.jpg     thumbnails (cache)
  annotations.json    ground truth in the internal format (activelabel.labels)
  curation.json       thresholds and near-duplicate groups
  staging/<token>/    annotation files waiting for a decision
```

## Screens

1. **Projects** — cards: name, `#id`, task, images (active / excluded), annotation status,
   date. Button *New project*.
2. **New project** (like Label Studio / CVAT) — name, description (the id is automatic);
   task as four cards with an icon and one line each (classification, detection,
   segmentation, keypoints); classes (optional, can come from the annotations).
3. **Project page** — header (name, `#id`, task, edit, delete) and the three steps:
   - **Images**: drop zone for files, folders and .zip; upload in chunks with a progress
     bar and *Cancel*; result: added, duplicates, rejected with reasons; annotation files
     found in the upload lead to step 2. Paged grid of thumbnails.
   - **Annotations**: upload files or a .zip → validation report: format, task found vs
     project task (clear sentence), images matched, classes (new ones marked), problems
     with examples; then the possible actions as buttons with one-line explanations
     (*Import*, *Change the task to X and import*, *Convert to boxes and import*,
     *Discard*); *Replace* or *Merge* when the project already has annotations.
   - **Curation**: *Analyze images* (job) → chips with counts per issue (filter the grid),
     thumbnails with issue tags and checkboxes, *Select suggested*, *Exclude selected*,
     *Restore*, *Delete permanently* (confirmation); *Excluded* tab; thresholds (advanced,
     collapsed; changing them re-flags without measuring again); *Open in FiftyOne* (opens
     the saved view of the current filter); *Bring exclusions from FiftyOne*.
4. **Jobs bar** — running and recent jobs of the project: progress, *Cancel*, result
   summary and *See result*.

## HTTP API

All under `/api`. Errors: `{"detail": "<readable message>"}` with 400/404/409.

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/projects` | | list of projects with counts |
| POST | `/projects` | `{name, description, task, classes}` | project |
| GET | `/projects/{id}` | | project + counts + step status |
| PATCH | `/projects/{id}` | any of `name, description, task, classes`; `annotations` (`convert`/`discard`) and `classes_action` (`remove_annotations`) answer a 409 (`task_change_annotations`, `classes_in_use`) | project (+ `annotations_result`, `classes_result`) |
| DELETE | `/projects/{id}` | `{confirm_name}` | `{deleted: id}` |
| POST | `/projects/{id}/images` | multipart `files` (images and/or .zip) | ingest report |
| POST | `/projects/{id}/images/folder` | `{path}` | job (folder import) |
| GET | `/projects/{id}/images` | `status, issue, offset, limit` | `{total, items}` |
| GET | `/projects/{id}/images/{image_id}/thumb` | | JPEG |
| GET | `/projects/{id}/images/{image_id}/file` | | original file |
| POST | `/projects/{id}/images/exclude` | `{ids, reason}` | `{changed}` |
| POST | `/projects/{id}/images/restore` | `{ids}` | `{changed}` |
| POST | `/projects/{id}/images/delete` | `{ids, confirm: true}` | `{deleted}` |
| POST | `/projects/{id}/annotations` | multipart `files` | validation report + `token` |
| POST | `/projects/{id}/annotations/{token}/apply` | `{action, mode}` | import summary |
| DELETE | `/projects/{id}/annotations/{token}` | | discard staged files |
| GET | `/projects/{id}/annotations` | | summary of the ground truth |
| POST | `/projects/{id}/curation/analyze` | `{recompute}` | job |
| GET | `/projects/{id}/curation` | | summary (counts, ids per issue, histograms, thresholds) |
| PUT | `/projects/{id}/curation/thresholds` | thresholds | summary (re-flagged) |
| GET | `/projects/{id}/curation/suggested` | | ids |
| POST | `/projects/{id}/fiftyone` | `{view}` | job → `{url}` |
| POST | `/projects/{id}/fiftyone/pull-exclusions` | | `{excluded}` |
| GET | `/projects/{id}/jobs` | | jobs |
| POST | `/jobs/{job_id}/cancel` | | job |
