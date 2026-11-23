# ActiveLabel — design

A project covers: create it, choose its task, upload images, load annotations (if any), check them
against the task, curate the images (corrupt, blurry, dark, low quality, near duplicates), split the
data into versions, train models, keep them, and pre-annotate new images with them. This document
describes the project page, the data on disk and the HTTP API.

Principles (from what went wrong in the previous tool):

- **Steps, not buttons.** A project page is a sequence of steps with a clear status each:
  two groups, *Data* (1 Images, 2 Annotations (optional), 3 Curation) and *Model* (4 Data split,
  5 Training, 6 Models). Names say what the person does.
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
  tasks.json          tasks: packages of images, split into jobs (docs/TASKS.md)
  datasets/v<N>.json  frozen dataset versions: train/val/test split + label copy (docs/DATASETS.md)
  datasets/counter.json  last version number (numbers are never reused)
  runs/r<N>/          training runs: config.yaml, run.json, metrics.jsonl, results.json (docs/TRAINING.md)
  runs/counter.json   last run number
  models/m<N>/        saved models: weights, config.yaml, results.json, model.json, ONNX export
  models/current.json the current model; models/counter.json: last model number (docs/TRAINING.md)
  preannotations.json pre-annotation runs; preannotations/p<N>.log: their logs (docs/PREANNOTATE.md)
  videos.json         source videos (id, name, fps, frames, duration, size) (docs/VIDEO.md)
  videos/<id>.<ext>   original videos; their selected frames become images
  images/<id>.<ext>   original files (never re-encoded)
  thumbs/<id>.jpg     thumbnails (cache)
  annotations.json    ground truth in the internal format (activelabel.labels)
  curation.json       thresholds and near-duplicate groups
  staging/<token>/    annotation files waiting for a decision (+ <token>.target.json: the video task)
```

## Screens

1. **Projects** — cards: name, `#id`, task, images (active / excluded), annotation status,
   date. Button *New project*.
2. **New project** (like Label Studio / CVAT) — name, description (the id is automatic);
   task as four cards with an icon and one line each (classification, detection,
   segmentation, keypoints); classes (optional, can come from the annotations).
3. **Project page** — header (name, `#id`, task, edit, delete) and the six steps (the three below,
   then *Data split*, *Training* and *Models*, see [DATASETS.md](DATASETS.md) and
   [TRAINING.md](TRAINING.md); the *Annotations* step also has the tabs *Pré-anotar*, *CVAT* and
   *FiftyOne*, see [PREANNOTATE.md](PREANNOTATE.md), [SYNC.md](SYNC.md) and
   [ANNOTATION.md](ANNOTATION.md)). The first three:
   - **Images**: drop zone for files, folders and .zip; upload in chunks with a progress
     bar and *Cancel*; result: added, duplicates, rejected with reasons; annotation files
     found in the upload lead to step 2. Paged grid of thumbnails.
   - **Annotations**: upload files or a .zip → validation report: format, task found vs
     project task (clear sentence), images matched, classes (new ones marked), problems
     with examples; then the possible actions as buttons with one-line explanations
     (*Import*, *Change the task to X and import*, *Convert to boxes and import*,
     *Discard*); *Replace* or *Merge* when the project already has annotations. When the project
     has video tasks, a *For task* selector matches files made for a video by frame number
     (docs/VIDEO.md, *Loading annotations of a video*).
   - **Curation**: *Analyze images* (job) → chips with counts per issue (filter the grid),
     thumbnails with issue tags and checkboxes, *Select suggested*, *Exclude selected*,
     *Restore*, *Delete permanently* (confirmation); *Excluded* tab; thresholds (advanced,
     collapsed; changing them re-flags without measuring again); *Open in FiftyOne* (opens
     the saved view of the current filter); *Bring exclusions from FiftyOne*.
4. **Activities card** — a small card in the bottom-right corner (minimisable, closable) with the running
   and recent jobs of the project: progress, *Cancel*, result summary and *See result*; finished ones
   stay in its history.

## HTTP API

All under `/api`. Errors: `{"detail": "<readable message>"}` with 400/404/409.

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/projects` | | list of projects with counts |
| POST | `/projects` | `{name, description, task, classes}` | project |
| GET | `/projects/{id}` | | project + counts + step status |
| PATCH | `/projects/{id}` | any of `name, description, task, classes`; `annotations` (`convert`/`discard`) and `classes_action` (`remove_annotations`) answer a 409 (`task_change_annotations`, `classes_in_use`) | project (+ `annotations_result`, `classes_result`) |
| DELETE | `/projects/{id}` | `{confirm_name}` | `{deleted: id}` |
| POST | `/projects/{id}/images` | multipart `files` (images and/or .zip); creates a new task | ingest report (+ `task`) |
| POST | `/projects/{id}/images/folder` | `{path}` | job (folder import) |
| GET | `/projects/{id}/images` | `status, issue, task, job, offset, limit` | `{total, items}` |
| GET | `/projects/{id}/images/{image_id}/thumb` | | JPEG |
| GET | `/projects/{id}/images/{image_id}/file` | | original file |
| POST | `/projects/{id}/images/exclude` | `{ids, reason}` | `{changed}` |
| POST | `/projects/{id}/images/restore` | `{ids}` | `{changed}` |
| POST | `/projects/{id}/images/delete` | `{ids, confirm: true}` | `{deleted}` |
| POST | `/projects/{id}/annotations` | multipart `files`, optional `task` (video task id), `class_name` | validation report + `token` (+ `task`) |
| POST | `/projects/{id}/annotations/{token}/apply` | `{action, mode}` | import summary |
| DELETE | `/projects/{id}/annotations/{token}` | | discard staged files |
| GET | `/projects/{id}/annotations` | | summary of the ground truth |
| POST | `/projects/{id}/curation/analyze` | `{recompute}` | job |
| GET | `/projects/{id}/curation` | | summary (counts, ids per issue, histograms, thresholds) |
| PUT | `/projects/{id}/curation/thresholds` | thresholds | summary (re-flagged) |
| GET | `/projects/{id}/curation/suggested` | | ids |
| POST | `/projects/{id}/fiftyone` | `{view, task?, job?}` | job → `{url}` |
| POST | `/projects/{id}/fiftyone/pull-exclusions` | | `{excluded}` |
| GET | `/projects/{id}/jobs` | | jobs |
| POST | `/jobs/{job_id}/cancel` | | job |
| GET | `/training/catalog` | `type` | models with hyperparameter specs, selection metrics |
| GET / POST | `/projects/{id}/runs` | POST: run configuration (docs/TRAINING.md) | runs / the new run, stored as `queued`; it starts as soon as no other heavy process is running |
| GET / DELETE | `/projects/{id}/runs/{n}` | DELETE: `{confirm: true}` (409 while running) | config, status, metrics, results / `{deleted}` |
| POST | `/projects/{id}/runs/{n}/reproduce` | | new run with the same configuration |

## Tasks and jobs

Images are added as **tasks** (one upload = one task), each split into fixed-size **jobs**, as in CVAT.
Storage, endpoints (`/projects/{id}/tasks`), the `task`/`job` image filters and the CVAT mirror are
specified in [TASKS.md](TASKS.md); the code is `activelabel.tasks_store` and `activelabel.web.tasks_api`.
Dataset versions (`/projects/{id}/datasets`, preview/create/get/delete, and the `dataset`/`part`/`fold`
image filters) are specified in [DATASETS.md](DATASETS.md); the code is `activelabel.datasets` and
`activelabel.web.datasets_api`.

Training runs, saved models, ONNX export and pre-annotation (`/projects/{id}/runs`, `/models`,
`/preannotations`, `/training/*`) are specified in [TRAINING.md](TRAINING.md) and
[PREANNOTATE.md](PREANNOTATE.md); the code is `activelabel.training`, `activelabel.preannotate` and
`activelabel.web.training_api`, `models_api` and `preannotate_api`.
