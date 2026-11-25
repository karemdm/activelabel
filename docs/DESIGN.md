# ActiveLabel — design

A project covers: create it, choose its type, upload images, load annotations (if any), check them
against the type, curate the images (blurry, dark, bright, low contrast, small, odd aspect ratio,
grayscale, near duplicates), split the data into versions, train models, keep them, and pre-annotate
new images with them. This document describes the project page, the data on disk and the HTTP API.
For the architecture and the flows as diagrams see [ARCHITECTURE.md](ARCHITECTURE.md).

Quoted names are the English interface; the Portuguese names are in `src/activelabel/i18n/pt.json`.

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
<workspace>/                       ACTIVELABEL_WORKSPACE (default ~/.activelabel)
  tools.json                       address and user of CVAT (secrets: system keyring only)
  projects/next_id.json            next project id (ids are never reused)
  projects/<id>/...                one folder per project (below)
  training/                        training environment (docs/TRAINING.md)
    env/ , env.json, install.log   the Python environment (env.json: only when it lives elsewhere, written when the environment is created)
    tmp/                           temporary folder of a run; removed when the run ends

<workspace>/projects/<id>/
  project.json        name, description, task (the project type), classes, dates
  images.json         one record per image (id = content hash, status, metrics, issues)
  tasks.json          tasks: packages of images, split into jobs (docs/TASKS.md)
  links.json          link with a remote project per tool, with the state at the last sync (docs/SYNC.md)
  sync/<plan>.json    comparisons made and not yet applied (the last 10 are kept)
  annotate.json       edit sessions in an annotation tool (docs/ANNOTATION.md)
  annotate/<session>/pulled.json   the annotations brought back by a session, until applied or discarded
  origin.json         where an imported project came from (docs/IMPORT.md)
  fiftyone.json       {"dataset": "<name>"}: the project's FiftyOne dataset (see below)
  annotations.json    ground truth in the internal format (activelabel.labels)
  annotations.<UTC time>Z[.<reason>].bak.json   backup made before a change that removes or rewrites annotations (a replacing import, class removal, a task change that discards labels, FiftyOne or CVAT apply, a class renamed in CVAT, a restore); the newest 20 are kept; see activelabel/backups.py
  curation.json       thresholds and near-duplicate groups
  datasets/v<N>.json  frozen dataset versions: train/val/test split + label copy (docs/DATASETS.md)
  datasets/counter.json  {"next": N+1}: numbers are never reused
  runs/r<N>/          training runs: config.yaml, run.json, metrics.jsonl, results.json, ... (docs/TRAINING.md)
  runs/counter.json   {"next": N+1}
  models/m<N>/        saved models: weights, config.yaml, results.json, model.json, ONNX export
  models/current.json the current model: {"model": N}; models/counter.json: {"next": N+1}
  preannotations.json pre-annotation runs; preannotations/p<N>.log: their logs (docs/PREANNOTATE.md)
  videos.json         source videos (id, name, fps, frames, duration, size) (docs/VIDEO.md)
  videos/<id>.<ext>   original videos; their selected frames become images
  images/<id>.<ext>   original files (never re-encoded)
  thumbs/<id>.jpg     thumbnails (cache)
  staging/<token>/    annotation files waiting for a decision (+ <token>.target.json: the video task)
```

`ACTIVELABEL_WORKSPACE` can be any folder (an empty variable counts as not set); the FiftyOne database is
not inside it (FiftyOne keeps its own, see its documentation).

**The project's FiftyOne dataset.** FiftyOne keeps every dataset of the machine in one database, so two
workspaces (for example a test server and the real one) must not share a name. A new project gets
`activelabel-<project id>-<8 random hex digits>`, chosen once and recorded in `<project>/fiftyone.json`.
A dataset made by an earlier version under the plain name `activelabel-<project id>` is kept under that
name (renaming it would break open FiftyOne App tabs and edit sessions), but only when its samples point
into this project's folder; a dataset that holds another workspace's images is never adopted. Deleting a
project deletes its recorded dataset (and the plain-named one only when it holds this project's images).

## Screens

1. **Projects** — cards that fill the row: name, `#id`, type, images (active / excluded), annotation
   coverage ("{done} of {total} images annotated"), date. Buttons *New project* and *Import project*.
2. **New project** — two columns: name and description (the id is automatic),
   then type as four cards with an icon and one line each (classification, detection,
   segmentation, keypoints) and classes (optional, can come from the annotations). Editing a project
   uses the same layout.
3. **Project page** — header (name, `#id`, type, edit, delete; the classes stay on one line with *See all*
   opening the full list) and the six steps in a compact **step bar**: two groups (*Data*, *Model*), a marker
   on each step (done, needs attention, in progress) and the current step scrolled into view on a narrow
   screen. Texts and dialogs share one pattern (title with a close button, sticky footer, Cancel focused in
   destructive ones, touch targets of at least 36 px, text of at least 12 px). The steps (the three below,
   then *Data split*, *Training* and *Models*, see [DATASETS.md](DATASETS.md) and
   [TRAINING.md](TRAINING.md); the *Annotations* step also has the tabs *Pre-annotate*, *CVAT* and
   *FiftyOne*, see [PREANNOTATE.md](PREANNOTATE.md), [SYNC.md](SYNC.md) and
   [ANNOTATION.md](ANNOTATION.md)). The first three:
   - **Images**: tabs *Gallery* and *Tasks and jobs*; drop zone for files, folders and .zip; upload
     in chunks with a progress bar and *Cancel*; result: added, duplicates, rejected with reasons;
     annotation files found in the upload lead to step 2. Paged grid of thumbnails.
   - **Annotations**: the tab *Summary* shows tiles (images annotated, annotations, classes with
     annotations, to review), the top classes with *Search class* and *Show all ({n})*. Tab
     *Import annotations*: upload files or a .zip → validation report (it takes the place of the drop zone;
     with nothing usable it offers *Choose other files* or *Discard*): format,
     type found vs project type (clear sentence), images matched, classes (new ones marked), problems
     with examples; then the possible actions as buttons with one-line explanations
     (*Import*, *Change the type to {task} and import*, *Convert to {task} and import*,
     *Discard*); *Replace* or *Merge* when the project already has annotations. When the project
     has video tasks, a *For task* selector matches files made for a video by frame number
     (docs/VIDEO.md, *Loading annotations of a video*).
   - **Curation**: the actions are at the top of the step: *Analyze images* (job), *Reset analysis*,
     *Open in FiftyOne*, *Bring exclusions from FiftyOne*; then chips with counts per issue (filter the
     grid; *Only with issues* shows the flagged images first), histograms with an axis legend,
     thumbnails with issue tags and checkboxes, a compact selection row (*All in filter*, *Suggested*,
     *Exclude selected*, *Restore selected*, *Delete permanently* with confirmation; *Select suggested*
     needs the analysis); *Excluded* tab; thresholds (*Thresholds*, collapsed; changing them re-flags
     without measuring again). *Reset analysis* forgets measures, flags and duplicate groups; *Open in
     FiftyOne* opens the saved view of the current filter. The issues are: blurry, dark, bright, low
     contrast, small, odd aspect ratio, grayscale, near duplicate. (Corrupt files are rejected at
     upload and never become images.)
4. **Activities card** — a small card in the bottom-right corner (minimisable, closable) with the running
   and recent jobs of the project: progress, *Stop*, result summary and *See result*; finished ones
   stay in its history.

## HTTP API

All under `/api`; the interactive documentation of the running service is at `/docs` and the schema at
`/openapi.json`.

### Errors

Every error body has `detail`, a readable English text. Errors the interface translates also carry a
stable `code` and the `params` the translated text needs:

```json
{"detail": "A project named 'Cats' already exists.", "code": "project_name_taken", "params": {"name": "Cats"}}
```

Status codes: 400 (rejected input), 404 (not found), 409 (conflict: a job of the project is running, or the
state changed since the person looked), 422 (the body or a parameter has the wrong shape; `detail` is the
list FastAPI gives), 500 for a damaged data file (`code: "file_damaged"`, `params.file`; the project
list itself survives one damaged project), 503 `fiftyone_unavailable` when the FiftyOne database did not answer
and 500 `internal_error` for any other unexpected error (its traceback stays in the server log; the body never
carries the exception text, and the interface never shows a bare "HTTP 500"). Three answers (two 409, one 400) carry an object in `detail` with its own
`code` the interface acts on: `task_change_annotations` and `classes_in_use` (PATCH of a project) and `no_images_added` (a task created from files that add nothing, with
the upload `report`). The `code` values of the other errors are:

| Status | Codes |
| --- | --- |
| 400 | **Projects, tasks, text:** `project_name_empty`, `project_name_taken`, `project_name_too_long`, `description_too_long`, `class_name_invalid`, `class_name_too_long`, `too_many_classes`, `type_invalid`, `task_name_empty`, `task_name_taken`, `task_name_too_long`, `job_size_invalid`, `control_characters`, `url_invalid`, `url_required`, `confirm_required`, `name_mismatch`, `no_images`, `task_required`, `job_required`. **Annotation files:** `annotation_format_unknown`, `annotation_unreadable`, `annotation_not_json` / `annotation_not_xml` (with `file`, `line`, `column`), `annotation_not_text`, `annotation_files_missing`, `annotation_zip_invalid`, `annotation_path_unsafe`, `annotation_too_large`, `staging_token_invalid`, `backup_name_invalid`, `backup_unreadable`, `annotations_not_convertible`, `video_task_required`. **Curation, scopes, tools, sync:** `unknown_thresholds`, `scope_invalid`, `scope_unknown`, `issue_required`, `no_images_match`, `direction_invalid`, `resolution_invalid`, `tool_not_configured`, `tool_unsupported`, `tool_error`, `pull_first`, `fiftyone_not_opened`. **Videos:** `one_file_only`, `video_type_unsupported`, `video_unreadable`, `video_no_frames`, `video_too_many_frames`, `video_not_enough_disk`, `video_range_invalid`, `video_seconds_range`, `video_step_range`, `frame_out_of_range`, `frame_unreadable`. **Data split** (docs/DATASETS.md): `dataset_required`, `not_whole_number`, `split_percentage_range`, `split_sum`, `split_train_empty`, `split_unknown_part`, `kfold_range`, `kfold_test_too_big`, `kfold_too_few_groups`, `background_not_bool`, `videos_mode`, `gap_not_number`, `gap_range`, `tasks_not_list`, `nothing_to_split`, `no_usable_labels`, `note_too_long`, `part_unknown`, `fold_required`, `fold_not_found`, `version_has_no_folds`. **Training and models** (docs/TRAINING.md): `dataset_classes_removed`, `dataset_incomplete`, `dataset_no_validation`, `dataset_train_empty`, `dataset_wrong_type`, `training_unavailable`, `public_model_wrong_type`, and the hyperparameter checks `param_max`, `param_min`, `param_not_bool`, `param_not_choice`, `param_not_finite`, `param_not_number`, `param_not_whole`, `param_too_large`. **Pre-annotation:** `model_kind_mismatch`, `model_not_in_catalogue`, `preannotation_kind_unsupported`, `threshold_range`, `no_images_for_choice` |
| 404 | `project_not_found`, `task_not_found`, `job_not_found`, `activity_not_found`, `image_not_found`, `dataset_not_found`, `run_not_found`, `child_not_found`, `model_not_found`, `public_model_not_found`, `public_model_no_weights`, `model_file_missing`, `download_unknown`, `preannotation_not_found`, `video_not_found`, `session_not_found`, `plan_not_found`, `staging_not_found`, `tool_unknown`, `not_linked` |
| 409 | `busy`, `busy_training`, `task_closed`, `already_applied`, `already_discarded`, `annotations_changed`, `session_finished`, `dataset_part_empty`, `dataset_in_use`, `dataset_used_by_runs`, `run_running`, `run_config_damaged`, `model_in_use`, `model_file_in_use`, `preannotation_running`, `video_in_use`, `video_task_running` |

`tool_error` (400) means a remote tool (CVAT) failed or refused: `detail` is its message and `params` has
`tool` and `reason`. `tool_unsupported` (400) is a session of a discontinued tool (docs/ANNOTATION.md).

Texts the interface does not know a code for are shown as the English `detail`.

A background job that fails also carries `error_code` and `error_params` next to its English `error` (the same
codes, `error.<code>` in the language files); its `result` is empty: tracebacks go to the log, not to the
interface. Failures of CVAT (`cvat_unreachable`, `cvat_login_failed`, `cvat_refused`, `cvat_plan_limit`,
`cvat_rate_limited`, `cvat_error`, `cvat_not_found`...), of the synchronisation (`remote_gone`,
`sync_remote_changed`, `sync_local_changed`, `sync_link_changed`, `sync_no_remote`) and of the import
(`cvat_no_images`, `cvat_item_not_found`, `import_kind_invalid`) have codes; an error of a tool without one is
sent as `tool_error` with `tool` and `reason`.

### Limits

| What | Limit |
| --- | --- |
| Project name | at most 100 characters, unique among projects |
| Project description | at most 2000 characters |
| Class name | at most 100 characters, not empty |
| Classes per project | at most 1000 |
| Task name | at most 100 characters, unique within the project |
| Images per job | a whole number from 1 to 100,000 |
| Names and text fields | no control characters (`control_characters`) |
| Frames taken from a video for one task | at most 20,000; the disk must have about twice the estimated size of the frames free (400 otherwise) |
| Image files | JPEG, PNG, BMP, TIFF or WebP, shortest side at least 8 px; the kind is decided by the content (a PNG sent as `photo.jpg` is stored as `.png`); a rejected file is listed with its reason (`empty`, `corrupt`, `too large`, `unsupported`, `too small`, `nested archive`) |
| Files in an uploaded archive | at most 100,000 entries and 20 GiB when expanded; larger ones are rejected with a reason |
| Annotation files kept aside before a decision | at most 1 GiB in all, removed after 24 hours and when the service starts |
| Tool address (CVAT) | must be `http://` or `https://` (`url_invalid`) |

Calls that change a project's data wait for its running jobs: they answer 409 "Wait for the running task
to finish, or cancel it" while an analysis, a synchronisation or another job of the project is queued or
running.

### Projects, images and jobs

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/projects` | | list of projects with counts |
| POST | `/projects` | `{name, description, task, classes}` (`task` is the project type: `classification`, `detection`, `segmentation`, `keypoints`) | project |
| GET | `/projects/{id}` | | project + counts + step status |
| PATCH | `/projects/{id}` | any of `name, description, task, classes`; `annotations` (`convert`/`discard`) and `classes_action` (`remove_annotations`) answer a 409 (`task_change_annotations`, `classes_in_use`) | project (+ `annotations_result`, `classes_result`) |
| DELETE | `/projects/{id}` | `{confirm_name}` | `{deleted: id}` |
| POST | `/projects/{id}/images` | multipart `files` (images and/or .zip); creates a new task with the next automatic name ("Task <n>", docs/TASKS.md) | ingest report (+ `task`) |
| POST | `/projects/{id}/images/folder` | `{path}`; needs `ACTIVELABEL_IMPORT_ROOT` (400 otherwise), the path must be inside it; API only, the interface has no button for it | job (folder import) |
| GET | `/projects/{id}/images` | `status` (`active` default, `excluded`, `all`), `issue` (a flag, or `any` for every flagged image), `ids` (comma separated), `task`, `job` (needs `task`), `dataset`, `part` and `fold` (need `dataset`), `offset`, `limit` (60 by default, at most 500) | `{total, items}` (classification projects: each item has `classes`) |
| GET | `/projects/{id}/images/{image_id}/thumb` | | JPEG |
| GET | `/projects/{id}/images/{image_id}/file` | | original file |
| POST | `/projects/{id}/images/exclude` | `{ids, reason}` | `{changed}` |
| POST | `/projects/{id}/images/restore` | `{ids}` | `{changed}` |
| POST | `/projects/{id}/images/delete` | `{ids, confirm: true}` | `{deleted}` |
| GET | `/projects/{id}/jobs` | | running and recent jobs, newest first |
| DELETE | `/projects/{id}/jobs` | | `{removed}`: the finished jobs leave the list |
| GET | `/jobs/{job_id}` | | the job (of any project; imports belong to project 0); `params` says what it is about (for example `{"number": 3}` for the run, export or pre-annotation it belongs to) |
| POST | `/jobs/{job_id}/cancel` | | job |

### Annotations and curation

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| POST | `/projects/{id}/annotations` | multipart `files`, optional `task` (video task id), `class_name` | validation report + `token` (+ `task`) |
| POST | `/projects/{id}/annotations/{token}/apply` | `{action, mode}` (`action`: `import`, `change_task_and_import`, `convert_and_import` or `discard`; `mode`: `replace` by default, or `merge`) | import summary |
| DELETE | `/projects/{id}/annotations/{token}` | | discard staged files |
| GET | `/projects/{id}/annotations` | | summary of the ground truth |
| GET | `/projects/{id}/annotations/backups` | | safety copies of the annotations, newest first: `name`, `created` (UTC), `images`, `labels`, `reason`, `valid` |
| POST | `/projects/{id}/annotations/backups/{name}/restore` | | makes that copy the annotations again (the current file is copied first; 409 while an activity runs; 400 `backup_name_invalid` / `backup_unreadable`, 404 `backup_not_found`) |
| POST | `/projects/{id}/curation/analyze` | `{recompute}` | job |
| GET | `/projects/{id}/curation` | | summary (counts, ids per issue, histograms, thresholds) |
| DELETE | `/projects/{id}/curation` | | `{reset}`: forgets measures, issues, hashes and duplicate groups; thresholds stay (409 while a job runs) |
| PUT | `/projects/{id}/curation/thresholds` | thresholds | summary (re-flagged) |
| GET | `/projects/{id}/curation/suggested` | | ids |
| POST | `/projects/{id}/fiftyone` | `{view, task?, job?}` | job → `{url}` |
| POST | `/projects/{id}/fiftyone/pull-exclusions` | | `{excluded}` |

### Tasks, videos, datasets, training, models

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET, POST | `/projects/{id}/tasks` | see [TASKS.md](TASKS.md) | tasks / the new task |
| PATCH, DELETE | `/projects/{id}/tasks/{task_id}` | see [TASKS.md](TASKS.md) | task / `{deleted}` |
| POST | `/projects/{id}/tasks/{task_id}/images` | multipart `files` (open task only) | upload report |
| POST, GET | `/projects/{id}/videos` | multipart `file` (one video) | video properties / the list |
| DELETE | `/projects/{id}/videos/{video_id}` | | `{deleted}` (409 while a task uses it) |
| GET | `/projects/{id}/videos/{video_id}/frame` | `n` (frame number), `w` (width, 16 to 4096, 320 by default) | JPEG |
| POST | `/projects/{id}/videos/{video_id}/plan` | `{start, stop, step, every_seconds}` | how many frames the selection gives |
| POST | `/projects/{id}/tasks/from-video` | `{video, name, job_size, start, stop, step, every_seconds}` | job |
| GET, POST | `/projects/{id}/datasets`, `/datasets/preview` | see [DATASETS.md](DATASETS.md) | versions / preview / the new version |
| GET, DELETE | `/projects/{id}/datasets/{n}` | see [DATASETS.md](DATASETS.md) | version / `{deleted}` |
| GET | `/training/catalog` | `type` | models with hyperparameter specs, selection metrics |
| GET | `/training/environment` | | installed packages, device found |
| GET / POST | `/projects/{id}/runs` | POST: run configuration (docs/TRAINING.md) | runs (each with `model_number` and `has_weights`) / the new run, stored as `queued`; it starts as soon as no other heavy process is running |
| GET / DELETE | `/projects/{id}/runs/{n}` | DELETE: `{confirm: true}` (409 while running) | config, status, metrics, results / `{deleted}` |
| POST | `/projects/{id}/runs/{n}/cancel`, `/reproduce` | | the run / a new run with the same configuration |
| GET | `/projects/{id}/runs/{n}/log`, `/children/{c}` | `lines` (200 by default) | log tail / one child of a grid or all-folds run |
| GET, POST | `/projects/{id}/models` | see [TRAINING.md](TRAINING.md) | models / save a run |
| GET, PATCH, DELETE, POST | `/projects/{id}/models/{n}`, `/export`, `/download/{kind}`, `/models/public`, `/models/public-catalog` | see [TRAINING.md](TRAINING.md) | |
| GET, POST | `/projects/{id}/preannotations`, `/accept`, `/{n}/cancel` | see [PREANNOTATE.md](PREANNOTATE.md) | |

### Annotation tools, sync and import

See [ANNOTATION.md](ANNOTATION.md) (`/annotation-tools`, `/projects/{id}/annotate`), [SYNC.md](SYNC.md)
(`/projects/{id}/links`, `/projects/{id}/sync/{tool}`, `/cvat/audit`) and [IMPORT.md](IMPORT.md) (`/import`).

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
