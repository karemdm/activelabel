# ActiveLabel — import: a project from CVAT, and annotation files

There are two kinds of import:

- **A project (or task) from CVAT**, on the *Import a project* page (*Importar um projeto*, reached by the *Import project* button of the project
  list and by *Import from CVAT* when the list is empty): it brings the images and their annotations and creates a normal ActiveLabel
  project: everything works the same afterwards (curation, annotation sessions, editing, task change, ...).
  The origin is only recorded for information. CVAT is the only source (`tool` must be `cvat`; any other name
  answers 404 `tool_unknown`); Label Studio is no longer supported (see [CHANGELOG.md](../CHANGELOG.md)).
- **Annotation files** (COCO, YOLO, Pascal VOC, CVAT XML, CVAT video, MOT, or class folders) loaded into a
  project that already has its images: the *Import annotations* tab of the *Annotations* step, described
  below. For the files of a video task see also docs/VIDEO.md.

The CVAT login is the same as the one of the synchronisation (docs/SYNC.md); it is set on the import page
(one *Set up* / *Settings* button on the CVAT card, no second one in the error message) or in the *CVAT* tab
(the password only in the system keyring, docs/ANNOTATION.md).

## What is imported from CVAT

The source is a CVAT project, or one CVAT task (of a project or outside any). Images: project, or task of a
project: frames downloaded through the sync engine (see below); task outside a project: FiftyOne
`import_annotations(..., download_media=True)`. Annotations: boxes, polygons, points/skeletons, tags.
Classes: the project/task labels.

**CVAT structure.** Importing a CVAT project keeps it: one local task per CVAT task (same name,
frames in order, `job_size` = the task's `segment_size`), the labels as classes, and a link with the
CVAT project (docs/SYNC.md) that records the tasks and jobs mapping (docs/TASKS.md), so the next
synchronisation has nothing to do. It reuses the sync engine (`make_plan` with `remote_tasks`, then
`apply_plan`), reading CVAT through `sync.registry.get_remote("cvat")`. Importing one task that
belongs to a project imports only that task (a project with one task) and links to the task's CVAT
project; the project's other tasks then show up as "added in CVAT" in the next synchronisation. A task
outside any project becomes one local task, with no tasks mapping.

**Task of the new project**: inferred from the annotations (`labels.task_of`); when several
kinds exist (e.g. boxes and polygons), the most frequent one wins and the import report says
how many annotations of other kinds were converted (polygon → box) or left out. Projects
without annotations import as the task the person picks (default detection). Name: the
remote name (made unique with " (2)", ...), editable before importing; description says where
it came from.

**One local project per CVAT project.** A CVAT project that already has a local project linked to it (one
imported before, or one made by ActiveLabel and sent) is marked in the list ("Already in ActiveLabel as
...", with a button that opens it) and is not imported again: importing it twice would link two projects
to the same CVAT project, and each would offer the other's tasks as new. `POST /api/import` answers 409
`cvat_already_imported` (`project_id`, `name`); the import of a task of such a project starts, and its job then fails with the same code (`error_code`, `error_params`). New
tasks of that project come with *Compare with CVAT* in the project that already exists. A CVAT project
without images is marked and cannot be chosen ("No images in CVAT yet"). `GET /api/import/{tool}/projects`
adds `linked_to: {id, name}` to the projects that have a local one.

Images keep their remote file names; identical files are added once (content hash). A project import (or a
task of a project) says when fewer images were added than CVAT listed (`images_missing`, an information
line). The import of a task outside any project reports identical files as `images_duplicate` and unreadable
ones as `images_rejected`. A CVAT project or task with no images is not imported (`cvat_no_images`). The import also counts what it cannot bring instead of dropping it
silently: CVAT shapes that are not converted, by kind (ellipses, masks, tracks...: `shapes_not_converted`; they
stay in CVAT), rotated rectangles, which are imported as the box around them (`rotated_boxes_flattened`; the
rotation stays in CVAT), a task whose original video could not be downloaded or used and so stays as images
(`video_as_images`, with the task names), and the annotations of images left out as identical or unreadable
(`duplicate_labels_dropped`).
Nothing is changed on the remote server.

## Annotation files (the *Import annotations* tab)

Annotation files (or a .zip) of an existing project follow a check-then-apply flow
(`POST /projects/{id}/annotations`, then `POST .../{token}/apply`, docs/DESIGN.md): the files are kept
aside in `staging/<token>/`, the format is detected, the files are matched with the images, and a report
says what was found and what was left out and why. Nothing changes until the person chooses an action. While
that report waits for a decision it is kept with the files and shown again after a reload (`GET .../annotations/pending`),
until it is applied or discarded; leaving the page (a link, the language switch, closing the tab) asks for
confirmation first.

- **Formats** are detected by content (`detect_format`): COCO (`.json` with `images` and `annotations`), YOLO
  (one `.txt` of label lines per image, with `data.yaml`, `classes.txt` or a `.names` file for the class names; also
  pose), Pascal VOC (`.xml`), CVAT XML,
  CVAT for video, MOT (`gt.txt`/`det.txt`), class folders and regression values (`.csv`, or a `.json` of values). Files of two different formats, or none usable,
  are refused (`annotation_format_unknown`). Images are matched by relative name, then by the same name ignoring
  case, then by file name alone; a name shared by several images is reported (`ambiguous_image_name`). The
  other names an image was uploaded with also match (docs/TASKS.md).
- **Class folders** (a .zip or folder of images with one sub-folder per class, and no annotation file): each image
  gets the name of its parent folder as its classification label. It needs at least two class folders, with at
  least 90 % of the images inside one; images at the top level get no class. The labels are read from the
  names of the images of the project, so the project's images must have been uploaded with those folder names;
  the uploaded copies in the check are only used to recognise the format. They are classification labels, so a
  project of another type is offered *Change the type to classification and import*.
- **Mode**: *replace* (the default) or *merge* (`mode` of the apply request). Merge only adds labels to images that
  have none yet, keeps the existing ones (`images_kept`), adds only the classes of the labels it added, and is
  refused when the existing labels are of another type; it makes no safety copy because nothing is removed.
- A check that finds nothing usable (unknown format, unreadable file, no labels) is not kept aside: the
  files are removed at once and the answer is a 400 with a code (`annotation_format_unknown`,
  `annotation_not_json`, `annotation_not_xml` or `annotation_not_text` with the file name and the line and
  column of the first error, `annotation_files_missing`, `annotation_zip_invalid`, `annotation_unreadable`; the
  interface shows each by its code, in its own language, never the parser's English text). Files that were kept and never decided are removed after 24 hours and when the
  service starts.
- **Excluded images are never changed by an import.** Replace swaps the labels of the *active* images only;
  the labels an excluded image already has are kept (also when the file has entries for it, which are
  skipped and reported as `excluded_image_entries`, not as images missing from the project), so excluding
  and restoring an image never loses its labels. The exception is *Change the type ... and import*: labels
  of the old type cannot stay beside the new ones, so they are left out (they are in the backup). Every
  replace first copies `annotations.json` (`annotations.<UTC time>Z.replace.bak.json`, see docs/DESIGN.md);
  the answer names it in `backup`.
- **Labels the project's type cannot use are reported and left out**, never imported as empty or broken
  labels: a box project does not take a label that has no box (`no_geometry`: "Some labels do not have the
  shape this project type needs (box, polygon or keypoints); they are left out."), and a keypoints project
  leaves out labels whose number of points differs from the most common one
  (`keypoints_count_mismatch`). When the project is a detection project and the file has polygons or
  keypoints, *Convert to {task} and import* turns them into boxes (`converted_to_box`).
- **YOLO lines that cannot be read are reported with their place**: the example is `<file>:<line>: <start of
  the line>` (an unknown class id, a polygon with too few points, a wrong number of keypoints, ...). A file
  called `labels.txt`, `classes.txt` or `names.txt` that holds label lines is the label file of an image of
  that name; one that only lists names gives the class names.
- **YOLO pose without `kpt_shape`**: a line with `5 + 3k` values is read as a pose when a visibility value is
  2 and as a polygon when one of those values is not 0, 1 or 2. When it could be either (every such value is
  0 or 1), the lines of the file set that are clearly one kind decide, and the report warns
  (`yolo_pose_ambiguous`). Add `kpt_shape: [points, 3]` to `data.yaml` to remove the doubt.
- A Pascal VOC file without `<filename>` (and without `<path>`) is matched by its own name: `c.xml` goes with
  `c.png`.
- A CVAT for video file numbered by **video frames** (the numbers CVAT's own export writes, for example
  `0, 15, 30...` with a frame step of 15) is mapped to the frames of the video task; when its `<meta>` gives
  the task's first and last frame, every task frame in that range with no objects counts as annotated with no
  objects (docs/VIDEO.md).
- CVAT shapes without a label are reported and left out, not imported as an empty class.
- **Rotated CVAT boxes** (`rotation` on a `box`, in images and in video tracks) are read as the axis-aligned
  box that encloses the turned rectangle, as the CVAT synchronisation does (docs/SYNC.md); the angle itself is not kept.
- **Other things the report can list** (each with a count and examples): `unmatched_entries` (annotations of
  images that are not in the project), `duplicate_image_entry`, `size_mismatch` (the file's image size differs
  from the uploaded image), `box_degenerate` (a box without area, skipped), `polygon_too_few_points` (a ring of
  fewer than 3 points, skipped), `unknown_class_id`, `missing_class_names`, `rle_bbox_only` (a COCO mask is
  kept as its box), `unsupported_shape`, `invalid_annotation` and `dropped_other_kind`; the texts are in the `problem.*` entries of the language files.
- **Shapes outside the image are reported**: a box is clipped to the image (`box_clipped`); a polygon vertex or
  a keypoint outside it is moved to its border and the report says so (`geometry_clipped`), since the shape
  changed.
- In a classification project an image with labels of several classes keeps the first one; the apply answer
  says how many images were reduced (`classes_reduced`).
- *Change the type to {task} and import* always rewrites the stored type of the saved annotations as well
  as the project's; changing the type or the classes of a project (PATCH) waits for training, ONNX export
  and pre-annotation of the project to finish (409 `busy_training`), because they read both.

## Regression values (CSV/JSON)

A regression project (docs/DESIGN.md, issue #619) has a number per target and image; its *classes* are the
targets (for example `age`). The values come in one `.csv` or `.json` file (format `regression`):

- **CSV**: a header row; one column names the image (`image`, `file`, `filename` or `path`, in any case; the
  first column when none of these is there) and every other column is a target, in column order. The delimiter
  (comma, semicolon or tab) is read from the header row. A decimal comma is read when the value is quoted
  (`"3,5"`) or the delimiter is a semicolon. An empty cell means no value for that target. A row with more cells
  than the header, or without an image name, is skipped (`invalid_annotation`).
- **JSON**: `{"a.jpg": {"age": 34.5, "height": 1.7}, ...}`, or a list `[{"image": "a.jpg", "age": 34.5}, ...]`. A
  JSON with `images` and `annotations` is COCO, not values.
- Images are matched by name like the other formats (unknown names: `unmatched_entries`). A value that is not a
  finite number (text, `nan`, `inf`) is skipped and reported (`value_invalid`, with examples).
- On import an image keeps the first value of each target (`repeated_target` reports the images that had more;
  `classes_reduced` in the answer counts them). Each value is a `Label(label=<target>, value=<number>)`.
- The values can also be edited one by one on the Images screen (click the number, type, Enter to save).
- The values are edited in ActiveLabel (Images screen) and shown in FiftyOne as one float field per target,
  view only (docs/ANNOTATION.md); they do not go to CVAT (docs/SYNC.md). Importing a CVAT project, or a task of
  one, with the type *Regression* is refused (400 `cvat_not_for_regression`): CVAT keeps no number per image,
  so use the CSV or JSON file.
- The data split balances bins of the values of the first target (docs/DATASETS.md).
- Not built yet for regression projects: training and pre-annotation answer 400
  `regression_not_supported_yet`.

## HTTP API

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/api/import/{tool}/projects` | `search?` | `{items: [{id, kind: "project"\|"task", name, images, created, classes: [...], linked_to?: {id, name} }]}` (newest first, max 200) |
| POST | `/api/import` | `{tool, remote_id, kind, name, description?, task?}` | job (`kind: "import_project"`, `project_id: 0`) |
| GET | `/api/jobs/{job_id}` | | the job (any project, including 0) |

Job result (CVAT projects add `tasks`, the tasks imported; `linked` says whether the new project is linked to CVAT, shown on the result screen, see [SYNC.md](SYNC.md#link)): `{project_id, name, task, images_added, images_rejected, labels, classes,
problems: [{level, code, message, count, examples}]}`. While running, `progress.update`
messages: "Listing CVAT data", "Reading annotations", "Comparing with CVAT", "Downloading images",
"Creating the project" (shown in the interface language); while the frames of a CVAT project are downloaded the job counts them
(`done` / `total`, message "Downloading images"), so the bar advances image by image. Cancelling stops at a safe point and removes the half-created project.
Errors: tool not configured → 400 "Configure <tool> first" (`tool_not_configured`); remote project not
found → 404 (`cvat_item_not_found`); any other failure of CVAT → 400 with the code of the failure and its message.
The import answers 501 `fiftyone_not_installed` when FiftyOne is not installed, also for a project, since the
check is made before the kind is known.

The address of the tool must be `http://` or `https://` (`url_invalid`). A frame that CVAT does not deliver
stops the import (`cvat_download_failed`, with the HTTP status) and the half-created project is removed.
Frames of a CVAT task are paired with the local images by position in the task, not by file name. A failure reported by CVAT keeps its own code (for
example `cvat_unreachable`, `cvat_login_failed`, `cvat_refused`, see docs/DESIGN.md); only an error with no code is
answered as 400 `tool_error` (`params.tool`, `params.reason`).
