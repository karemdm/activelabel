# ActiveLabel — import a project from CVAT

Besides creating a project, people can **import** one that already exists in CVAT. The import brings
the images and their annotations and creates a normal ActiveLabel project: everything works the same
afterwards (curation, annotation sessions, editing, task change, ...). The origin is only recorded for
information.

The CVAT login is the same as the one of the synchronisation (docs/SYNC.md); it is set on the import page
or in the *CVAT* tab (the password only in the system keyring, docs/ANNOTATION.md).

## What is imported

The source is a CVAT project, or a task that is not in a project. Images: project, or task of a project:
frames downloaded through the sync engine (see below); task outside a project: FiftyOne
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

Images keep their remote file names; identical files are added once (content hash); unreadable
files are listed in the report. Nothing is changed on the remote server.

## Annotation files (the *Import annotations* tab)

Annotation files (or a .zip) of an existing project follow a check-then-apply flow
(`POST /projects/{id}/annotations`, then `POST .../{token}/apply`, docs/DESIGN.md): the files are kept
aside in `staging/<token>/`, the format is detected, the files are matched with the images, and a report
says what was found. Nothing changes until the person chooses an action.

- A check that finds nothing usable (unknown format, unreadable file, no labels) is not kept aside: the
  files are removed at once and the answer is a 400 with a code (`annotation_format_unknown`,
  `annotation_unreadable`). Files that were kept and never decided are removed after 24 hours and when the
  service starts.
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
- CVAT shapes without a label are reported and left out, not imported as an empty class.
- *Change the type to {task} and import* always rewrites the stored type of the saved annotations as well
  as the project's; changing the type or the classes of a project (PATCH) waits for training, ONNX export
  and pre-annotation of the project to finish (409 `busy_training`), because they read both.

## HTTP API

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/api/import/{tool}/projects` | `search?` | `{items: [{id, kind: "project"\|"task", name, images, created, classes: [...] }]}` (newest first, max 200) |
| POST | `/api/import` | `{tool, remote_id, kind, name, description?, task?}` | job (`kind: "import_project"`, `project_id: 0`) |
| GET | `/api/jobs/{job_id}` | | the job (any project, including 0) |

Job result (CVAT projects add `tasks`, the tasks imported, and `linked`): `{project_id, name, task, images_added, images_rejected, labels, classes,
problems: [{level, code, message, count, examples}]}`. While running, `progress.update`
messages: "Listing <tool> data", "Reading annotations", "Comparing with CVAT", "Downloading images",
"Creating the project". Cancelling stops at a safe point and removes the half-created project.
Errors: tool not configured → 400 "Configure <tool> first" (`tool_not_configured`); remote project not
found → 404; `ToolError` → 400 with its message.

The address of the tool must be `http://` or `https://` (`url_invalid`). Images that cannot be downloaded
are listed in the report (`image_download_failed`) and the import goes on. Frames of a CVAT task are paired
with the local images by position in the task, not by file name. A failure reported by CVAT is answered as
400 `tool_error` (`params.tool`, `params.reason`).
