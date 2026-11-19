# ActiveLabel — import a project from CVAT or Label Studio

Besides creating a project, people can **import** one that already exists in CVAT or Label
Studio. The import brings the images and their annotations and creates a normal ActiveLabel
project: everything works the same afterwards (curation, annotation sessions in any tool,
editing, task change, ...). The origin is only recorded for information.

The tools use the same settings and keyring login as the annotation tools (docs/ANNOTATION.md).

## What is imported

| | CVAT | Label Studio |
| --- | --- | --- |
| Source | a project, or a task that is not in a project | a project |
| Images | downloaded from the server (FiftyOne `import_annotations(..., download_media=True)`) | downloaded through the API with the login (uploaded files and URLs the server serves) |
| Annotations | boxes, polygons, points/skeletons, tags | rectanglelabels, polygonlabels, keypointlabels, choices (latest non-cancelled annotation per task) |
| Classes | the project/task labels | the labels of the labeling config |

**Task of the new project**: inferred from the annotations (`labels.task_of`); when several
kinds exist (e.g. boxes and polygons), the most frequent one wins and the import report says
how many annotations of other kinds were converted (polygon → box) or left out. Projects
without annotations import as the task the person picks (default detection). Name: the
remote name (made unique with " (2)", ...), editable before importing; description says where
it came from.

Images keep their remote file names; identical files are added once (content hash); unreadable
files are listed in the report. Nothing is changed on the remote server.

## HTTP API

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/api/import/{tool}/projects` | `search?` | `{items: [{id, kind: "project"\|"task", name, images, created, classes: [...] }]}` (newest first, max 200) |
| POST | `/api/import` | `{tool, remote_id, kind, name, description?, task?}` | job (`kind: "import_project"`, `project_id: 0`) |
| GET | `/api/jobs/{job_id}` | | the job (any project, including 0) |

Job result: `{project_id, name, task, images_added, images_rejected, labels, classes,
problems: [{level, code, message, count, examples}]}`. While running, `progress.update`
messages: "Listing <tool> data", "Downloading images", "Reading annotations",
"Creating the project". Cancelling stops at a safe point and removes the half-created project.
Errors: tool not configured → 400 "Configure <tool> first"; remote project not found → 404;
`ToolError` → 400 with its message.
