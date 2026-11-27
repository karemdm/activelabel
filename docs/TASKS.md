# ActiveLabel — tasks and jobs (aligned with CVAT)

A project is organised like a CVAT project:

```
project ── task 1 ── job 1, job 2, …
        ├─ task 2 ── job 1, …
        └─ …
```

- **Task**: a package of images added together (one upload = one new task). A project has any
  number of tasks; adding images always means adding a task. Like in CVAT, the images of a task are
  fixed once it is created (an image can still be excluded, which CVAT calls a deleted frame).
- **Job**: a fixed-size chunk of a task, in the task's image order. The size ("images per job") is
  chosen when the task is created and never changes, like CVAT's `segment_size`. Empty = one job
  with every image of the task. Example: 100 images, 10 per job → jobs 1–10.

Names in the interface: *Task* and *Job* in every language (as in CVAT). The project's annotation
kind (classification, detection…) is called **Type** (*Tipo*), and the card of background processing (bottom-right
corner of the project page) is called **Activities** (*Atividades*). In the code, `Project.task` (the type) and `activelabel.jobs` (background
processing) keep their names; the new concepts live in `activelabel.tasks_store` (`ImageTask`,
`TaskJob`) to avoid confusion.

## Storage

`<project>/tasks.json`:

```json
{"next_id": 3,
 "tasks": [
   {"id": 1, "name": "Task 1", "created": "<iso>", "job_size": 10, "open": false, "source": null,
    "images": ["<image id>", "…"]},
   {"id": 2, "name": "Field trip", "created": "<iso>", "job_size": null, "open": false,
    "source": {"video": "<video id>", "start": 0, "stop": 8999, "step": 30, "fps": 30.0},
    "images": ["…"]}
 ]}
```

- Every image of the project (active or excluded) is in exactly one task; `images` keeps the order
  of the upload (the order of the files in the upload, as in CVAT). An image is stored once, by its content:
  a file the project already has is skipped as a duplicate and does not enter the new task. The name it was
  uploaded with is kept as another name of the image (`aliases` in the manifest, at most 20 per image), so
  annotation files that use either name still match it (docs/IMPORT.md).
- Jobs are not stored: job *k* (1-based) of a task is `images[(k-1)*job_size : k*job_size]`.
- `open` is true while a chunked upload may still append images (see below); `incomplete` is true when
  that upload was cut off (see below); `touched` (saved, not sent to the interface) is the time of the last
  chunk of an open task. `source` is set for a task made from a video (docs/VIDEO.md) and `null` otherwise.
- Task names are unique within the project and at most 100 characters (400 `task_name_taken`,
  `task_name_too_long`); the default name is "Task <n>", one more than the highest number among the tasks named that way
  (so an upload that added nothing, or a deleted task, does not use up a number; the id still advances, and
  a name that is taken gets a number added: `Name (2)`). The job size is a whole number
  from 1 to 100,000 (`job_size_invalid`).
- A task is never empty. An upload that adds nothing (no readable image, only duplicates) leaves no task:
  - a task created **closed** (`open=false`, the way for a one-shot upload through the API) with no image
    added is refused: 400 `no_images_added` with the upload report;
  - a task created **open** (the default, used by the chunked upload of the interface) that ends empty is
    **removed when it is closed**: `PATCH …/tasks/{tid}` with `{open: false}` deletes it and answers the task
    view with `removed: true`;
  - the interface shows the reason of each rejected file.
- Ids are never reused; a deleted task's id is not given again.
- **Migration**: a project without `tasks.json` that has images gets one task, `"Task 1"`, with
  every image in manifest order and `job_size: null`. Done on first read, then saved.
- Deleting a task deletes its images (files, annotations, thumbnails), after confirmation.
- Deleting images (the existing *delete* action) removes them from their task; a task left empty
  is removed.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/tasks` | query `refresh=true` (optional) | `{tasks: [task view]}`; closes the tasks whose upload was cut off (see below); `refresh` reads the CVAT job states again instead of reusing a read of the last minute |
| POST | `/api/projects/{id}/tasks` | multipart: `name` (optional; default "Task <n>", see above), `job_size` (optional), `open` (default `true`), `files` | `{task: task view, report: <upload report>}`; `open=false` with no image added is refused (400 `no_images_added`); an open task that stays empty is removed when it is closed (see below) |
| POST | `/api/projects/{id}/tasks/{tid}/images` | multipart `files` | upload report — only while the task is being created by a chunked upload (see below) |
| PATCH | `/api/projects/{id}/tasks/{tid}` | `{name?, open?}` (`open: false` closes the task and, when it holds no image, removes it: the answer has `removed: true`; reopening a closed task is refused: 409 `task_closed`) | task view |
| POST | `/api/projects/{id}/tasks/from-video` | see docs/VIDEO.md | job (the task is created closed) |
| DELETE | `/api/projects/{id}/tasks/{tid}` | `{confirm: true}` | `{deleted: n images}` |

Task view: `{id, name, created, job_size, open, incomplete, source, images, active, annotated, jobs: [{n, first,
last, images, active, annotated}], remote?: {...}, cvat}` (`first`/`last`: 1-based positions in the task;
`remote` only when the project is linked to CVAT, see below). `annotated` counts the images with at least
one reviewed label, the same number as on the step bar: an image whose labels are all pre-annotations is not
counted until it is reviewed, and an image reviewed with no objects is counted apart in the step status
(*sem objetos*, docs/ANNOTATION.md). `cvat` is `ready` when `remote` is there,
otherwise the reason the task cannot be opened in CVAT: `not_configured` (no CVAT connection),
`not_linked` (the project is not linked), `not_synchronised` (the task is not in CVAT yet), `removed_remote`
(CVAT answered without it: it was deleted there), `remote_gone` (the CVAT project itself was deleted) or
`unreachable` (it is there, but CVAT did not answer). In the *Open in…* menu of a task or job, the first four
lead to *Annotations → CVAT* (*Go to CVAT in Annotations*), where a comparison creates the project or the
task in CVAT, or decides what to do with what was deleted there; `not_configured` and `unreachable` keep the
item disabled with the reason. The project header's *Open in CVAT* does the same once the last comparison
found the CVAT project deleted, and its address is built on the CVAT server set up now
(`GET /projects/{id}/links`), so it follows a change of that address.

Chunked upload: the browser sends big uploads in chunks. The first chunk creates the task
(`POST …/tasks` returns its id), later chunks append with `POST …/tasks/{tid}/images` while the
task is `open` (`"open": true` in tasks.json until the browser calls `PATCH …/tasks/{tid}`
`{open: false}`, or the next task is created). Once closed, the images of a task are fixed.

**An upload cut off midway** (the page was closed, the connection lost) never calls the closing `PATCH`, so
the app closes the task itself and marks it `incomplete: true` so the interface can say that images may be
missing (the images that arrived stay; further chunks are refused, 409 `task_closed`): when the next task
is created, or when the task list is read and the last chunk is older than 30 minutes. Start-up does not
touch the tasks. A task closed by the browser is `incomplete: false`. The card of an incomplete task says
"Upload cut off: images may be missing" (*tasks.incomplete*), and the missing images go in a new task. In the
browser, while an upload runs the new-task dialog cannot be hidden with Esc (a toast says to use *Cancel*), and
leaving the page (a link, the language switch, closing the tab) asks for confirmation. *Cancel* in the upload
stops sending, closes the task with the images that arrived (`incomplete: false`: the person chose to stop) and
says how many files had been sent.

`GET /api/projects/{id}/images` gains `task` and `job` query parameters (filter). The existing
`POST /api/projects/{id}/images` (upload) creates a new task with the next automatic name ("Task <n>"), for compatibility.

## CVAT mirror

A project linked to CVAT has one **mirror project** there: same name rule as today, one CVAT task
per ActiveLabel task (same name, same images in the same order, `segment_size = job_size`), so the
jobs match one to one. The link (`links.json`, docs/SYNC.md) records the mapping:

```json
"tasks": {"<task id>": {"remote_task": 123, "name": "Field trip", "jobs": {"1": 456, "2": 457}}}
```

and items keep the key `"<remote task>:<frame>"`. `name` is the task name as it was at the last
synchronisation.

**Task names** follow the same three-way rule as the labels, against that recorded name: renamed only
here, the CVAT task is renamed (`PATCH /api/tasks/{id}` with the name only); renamed only in CVAT, the
local task takes that name (not in a `push`); renamed on both sides to different names, both names are
kept and the synchronisation reports a warning until one side is changed. `push` always sends the local
name and `pull` always takes the CVAT one. A rename that fails (for example a name already used by
another local task) is a warning and is tried again at the next synchronisation. Renaming a task here
does not contact CVAT by itself: the new name reaches CVAT on the next synchronisation.

**Sending to CVAT is synchronising.** The annotation sessions that created separate 50-image CVAT
tasks are replaced, for CVAT, by the sync with the mirror project. The person chooses the scope:

- **Project**: every task.
- **Task**: one task.
- **Job**: one job of a task.

`POST /api/projects/{id}/sync/cvat` body `{scope: "project"|"task"|"job", task?, job?}` → plan
limited to the images of the scope; apply as today. Rules:

The sync covers **images and annotations**, in both directions, for what was **added, removed or
modified** on each side since the last sync (three-way comparison with the state recorded in the
link, as in docs/SYNC.md):

| What changed | in ActiveLabel → CVAT | in CVAT → ActiveLabel |
| --- | --- | --- |
| Task added | created in CVAT whole (all its images, in order, `segment_size = job_size`) | new local task: its images are downloaded, `job_size = segment_size` (project scope) |
| Task deleted | the CVAT task is deleted | the local task's images are excluded (reason "deleted in CVAT"); nothing is deleted locally |
| Image removed (excluded or deleted) | the frame is deleted in CVAT (`deleted_frames`) | the image is excluded (reason "deleted in CVAT") |
| Image restored | the frame is restored in CVAT | the image is restored |
| Annotations added / modified / removed (objects of an image) | written to CVAT | written to the project |

- The link records, per image, the labels **and** whether it was removed at the last sync, so each
  side's change is known. An image changed on both sides in different ways (e.g. labels edited in
  CVAT and the image removed in ActiveLabel, or different labels on both sides) is a **conflict**:
  the person chooses per image (or all one side).
- The review before applying lists, per direction: tasks added/deleted, images added/removed/
  restored, and annotations per class (added, modified, removed), with examples.
- A local task not yet in CVAT is created there whole (CVAT needs every image of a task when it is
  created), even when the scope is one of its jobs; its jobs then exist on both sides.
- Deletions in CVAT never delete local files: they exclude (reversible). Deleting a CVAT task from
  ActiveLabel asks for confirmation in the review.
- Backup of the project annotations before applying, as in docs/SYNC.md.

The task view's `remote` part is `{task, url, jobs: [{n, id, url, state, stage, assignee}]}`: per job, the
CVAT job id, link, state (`new`, `in progress`, `completed`, `rejected`), stage (`annotation`, `validation`,
`acceptance`) and assignee, read from CVAT when the tasks are listed. A read is reused for a minute
(`REMOTE_TTL`), and the task list never waits for CVAT longer than 8 seconds (`REMOTE_TIMEOUT`): if CVAT is
slow or unreachable the tasks show without `remote` (their `cvat` is then `unreachable`). The *Update*
(*Atualizar*) button of the *Tasks and jobs* tab, shown only when at least one task is linked to CVAT, reads
the states again at once (`?refresh=true`) and confirms with a notice, even when nothing changed. Each job row
shows its state, its stage and who it is assigned to.

## Import from CVAT

Importing a CVAT project creates one local task per CVAT task (same name, same order of frames,
`job_size = segment_size`) and links them (mapping above). Importing a single CVAT task creates a
project with that one task, linked to the task's project in CVAT when it has one.

## Interface

- **Images** step, tab *Tasks and jobs* (*Tasks e jobs*): the project's tasks as cards (name, images,
  annotated, jobs). *New task*: name, images per job (optional), files, folder or video (docs/VIDEO.md).
  A task card lists its jobs (images 1–10, 11–20…; CVAT state when linked) and opens the tab *Gallery*
  (*Galeria*) filtered on the task or job; *Gallery* also has task and job filters of its own.
  The open icon on a task, and on each job row, is a menu (button with `aria-haspopup`; arrows, Home, End,
  Esc): *Edit in FiftyOne* (*Editar no FiftyOne*) starts an edit session (docs/ANNOTATION.md) on exactly the
  active images of the task or job; *Open in CVAT* (*Abrir no CVAT*) opens the linked CVAT task or job. An
  item that cannot be used stays in the menu, disabled, with the reason written under it (CVAT not set up,
  project not linked, task not synchronised yet, CVAT did not answer; the texts are `tasks.cvat_*`).
  Keypoints cannot be edited in FiftyOne: in a keypoints project *Edit in FiftyOne* is disabled ("FiftyOne
  does not edit keypoints: use CVAT") and the menu adds *View in FiftyOne*, which opens the task for a look
  only (`POST /projects/{id}/fiftyone`); the API refuses the edit session as well (docs/ANNOTATION.md).
  *Edit in FiftyOne* needs nothing set up by the person: the label schema the App requires to edit
  `ground_truth` is written each time the dataset is refreshed (docs/ANNOTATION.md, *How to edit in the App*).
- A task card also has *Rename* and *Delete* (which asks for a confirmation). When one of these
  answers 404 or 409 because the task changed in another tab, the interface reloads the project and the task
  list so no dead card stays.
- *New task* dialog: the footer button reads *Cancel* until something was created and *Close* afterwards; while
  files are being sent it is hidden and the upload has its own *Cancel* (see *Chunked upload*). The name field
  shows "Task <n>" as its placeholder (the name is optional), and the *Images per job* hint says it cannot change after
  creation. A task that adds no image reports no success: the dialog goes back to the form and the report says
  why each file was rejected.
- **Annotations** step, tabs *CVAT* and *FiftyOne*: tool + scope (project / task / job). For CVAT the
  button is *Compare with CVAT* (*Comparar com o CVAT*): it makes the plan, you review it, and *Apply
  synchronisation* (*Aplicar a sincronização*) carries it out; for FiftyOne the session works as in
  docs/ANNOTATION.md, limited to the scope.
- Wherever images can be chosen (curation, annotate), the task and job filters are available.
