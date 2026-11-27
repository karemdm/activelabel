# ActiveLabel — synchronisation with CVAT

A project can be **linked** to one remote project in CVAT. Synchronising brings both sides to the same
annotations, with a review before anything changes. In the interface this is the *CVAT* tab of
*Annotations*; it is also the only way to review in CVAT (there are no CVAT edit sessions). The API takes
`cvat` as `{tool}`; any other name answers 404 `tool_unknown`. When CVAT is set up, the tab says which
server and user it is connected to (*Connected to ... as ...*); the connection is set as described in
docs/ANNOTATION.md (*Credentials*).

A link written for a tool that cannot be synchronised any more (an old workspace; see
[CHANGELOG.md](../CHANGELOG.md)) is listed by `GET /api/projects/{id}/links` with `supported: false` and shown as "Linked to a
discontinued tool" (*Vinculado a uma ferramenta descontinuada*): it cannot be synchronised and can only be
unlinked, which changes nothing in that tool.

## Link

`<project>/links.json`: `{"<tool>": {url, remote_id, remote_name, created, last_sync, revision,
tasks?, classes?, items: {"<image_id>": {"key": "<remote item key>", "base": "<hash>", "labels": [...],
"removed"?: true}}}}`
(one link per tool; `key` identifies the image in the remote project, `"<task_id>:<frame>"` in CVAT;
`revision` goes up on every save, so a comparison made before
another synchronisation is refused; `tasks` is the mapping of docs/TASKS.md; `classes` is
`{"<class id in the tool>": "<name>"}` at the last sync, to notice a class renamed there). For each image the link
keeps the labels as they were at the last sync (`labels`) and their hash (`base`, a stable SHA-256 of
the labels in the internal format, sorted, coordinates rounded to 4 decimals); `removed` marks an image
removed on both sides. Whether a side changed is decided by comparing its labels with `labels` with a
tolerance of 0.2 % of the image side, so geometry noise from the round trip does not count as a change;
links written by older versions only have the hash and are compared exactly. For a label with polygons only the polygons are compared: its box is derived from them (a COCO box is a pixel wider
than the one the tool computes).

A link is created:
- when a project is **imported** (the remote project/task it came from; every image mapped);
- when the person chooses *Synchronise* (*Sincronizar*) for an existing remote project with the same name
  (images matched by file name; unmatched ones are reported).

**After an import from CVAT.** A CVAT project, or a task that belongs to one, is imported through the
synchronisation engine and is **linked right away**: one local task per CVAT task (same jobs), every image
mapped, nothing to do in the next comparison. From the project page the person can at once:
- open the project, a task or a job in CVAT (the *CVAT* tab, or the menu of each task in *Tasks and jobs*);
- *Compare* in the *CVAT* tab and synchronise in **both directions** (edits made in CVAT come back,
  edits made here go there, conflicts are shown before anything changes);
- *Edit in FiftyOne* (the FiftyOne tab of a task or job): the FiftyOne dataset is made on the first
  send; edits applied from FiftyOne become project edits, which the next synchronisation sends to CVAT.

So an edit made in either tool reaches the other: FiftyOne → project (apply) → CVAT (synchronise), and
CVAT → project (synchronise) → FiftyOne (next send). A **task outside any CVAT project** is imported as a
copy, **not linked** (synchronisation works with CVAT projects); the import result says so. To keep it in
step, move the task into a project in CVAT and import that project.

**Task guard.** In CVAT every task lives **inside** the linked CVAT project. The CVAT client refuses to
create a task without a `project_id` (`POST /api/tasks` with none is stopped before anything is sent, with
the text "Refusing to create a CVAT task outside a project"): a task outside any project would be invisible
to the synchronisation and would keep counting in the plan. If an upload of later batches fails, only the
half-made task is deleted.

**Audit of the server.** The *Tasks on the server* (*Tasks no servidor*) button of the *CVAT* tab opens a
**read-only audit** (`GET /api/cvat/audit`, 400 `tool_error` when CVAT cannot be reached or refuses the
login): the tasks of every project of the configured server, saying which project is linked to one of
ours, and the tasks that are in no project, grouped by name pattern (digits shown as `#`, e.g.
`street_round_#`; the biggest group first). ActiveLabel only creates tasks inside a project, so those come
from other tools or older versions. The audit never moves or deletes anything. A list CVAT does not
offer (it answers 404, as an old version or an address that is not the root of CVAT does) is left out and
named in `missing` (`projects` or `tasks`), the rest is shown; when neither list exists the answer is 400
`cvat_audit_unavailable` with the address.

## A linked project that was deleted in CVAT

When the person compares (*Synchronise*) with a linked project that was deleted in CVAT, nothing is
changed: the comparison fails with the code `remote_gone` and the *CVAT* tab of *Annotations* says, in the
interface language, that the project no longer exists, with an *Unlink* (*Desvincular*) button; after
unlinking, a new comparison creates a new project in CVAT.

A job that fails with a coded error (`remote_gone` here; `sync_remote_changed`, `sync_local_changed`,
`sync_link_changed`, `sync_task_has_new_images` or `plan_not_found` when something changed after the
comparison; `sync_no_remote`, ...) carries `error_code` and `error_params` next to its `error` text, so the
interface can translate it. When an apply fails with one of the "changed since the comparison" codes, the
card says so and offers *Synchronise again* (*Sincronizar de novo*) instead of a plain *Close*.

## Synchronise

In the interface the plan is made and reviewed in the *CVAT* tab of the *Annotations* step: choose what to
compare (*Whole project*, *Task* or *Job*) and the direction, click *Compare with CVAT* (*Comparar com o
CVAT*), review, and click *Apply synchronisation* (*Aplicar a sincronização*).

1. **Plan** (job `sync_plan`): read every remote item (labels converted to the internal
   format) and compare, per image, three versions: `base` (hash at last sync), `local` (project
   ground truth) and `remote`:

   | local changed | remote changed | action |
   | --- | --- | --- |
   | no | no | nothing |
   | no | yes | **pull** (remote → ActiveLabel) |
   | yes | no | **push** (ActiveLabel → remote) |
   | yes | yes, same result | nothing (hashes equal) |
   | yes | yes, different | **conflict** — the person chooses per image (or "all ActiveLabel" / "all remote") |

   Plus: active images not yet in the remote project → **add** (uploaded with their labels);
   remote items with no local image → reported as `remote_only` (count + names); images and tasks
   that exist only here are listed in `only_here` and left alone in `pull`. Removal and restoring of
   images and tasks follow docs/TASKS.md (*CVAT mirror*).
   Result: `{plan_id, tool, remote, create, scope, direction, only_here: {images, tasks}, pull, push,
   add, images_new, labels_new, remove_remote, remove_local, restore_remote, restore_local, tasks_create, skipped_tasks,
   tasks_new, tasks_delete, tasks_removed_remote: [{task, name, images}] (tasks deleted in the tool: their images go), tasks_rename: [{task, remote_task, old, new, direction: "push"|"pull"}],
   tasks_rename_conflicts: [{task, remote_task, local, remote}], project_rename: {old, new, direction: "push"|"pull"}|null,
   project_rename_conflict: {local, remote}|null, conflicts: [{image_id, name, local: {labels, per_class}, remote: {labels,
   per_class}, ...}], remote_only, unchanged, renamed_classes: [{from, to}], not_converted: {kind: n},
   diff_pull: <diff as in ANNOTATION.md>, diff_push:
   <same shape>}` saved in `<project>/sync/<plan_id>.json` with the remote labels it read (the last 10
   plans are kept). `images_new` and `labels_new` count what is sent to CVAT for the first time (the
   images of tasks created there and the added images, and their labels); the review lists them as *New
   labels to send* (*Rótulos novos a enviar*), next to the images with annotation changes. The column of what
   comes from CVAT also lists the classes renamed there (`diff_pull` shows their labels under the old name,
   so they are not a new class nor an edited label) and, from `not_converted`, the shapes CVAT keeps as they
   are and the project cannot hold (ellipses, masks, open lines, cuboids, tracks; rotated boxes, read here as
   their enclosing box), by kind. A conflict whose two sides have the same classes and counts says that the
   position or the shape of an object differs. A comparison that
   finds nothing to bring, send, rename or decide says "Everything is already in sync" (*Tudo já está
   sincronizado*) in the review, with nothing to apply, and raises no "comparison ready" notice; the job
   result reads "Already in sync". Discarding a comparison in the review deletes the saved plan on the server
   (`DELETE`, see *HTTP API*), so no other tab can apply it afterwards; applying a comparison that no
   longer exists answers 404 `plan_not_found` and the interface asks to compare again.
2. **Apply** (job `sync_apply`) with `{plan_id, resolutions: {"<image_id>": "local"|"remote"},
   default: "local"|"remote"|null, delete_tasks: bool}` (`delete_tasks`: also delete in the tool the
   tasks deleted here; false by default; a task left in the tool is remembered in the link as
   `kept_remote` and is neither offered for deletion again nor imported back): re-reads only what it needs, refuses if the remote, the
   project's annotations or the link changed again since the plan ("Synchronise again"), then: backup + write pulled labels into the
   project ground truth; write pushed labels to the remote; upload added images; update the
   link (`base` hashes, keys, `last_sync`). Result: counts (`pulled`, `pushed`, `added`, `images_new` and
   `labels_new` sent for the first time, `removed_*`, `restored_*`, `tasks_created`, `tasks_imported`,
   `tasks_deleted`, `tasks_renamed`, `project_renamed`, `conflicts_left`, `unchanged`, `new_classes`, `renamed_classes`,
   `backup`, `skipped`, `warnings`); the result card shows *Images sent to CVAT* and *Labels sent to CVAT*
   from `images_new` and `labels_new`.
   Task renames follow docs/TASKS.md (*CVAT mirror*): the local name is sent, or the remote one pulled,
   against the name recorded at the last synchronisation; both changed = both kept and a warning.
   The **project name** follows the same rule against the `remote_name` of the link (project scope only):
   renamed only here, the CVAT project is renamed (`PATCH /api/projects/{id}`); renamed only in CVAT, the
   project takes that name here. A project made here is called `ActiveLabel <id> - <name>` in CVAT, and the
   prefix is added or removed on the way; a project imported from CVAT keeps the plain name on both sides.
   A name typed differently at the import counts as a rename made here and is offered on the first
   synchronisation. Renaming is never blocked while CVAT is down: the name is sent at the next
   synchronisation that can reach it. A rename that fails is a warning, tried again next time.

Classes: classes found remotely but missing locally are added to the project; local classes
missing in the remote project are added to it (CVAT labels). Class names are compared ignoring case,
like the project's classes: a CVAT label `Cat` is the class `cat` (no second label is created, and a
shape of `Cat` stays on that label when its frame is pushed). A class **renamed in CVAT** keeps its
label id there: the link remembers the ids (`classes`), so the plan lists it in `renamed_classes` and
the apply renames the class here too, in the class list and in every label of the ground truth (after a
backup `annotations.<UTC time>Z.rename.bak.json`); it is not created again in CVAT. A rename to a name that
is already another class here (two classes merged) is not done automatically.

## Direction

The person chooses the direction before comparing (`direction` in `POST /sync/{tool}`):

| Direction | What happens |
| --- | --- |
| `both` (default) | each change goes the way it was made; a change on both sides is a conflict the person resolves |
| `pull` | the tool wins: this project becomes like it (labels brought, images removed/restored as there); no conflicts; images and tasks that exist only here are left alone and listed (`only_here`) |
| `push` | this project wins: the tool becomes like it; no conflicts; tasks that exist only in the tool are left alone |

A task deleted here is deleted in the tool only in `both`/`push` and only when confirmed.

## Tracks (CVAT)

CVAT objects marked with *tracks* (interpolation) are synchronised as one label per frame where the
object is visible, all with `track` = the CVAT track id (FiftyOne: the `index` attribute). Reading:
keyframes as they are; between two keyframes rectangles and points are interpolated linearly like
CVAT, **polygons use the previous keyframe's points** (approximation); an `outside` keyframe hides the
object until the next visible keyframe; after the last keyframe the object continues unless it is
outside. Skeleton tracks and tracks of other shape types are not read and are kept as they are
(counted in the plan's `not_converted`).
Writing: a label with a `track` of the task changes that track (a keyframe on that frame, replacing
or inserted; CVAT interpolates the frames around it again); a track missing from a frame's labels gets
an `outside` keyframe there, plus keyframes on the neighbouring frames so they keep their shape;
labels whose track is gone (or whose class/shape type no longer fits the track) are written as plain
shapes. Pushing what was read changes nothing. A changed track is sent as an update of that track
(same id).

After a push, CVAT re-interpolates the frames next to a new keyframe. The tasks the push touched are
therefore read again, and their tracked frames that had no pending change here take the labels CVAT
has now (re-interpolated boxes, and a new track id if the server gave one), so the next comparison has
nothing to do. Note that CVAT splits a track at the end of each job: one object that crosses jobs has
one track id per job.

## Shapes and how they are read

| CVAT | ActiveLabel |
| --- | --- |
| rectangle | box (detection) |
| rectangle with `rotation` | the axis-aligned box that **encloses** the turned rectangle (detection trains on axis-aligned boxes). The rectangle stays turned in CVAT: a push of its frame leaves it alone while its enclosing box is unchanged; moving the box moves the rectangle and keeps its turn; resizing it replaces the rectangle by the new box, without a turn. In tracks the turn is interpolated between keyframes, the short way round |
| polygon | polygon and its box (segmentation) |
| points | one pose object (keypoints). A point at the top-left corner is *not labeled* (0). CVAT has one `occluded` flag per shape: a pose whose labeled points are all *labeled but not visible* (1) is an occluded shape; a pose that mixes 1 and 2 keeps each point's visibility in the text attribute `visibility` of its CVAT label (`"2,1,0"`; added to that label the first time a pose needs it). CVAT has no box for a pose: a pose whose points did not change keeps the box the project has (from COCO, for example), so the box alone is never a change; when the points change the box becomes the box of the points |
| skeleton | one pose object (keypoints projects): one point per sublabel of the skeleton label, in the order of the sublabels; a missing or `outside` part is *not labeled* (0), an `occluded` one is 1. The class is the skeleton label. A pose of a skeleton class is written back as a skeleton (each part keeps its id) |
| tag | image class (classification) |

Shapes of other types (ellipse, mask, polyline, cuboid), skeletons outside a keypoints project and tracks
that are not read (skeleton tracks, tracks of other types) stay in CVAT as they are and never reach the
project or the training. The plan says so: `not_converted` counts them by kind (`{"ellipse": 2,
"mask": 1, "tracks": 1, ...}`), and `rotated_rectangle` counts the turned rectangles read as their
enclosing box.

## What a push sends to CVAT

A push never replaces the annotations of a task. For each task it reads the annotations, works out
the differences on the frames being pushed and sends them as `PATCH /api/tasks/{id}/annotations` with
`action=delete`, then `update`, then `create` (only the non-empty ones):

- a shape that matches a label (same class, type and geometry within the tolerance) is not sent: it
  keeps its id, attributes, `occluded`, `z_order`, `group`, rotation and source;
- an edited label takes the old shape of the same class and type it overlaps most (IoU of their boxes
  above 0.5): that shape is **updated** with the new points and keeps its id and everything else;
- a shape left without a label is deleted, and a label left without a shape is created with CVAT's
  defaults (no attributes);
- only objects read just before can be deleted: what someone draws in CVAT meanwhile, on any frame,
  stays; shapes of types ActiveLabel does not convert, and frames that are not pushed, are never sent.

A new task (*add*) is filled with one `PUT`, since it has no annotations yet.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/links` | | `{links: [{tool, url, remote_id, remote_name, last_sync, images, supported}]}` |
| DELETE | `/api/projects/{id}/links/{tool}` | | `{unlinked: true}` (remote untouched); 409 while a job of the project runs; 404 `not_linked` |
| POST | `/api/projects/{id}/sync/{tool}` | `{create?, remote_id?, scope: "project"\|"task"\|"job" (default project), task?, job?, direction: "both"\|"pull"\|"push" (default both)}` | job `sync_plan` |
| GET | `/api/cvat/audit` | | `{url, projects: [{id, name, url, linked_to: {id, name}\|null, tasks: [{id, name, size, url}]}], loose: [{pattern, tasks}], missing: ["projects"\|"tasks"]}`; read-only |
| DELETE | `/api/projects/{id}/sync/{tool}/{plan_id}` | | `{discarded: true}`: the comparison is deleted, so no tab can apply it; 404 `plan_not_found` when it does not exist |
| POST | `/api/projects/{id}/sync/{tool}/{plan_id}/apply` | `{resolutions, default, delete_tasks}` | job `sync_apply`; 404 when the comparison no longer exists |

## Remote adapters (Python)

```python
class RemoteProject(Protocol):          # activelabel/sync/remote.py
    tool: str
    def check(self, settings) -> str                                  # login test, "Connected to ..."
    def audit(self, settings) -> dict            # {projects: [...], loose: [...], missing: [...]}, read-only
    def find(self, settings, name: str) -> RemoteInfo | None
    def get(self, settings, remote_id: str) -> RemoteInfo            # {id, name, url, images}
    def create(self, settings, project: Project) -> RemoteInfo
    def items(self, settings, remote_id: str, task: Task,
              task_ids: list[str] | None = None) -> list[RemoteItem]  # None: the whole project
    def tasks(self, settings, remote_id: str) -> list[RemoteTask]     # tasks and jobs (state, stage, assignee)
    def add_task(self, settings, remote_id: str, project: Project, name: str, job_size: int | None,
                 images: list[tuple[ImageRecord, Path, list[Label]]],
                 video: tuple[Path, dict] | None = None) -> tuple[RemoteTask, dict[str, str]]
    def add(self, settings, remote_id: str, project: Project,
            images: list[tuple[ImageRecord, Path, list[Label]]]) -> dict[str, str]   # image_id -> key
    def write(self, settings, remote_id: str, project: Project,
              updates: dict[str, list[Label]]) -> None                # key -> labels the item must end with
    def set_frames(self, settings, task_id: str, delete: list[int], restore: list[int]) -> None
    def rename_task(self, settings, task_id: str, name: str) -> None
    def rename_project(self, settings, project_id: str, name: str) -> None
    def delete_task(self, settings, task_id: str) -> None
    def frames(self, settings, task_id: str) -> list[tuple[str, bool]]   # (file name, deleted) in order
    def frame(self, settings, task_id: str, frame: int) -> bytes
    def video_source(self, settings, task_id: str, dest_dir: Path) -> tuple[Path, dict] | None
    def classes(self, settings, remote_id: str) -> dict[str, str]      # {class id: name}
    def ensure_classes(self, settings, remote_id: str, classes: list[str]) -> None
```
`RemoteItem = {key, file_name, width, height, labels: list[Label], not_converted: {kind: n}}` (labels
relative 0..1).

## Tests

The CVAT adapter is tested in the normal CI against a strict in-memory fake of the CVAT REST API
(`tests/helpers/cvat_fake.py`). A separate nightly workflow (`.github/workflows/cvat-live.yml`) runs
`tests/activelabel/sync/test_cvat_live.py` against a real CVAT started in containers for the occasion, to
catch what a fake cannot: the real serializer, the ids CVAT assigns, differences between CVAT versions and
real interpolation. It never touches anyone's server.
