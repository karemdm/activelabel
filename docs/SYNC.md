# ActiveLabel — synchronisation with CVAT

A project can be **linked** to one remote project in CVAT. Synchronising brings both sides to the same
annotations, with a review before anything changes. In the interface this is the *CVAT* tab of
*Annotations*; it is also the only way to review in CVAT (there are no CVAT edit sessions). The API takes
`cvat` as `{tool}`; any other name answers 404 `tool_unknown`.

A link written for a tool that cannot be synchronised any more (an old workspace; see
[CHANGELOG.md](../CHANGELOG.md)) is listed by `GET /api/projects/{id}/links` with `supported: false` and shown as "Linked to a
discontinued tool" (*Vinculado a uma ferramenta descontinuada*): it cannot be synchronised and can only be
unlinked, which changes nothing in that tool.

## Link

`<project>/links.json`: `{"<tool>": {url, remote_id, remote_name, created, last_sync, revision,
tasks?, items: {"<image_id>": {"key": "<remote item key>", "base": "<hash>", "labels": [...],
"removed"?: true}}}}`
(one link per tool; `key` identifies the image in the remote project, `"<task_id>:<frame>"` in CVAT;
`revision` goes up on every save, so a comparison made before
another synchronisation is refused; `tasks` is the mapping of docs/TASKS.md). For each image the link
keeps the labels as they were at the last sync (`labels`) and their hash (`base`, a stable SHA-256 of
the labels in the internal format, sorted, coordinates rounded to 4 decimals); `removed` marks an image
removed on both sides. Whether a side changed is decided by comparing its labels with `labels` with a
tolerance of 0.2 % of the image side, so geometry noise from the round trip does not count as a change;
links written by older versions only have the hash and are compared exactly.

A link is created:
- when a project is **imported** (the remote project/task it came from; every image mapped);
- when the person chooses *Synchronise* (*Sincronizar*) for an existing remote project with the same name
  (images matched by file name; unmatched ones are reported).

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
from other tools or older versions. The audit never moves or deletes anything.

## A linked project that was deleted in CVAT

When the person compares (*Synchronise*) with a linked project that was deleted in CVAT, nothing is
changed: the *CVAT* tab of *Annotations* says the project no longer exists and suggests *Unlink*
(*Desvincular*); after unlinking, a new comparison creates a new project in CVAT.

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
   add, remove_remote, remove_local, restore_remote, restore_local, tasks_create, skipped_tasks,
   tasks_new, tasks_delete, conflicts: [{image_id, name, local: {labels, per_class}, remote: {labels,
   per_class}, ...}], remote_only, unchanged, diff_pull: <diff as in ANNOTATION.md>, diff_push:
   <same shape>}` saved in `<project>/sync/<plan_id>.json` with the remote labels it read (the last 10
   plans are kept).
2. **Apply** (job `sync_apply`) with `{plan_id, resolutions: {"<image_id>": "local"|"remote"},
   default: "local"|"remote"|null, delete_tasks: bool}` (`delete_tasks`: also delete in the tool the
   tasks deleted here; false by default): re-reads only what it needs, refuses if the remote, the
   project's annotations or the link changed again since the plan ("Synchronise again"), then: backup + write pulled labels into the
   project ground truth; write pushed labels to the remote; upload added images; update the
   link (`base` hashes, keys, `last_sync`). Result: counts.

Classes: classes found remotely but missing locally are added to the project; local classes
missing in the remote project are added to it (CVAT labels).

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
outside. Skeleton tracks and tracks of other shape types are not read and are kept as they are.
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
| GET | `/api/projects/{id}/links` | | `{links: [{tool, url, remote_id, remote_name, last_sync, images}]}` |
| DELETE | `/api/projects/{id}/links/{tool}` | | `{unlinked: true}` (remote untouched); 409 while a job of the project runs; 404 `not_linked` |
| POST | `/api/projects/{id}/sync/{tool}` | `{create?, remote_id?, scope: "project"\|"task"\|"job" (default project), task?, job?, direction: "both"\|"pull"\|"push" (default both)}` | job `sync_plan` |
| GET | `/api/cvat/audit` | | `{url, projects: [{id, name, url, linked_to: {id, name}\|null, tasks: [{id, name, size, url}]}], loose: [{pattern, tasks}]}`; read-only |
| POST | `/api/projects/{id}/sync/{tool}/{plan_id}/apply` | `{resolutions, default, delete_tasks}` | job `sync_apply`; 404 when the comparison no longer exists |

## Remote adapters (Python)

```python
class RemoteProject(Protocol):          # activelabel/sync/remote.py
    tool: str
    def check(self, settings) -> str                                  # login test, "Connected to ..."
    def find(self, settings, name: str) -> RemoteInfo | None
    def get(self, settings, remote_id: str) -> RemoteInfo            # {id, name, url, images}
    def create(self, settings, project: Project) -> RemoteInfo
    def items(self, settings, remote_id: str, task: Task) -> list[RemoteItem]
    def add(self, settings, remote_id: str, project: Project,
            images: list[tuple[ImageRecord, Path, list[Label]]]) -> dict[str, str]   # image_id -> key
    def write(self, settings, remote_id: str, project: Project,
              updates: dict[str, list[Label]]) -> None                # key -> labels (replace)
    def ensure_classes(self, settings, remote_id: str, classes: list[str]) -> None
    def audit(self, settings) -> dict            # {projects: [...], loose: [...]}, read-only
```
`RemoteItem = {key, file_name, width, height, labels: list[Label]}` (labels relative 0..1).
