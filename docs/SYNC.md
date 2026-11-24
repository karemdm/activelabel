# ActiveLabel — synchronisation with CVAT and Label Studio

A project can be **linked** to one remote project in CVAT or Label Studio. Synchronising
brings both sides to the same annotations, with a review before anything changes. In the interface
this is offered for CVAT only (the *CVAT* tab of *Annotations*); Label Studio is switched off in the
interface for now, while the code and the API below accept both tools.

## Link

`<project>/links.json`: `{"<tool>": {url, remote_id, remote_name, created, last_sync, revision,
tasks?, items: {"<image_id>": {"key": "<remote item key>", "base": "<hash>", "labels": [...],
"removed"?: true}}}}`
(one link per tool; `key` identifies the image in the remote project, e.g. `"<task_id>:<frame>"`
in CVAT or the task id in Label Studio; `revision` goes up on every save, so a comparison made before
another synchronisation is refused; `tasks` is the mapping of docs/TASKS.md). For each image the link
keeps the labels as they were at the last sync (`labels`) and their hash (`base`, a stable SHA-256 of
the labels in the internal format, sorted, coordinates rounded to 4 decimals); `removed` marks an image
removed on both sides. Whether a side changed is decided by comparing its labels with `labels` with a
tolerance of 0.2 % of the image side, so geometry noise from the round trip does not count as a change;
links written by older versions only have the hash and are compared exactly.

A link is created:
- when a project is **imported** (the remote project/task it came from; every image mapped);
- when the person **sends** to CVAT/Label Studio and no link exists yet (the remote project is
  created and linked);
- when the person chooses *Synchronise* (*Sincronizar*) for an existing remote project with the same name
  (images matched by file name; unmatched ones are reported).

## Send when the remote project already exists

`POST /api/projects/{id}/annotate` with a CVAT/Label Studio tool first checks for a link or a
remote project with the same name (`"ActiveLabel <id> - <name>"`, or the imported remote name).
If one exists → **409** with detail
`{code: "remote_exists", tool, remote: {id, name, url, images}, linked: bool, options: ["cancel", "sync"]}`
and nothing is sent. The interface asks: *Cancel* or *Synchronise* (opens the sync preview).

A link whose remote project was deleted in the tool (the tool answers "not found") is dropped, and
the send goes on as if there were no link; when the tool cannot be reached the link stands. When
the person compares (*Synchronise*) with a linked project that was deleted in CVAT, nothing is
changed: the *CVAT* tab of *Annotations* says the project no longer exists and suggests *Unlink* (*Desvincular*);
after unlinking, a new comparison creates a new project in CVAT. Closing
an annotation session with *delete in the tool* also drops the link when its remote project went
with the session.

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
missing in the remote project are added to it (CVAT labels / Label Studio config).

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
shapes. Pushing what was read changes nothing. CVAT may assign new track ids when it stores the
annotations (the PUT carries no ids).

After a push, CVAT renumbers the tracks it stores and re-interpolates the frames next to a new
keyframe. The tasks the push touched are therefore read again, and their tracked frames that had no
pending change here take the labels CVAT has now (new track ids, re-interpolated boxes), so the next
comparison has nothing to do. Note that CVAT splits a track at the end of each job: one object that
crosses jobs has one track id per job.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/links` | | `{links: [{tool, url, remote_id, remote_name, last_sync, images}]}` |
| DELETE | `/api/projects/{id}/links/{tool}` | | `{unlinked: true}` (remote untouched) |
| POST | `/api/projects/{id}/sync/{tool}` | `{create?, remote_id?, scope: "project"\|"task"\|"job" (default project), task?, job?, direction: "both"\|"pull"\|"push" (default both)}` | job `sync_plan` |
| POST | `/api/projects/{id}/sync/{tool}/{plan_id}/apply` | `{resolutions, default, delete_tasks}` | job `sync_apply`; 404 when the comparison no longer exists |

## Remote adapters (Python)

```python
class RemoteProject(Protocol):          # activelabel/sync/remote.py
    tool: str
    def find(self, settings, name: str) -> RemoteInfo | None
    def get(self, settings, remote_id: str) -> RemoteInfo            # {id, name, url, images}
    def create(self, settings, project: Project) -> RemoteInfo
    def items(self, settings, remote_id: str, task: Task) -> list[RemoteItem]
    def add(self, settings, remote_id: str, project: Project,
            images: list[tuple[ImageRecord, Path, list[Label]]]) -> dict[str, str]   # image_id -> key
    def write(self, settings, remote_id: str, project: Project,
              updates: dict[str, list[Label]]) -> None                # key -> labels (replace)
    def ensure_classes(self, settings, remote_id: str, classes: list[str]) -> None
```
`RemoteItem = {key, file_name, width, height, labels: list[Label]}` (labels relative 0..1).
