# ActiveLabel — synchronisation with CVAT and Label Studio

A project can be **linked** to one remote project in CVAT or Label Studio. *Synchronise*
brings both sides to the same annotations, with a review before anything changes.

## Link

`<project>/links.json`: `{"<tool>": {url, remote_id, remote_name, created, last_sync,
items: {"<image_id>": {"key": "<remote item key>", "base": "<hash of the labels at last sync>"}}}}`
(one link per tool; `key` identifies the image in the remote project, e.g. `"<task_id>:<frame>"`
in CVAT or the task id in Label Studio). The hash is a stable SHA-256 of the image's labels in
the internal format (sorted, coordinates rounded to 4 decimals), so geometry noise from the
round trip does not count as a change.

A link is created:
- when a project is **imported** (the remote project/task it came from; every image mapped);
- when the person **sends** to CVAT/Label Studio and no link exists yet (the remote project is
  created and linked);
- when the person chooses *Synchronise* for an existing remote project with the same name
  (images matched by file name; unmatched ones are reported).

## Send when the remote project already exists

`POST /api/projects/{id}/annotate` with a CVAT/Label Studio tool first checks for a link or a
remote project with the same name (`"ActiveLabel <id> - <name>"`, or the imported remote name).
If one exists → **409** with detail
`{code: "remote_exists", tool, remote: {id, name, url, images}, linked: bool, options: ["cancel", "sync"]}`
and nothing is sent. The interface asks: *Cancel* or *Synchronise* (opens the sync preview).

A link whose remote project was deleted in the tool (the tool answers "not found") is dropped, and
the send goes on as if there were no link; when the tool cannot be reached the link stands. Closing
an annotation session with *delete in the tool* also drops the link when its remote project went
with the session.

## Synchronise

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
   remote items with no local image → reported as `remote_only` (count + names; optionally
   imported); images excluded locally are left alone.
   Result: `{plan_id, tool, remote: {...}, pull: n, push: n, add: n, conflicts: [{image_id, name,
   local: {labels, per_class}, remote: {labels, per_class}}], remote_only: {count, examples},
   diff_pull: <diff as in ANNOTATION.md>, diff_push: <same shape>}` saved in
   `<project>/sync/<plan_id>.json` with the remote labels it read.
2. **Apply** (job `sync_apply`) with `{plan_id, resolutions: {"<image_id>": "local"|"remote"},
   default: "local"|"remote"|null}`: re-reads only what it needs, refuses if the remote changed
   again since the plan (409 "Synchronise again"), then: backup + write pulled labels into the
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
| POST | `/api/projects/{id}/sync/{tool}` | `{create?: bool, remote_id?}` | job `sync_plan` |
| POST | `/api/projects/{id}/sync/{tool}/{plan_id}/apply` | `{resolutions, default}` | job `sync_apply` |

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
