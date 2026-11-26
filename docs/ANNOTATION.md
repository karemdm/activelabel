# ActiveLabel — annotation with external tools

People draw and correct annotations in a tool of their choice. ActiveLabel stays the source of truth: work
is **sent** to the tool and **brought back**; nothing changes in the project until the person reviews a
summary of the changes and applies it.

| Tool | What it is | Used for |
| --- | --- | --- |
| FiftyOne | The FiftyOne App, *Annotate* mode (draw/edit boxes and polygons, change class, delete; keypoints are shown but not edited) | **edit sessions** (this page) |
| CVAT | CVAT (online or self-hosted) | **synchronisation** of the linked project (docs/SYNC.md); it has no edit sessions |

FiftyOne is the hub for inspection: every project has a FiftyOne dataset, named
`activelabel-<id>-<random>` and recorded in `<project>/fiftyone.json` (docs/DESIGN.md). In the interface
this is the step **2 · Annotations** (*Anotações*): the tab *CVAT* (synchronisation and review in CVAT) and
the tab *FiftyOne* (edit sessions, described here). Review in CVAT happens only through the
synchronisation.

**Discontinued tools.** Label Studio support was removed, and so were the CVAT edit sessions (CVAT is
reviewed through the synchronisation; see [CHANGELOG.md](../CHANGELOG.md)). A workspace made before that may still hold their sessions, links or
imported-from records: they are shown as belonging to a discontinued tool and can only be **closed**
(sessions) or **unlinked** (links); nothing is changed in the tool, and no request to it is made.

Same functions in the tool: add, edit, delete, change class — for classification (one class per image),
detection (bounding boxes) and segmentation (polygons). **Keypoints cannot be edited in the FiftyOne App**
(its Annotate mode edits classifications, boxes and polygons of images only): a keypoints project sees its
points in FiftyOne and is edited in CVAT (docs/SYNC.md). The API refuses a FiftyOne edit session in such a
project (400 `fiftyone_keypoints_not_editable`: "The FiftyOne App does not edit keypoints: edit them in
CVAT."), and in the interface the *Edit in FiftyOne* entry of a task's open menu is disabled, with the reason
written under it, while *View in FiftyOne* stays (docs/TASKS.md, *Interface*).

**How to edit in the App.** The App only edits a field that has an active *label schema*. ActiveLabel
gives `ground_truth` one each time it refreshes the dataset (type of the project, classes of the project
in a list), so nothing has to be set up: open the session's link, click an image, choose the *Annotate*
tab of the image window, pick `ground_truth` and draw, move, relabel or delete. Edits are saved in the
dataset as you make them; bring them back from the *FiftyOne* tab. Edits made in a view that is not a
session (a task or a curation view opened for a look) are kept too — a refresh never overwrites a sample
edited in FiftyOne — and come back with the next session that holds those images.

**Polygons.** A segmentation polygon is sent as a FiftyOne polyline with `closed=True, filled=True`, and the
label schema of `ground_truth` makes a polyline drawn in the App closed and filled too (the App's own default
is an open, unfilled line, which the schema's attribute defaults override). When labels are brought back,
`closed` and `filled` are not read for a ring of 3 points or more: every label of a segmentation project is a
polygon, so such a ring is one (closed implicitly), even when it was saved as `closed=False, filled=False`.
Only a polyline that is neither closed nor filled and has no ring of 3 points is a line, and is skipped
(`open_polyline`).

The FiftyOne App runs on port 5161 (or the next free one; a port where an App of another program already answers is skipped, so a link never shows another program's App and database). Its links point to `http://localhost:<port>`.

**Images with no objects.** An image can be listed in the project's annotations with an empty list of
labels: reviewed, with nothing to label (a background image, for example). It is not "annotated", which
everywhere (step bar, tasks, jobs, summary) means at least one reviewed label; the step status and the
project card count it apart as "N without objects" (*N sem objetos*), and it is enough for the
*Annotations* step to count as loaded (`counts.empty` in the project overview, `images_empty` in
`GET /projects/{id}/annotations`). Images that only hold pre-annotations are counted apart too, as waiting
for review ([PREANNOTATE.md](PREANNOTATE.md)).

## Flow (FiftyOne)

1. **Choose which images** (*Which images*): all active images, only images without annotations, the
   images selected in the curation grid, one issue filter, or only the images whose labels are all
   pre-annotations (`preannotated`, see [PREANNOTATE.md](PREANNOTATE.md)). The choice starts on *All
   images*, or on the one the person chose last in this project (kept in the browser); *Only images without
   annotations* means images that are not listed in the project's annotations, so an image reviewed with no
   objects (see below) is not in it. A task or one of its jobs can also be sent from its open menu in the
   *Tasks and jobs* tab (*Edit in FiftyOne*): scope `task`, exactly its active images; the session records
   the task (and job) and its card names it.
2. **Send** (*Send to FiftyOne*, a job): the FiftyOne dataset is refreshed from the project (images +
   current ground truth) and the App opens on a saved view with those images, field `ground_truth`.
   A **session** is recorded in `<project>/annotate.json` (and `annotate/<session>/`): id, tool, image
   ids, created, status (`open` → `pulled` → `applied` / `discarded`), links. Each sample records what
   ActiveLabel wrote in its `ground_truth` (`activelabel_written`: a fingerprint of those labels;
   `activelabel_labels`: the hash of the project labels they came from). An image of a session that is
   still open (not applied or discarded) whose `ground_truth` no longer matches its fingerprint was
   **edited in FiftyOne**: it keeps the FiftyOne version, and the result says how many
   (`kept_fiftyone_edits`). Every other image shows the project's current labels, so a project edit made
   meanwhile (re-import, CVAT sync, accepted pre-annotations) reaches FiftyOne. A dataset written by an
   earlier version has no fingerprints: its session images count as edited (nothing is lost).
3. **Bring back** (*Bring annotations back*, a job): read the `ground_truth` field of those samples,
   convert it to the internal format (`activelabel.labels`) and **diff** it against the project ground
   truth for the session images: per image, labels added / removed / edited (class change, geometry
   change). Also validate against the type and the class list (unknown classes are reported) and the
   geometry: a polygon ring of fewer than 3 points (`polygon_too_few_points`), an open line (not closed
   nor filled) without a ring of 3 points (`open_polyline`), a pose with another number of points than the
   project's (`keypoints_count_mismatch`), a box without area or entirely outside the image (`box_invalid`) or
   a label without the shape the project type needs (`no_geometry`) is listed as a problem and skipped on
   apply, never stored silently; an image with more than one class in a classification project keeps the
   first (`multiple_classes`); a detection mask keeps only its box (`mask_only`). Only the
   images edited in FiftyOne bring FiftyOne's version; the others keep the project's current labels, so
   a project edit is never undone by an image nobody touched in FiftyOne. **Conflicts**: an image edited
   in FiftyOne whose project labels also changed since they were written into FiftyOne is listed under
   the problem `edited_on_both_sides`; applying gives it FiftyOne's version, so the person decides on the
   summary (apply, or discard and send again).
4. **Review and apply**: the person sees the summary (counts per class, images changed, problems) and
   chooses *Apply changes* (the project ground truth for those images becomes the tool's version; new
   classes are added to the project) or *Discard these changes*. A safety copy of `annotations.json` is made
   before applying (see *Safety copies* below). If the project's labels of those images changed after they were brought back,
   applying is refused (409) and they must be brought back again. After a successful apply the session's
   samples record the fingerprint of their current `ground_truth` and the hash of the new project labels:
   FiftyOne and the project are the same, so the next session shows no edited images and no conflicts, and
   a later change of the project reaches those images. Anything FiftyOne held that apply skipped (a problem
   above) is not in the project; it is replaced by the project's labels at the next send.
   *Discard these changes* does the opposite for FiftyOne: it rewrites `ground_truth` (and the
   fingerprints) of every image of the session with the project's current labels, so the edits made in the
   App are undone and "discard and send again" starts from the project. Nothing in the project changes.
   (A discard cannot reach FiftyOne when its dataset is missing; the edits then stay.)
   Applying also records the session in the annotations' `source` (`GET /projects/{id}/annotations`) as
   `last_session: {tool, session, at}`, next to the import information (`format`, `files`), which is kept.
5. **Close** a session (*Close session*): forget it.

## Safety copies

Before anything that removes or rewrites annotations in bulk, `annotations.json` is copied next to itself as
`annotations.<UTC time>Z.<reason>.bak.json`. The reason is one of: `classes` (a used class was removed),
`task` (a task change that discards the labels), `apply` (FiftyOne edits applied), `sync` (CVAT synchronised),
`rename` (a class renamed in CVAT), `replace` (an import that replaces) or `restore` (the file a restore
replaced). Copies made by older versions have no `Z`; their time is the file's modification time. Only the
newest 20 copies of a project are kept (`KEEP_BACKUPS` in `activelabel/backups.py`): making a new one deletes
the oldest beyond that.

The *Annotations* step lists them under the summary (*Safety copies*): time, reason, images and labels, and
*Restore*, after a confirmation. Restoring makes the copy the annotations again, after making a copy of the
current file (so a restore can be undone the same way). The project takes the task of the copy and gets the
classes of the copy that it does not have, so no label is left with an unknown class. The API is
`GET /projects/{id}/annotations/backups` and `POST .../backups/{name}/restore` (docs/DESIGN.md); the name is
checked against the names the app writes, so no other file can be reached.

## Credentials

Only CVAT needs them. Set in the interface: the *Settings* (*Ajustes*) button of the *CVAT* tab (*Set up CVAT*,
*Configurar o CVAT*, before it is set up) or the *Set up* / *Settings* button of the CVAT card on the import page
(docs/IMPORT.md). The form starts with the address empty and an example of one as its placeholder (the address of CVAT
online); the password or token is one field, called *Password or token* (*Senha ou token*), and when one is
already saved the field says so and leaving it empty keeps it. Once CVAT is set up, the *CVAT* tab shows
"Connected to <server> as <user>". URL and username are stored in
`<workspace>/tools.json`; the password / token **only** in the system keyring (service
`activelabel`, key `<tool>:<url>`). They are never written to a file, never returned by the API
and never logged. The URL is trimmed and must start with `http://` or `https://`, name a server and have no
spaces (400 `url_invalid`), and one with a user or password in it (400 `url_has_credentials`: they go in their own fields). *Save and test connection* (*Salvar e testar conexão*) saves them and checks the login;
a result message about an old value is cleared as soon as a field is edited.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/annotation-tools` | | `[{tool, configured, url, username, needs}]` for `fiftyone` and `cvat` (no secrets) |
| PUT | `/api/annotation-tools/{tool}` | `{url, username?, secret?}` | the tool entry (secret kept in keyring when given) |
| POST | `/api/annotation-tools/{tool}/test` | | `{ok, message}`; a failed test also has `code` and `params` |
| DELETE | `/api/annotation-tools/{tool}` | | `{removed: true}` (also removes the keyring secret) |
| GET | `/api/projects/{id}/annotate` | | `{sessions: [...]}` newest first |
| POST | `/api/projects/{id}/annotate` | `{tool: "fiftyone", scope: "all"\|"unlabeled"\|"selected"\|"issue"\|"preannotated"\|"task", ids?, issue?, task?, job?}` (`task`, with optional `job`, for scope `task`; 404 `task_not_found`, 400 `job_not_found`) | job; result `{session, links, kept_fiftyone_edits?}`; 400 `tool_unsupported` for any other tool; 400 `fiftyone_keypoints_not_editable` for a keypoints project (the App cannot edit keypoints) |
| POST | `/api/projects/{id}/annotate/{session}/pull` | | job; result `{session, diff}`; 409 when the session was already applied or discarded; 400 `tool_unsupported` for a session of a discontinued tool |
| POST | `/api/projects/{id}/annotate/{session}/apply` | | `{applied: {...counts}}`; 400 when nothing was brought back yet, 409 when the labels changed since the pull or it was already applied or discarded |
| POST | `/api/projects/{id}/annotate/{session}/discard` | | `{discarded: true}`; puts the session images in FiftyOne back to the project labels; 409 when already applied (apply and discard of one session take the same lock: exactly one wins) |
| DELETE | `/api/projects/{id}/annotate/{session}` | | `{closed: true}` (a session of a discontinued tool is only forgotten) |

`diff` = `{images_changed, added, removed, edited, per_class: {name: {added, removed, edited}},
unknown_classes: [...], problems: [{level, code, message, count, examples}], examples:
[{image_id, name, added, removed, edited}] (max 20)}`.

Session (as returned): `{id, tool, scope, images, created, status, links: [{label, url}], task?, job?, diff?}`.
