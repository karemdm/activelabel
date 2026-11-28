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
CVAT."). In the interface the button of the *FiftyOne* tab then reads *Open in FiftyOne (view only)* and opens
the App to look, with the reason on one line under it; the *Edit in FiftyOne* entry of a task's open menu is
disabled, with the reason written under it, while *View in FiftyOne* stays (docs/TASKS.md, *Interface*).

**How to edit in the App.** The App only edits a field that has an active *label schema*. ActiveLabel
gives `ground_truth` one each time it refreshes the dataset (type of the project, classes of the project
in a list), so nothing has to be set up: click *Edit in FiftyOne* and click an image. The image window opens
in the *Annotate* tab, ready to edit, with the mouse alone:

| To | Do |
| --- | --- |
| Select a box or polygon | Click it (or its row in the *LABELS* list) |
| Move it | Drag it |
| Resize a box, move a polygon point | Drag a corner or a point |
| Change the class | Pick another *label* in the panel on the right (the panel shows only *label* and *tags*: the fields a person never edits, `confidence`, `index`, `mask_path` and the free `attributes`, are left out of the label schema, `fiftyone_bridge.HIDDEN_ATTRIBUTES`; their values stay on the labels) |
| Delete it | Press *Delete*, or *⋮ → Delete label* in the panel |
| Delete several at once | The trash button in the image window's toolbar (*Apagar rótulos* / *Delete labels*): tick the labels, then *Execute*. Outside the image window it lists the labels selected in the App |
| Undo that deletion | The trash-with-an-arrow button next to it (*Desfazer a remoção* / *Undo the deletion*), shown while there is one to undo; the labels come back with their ids |
| Draw a new box or polygon | Pick the rectangle or polygon tool under *Create*, then drag (or click the points) on the image |
| Undo / redo | The arrows next to *Create*, or Ctrl+Z / Ctrl+Shift+Z |

The card of an open FiftyOne session repeats these moves in one line (*annotate.fo_mouse_hint*), so no
shortcut is hidden. The App keeps its drawing tool on while a label is open and after it is deleted; the plugin
picks the select tool (the arrow) again after a deletion (*Delete*/*Backspace* or *Delete label* in the label's
menu), so the next click selects a box instead of drawing one. Backspace while drawing a polygon still removes
its last point. After drawing a new shape the rectangle or polygon tool stays on, to draw the next one. Edits
are saved in the dataset as you make them; review and apply them from the *FiftyOne* tab.

**Deleting several labels.** The two buttons above are Python operators of the same plugin
(`activelabel_annotate/__init__.py`). The deletion is done on the server, so the App's own Ctrl+Z does not
see it: the labels deleted last are kept in the dataset's execution store (`activelabel_annotate`,
`last_deleted`) for *Undo the deletion*, and the page reloads after each of them (the image window comes
back in Annotate mode) so the App shows the change. Nothing reaches the project before *Review and apply*.

**Why the window opens in Annotate.** The App's own default is *Explore*, where the labels are only shown
(hovering a box shows its class and id, nothing moves), and it remembers the tab per dataset in the
browser. ActiveLabel turns on its App plugin `@activelabel/annotate` (`src/activelabel/fiftyone_plugin/`, plain
script, no build) for the App it starts: FiftyOne's plugins folder is set for the ActiveLabel process
only, never in `~/.fiftyone`. When a dataset that ActiveLabel marked editable (`activelabel_editable` in the
dataset info, set by `enable_editing`) opens, the plugin switches the image window to *Annotate*; the
*Explore* tab is still one click away. A keypoints dataset is not marked, so it opens in *Explore*. The plugin uses an App hook that is not a public API (`useModalModeController`); it was checked
with FiftyOne 1.22.1 (`fiftyone_plugin.VERIFIED_FIFTYONE`). With another version ActiveLabel logs a warning,
a test fails until the plugin is checked again, and if the hook is gone the plugin writes one console warning
and the App opens images in *Explore* as usual (click *Annotate* to edit).

**What the App shows.** Only `ground_truth` is drawn on an image (the other fields can be turned on in the
sidebar), and the fields ActiveLabel keeps for itself (`activelabel_written`, `activelabel_labels`, `activelabel_label_refs`,
`duplicate_group`, `position`, `task`, `job`, `video`, `video_name`, `frame_number`) sit in a collapsed
sidebar group, *internal (do not edit)* (`fiftyone_bridge.tidy_app`, the same idea as the first CVAT-FiftyOne
project). When ActiveLabel refreshes a dataset that the App has open (a send, an open, a new session), it
asks that App to reload (`Session.refresh()`), so the person does not press F5 (the App session's port is its `server_port`); if the App does not answer,
the refresh of the dataset still succeeds and F5 is needed. Edits made in a view that is not a
session (a task or a curation view opened for a look) are kept too — a refresh never overwrites a sample
edited in FiftyOne — and come back with the next session that holds those images.

**Polygons.** A segmentation polygon is sent as a FiftyOne polyline with `closed=True, filled=True`, and the
label schema of `ground_truth` makes a polyline drawn in the App closed and filled too (the App's own default
is an open, unfilled line, which the schema's attribute defaults override). When labels are brought back,
`closed` and `filled` are not read for a ring of 3 points or more: every label of a segmentation project is a
polygon, so such a ring is one (closed implicitly), even when it was saved as `closed=False, filled=False`.
Only a polyline that is neither closed nor filled and has no ring of 3 points is a line, and is skipped
(`open_polyline`).

The FiftyOne App runs on port 5161 (or the one in the environment variable `ACTIVELABEL_FIFTYONE_PORT`; either way, the next free one when it is busy; a port where an App of another program already answers is skipped, so a link never shows another program's App and database). Its links point to `http://localhost:<port>`. Starting the App waits at most 2 minutes and switching it to another dataset at most 30 seconds; an App that does not answer in time is dropped and the activity fails with a message instead of waiting for ever (#525).

**Images with no objects.** An image can be listed in the project's annotations with an empty list of
labels: reviewed, with nothing to label (a background image, for example). It is not "annotated", which
everywhere (step bar, tasks, jobs, summary) means at least one reviewed label; the step status and the
project card count it apart as "N without objects" (*N sem objetos*), and it is enough for the
*Annotations* step to count as loaded (`counts.empty` in the project overview, `images_empty` in
`GET /projects/{id}/annotations`). Images that only hold pre-annotations are counted apart too, as waiting
for review ([PREANNOTATE.md](PREANNOTATE.md)).

## Flow (FiftyOne)

1. **Edit in FiftyOne** (a button, the only one in the *FiftyOne* tab): it sends the images chosen and opens
   the App in a new tab. The choice (*Choose images*, folded away by default) is all active images, only images
   without annotations, the images selected in the curation grid, one issue filter, or only the images whose
   labels are all pre-annotations (`preannotated`, see [PREANNOTATE.md](PREANNOTATE.md)). It starts on *All
   images*, or on the one the person chose last in this project (kept in the browser); *Only images without
   annotations* means images that are not listed in the project's annotations, so an image reviewed with no
   objects (see below) is not in it. A task or one of its jobs can also be sent from its open menu in the
   *Tasks and jobs* tab (*Edit in FiftyOne*): scope `task`, exactly its active images; the session records
   the task (and job) and its card names it. *Open in FiftyOne (view only)*, in *Curation*, opens the dataset
   to look, with no session.
2. **Send** (a job, started by the button): the FiftyOne dataset is refreshed from the project (images +
   current ground truth) and the App opens on a saved view with those images, field `ground_truth`.
   Sending images that an **open session** already holds (same tool, scope, task, job and images) does not
   make a second session: the same one is used again, its view and link renewed (a session that was already
   brought back is not reused).
   The job reports real progress on a scale of 100 (`done` of `total`): the images written to the dataset fill
   the first 70, then come the stages *Sending to FiftyOne* (75) and finished (100). The interface opens a tab
   at the click (a browser only allows that straight after a click), shows a waiting line in it and, when the
   job ends, takes it to the App; when the browser gave no tab, the address stays on screen as a link until it
   is closed; when the job fails or is stopped, the tab is closed and the reason is shown. *Open in
   FiftyOne* (the job `fiftyone`, result `{url, view, step}`) works the same way.
   A **session** is recorded in `<project>/annotate.json` (and `annotate/<session>/`): id, tool, image
   ids, created, status (`open` → `pulled` → `applied` / `discarded`), links. Each sample records what
   ActiveLabel wrote in its `ground_truth` (`activelabel_written`: a fingerprint of those labels;
   `activelabel_labels`: the hash of the project labels they came from). An image of a session that is
   still open (not applied or discarded) whose `ground_truth` no longer matches its fingerprint was
   **edited in FiftyOne**: it keeps the FiftyOne version, and the result says how many
   (`kept_fiftyone_edits`). Every other image shows the project's current labels, so a project edit made
   meanwhile (re-import, CVAT sync, accepted pre-annotations) reaches FiftyOne. A dataset written by an
   earlier version has no fingerprints: its session images count as edited (nothing is lost).
3. **Read the changes** (the first half of *Review and apply*, a job): read the `ground_truth` field of those samples,
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
   **Pairing.** A label is matched with the project label it was written from by the id FiftyOne keeps for
   it (ActiveLabel records, in `activelabel_label_refs` on each sample, which project label each FiftyOne
   label shows), so a box or polygon moved far in the App, resized, or given another class is one **edited**
   label, not an added one plus a removed one. Only the labels without a known origin (drawn in the App, or
   brought by a project that was not sent with this version) are paired by overlap as before, and a label
   drawn where another one used to be stays an addition. After an apply the labels FiftyOne holds exactly as
   the project does keep their reference, so the next session starts from the same pairing.
   **Tags** come back too: the tags of each sample (except `excluded`, which ActiveLabel sets, and
   `exclude`, which asks for an exclusion through `POST /projects/{id}/fiftyone/pull-exclusions`) and the
   tags of each label. The summary counts the tags gained and lost (`tags: {added, removed}`, and `tags_added` / `tags_removed` per image in
   `examples`); an image whose only change is its tags counts as changed. A label tag is an edit of the
   label (the fingerprint of a `ground_truth` with tags covers them), so the image brings FiftyOne's
   version. The review shows the tags gained and lost as one more figure next to the counts, and a small
   `Tags +n −m` chip on each example image that changed its tags.
4. **Review and apply**: the session card has one main button, *Review and apply*, that reads the changes
   (step 3) and shows the summary (counts per class, images changed, problems) on the card; it can be clicked
   again to read the changes again. Applying stays a separate step, because it writes to the project: the
   person chooses *Apply changes* (the project ground truth for those images becomes the tool's version, label
   tags included; the sample tags become the image's tags in `<project>/tags.json`; new classes are added
   to the project) or *Discard these changes*. A safety copy of `annotations.json` is made
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
   Each task card in *Tasks and jobs* shows the newest edit session made for that task (or one of its
   jobs): the tool, when it started and how it stands (open, ready to review, applied, discarded);
   `GET /projects/{id}/tasks` gives it as `last_session: {tool, at, status}`.
5. **Close** a session (*Close session*): forget it.

The flow is the old one of the first CVAT-FiftyOne project (edit in the App, everything saved at once) with a
review layer on top: sessions, the summary of changes, conflicts, safety copies and *Discard* stay.

**Regression projects** (#619) have no `ground_truth` label and no edit session in FiftyOne. Each target is a
float field `ground_truth_<target>` (the target name in lower case, every run of other characters than letters
and digits replaced by `_`, `_2`, `_3`... when two names give the same field), recorded in the dataset info
(`activelabel_targets`). The field is view only: the App sorts, filters and charts the values and finds the
outliers (*Open in FiftyOne (view only)*), but there is no label schema, and `POST /projects/{id}/annotate` with
`tool` `fiftyone` answers 400 `fiftyone_regression_not_editable`. The values are edited in ActiveLabel (Images
screen) and loaded from CSV or JSON (docs/IMPORT.md). A target removed from the project loses its field; a
project that changes type loses the fields of the old one.

## Editing the labels of one image in ActiveLabel

`GET /api/projects/{id}/images/{image_id}/labels` gives `{labels, classes, revision, tags}`: the labels of the
image in the internal format (`label`, and `bbox`, `polygons`, `keypoints`, `track`, `score`, `model`, `tags`
when present), the project's classes, the image's own tags (see *Flow*, step 3) and the revision of those labels (16 hex digits that change whenever any field of any
label changes). `PUT` on the same path takes `{labels, revision, add_classes?}`, the whole new list of the
image (an empty list leaves the image listed with no labels: reviewed, nothing in it), and returns
`{labels, revision, classes}`. A label sent back with `track`, `score`, `model` and `tags` keeps them, so a tracked
object keeps its identity; a label sent without `model` counts as made by a person.

The save is refused, and nothing changes, when:

- the revision is not the current one (409 `labels_changed`, `params.revision` is the current one): another
  edit, a synchronisation, an import, a session apply or a pre-annotation changed the image meanwhile;
- a label does not pass the checks of an import (400 `labels_invalid`, with `count`, the first `index`
  counted from 1, its `problem` and up to 20 `problems: [{index, problem}]` counted from 0): `shape_invalid`
  (an empty class, a box that is not 4 numbers, a polygon point that is not 2, a keypoint that is not
  `[x, y, visibility]` with visibility 0, 1 or 2, a number that is not finite), `geometry_not_for_task` (a
  shape the project type does not use: any shape in classification, polygons or keypoints in detection,
  keypoints in segmentation, polygons in keypoints), `no_geometry`, `box_invalid`,
  `polygon_too_few_points`, `keypoints_count_mismatch` (another number of points than the other images
  of the class have), `outside_image` (a box, polygon point or labelled keypoint past the border of the image)
  and `multiple_classes` (more than one label in classification);
- a class is not in the project and `add_classes` is not true (409 `classes_new`, with `classes` and
  `count`); with `add_classes: true` they are added at the end of the class list. Class names are matched
  ignoring case and saved with the project's spelling.

It waits for the project's running activities like any change (409 `busy`). No safety copy is made for one
image. The edit is a local change for the other ways of editing:

- **CVAT**: every link that holds the image marks it as edited here (docs/SYNC.md, *Link*), so the next
  comparison sends it even when the change is smaller than the comparison's tolerance.
- **FiftyOne**: when the project has a FiftyOne dataset, its sample is rewritten with the new labels after
  the answer is sent, unless it holds edits made in the App that were not brought back yet. An open edit
  session then brings back the project's labels for that image (it was not edited in the App); an image
  edited on both sides is listed as `edited_on_both_sides` in the review. A session already brought back
  answers 409 `annotations_changed` on apply: bring it back again.

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
already saved the field says so and leaving it empty keeps it, also when only the address changes (the secret
moves to the keyring key of the new address). When *Save and test* saved the address but the test then failed, the
page still shows the new address. Once CVAT is set up, the *CVAT* tab shows
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
| POST | `/api/annotation-tools/{tool}/test` | | `{ok, message}`; for CVAT also `code` and `params`: `test_connected` with `url` and `user` when it worked, or the code of the failure (for example `cvat_login_failed`) when it did not |
| DELETE | `/api/annotation-tools/{tool}` | | `{removed: true}` (also removes the keyring secret) |
| GET | `/api/projects/{id}/annotate` | | `{sessions: [...]}` newest first |
| POST | `/api/projects/{id}/annotate` | `{tool: "fiftyone", scope: "all"\|"unlabeled"\|"selected"\|"issue"\|"preannotated"\|"task", ids?, issue?, task?, job?}` (`task`, with optional `job`, for scope `task`; 404 `task_not_found`, 400 `job_not_found`) | job; result `{session, links, kept_fiftyone_edits?}` (an open session with the same scope and images is reused, not duplicated); 400 `tool_unsupported` for any other tool; 400 `fiftyone_keypoints_not_editable` for a keypoints project (the App cannot edit keypoints) |
| POST | `/api/projects/{id}/annotate/{session}/pull` | | job; result `{session, diff}`; 409 when the session was already applied or discarded; 400 `tool_unsupported` for a session of a discontinued tool |
| POST | `/api/projects/{id}/annotate/{session}/apply` | | `{applied: {...counts}}`; 400 when nothing was brought back yet, 409 when the labels changed since the pull or it was already applied or discarded |
| POST | `/api/projects/{id}/annotate/{session}/discard` | | `{discarded: true}`; puts the session images in FiftyOne back to the project labels; 409 when already applied (apply and discard of one session take the same lock: exactly one wins) |
| DELETE | `/api/projects/{id}/annotate/{session}` | | `{closed: true}` (a session of a discontinued tool is only forgotten) |
| GET | `/api/projects/{id}/images/{image_id}/labels` | | `{labels, classes, revision, tags}`; 404 `image_not_found` |
| PUT | `/api/projects/{id}/images/{image_id}/labels` | `{labels: [{label, bbox?, polygons?, keypoints?, track?, score?, model?, tags?}], revision, add_classes?}` | `{labels, revision, classes}`; 409 `labels_changed`, 409 `classes_new`, 400 `labels_invalid`, 409 `busy` (see *Editing the labels of one image in ActiveLabel*) |

`diff` = `{images_changed, added, removed, edited, per_class: {name: {added, removed, edited}},
tags: {added, removed}, unknown_classes: [...], problems: [{level, code, message, count, examples}], examples:
[{image_id, name, added, removed, edited, tags_added, tags_removed}] (max 20)}`; `applied` has the same
`tags` counts.

The project writes its tags back into FiftyOne: the labels carry their tags whenever `ground_truth` is
written, and a sample created anew (a new image, or the dataset rebuilt) starts with the image's tags from
`tags.json`; an existing sample keeps the tags it has in FiftyOne. CVAT has no label tags: an image whose labels
come from CVAT in a synchronisation gets CVAT's labels, without tags.

Session (as returned): `{id, tool, scope, images, created, status, links: [{label, url}], task?, job?, diff?}`.
