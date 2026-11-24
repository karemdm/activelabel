# ActiveLabel — annotation with external tools

People draw and correct annotations in a tool of their choice. ActiveLabel stays the source of truth: work
is **sent** to the tool and **brought back**; nothing changes in the project until the person reviews a
summary of the changes and applies it.

| Tool | What it is | Used for |
| --- | --- | --- |
| FiftyOne | The FiftyOne App, *Annotate* mode (draw/edit boxes, polygons, keypoints; change class; delete) | **edit sessions** (this page) |
| CVAT | CVAT (online or self-hosted) | **synchronisation** of the linked project (docs/SYNC.md); it has no edit sessions |

FiftyOne is the hub for inspection: every project has a FiftyOne dataset, named
`activelabel-<id>-<random>` and recorded in `<project>/fiftyone.json` (docs/DESIGN.md). In the interface
this is the step **2 · Annotations** (*Anotações*): the tab *CVAT* (synchronisation and review in CVAT) and
the tab *FiftyOne* (edit sessions, described here). Review in CVAT happens only through the
synchronisation.

**Discontinued tools.** Label Studio support was removed, and so were the CVAT edit sessions (CVAT is
reviewed through the synchronisation). A workspace made before that may still hold their sessions, links or
imported-from records: they are shown as belonging to a discontinued tool and can only be **closed**
(sessions) or **unlinked** (links); nothing is changed in the tool, and no request to it is made.

Same functions in the tool: add, edit, delete, change class — for classification (one class per image),
detection (bounding boxes), segmentation (polygons) and keypoints.

## Flow (FiftyOne)

1. **Choose which images** (*Which images*): all active images, only images without annotations, the
   images selected in the curation grid, one issue filter, or only the images whose labels are all
   pre-annotations (`preannotated`, see [PREANNOTATE.md](PREANNOTATE.md)).
2. **Send** (*Send to FiftyOne*, a job): the FiftyOne dataset is refreshed from the project (images +
   current ground truth) and the App opens on a saved view with those images, field `ground_truth`.
   A **session** is recorded in `<project>/annotate.json` (and `annotate/<session>/`): id, tool, image
   ids, created, status (`open` → `pulled` → `applied` / `discarded`), links. Images that already have
   edits in FiftyOne from an earlier session keep them.
3. **Bring back** (*Bring annotations back*, a job): read the `ground_truth` field of those samples,
   convert it to the internal format (`activelabel.labels`) and **diff** it against the project ground
   truth for the session images: per image, labels added / removed / edited (class change, geometry
   change). Also validate against the type and the class list (unknown classes are reported).
4. **Review and apply**: the person sees the summary (counts per class, images changed, problems) and
   chooses *Apply changes* (the project ground truth for those images becomes the tool's version; new
   classes are added to the project) or *Discard these changes*. A backup of `annotations.json` is kept
   before applying. If the project's labels of those images changed after they were brought back,
   applying is refused (409) and they must be brought back again.
5. **Close** a session (*Close session*): forget it.

## Credentials

Only CVAT needs them. Set in the interface: the *Settings* (*Ajustes*) button on the tool card, or *Set up
CVAT* (*Configurar o CVAT*) in the *CVAT* tab. URL and username are stored in
`<workspace>/tools.json`; the password / token **only** in the system keyring (service
`activelabel`, key `<tool>:<url>`). They are never written to a file, never returned by the API
and never logged. *Save and test connection* (*Salvar e testar conexão*) saves them and checks the login.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/annotation-tools` | | `[{tool, configured, url, username, needs}]` for `fiftyone` and `cvat` (no secrets) |
| PUT | `/api/annotation-tools/{tool}` | `{url, username?, secret?}` | the tool entry (secret kept in keyring when given) |
| POST | `/api/annotation-tools/{tool}/test` | | `{ok, message}` |
| DELETE | `/api/annotation-tools/{tool}` | | `{removed: true}` (also removes the keyring secret) |
| GET | `/api/projects/{id}/annotate` | | `{sessions: [...]}` newest first |
| POST | `/api/projects/{id}/annotate` | `{tool: "fiftyone", scope: "all"\|"unlabeled"\|"selected"\|"issue"\|"preannotated", ids?, issue?}` | job; result `{session, links}`; 400 `tool_unsupported` for any other tool |
| POST | `/api/projects/{id}/annotate/{session}/pull` | | job; result `{session, diff}`; 409 when the session was already applied or discarded; 400 `tool_unsupported` for a session of a discontinued tool |
| POST | `/api/projects/{id}/annotate/{session}/apply` | | `{applied: {...counts}}`; 400 when nothing was brought back yet, 409 when the labels changed since the pull or it was already applied |
| POST | `/api/projects/{id}/annotate/{session}/discard` | | `{discarded: true}`; 409 when already applied |
| DELETE | `/api/projects/{id}/annotate/{session}` | | `{closed: true}` (a session of a discontinued tool is only forgotten) |

`diff` = `{images_changed, added, removed, edited, per_class: {name: {added, removed, edited}},
unknown_classes: [...], problems: [{level, code, message, count, examples}], examples:
[{image_id, name, added, removed, edited}] (max 20)}`.

Session (as returned): `{id, tool, scope, images, created, status, links: [{label, url}], diff?}`.
