# ActiveLabel — annotation with external tools

People draw and correct annotations in a consolidated tool of their choice. FiftyOne is the
hub: every project already has a FiftyOne dataset (named `activelabel-<id>-<random>` and recorded in `<project>/fiftyone.json`, docs/DESIGN.md), and FiftyOne
integrates natively with CVAT and Label Studio. ActiveLabel stays the source of truth: work
is **sent** to a tool and **brought back**; nothing changes in the project until the person
reviews a summary of the changes and applies it.

| Tool | What it is | Needs |
| --- | --- | --- |
| `fiftyone` | The FiftyOne App, *Annotate* mode (draw/edit boxes, polygons, keypoints; change class; delete) | nothing |
| `cvat` | CVAT (app.cvat.ai or self-hosted), through FiftyOne's CVAT backend | URL, username, password |
| `labelstudio` | Label Studio, through FiftyOne's Label Studio backend | URL, API token |

In the interface, this is the step **2 · Annotations** (*Anotações*): the tab *CVAT* (synchronisation,
docs/SYNC.md) and the tab *FiftyOne* (edit sessions, described here). The edit sessions of this page
are what the interface offers for FiftyOne; CVAT is handled by the synchronisation of the *CVAT* tab,
although the API below still accepts `cvat` (and `labelstudio`) as a tool. Label Studio is switched
off in the interface for now (only importing a project from it is offered, docs/IMPORT.md); the code
and the API stay.

Same functions for every tool: add, edit, delete, change class — for classification (one
class per image), detection (bounding boxes), segmentation (polygons) and keypoints.

## Flow

1. **Choose the tool** and **which images** (*Which images*):
   all active images, only images without annotations, the images selected in the curation
   grid, one issue filter, or only the images whose labels are all pre-annotations
   (`preannotated`, see [PREANNOTATE.md](PREANNOTATE.md)).
2. **Send** (*Send to FiftyOne*, a job): the FiftyOne dataset is refreshed from the project (images + current
   ground truth); then
   - `fiftyone`: the App opens on a saved view with those images, field `ground_truth`;
   - `cvat` / `labelstudio`: `dataset.annotate(anno_key, backend=..., label_field="ground_truth", ...)`
     creates the tasks/project in the tool with the current annotations pre-loaded; the result
     has the links to open them.
   A **session** is recorded in `<project>/annotate.json`: id, tool, anno_key, image ids,
   created, status (`open` → `pulled` → `applied` / `discarded`), links.
3. **Bring back** (*Bring annotations back*, a job): read the annotations from the tool
   (`fiftyone`: the `ground_truth` field of those samples; others: `dataset.load_annotations`
   into a temporary field), convert them to the internal format (`activelabel.labels`) and
   **diff** them against the project ground truth for the session images: per image, labels
   added / removed / edited (class change, geometry change). Also validate against the task
   and the class list (unknown classes are reported).
4. **Review and apply**: the person sees the summary (counts per class, images changed,
   problems) and chooses *Apply changes* (the project ground truth for those images becomes the
   tool's version; new classes are added to the project) or *Discard these changes*. A backup of
   `annotations.json` is kept before applying. If the project's labels of those images changed
   after they were brought back, applying is refused (409) and they must be brought back again.
5. **Close** a session (*Close session*): forget it; for CVAT/Label Studio optionally delete the
   remote tasks (*Also delete the tasks in {tool}*).

## Credentials

Set in the interface: the *Settings* (*Ajustes*) button on the tool card, or *Set up {tool}*
(*Configurar o CVAT*) in the *CVAT* tab. URL and username are stored in
`<workspace>/tools.json`; the password / token **only** in the system keyring (service
`activelabel`, key `<tool>:<url>`). They are never written to a file, never returned by the API
and never logged. *Save and test connection* (*Salvar e testar conexão*) saves them and checks the login.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/annotation-tools` | | `[{tool, configured, url, username, needs}]` (no secrets) |
| PUT | `/api/annotation-tools/{tool}` | `{url, username?, secret?}` | the tool entry (secret kept in keyring when given) |
| POST | `/api/annotation-tools/{tool}/test` | | `{ok, message}` |
| DELETE | `/api/annotation-tools/{tool}` | | `{removed: true}` (also removes the keyring secret) |
| GET | `/api/projects/{id}/annotate` | | `{sessions: [...]}` newest first |
| POST | `/api/projects/{id}/annotate` | `{tool, scope: "all"\|"unlabeled"\|"selected"\|"issue"\|"preannotated", ids?, issue?}` | job; result `{session, links}` |
| POST | `/api/projects/{id}/annotate/{session}/pull` | | job; result `{session, diff}`; 409 when the session was already applied or discarded |
| POST | `/api/projects/{id}/annotate/{session}/apply` | | `{applied: {...counts}}`; 400 when nothing was brought back yet, 409 when the labels changed since the pull or it was already applied |
| POST | `/api/projects/{id}/annotate/{session}/discard` | | `{discarded: true}`; 409 when already applied |
| DELETE | `/api/projects/{id}/annotate/{session}` | `{delete_remote: bool}` | `{closed: true}` |

`diff` = `{images_changed, added, removed, edited, per_class: {name: {added, removed, edited}},
unknown_classes: [...], problems: [{level, code, message, count, examples}], examples:
[{image_id, name, added, removed, edited}] (max 20)}`.

Session (as returned): `{id, tool, scope, images, created, status, links: [{label, url}], diff?}`.
