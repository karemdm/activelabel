# ActiveLabel — architecture

An engineering overview with diagrams. Details of each part are in the other documents of this folder;
the strengths, weaknesses and a roadmap are in [ASSESSMENT.md](ASSESSMENT.md). The diagrams are Mermaid
and render on GitHub.

Interface names are the English ones; the Portuguese names are in `src/activelabel/i18n/pt.json`.

## 1. Components

```mermaid
flowchart TB
  UI["Browser<br/>pages and JS modules"]
  API["FastAPI routers<br/>web/*_api.py"]
  DOM["Domain code<br/>labels, datasets, sync,<br/>curation, training"]
  JOB["Light jobs<br/>one worker thread"]
  RUN["Training runner<br/>own queue, one worker thread"]
  ENV["Training environment<br/>separate Python and workers"]
  DISK[("Workspace on disk<br/>JSON, images, weights")]
  EXT["CVAT, FiftyOne,<br/>system keyring"]
  UI --> API
  API --> DOM
  API --> JOB
  API --> RUN
  JOB --> DOM
  DOM --> DISK
  DOM --> EXT
  RUN --> ENV
  RUN --> DISK
  ENV --> DISK
```

Arrows mean "calls or uses". The browser talks to the routers over HTTP (`/api`, `/static`, pages); the
domain code reads and writes the workspace (section 2) and reaches CVAT and FiftyOne through
`sync/cvat_remote.py`, `annotate/cvat_tool.py` and `fiftyone_bridge.py`; the runner creates and fills the
training environment, starts the workers as subprocesses with a `spec.json` and reads their JSON lines.

Rules that shape the design:

- The routers hold no business logic: they validate, take the project lock/`busy` check, and call the
  domain code or submit a job.
- There is **one heavy process at a time** for the whole app (`TrainingRunner`), shared by training runs,
  ONNX exports and pre-annotation. The light `JobManager` has one worker too; `AppState.busy(project)`
  answers 409 to calls that change a project while one of its jobs is queued or running.
- Workers never import ActiveLabel: they read a `spec.json` and talk back with one JSON object per line
  (`log`, `warning`, `progress`, `epoch`, `done`, `error`; `prediction` for pre-annotation).
- Secrets live only in the system keyring; `tools.json` has addresses and user names.

## 2. Data model on disk

An arrow means "is used by" or "feeds": images come from files or from the frames of a video and are listed
by tasks; labels belong to images; a dataset version freezes images and labels; runs train on a version;
a model is saved from a run; pre-annotation runs use a model and write labels for review. A link (sync
with CVAT) and an edit session refer to images.

```mermaid
flowchart TB
  VID["Video<br/>videos.json"] --> IMG
  TSK["Task<br/>tasks.json"] --> IMG["Image<br/>images.json"]
  IMG --> LAB["Labels<br/>annotations.json"]
  IMG --> DS["Dataset version<br/>datasets/vN.json"]
  LAB --> DS
  DS --> RUN["Run<br/>runs/rN"]
  RUN --> MOD["Model<br/>models/mN"]
  MOD --> PRE["Pre-annotation run<br/>preannotations.json"]
  PRE --> LAB
  LNK["Link<br/>links.json"] --> IMG
  SES["Edit session<br/>annotate.json"] --> IMG
```

- Image ids are the first 16 hex digits of the SHA-256 of the file, so identical files are one image. The image keeps
  the name of its first upload; the other names it was uploaded with are kept as `aliases`, so annotation files
  that use them still match.
- A label is `{label, bbox | polygons | keypoints, score?, track?, model?}` with coordinates relative to
  the image (0 to 1). A `score` marks a pre-annotation that nobody reviewed.
- Numbers (projects, tasks, dataset versions, runs, models) are never reused.
- The folder layout and the files are listed in [DESIGN.md](DESIGN.md).

## 3. User workflow

```mermaid
flowchart TB
  S(["New or imported project"]) --> A1
  subgraph Data
    direction LR
    A1["1 Images"] --> A2["2 Annotations"] --> A3["3 Curation"]
  end
  subgraph Model
    direction LR
    B4["4 Data split"] --> B5["5 Training"] --> B6["6 Models"]
  end
  A3 --> B4
  B6 --> PRE["Pre-annotate"]
  NEW["New images"] --> PRE
  PRE --> REV{"Review"}
  REV --> CV["CVAT sync"]
  REV --> FO["FiftyOne session"]
  REV --> AC["Accept as they are"]
  CV --> LAB["Reviewed labels"]
  FO --> LAB
  AC --> LAB
  LAB --> B4
```

Pre-annotations (labels with a score) are ignored by a dataset version until they are reviewed, so
the loop never trains on unchecked labels. Without any training, a public model (*Add a public model*)
can start the loop.

## 4. Synchronisation with CVAT (three-way)

```mermaid
sequenceDiagram
  actor P as Person
  participant UI as Browser
  participant API as sync_api
  participant E as sync/engine
  participant L as links.json (last sync)
  participant G as annotations.json (project)
  participant C as CVAT

  P->>UI: Compare with CVAT (scope, direction)
  UI->>API: POST /sync/cvat
  API->>E: job sync_plan
  E->>C: read tasks, frames and labels
  E->>L: read labels and keys at the last sync
  E->>G: read project labels
  Note over E: per image compare base, local, remote<br/>pull, push, same, conflict, add, remove, restore
  E-->>API: plan saved in sync/plan_id.json
  API-->>UI: result: counts, diffs, conflicts
  P->>UI: review, choose a side for each conflict
  P->>UI: Apply synchronisation
  UI->>API: POST /sync/cvat/plan_id/apply
  API->>E: job sync_apply
  E->>C: read again what it needs
  alt remote, project or link changed since the plan
    E-->>API: failed: Synchronise again
  else unchanged
    E->>G: backup, then write pulled labels
    E->>C: write pushed labels, add or remove tasks and images
    E->>L: new base labels, keys, last_sync, revision
    E-->>API: counts
  end
```

The decision per image:

| Changed since the last sync | Action |
| --- | --- |
| nothing | nothing |
| only the remote | pull: remote to project |
| only the project | push: project to remote |
| both, same result | nothing |
| both, different | conflict: the person chooses |

With `direction: pull` or `push` one side wins and there are no conflicts; what exists only on the other
side is left alone and listed (docs/SYNC.md).

## 5. Training run lifecycle

```mermaid
stateDiagram-v2
  direction LR
  [*] --> queued
  queued --> running: start
  queued --> cancelled: cancel
  running --> done: finish
  running --> failed: error
  running --> cancelled: cancel
  done --> [*]
  failed --> [*]
  cancelled --> [*]
```

While a run is `running` it goes through these phases (`phase` in `run.json`):

```mermaid
flowchart LR
  E["environment"] --> P["preparing"] --> T["training"] --> V["evaluating"]
```

A run that was running when the service stopped ends as `failed` (`interrupted`) at the next start;
queued runs are queued again.

A grid search or an all-folds run is one run with **children** trained one after another in the same
slot:

```mermaid
flowchart LR
  C1["child c1"] --> C2["child c2"] --> C3["child c3 ..."] --> Q{"any done?"}
  Q -- yes --> OK["rank, keep best"]
  Q -- no --> NO["run failed"]
```

If the run is cancelled, children not started are marked `cancelled`. A child that fails does not stop
the others. A run saved as a model (`models/m<N>`) hands its best weights over to the model (they move, they are not copied); ONNX export and pre-annotation
are queued in the same slot and run the same way (`queued`, `running`, `done`, `failed`).

## 6. Job lifecycle (light jobs)

```mermaid
stateDiagram-v2
  direction LR
  [*] --> queued
  queued --> running: taken
  queued --> cancelled: cancel
  running --> done: returns
  running --> failed: exception
  running --> cancelled: cancel
```

A job function receives a `Progress` handle, reports with `update(...)` and calls `check()` between
steps; cancelling sets an event that `check()` turns into `Cancelled` at the next safe point.
