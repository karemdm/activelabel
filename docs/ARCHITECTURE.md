# ActiveLabel — architecture

An engineering overview with diagrams. Details of each part are in the other documents of this folder;
the strengths, weaknesses and a roadmap are in [ASSESSMENT.md](ASSESSMENT.md). The diagrams are Mermaid
and render on GitHub.

Interface names are the English ones; the Portuguese names are in `src/activelabel/i18n/pt.json`.

## 1. Components

```mermaid
flowchart TB
  subgraph Browser["Browser (no framework)"]
    UI["Pages: templates/*.html<br/>app.js + app.css<br/>texts from i18n pt/en"]
  end

  subgraph Service["ActiveLabel service (FastAPI, one process)"]
    direction TB
    R["Routers (web/): api, tasks_api, video_api, datasets_api,<br/>annotate_api, sync_api, import_api,<br/>training_api, models_api, preannotate_api, pages"]
    S["AppState: settings, ProjectStore,<br/>JobManager (light jobs), TrainingRunner"]
    D["Domain: labels, annotations (importers), curation,<br/>datasets (split), tasks_store, images, sync, annotate,<br/>importing, preannotate, training (catalog, config, runs, models)"]
    JM["JobManager<br/>one worker thread:<br/>analysis, uploads, sync, FiftyOne, import"]
    TR["TrainingRunner<br/>its own queue, one worker thread:<br/>training, ONNX export, pre-annotation"]
    R --> S
    R --> D
    S --> JM
    S --> TR
  end

  subgraph Disk["Workspace on disk (JSON, YAML, images, weights)"]
    P["projects/ID/... (see section 2)"]
    T["tools.json (addresses and users)"]
    TE["training/env: Python environment<br/>training/tmp: temporary folder of a run"]
  end

  subgraph Workers["Training environment (separate Python)"]
    W["worker scripts (training/workers):<br/>classification, detection, segmentation,<br/>predict, export_onnx<br/>read spec.json, write JSON lines"]
  end

  K["System keyring<br/>(passwords, tokens)"]
  FO["FiftyOne App and database"]
  CV["CVAT (online or own server)"]
  LS["Label Studio<br/>(import only in the interface)"]

  UI -- "HTTP /api, /static, pages" --> R
  D --> P
  S --> T
  TR -- "creates, pip installs" --> TE
  TR -- "subprocess + spec.json" --> W
  W -- "JSON lines on stdout" --> TR
  W --> TE
  TR --> P
  D --> K
  D -- "fiftyone_bridge" --> FO
  D -- "sync/cvat_remote,<br/>annotate/cvat_tool" --> CV
  D --> LS
```

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

```mermaid
erDiagram
  PROJECT ||--o{ IMAGE : "images.json"
  PROJECT ||--o{ TASK : "tasks.json"
  PROJECT ||--o| ANNOTATIONS : "annotations.json"
  PROJECT ||--o{ DATASET_VERSION : "datasets/v N .json"
  PROJECT ||--o{ RUN : "runs/r N"
  PROJECT ||--o{ MODEL : "models/m N"
  PROJECT ||--o{ PREANNOTATION_RUN : "preannotations.json"
  PROJECT ||--o{ VIDEO : "videos.json"
  PROJECT ||--o{ LINK : "links.json (one per tool)"
  PROJECT ||--o{ SESSION : "annotate.json"
  TASK }o--o{ IMAGE : "ordered list of image ids"
  TASK }o--o| VIDEO : "source (video tasks)"
  IMAGE }o--o| VIDEO : "video, frame, time"
  ANNOTATIONS ||--o{ LABEL : "per image id"
  DATASET_VERSION ||--o{ LABEL : "frozen copy"
  DATASET_VERSION }o--o{ IMAGE : "parts: train, val, test, folds"
  RUN }o--|| DATASET_VERSION : "config.yaml: dataset"
  RUN ||--o{ CHILD_RUN : "grid or folds: c1, c2..."
  MODEL }o--|| RUN : "saved from (or public model)"
  PREANNOTATION_RUN }o--|| MODEL : "uses"
  PREANNOTATION_RUN }o--|| TASK : "labels the images of"
  LINK }o--o{ IMAGE : "items: remote key, labels at last sync"
  SESSION }o--o{ IMAGE : "image_ids"
```

- Image ids are the first 16 hex digits of the SHA-256 of the file, so identical files are one image.
- A label is `{label, bbox | polygons | keypoints, score?, track?, model?}` with coordinates relative to
  the image (0 to 1). A `score` marks a pre-annotation that nobody reviewed.
- Numbers (projects, tasks, dataset versions, runs, models) are never reused.
- The folder layout and the files are listed in [DESIGN.md](DESIGN.md).

## 3. User workflow

```mermaid
flowchart TB
  start(["New project or Import project"]) --> A1

  subgraph Data
    A1["1 Images<br/>Gallery, Tasks and jobs:<br/>files, folder, .zip, video"]
    A2["2 Annotations (optional)<br/>Summary, Pre-annotate, CVAT,<br/>FiftyOne, Import annotations"]
    A3["3 Curation<br/>Analyze images, filters,<br/>Exclude selected"]
    A1 --> A2 --> A3
  end

  subgraph Model
    B4["4 Data split<br/>New version: train, val, test<br/>or K folds"]
    B5["5 Training<br/>New run: single, grid, all folds"]
    B6["6 Models<br/>Save as model, Make current,<br/>download, ONNX, public model"]
    B4 --> B5 --> B6
  end

  A3 --> B4
  A2 -. "labels may also come<br/>from CVAT or FiftyOne" .-> B4

  B6 -- "current model" --> PRE["Pre-annotate<br/>(Annotations tab)"]
  NEW["New task with new images<br/>(step 1)"] --> PRE
  PRE --> REV{"Review"}
  REV -- "Review in CVAT: Compare with CVAT,<br/>Apply synchronisation" --> LAB["Reviewed labels<br/>in annotations.json"]
  REV -- "Review in FiftyOne: Send, Bring back,<br/>Apply changes" --> LAB
  REV -- "Accept as they are" --> LAB
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

```mermaid
flowchart LR
  Q{"What changed since the last sync?"}
  Q -- "nothing" --> N["nothing"]
  Q -- "only the remote" --> PU["pull: remote to project"]
  Q -- "only the project" --> PS["push: project to remote"]
  Q -- "both, same result" --> N
  Q -- "both, different" --> CF["conflict: the person chooses"]
```

With `direction: pull` or `push` one side wins and there are no conflicts; what exists only on the other
side is left alone and listed (docs/SYNC.md).

## 5. Training run lifecycle

```mermaid
stateDiagram-v2
  [*] --> queued: POST /runs (or Repeat)
  queued --> running: the slot is free
  queued --> cancelled: cancel while waiting
  running --> done: worker finished
  running --> failed: error, or interrupted by a restart
  running --> cancelled: cancel (process tree killed)
  failed --> [*]
  done --> [*]
  cancelled --> [*]
  queued --> queued: restart puts it back in line

  state running {
    [*] --> environment
    environment --> preparing: packages installed
    preparing --> training: spec.json written
    training --> evaluating: best epoch, test part
    evaluating --> [*]
  }
```

A grid search or an all-folds run is one run with **children** trained one after another in the same
slot:

```mermaid
flowchart TB
  R["run r N: running"] --> C1["c1: queued, running, done or failed"]
  C1 --> C2["c2"] --> C3["c3 ..."]
  C3 --> F{"At least one child done?"}
  F -- "yes" --> RK["rank combinations by the mean<br/>of the selection metric over folds;<br/>copy the best child's curves, results<br/>and weights to r N; other weights removed"]
  F -- "no" --> X["run failed:<br/>every training failed"]
  RK --> D["run done"]
```

If the run is cancelled, children not started are marked `cancelled`. A child that fails does not stop
the others. A run saved as a model (`models/m<N>`) hands its best weights over to the model (they move, they are not copied); ONNX export and pre-annotation
are queued in the same slot and run the same way (`queued`, `running`, `done`, `failed`).

## 6. Job lifecycle (light jobs)

```mermaid
stateDiagram-v2
  [*] --> queued: submit
  queued --> running: the worker takes it
  queued --> cancelled: cancelled before it starts
  running --> done: function returns (result)
  running --> failed: exception (error, traceback)
  running --> cancelled: Progress.check raises Cancelled
```

A job function receives a `Progress` handle, reports with `update(...)` and calls `check()` between
steps; cancelling sets an event that `check()` turns into `Cancelled` at the next safe point.
