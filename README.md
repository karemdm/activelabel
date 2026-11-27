# ActiveLabel

Image datasets from curation to trained models, step by step, in a simple web interface
(English and Portuguese, light and dark theme).

A project goes from images to a trained model in six steps, in two groups: **Data** (images,
annotations, curation) and **Model** (data split, training, models). A trained model can then
pre-annotate new images, which a person reviews in CVAT or FiftyOne and feeds back into the next
training.

```mermaid
flowchart TB
  subgraph G1["Data"]
    direction LR
    A["1 Images"] --> B["2 Annotations"] --> C["3 Curation"]
  end
  subgraph G2["Model"]
    direction LR
    D["4 Data split"] --> E["5 Training"] --> F["6 Models"]
  end
  subgraph G3["Improve the model"]
    direction LR
    N["New images"] --> P["Pre-annotate"] --> R["Review"]
  end
  G1 --> G2
  G2 -- "current model" --> G3
  G3 -- "reviewed images" --> G2
```

The current model pre-annotates new images, a person reviews them, and the reviewed images go into a new
split and a new training (the arrows between *Model* and *Improve the model*).

Guia em português: [docs/GUIA.pt-BR.md](docs/GUIA.pt-BR.md). Engineering overview with diagrams:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

**Names in the interface.** This page and the other documents quote the English interface. The
Portuguese interface has the same screens; its names are given once in parentheses where they are not
obvious (for example *Gallery* (*Galeria*)). *Task*, *Job* and *Fold* are the same word in both languages,
as in CVAT. The full list is in
[`src/activelabel/i18n/pt.json`](src/activelabel/i18n/pt.json).

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -e .              # the core
python -m pip install -e ".[fiftyone]"  # the core plus FiftyOne (".[all]" is the same; ".[dev]" includes it)
python -m pip install -e ".[dev]"       # to develop: FiftyOne plus the test and lint tools
```

The core is a light install: projects, images, curation measures and the similarity map, annotations,
tasks, videos, CVAT synchronisation, datasets, training and pre-annotation. The extra adds FiftyOne:

| Install | What you get |
| --- | --- |
| `pip install activelabel` | everything above; the training packages are not part of it, they go to a separate training environment installed on first use ([docs/TRAINING.md](docs/TRAINING.md)) |
| `pip install "activelabel[fiftyone]"` (or `[all]`) | also the FiftyOne App, the FiftyOne annotation tool, the import of CVAT projects (read with FiftyOne's CVAT reader) and the similarity map in the App |

Without FiftyOne, the features of the extra answer HTTP 501 `fiftyone_not_installed` with the install
command, and the rest works. *GET /api/features* tells what is installed and how much space the training
environment takes; *DELETE /api/training/environment* removes that environment (the next run installs what
it needs again). See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), *Light install*.

## Run

```bash
activelabel serve --open
```

Options of `serve`: `--host` (default `127.0.0.1`), `--port` (default `8090`) and `--open` (opens
the browser). Without activating the environment, use the launcher of the repository, which runs
the Python of the `.venv` folder next to it: `activelabel.cmd serve --open` (Windows) or
`./activelabel.sh serve --open` (Linux, macOS). The interface opens at `http://127.0.0.1:8090`.

The service has no login: keep the default host `127.0.0.1` unless you are on a trusted network.

## How it works

**New project** (not a step): name, description and **type** (classification, detection,
segmentation or keypoints). The project number is automatic. Classes are optional; they can
come from the annotation files, or be typed (Enter or a comma adds a class) or pasted as a list. A project that already exists in CVAT can
be imported instead ([docs/IMPORT.md](docs/IMPORT.md)). The project page then shows a compact
bar with the six steps; each step shows a marker (done, needs attention, in progress) and its status.
Inside a project the accent colour is the one of its type (for example teal for classification, violet for
detection), always with the type's icon and name; the home page stays blue.
(In ActiveLabel a *task* is a package of images, as in CVAT; the kind of
project is its *type*.)

### Data

1. **Images** has two tabs. *Gallery* (*Galeria*) is the grid of images, filtered by task and job.
   *Tasks and jobs* (*Tasks e jobs*) lists the tasks (one upload, or one video, is one task) split
   into jobs of a chosen size, as in CVAT; *New task* takes files, a folder, a .zip or a video.
   Files are stored unchanged (the kind of an image is decided by its content, not its extension),
   identical images are added once and unreadable files are listed with the reason. Each task and job has
   an open menu: *Edit in FiftyOne* (*Editar no FiftyOne*) and *Open in CVAT* (*Abrir no CVAT*), an entry
   that cannot be used yet being disabled with the reason (for example, project not linked to CVAT); FiftyOne does not edit keypoints, so a keypoints project offers *View in
   FiftyOne* and is edited in CVAT. See [docs/TASKS.md](docs/TASKS.md) and [docs/VIDEO.md](docs/VIDEO.md).
2. **Annotations** (optional) has five tabs. *Open in CVAT* (*Abrir no CVAT*), in the CVAT tab, opens the linked CVAT
   project (or the task or job chosen there), and *Edit in FiftyOne* (*Editar no FiftyOne*), in the FiftyOne tab, sends the images and opens the App to edit them;
   before the project is in CVAT the button says why and leads to the CVAT tab. The edits come back as before:
   through *Compare with CVAT* and through a FiftyOne edit session.
   - *Summary* (*Resumo*): what the project has, as tiles (images annotated, annotations, classes with
     annotations, to review) and the top classes, with a search and *Show all*.
   - *Pre-annotate* (*Pré-anotar*): a saved model, or a public pretrained one (COCO), suggests
     labels for the images of a task; see [docs/PREANNOTATE.md](docs/PREANNOTATE.md).
   - *CVAT*: link the project to CVAT and synchronise annotations in both directions with
     *Compare with CVAT* (*Comparar com o CVAT*), a review, and *Apply synchronisation*
     (*Aplicar a sincronização*); see [docs/ANNOTATION.md](docs/ANNOTATION.md) and
     [docs/SYNC.md](docs/SYNC.md).
   - *FiftyOne*: edit annotations in the FiftyOne App and bring them back after a review; the label
     schema the App needs is written by ActiveLabel, and a safety copy of the annotations is made before
     anything is replaced (the newest 20 are kept).
   - *Import annotations* (*Importar anotações*): drop annotation files (or a .zip). The format is
     detected (COCO JSON, YOLO txt with `data.yaml`/`classes.txt`, Pascal VOC XML, CVAT for images or
     video XML, MOT, class folders), the files are matched with the images and a
     report (which takes the place of the drop zone) shows what was found and any problem. Labels that do
     not have the shape of the project's type are reported and left out. Nothing is imported until you choose *Import*,
     *Change the type to {task} and import*, *Convert to {task} and import* (for example outlines to
     boxes) or *Discard*; when the project already has annotations you also choose *Merge* or
     *Replace*. Files made for a whole video go to a video task by frame number
     ([docs/VIDEO.md](docs/VIDEO.md)).

   Review in CVAT goes only through the synchronisation (docs/SYNC.md).
3. **Curation**: *Analyze images* measures brightness, contrast, sharpness, size and a
   perceptual hash of each image. Filters per issue (blurry, dark, bright, low contrast, small, odd
   aspect ratio, grayscale, near duplicate) show the flagged images; *Select suggested* picks the
   ones that are probably worth excluding; select and *Exclude selected* (reversible, *Excluded*
   tab, *Restore selected*) or *Delete permanently* (asks first). The limits of each issue can be
   changed under *Thresholds* without measuring again, and *Reset analysis* forgets the
   measures, flags and near-duplicate groups (images, exclusions and annotations stay).
   *Open in FiftyOne* opens the same filter in the FiftyOne App; images tagged `exclude` there can
   be brought back with *Bring exclusions from FiftyOne*. (Corrupt files never get here: they are
   rejected when uploaded. *Analyze images* also checks that each stored file is still the imported one:
   a file deleted or replaced outside ActiveLabel is flagged *File missing* or *File changed*, and a
   training refuses such images with `dataset_images_missing` / `dataset_images_changed`.) The *Similarity map* groups similar images: *Compute map* reads every
   active image with a small ImageNet network in the training environment (installed on first use,
   queued with the trainings), places them on a 2-D map (t-SNE, or PCA for very small or large sets)
   and splits them into groups (k-means), each shown with its size and most typical image. The same
   map opens in the FiftyOne App (Embeddings panel, run `activelabel_similarity`, coloured by the
   field `similarity_group`).

### Model

4. **Data split** freezes a *dataset version*: train, validation and test parts (or K folds
   plus a test part) that keep frames of one video and near duplicates together, balanced by
   class, with rare classes placed in every part first. Videos stay whole when that works and are cut
   into distant segments with a safety gap (a switch, on by default) when it does not. The preview shows
   the classes by rarity and warns about missing classes (saying whether it could be avoided), empty
   parts and labels left out; a version whose training or validation part would be empty is refused; a
   version whose labels changed afterwards says so. Each version card has *Train with this version*;
   see [docs/DATASETS.md](docs/DATASETS.md).
5. **Training** runs a model of the catalogue on one dataset version (transfer learning,
   fine-tuning or from scratch; single run, grid search or all folds) with live curves (loss, metrics and
   the learning rate) and evaluation on the test part. The dialog chooses the family, then the architecture
   (or YOLO generation) and the size; the task comes from the project type. The families:
   classification, detection, semantic segmentation, instance segmentation (Mask R-CNN) and keypoints
   (torchvision) are available; the YOLO models of Ultralytics (AGPL-3.0) are offered once the licence is
   read and accepted in the same dialog (docs/TRAINING.md, *Licences*). A run chooses its device
   (automatic, CPU, a GPU or Apple MPS), the resize policy (stretch, letterbox or centre crop), the learning
   rate schedule with warmup, mixed precision on a CUDA GPU and the data augmentation techniques; when the
   GPU runs out of memory the batch is halved and the run says so. A run that failed or was cancelled can
   be resumed from its last checkpoint, and warnings and failures of a run are shown by code in the
   interface language. Packages are installed on demand, at pinned and tested versions, in a separate
   environment. See [docs/TRAINING.md](docs/TRAINING.md).
6. **Models** keeps the runs you save (*Save as model*): compare the ones you tick (finished runs can be
   compared the same way in Training), choose the current one
   (*Make current*), download the weights and export ONNX (classification and semantic
   segmentation). *Add a public model* adds a pretrained COCO model that needs no training (detection
   and keypoints projects). The
   current model pre-annotates new images (*Annotations*, *Pre-annotate*, or *Pre-annotate with this
   one* on its card) for classification, detection, keypoints and both kinds of segmentation (masks
   become polygons).

### Activities

Background jobs (analysis, uploads, synchronisation, training, export, pre-annotation) appear as a
small pill in the bottom-right corner of the project page; it opens a card with progress, *Stop*
(*Parar*) and *See result*. A success fades; a failure stays red until it is opened or dismissed;
finished activities stay in the history. Heavy work (training, pre-annotation, export, curation
analysis, video extraction) runs one job at a time per device (CPU, each GPU); the others wait in a
global queue that survives a restart. *Queue* (*Fila*) in the top bar, on every page (shortcut Alt+Q),
shows the heavy work of every project that runs or waits, with its place in the queue, progress and
*Cancel*.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (architecture, data model and flows),
[docs/DESIGN.md](docs/DESIGN.md) (data on disk and the HTTP API),
[docs/ASSESSMENT.md](docs/ASSESSMENT.md) (strengths, risks and a roadmap),
[CHANGELOG.md](CHANGELOG.md) (notable changes, including removed features),
[RELEASING.md](RELEASING.md) (how a release is published),
[docs/FUTURE-LLM.md](docs/FUTURE-LLM.md) (planned use of language models) and
[docs/GUIA.pt-BR.md](docs/GUIA.pt-BR.md) (user guide in Portuguese).

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `ACTIVELABEL_WORKSPACE` | `~/.activelabel` | Where projects, images and annotations are stored |
| `ACTIVELABEL_LANG` | `en` | Default interface language (`en` or `pt`) |
| `ACTIVELABEL_IMPORT_ROOT` | (off) | Folder of this machine whose sub-folders may be imported by path (`POST /api/projects/{id}/images/folder`; there is no button for it in the interface) |
| `ACTIVELABEL_ALLOWED_HOSTS` | (none) | Other host names, separated by commas, the server answers at, e.g. the name of this machine on the local network when it serves with `--host` |
| `ACTIVELABEL_QUEUE_CPU` | `1` | Jobs that may run at the same time on the processor lane of the work queue (empty, invalid or below 1: 1) |
| `ACTIVELABEL_QUEUE_GPU` | `1` | Jobs that may run at the same time on each GPU lane (`cuda:N`, `mps`) of the work queue (empty, invalid or below 1: 1) |

The address and user of CVAT are kept in `<workspace>/tools.json`; passwords and tokens only in the
system keyring.

### Calling the API from a script

The server has no login, so it answers only calls made to its own address, to protect it from other sites
open in the same browser: the `Host` must be `127.0.0.1`, `localhost`, `[::1]` or the address it listens on
(`--host`), with its port, and a change (POST, PUT, PATCH, DELETE) coming from a browser page must come
from one of its own pages (`Origin`/`Referer`). Anything else is refused with HTTP 403
(`host_not_allowed`, `origin_not_allowed`). A script sends neither `Origin` nor `Referer`, so it works as
is when it uses the server's address:

```bash
curl -X POST http://127.0.0.1:8090/api/projects -H "Content-Type: application/json"   -d '{"name": "Cats", "task": "classification", "classes": ["cat"]}'
```

A script must not send an `Origin` or `Referer` of another site. To call the server by another name (for
example `my-pc.local` on the local network), list it in `ACTIVELABEL_ALLOWED_HOSTS`.

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) (set-up, tests, lint, types, structure, conventions,
how the training environment and workers work) and [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pre-commit install
python -m pytest -m "not integration"   # fast tests
python -m pytest -m integration         # tests that use FiftyOne
```

Continuous integration (`.github/workflows/ci.yml`) runs on every push to `main` and on pull requests:
ruff, mypy and the fast tests with coverage on Linux and Windows (and macOS once a change is merged),
from a path with accents and spaces, the FiftyOne tests, a check of the built wheel, the training workers with a CPU PyTorch, and the
interface tests (JavaScript unit tests and Playwright). A nightly workflow tests the CVAT synchronisation
against a real CVAT. The list of jobs and which ones must pass is in docs/DEVELOPMENT.md. The version comes
from the git tag; releases are described in [RELEASING.md](RELEASING.md).

## License

Apache License 2.0, see [LICENSE](LICENSE).
