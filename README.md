# ActiveLabel

Image datasets from curation to trained models, step by step, in a simple web interface
(English and Portuguese, light and dark theme).

A project goes from images to a trained model in six steps, in two groups: **Data** (images,
annotations, curation) and **Model** (data split, training, models). A trained model can then
pre-annotate new images, which a person reviews in CVAT or FiftyOne and feeds back into the next
training.

Guia em português: [docs/GUIA.pt-BR.md](docs/GUIA.pt-BR.md). Design notes: [docs/DESIGN.md](docs/DESIGN.md).

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run

```bash
activelabel serve --open
```

Or, without activating the environment: `activelabel.cmd serve --open` (Windows) or
`./activelabel.sh serve --open` (Linux, macOS). The interface opens at
`http://127.0.0.1:8090`. The service has no login: keep the default host `127.0.0.1`
unless you are on a trusted network.

## How it works

**New project** (not a step): name, description and task (classification, detection,
segmentation or keypoints). The project number is automatic. Classes are optional; they can
come from the annotation files. A project that already exists in CVAT or Label Studio can
be imported instead ([docs/IMPORT.md](docs/IMPORT.md)). The project page then shows a compact
bar with the six steps.

### Data

1. **Images** has two tabs. *Galeria* is the grid of images, filtered by task and job.
   *Tasks e jobs* lists the tasks (one upload, or one video, is one task) split into jobs of a
   chosen size, as in CVAT; *New task* takes files, a folder, a .zip or a video. Files are
   stored unchanged, identical images are added once and unreadable files are listed with the
   reason. See [docs/TASKS.md](docs/TASKS.md) and [docs/VIDEO.md](docs/VIDEO.md).
2. **Annotations** (optional) has five tabs:
   - *Resumo*: what the project has (annotated, to review, classes).
   - *Pré-anotar*: a saved model, or a public pretrained one (COCO), suggests labels for the
     images of a task; see
     [docs/PREANNOTATE.md](docs/PREANNOTATE.md).
   - *CVAT*: send images to CVAT and synchronise annotations in both directions, with a review
     before anything changes; see [docs/ANNOTATION.md](docs/ANNOTATION.md) and
     [docs/SYNC.md](docs/SYNC.md).
   - *FiftyOne*: edit annotations in the FiftyOne App and bring them back.
   - *Importar anotações*: drop annotation files (or a .zip). The format is detected (COCO JSON,
     YOLO txt with `data.yaml`/`classes.txt`, Pascal VOC XML, CVAT for images or video XML, MOT,
     Label Studio JSON, class folders), the files are matched with the images and a report shows what was found
     and any problem. Nothing is imported until you choose *Import*, *Change the project task
     and import*, *Convert and import* (for example outlines to boxes) or *Discard*. Files made for
     a whole video go to a video task by frame number ([docs/VIDEO.md](docs/VIDEO.md)).
3. **Curation**: *Analyze images* measures brightness, contrast, sharpness, size and a
   perceptual hash of each image. Filters per issue (corrupt, blurry, dark, overexposed, low
   contrast, small, near duplicate) show the flagged images; select and *Exclude* them
   (reversible, *Excluded* tab) or delete them for good (asks first). *Open in FiftyOne* opens
   the same filter in the FiftyOne App; images tagged `exclude` there can be brought back with
   *Bring exclusions from FiftyOne*.

### Model

4. **Data split** freezes a *dataset version*: train, validation and test parts (or K folds
   plus a test part) that keep frames of one video and near duplicates together, balanced by
   class; see [docs/DATASETS.md](docs/DATASETS.md).
5. **Training** runs a model of the catalogue on one dataset version (transfer learning,
   fine-tuning or from scratch; single run, grid search or all folds) with live curves and
   evaluation on the test part. Classification, detection, semantic segmentation and
   keypoints (torchvision) are available; YOLO/Ultralytics models are planned, not available
   yet. Packages are installed on demand in a separate environment. See
   [docs/TRAINING.md](docs/TRAINING.md).
6. **Models** keeps the runs you save as models: compare them, choose the current one,
   download the weights and export ONNX (classification and semantic segmentation). The
   current model pre-annotates new images (*Anotações*, *Pré-anotar*).

### Activities

Background jobs (analysis, uploads, synchronisation, training, export, pre-annotation) appear
in a small card in the bottom-right corner of the project page, with progress, *Cancel* and
*See result*. It can be minimised or closed; finished activities stay in its history. Only one
heavy process (training, export or pre-annotation) runs at a time; the others wait in a queue.

More: [docs/FUTURE-LLM.md](docs/FUTURE-LLM.md) (planned use of language models),
[docs/DESIGN.md](docs/DESIGN.md) (data on disk and the HTTP API) and
[docs/GUIA.pt-BR.md](docs/GUIA.pt-BR.md) (user guide in Portuguese).

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `ACTIVELABEL_WORKSPACE` | `~/.activelabel` | Where projects, images and annotations are stored |
| `ACTIVELABEL_LANG` | `en` | Default interface language (`en` or `pt`) |
| `ACTIVELABEL_IMPORT_ROOT` | (off) | Folder of this machine whose sub-folders may be imported by path |

## Development

```bash
pre-commit install
python -m pytest -m "not integration"   # fast tests
python -m pytest -m integration         # tests that use FiftyOne
```

Code lives in `src/activelabel`; tests in `tests/activelabel` mirror it. Formatting and lint
with ruff, types with mypy; pre-commit runs them and the fast tests on every commit.

Continuous integration (`.github/workflows/ci.yml`) runs the same checks on Linux and
Windows on every push to `main` and on pull requests, plus the FiftyOne tests on Linux.
macOS runs only on demand (Actions tab, *ci*, *Run workflow*).
