# ActiveLabel

Image datasets from curation to trained models, step by step, in a simple web interface
(English and Portuguese, light and dark theme).

**Stage 1 (this version):** create a project, choose its task, upload images, load
existing annotations (checked against the task) and curate the images: find corrupt,
blurry, dark, overexposed, low-contrast, small and near-duplicate images, and exclude them.
Later stages (annotation, training several models, evaluation, review of predictions as the
next ground truth) build on this.

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

1. **New project**: name, description and task (classification, detection, segmentation
   or keypoints). The project number is automatic. Classes are optional; they can come
   from the annotation files.
2. **Images**: drop files, a folder or a .zip. Files are stored unchanged; identical
   images are added once; unreadable files are listed with the reason. Annotation files
   sent together with the images are pointed to step 2. Each upload is added as a task
   (split into jobs of a chosen size, see docs/TASKS.md).
3. **Annotations** (optional): drop the annotation files (or a .zip). The format is
   detected (COCO JSON, YOLO txt with `data.yaml`/`classes.txt`, Pascal VOC XML, CVAT for
   images XML, Label Studio JSON), the files are matched with the project images and a
   report shows what was found and any problem. Nothing is imported until you choose:
   *Import*, *Change the project task and import* (when the annotations are of another
   task), *Convert and import* (for example outlines to boxes) or *Discard*.
4. **Curation**: *Analyze images* measures brightness, contrast, sharpness, size and a
   perceptual hash of each image (a task with progress that can be cancelled). Filters
   per issue show the flagged images; select and *Exclude* them (reversible, *Excluded*
   tab) or delete them for good (asks first). *Open in FiftyOne* opens the same filter in
   the FiftyOne App; images tagged `exclude` there can be brought back with *Bring
   exclusions from FiftyOne*.

Long tasks appear in the **Tasks** bar at the bottom of the project page, with progress,
*Cancel* and *See result*.

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
