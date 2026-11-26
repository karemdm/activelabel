# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the versions follow [Semantic Versioning](https://semver.org/). The project has no tagged release yet.

The version of the package is read from the last git tag (`vMAJOR.MINOR.PATCH`), so there is no version
number to edit in the code; a build between two releases carries the number of the last tag. How a release
is made is in [RELEASING.md](RELEASING.md): move the "Unreleased" entries under the new version heading,
tag, and publish the draft release that the `release` workflow creates. The detailed list of changes is in
the release notes, generated from the merged pull requests and grouped by label (`.github/release.yml`).

## Unreleased

Everything below goes into the first release, 0.1.0.

### Added

- **Projects, images and tasks.** Projects of type classification, detection, segmentation or keypoints;
  images uploaded as files, a folder, a .zip or a video; tasks and jobs inside a project as in CVAT, with
  long videos cut into time-distant segments when needed (docs/TASKS.md, docs/VIDEO.md).
- **YOLO training (Ultralytics).** YOLO26, YOLO11 and YOLOv8 (sizes n to x) detection, segmentation, pose and
  classification models train with their own worker once the AGPL-3.0 licence of Ultralytics is accepted; the acceptance is kept with its date in
  the workspace settings, and nothing is installed or trained before it (docs/TRAINING.md, *Licences*).
- **Import.** Annotation files in COCO, YOLO (also pose), Pascal VOC, CVAT and MOT, class folders as
  classification labels, and whole CVAT projects with their tasks and jobs, with a report of what was
  left out and why (docs/IMPORT.md).
- **Annotation and review.** Edit sessions in the FiftyOne App, reviewed before they are applied;
  synchronisation with CVAT by project, task or job in the chosen direction, sending only differences,
  following renamed classes and tasks both ways, reading skeletons as poses, plus a read-only audit of
  the server's tasks and a guard that refuses a CVAT task outside a project (docs/ANNOTATION.md,
  docs/SYNC.md).
- **Curation.** Brightness, contrast, sharpness, size and near-duplicate checks with adjustable limits,
  reversible exclusion and the same filters in FiftyOne.
- **Dataset versions.** Train/validation/test and K-fold splits that bring rare classes into every part,
  video-aware splits with a safety gap, and warnings for labels left out, empty parts and labels changed
  after a version (docs/DATASETS.md).
- **Training.** A separate training environment installed on demand (CPU, NVIDIA GPU or Apple silicon)
  with pinned, tested package versions; classification (all 80 torchvision networks and 102 timm networks, grouped by architecture), detection, instance, semantic
  and keypoint models; grid search and K-fold runs; data augmentation choices; the device (automatic, CPU,
  GPU or Apple MPS), the resize policy, learning rate schedules with warmup, mixed precision and a batch
  that is halved when the GPU runs out of memory; resume of a failed or cancelled run from its checkpoint;
  warnings and failures of a run shown by code in the interface language; metrics such as mAP, OKS mAP,
  mask mAP and PCK (docs/TRAINING.md).
- **Models and pre-annotation.** Saved models with PyTorch and ONNX weights, public pretrained models with
  a class mapping, and pre-annotation of new images for review, also with segmentation models
  (docs/PREANNOTATE.md).
- **Interface.** A step bar grouped in Dados and Modelo, Portuguese and English texts, API errors with
  stable codes that the interface translates, light and dark colours, a phone layout with 44 px touch
  targets, and an activity pill for running jobs (docs/DESIGN.md).
- **Local server guard.** The server answers only calls made to its own address and refuses changes that
  come from another site (403 `host_not_allowed`, `origin_not_allowed`); `ACTIVELABEL_ALLOWED_HOSTS` adds
  names; scripts that send no `Origin` keep working (README).
- **Quality and releases.** Tests that mirror the source tree with coverage of 80% or more; CI on Linux and
  Windows, and on macOS after a merge, from a path with accents and spaces; JavaScript and Playwright tests; a nightly run
  against a real CVAT; a wheel check in a clean environment; one version source (the git tag) and a
  release workflow that creates a draft release (RELEASING.md).
- **Format version of the stored data.** JSON files carry `format_version`; older files are read as they are,
  and a migration at start-up copies the files first, only adds fields and can run again safely
  (docs/DESIGN.md, *Format version*).

### Removed

- **Label Studio support.** Importing a project from Label Studio, its synchronisation, its edit sessions
  and reading its JSON export are gone. A workspace made before keeps its data: an old link, an
  imported-from record or a session of that tool is shown as belonging to a **discontinued tool**
  ("Linked to a discontinued tool", "Imported from a discontinued tool"); a link can only be **unlinked**
  and a session can only be **closed**, which changes nothing in that tool.
- **CVAT edit sessions.** Review in CVAT goes only through the synchronisation of the linked project
  (docs/SYNC.md); the *FiftyOne* tab keeps the edit sessions (docs/ANNOTATION.md). Old CVAT sessions can
  only be closed.
