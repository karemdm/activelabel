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

- **Editing in FiftyOne is one click (#699).** The *FiftyOne* tab has one *Edit in FiftyOne* button that sends the
  images chosen (all, by default; the choice is folded under *Choose images*) and opens the App; *Open in
  FiftyOne (view only)* stays in *Curation* and in a keypoints project. Sending the same images again reuses the
  open session; the session card has one *Review and apply* that reads the changes and shows the difference (applying
  stays a separate step); the App shows only `ground_truth`, keeps the internal fields in a collapsed group, and
  reloads itself after the dataset is refreshed, so F5 is not needed (docs/ANNOTATION.md).
- **Light install.** `pip install activelabel` is the core (projects, images, curation, annotations, tasks,
  videos, CVAT synchronisation, datasets, training and pre-annotation); `pip install "activelabel[fiftyone]"`
  (or `[all]`) adds the FiftyOne App, its annotation tool, the import of CVAT projects and the similarity map
  in the App, which answer HTTP 501 `fiftyone_not_installed` without it. `GET /api/features` tells what is
  installed and `DELETE /api/training/environment` removes the training environment (docs/DEVELOPMENT.md,
  *Light install*).
- **YOLO26 sizes.** The YOLO26 models show their parameters and file size in the catalogue, like YOLO11 and YOLOv8.

- **Projects, images and tasks.** Projects of type classification, detection, segmentation or keypoints;
  images uploaded as files, a folder, a .zip or a video; tasks and jobs inside a project as in CVAT, with
  long videos cut into time-distant segments when needed (docs/TASKS.md, docs/VIDEO.md).
- **Regression projects** (#619). A number per image and target (for example age or a count): targets
  instead of classes, values imported from CSV or JSON, synchronised with CVAT (a tag per target with a
  `value` attribute) and edited in FiftyOne (a float field per target) (docs/IMPORT.md, docs/SYNC.md,
  docs/ANNOTATION.md), and split into versions balanced by bins of the values (docs/DATASETS.md). Training and
  pre-annotation of these projects come next.
- **YOLO training (Ultralytics).** YOLO26, YOLO11 and YOLOv8 (sizes n to x) detection, segmentation, pose and
  classification models train with their own worker once the AGPL-3.0 licence of Ultralytics is accepted; the acceptance is kept with its date in
  the workspace settings, and nothing is installed or trained before it (docs/TRAINING.md, *Licences*).
- **Import.** Annotation files in COCO, YOLO (also pose), Pascal VOC, CVAT and MOT, class folders as
  classification labels, and whole CVAT projects with their tasks and jobs, with a report of what was
  left out and why (shapes, rotated boxes, videos kept as images); a CVAT project already in ActiveLabel
  or without images cannot be imported again (docs/IMPORT.md).
- **Annotation and review.** Edit sessions in the FiftyOne App, reviewed before they are applied;
  synchronisation with CVAT by project, task or job in the chosen direction, sending only differences,
  following renamed classes, tasks and the project name both ways, the last edit session on each task
  card, reading skeletons as poses, plus a read-only audit of the server's tasks and a guard that refuses a CVAT task outside a project (docs/SYNC.md).
- **Curation.** Brightness, contrast, sharpness, size and near-duplicate checks with adjustable limits,
  reversible exclusion and the same filters in FiftyOne.
- **Dataset versions.** Train/validation/test and K-fold splits that bring rare classes into every part,
  video-aware splits with a safety gap, and warnings for labels left out, empty parts and labels changed
  after a version (docs/DATASETS.md).
- **Training.** A separate training environment installed on demand (CPU, NVIDIA GPU or Apple silicon)
  with pinned, tested package versions; classification (all 80 torchvision networks and 102 timm networks, grouped by architecture), detection, instance, semantic (torchvision and Segmentation Models, smp)
  and keypoint models, and Hugging Face transformers models (SegFormer, DETR, RT-DETR, Mask2Former); grid search and K-fold runs; data augmentation choices; the device (automatic, CPU,
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
- **Editing the labels of one image.** `GET` and `PUT /projects/{id}/images/{image_id}/labels` read and
  replace the labels of an image with a revision check (409 `labels_changed`), the checks of an import
  (400 `labels_invalid`) and new classes added only on request (409 `classes_new`); the edit is sent to CVAT
  at the next comparison and shown in FiftyOne (#588, docs/ANNOTATION.md).
- **A box moved far in FiftyOne is an edit.** Brought-back labels are paired with the project label they
  were written from by the id FiftyOne keeps (`activelabel_label_refs`), so a box or polygon moved far, resized
  or reclassified counts as *edited* in the review and on apply, not as *added* plus *removed* (#588). The
- **Tags in the review.** The summary of a brought-back session shows the tags gained and lost, and the
  images that changed them show a `Tags +n −m` chip (#588).
- **Tags from FiftyOne.** Sample tags and label tags made in the App come back with *Bring annotations
  back*, are counted in the review (`tags: {added, removed}`) and are kept in the project (`tags.json`,
  the labels' `tags`), and written back into FiftyOne (#588). The stored format is now version 2 (new
  optional keys only: nothing is rewritten and older code ignores them).

### Changed

- **Import name.** The name typed in the import form stays only in ActiveLabel: the first comparison no
  longer offers it as a rename of the CVAT project (the link records both names) (#695).
- **Import report.** The report card of an imported project uses the whole width and its summary cells line
  up (label above value, all to the left) (#695).
- **CVAT: project deleted there.** The CVAT tab says what to do there (unlink, or compare again to create the
  project) and no longer links to the deleted project (#695).
- **CVAT: task deleted there.** The review of a task deleted in CVAT names the task next to "Images to remove", and
  the task list stops offering "Open in CVAT" for it right after the comparison (the cached state is dropped).

- **Open in CVAT and in FiftyOne** moved from the project header into the CVAT and FiftyOne tabs of
  Annotations; the menu of each task keeps its own (#602).

### Fixed

- **The steps of the project page keep the same spacing whatever their text.** Each card (data, model) is a grid of three equal columns, so a long summary (for example the annotations count) no longer pushes a step to a second row; the summary wraps inside its own cell (two lines, then an ellipsis, with the full text in a tooltip) and both cards have the same height. Under 900 px the two cards stack, and on a phone the steps stack in one column, with no sideways scroll.
- **The FiftyOne App opens an image ready to edit.** Clicking an image in the App started by ActiveLabel
  opened it in *Explore*, where boxes are only shown (hover shows class and id) and nothing can be dragged;
  editing needed the *Annotate* tab, which the App remembers per dataset and browser, so each new project
  started in *Explore* again. ActiveLabel's App plugin `@activelabel/annotate` now switches the image window
  to *Annotate* for every editable dataset: a click selects a box or polygon, and dragging moves it, its
  corners or its points (docs/ANNOTATION.md, *How to edit in the App*).
- **The App's edit panel shows only the class and the tags.** `confidence`, `index`, `mask_path` and the free
  `attributes` are left out of the label schema of `ground_truth`; their values stay on the labels (#706).
- **How to edit is on screen.** The card of an open FiftyOne session says, in one line, how to select, move,
  resize, relabel, delete, undo and draw with the mouse (#706).
- **The Annotate-mode plugin fails quietly.** If a later FiftyOne App lacks the hook it uses, the plugin writes
  one console warning and the App works as usual (images open in *Explore*); ActiveLabel logs a warning when the
  installed FiftyOne is not the version the plugin was checked with (`fiftyone_plugin.VERIFIED_FIFTYONE`, 1.22.1),
  and a test fails until an upgrade is checked again (#706).

- **Class field and IME (#637).** The Enter and comma that confirm a character in an input method (Japanese,
  Chinese, Korean) no longer add a half-typed class.
- **The API refuses what the form could not type (#637).** Creating or changing a project with a class name that has
  a comma (`a,b`) or that is `background` / `__background__` answers HTTP 400 `class_name_has_comma` or
  `class_name_reserved`; a project that already has such a name keeps it.
- **A taken project name stays marked (#637).** In the new-project form the message stays under the name field,
  the field is marked as invalid and focused, until the name is edited; it was only a toast that faded.
- **Repeated class names are no longer dropped silently (#637).** Pasting or typing a name already in the list
  (in any case) leaves it out and a note under the field says which names were not added; the example text
  of the field hides once there is a chip.
- **Sending a project to FiftyOne.** The progress bar of *Open in FiftyOne* and *Edit in FiftyOne* now advances
  (images written to the dataset, then the App; the activity pill also names the stage), and the App really
  opens: the tab is opened at the click and taken to the App when the job ends, instead of a window opened
  after the wait, which browsers block (and the old "blocked" notice appeared even when the tab had opened).
  Sending images to an edit session opens the App too. If the browser gives no tab, the address stays on
  screen as a link; if the job fails, the reason is shown. The first port of the App can be set with
  `ACTIVELABEL_FIFTYONE_PORT` (default 5161).

### Removed

- **Label Studio support.** Importing a project from Label Studio, its synchronisation, its edit sessions
  and reading its JSON export are gone. A workspace made before keeps its data: an old link, an
  imported-from record or a session of that tool is shown as belonging to a **discontinued tool**
  ("Linked to a discontinued tool", "Imported from a discontinued tool"); a link can only be **unlinked**
  and a session can only be **closed**, which changes nothing in that tool.
- **CVAT edit sessions.** Review in CVAT goes only through the synchronisation of the linked project
  (docs/SYNC.md); the *FiftyOne* tab keeps the edit sessions (docs/ANNOTATION.md). Old CVAT sessions can
  only be closed.
