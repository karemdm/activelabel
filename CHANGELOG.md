# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the versions follow [Semantic Versioning](https://semver.org/).

The version of the package is read from the last git tag (`vMAJOR.MINOR.PATCH`), so there is no version
number to edit in the code; a build between two releases carries the number of the last tag. How a release
is made is in [RELEASING.md](RELEASING.md): move the "Unreleased" entries under the new version heading,
tag, and publish the draft release that the `release` workflow creates. The detailed list of changes is in
the release notes, generated from the merged pull requests and grouped by label (`.github/release.yml`).

## Unreleased

### Added

- **Distributions of the annotations (#623).** *Curation* shows *Distribution of the annotations* (objects and images
  per class, images by number of objects, box sizes by share of the image) from
  `GET /projects/{id}/curation/distributions`; the split preview draws the class balance of every part as bars; and
  the FiftyOne App gets the saved workspace *ActiveLabel: distribuições* (grid, Histograms and, with the similarity
  map, Embeddings), saved when the App is opened and never made the default.
- **Near duplicates from the similarity map (#608).** A new similarity map keeps the pairs of images whose
  embeddings are almost the same (cosine distance up to 0.1) and the curation adds them to its near duplicates:
  crops, brightness changes and neighbouring video frames the perceptual hash misses. The threshold is
  *embedding_distance* under *Thresholds* (default 0.02, 0 turns it off); the issues are flagged again when a map is
  ready. Identical files were already one image (images are stored by content), so #622 needs nothing more.
- **Edits go to the project from the App in one click (#706, #703).** *Enviar ao ActiveLabel* is also a button of the
  image window (no dialog). It applies the edits at once when nothing needs a look (no unreadable shape, conflict or
  unknown class) and says what changed; otherwise, or with *Revisar antes de aplicar* on (FiftyOne tab), the review
  opens as before. An applied session card has *Desfazer*. Fixing one box takes 5 clicks instead of 10.

- **Show similar images (#613).** With a similarity map, *Show similar* (*Mostrar parecidas*) in an image's window in
  Curation keeps that image and the 24 closest to it in the grid, nearest first, with the difference shown in the
  window; *Show all* clears it. `GET /projects/{id}/images` takes `similar_to` (and `similar_count`). The FiftyOne
  App gets the similarity index `activelabel_similar` (scikit-learn backend over the saved embeddings, no PyTorch), so
  its *sort by similarity* works.

- **Edit one image in FiftyOne from its window (#706).** Clicking an image in the *Images* gallery shows *Edit in
  FiftyOne*: it sends that image and the App opens straight on it, ready to edit (fewer clicks for the common case).

- **"Enviar ao ActiveLabel" inside the FiftyOne App (#703).** A FiftyOne plugin operator, loaded by the App that
  ActiveLabel starts (for that process only; nothing is written to `~/.fiftyone`), reads the edits of the open
  edit session back and opens the ActiveLabel review. It works only for ActiveLabel datasets, calls only the local
  server recorded in `dataset.info`, and says in the App when that server is not reachable (docs/ANNOTATION.md).
- **Evaluate a saved model in FiftyOne (#611).** *Evaluate* on a model card in *Models* predicts the test part of
  the dataset version the model was trained on (the validation part without a test one) in the global queue and
  lets FiftyOne compare it with the labels (`evaluate_detections` with mAP, or `evaluate_classifications`, under
  the evaluation key `eval_m<N>`, which the App's *Model Evaluation* panel lists). The card shows the main
  metrics, the metrics of each class, the confusion matrix, *See the worst images* (they filter *Images*) and
  *Open in FiftyOne*. API: `GET/POST /api/projects/{id}/models/{n}/evaluation` and `.../evaluation/cancel`
  (docs/TRAINING.md, *Models*). On its own K-fold version a fold model is evaluated only on its fold's validation
  part, and a model trained on every fold only on a test part; public models can be evaluated too. The evaluation
  has a worker of its own, so cancelling it never stops a label check. Deleting a model removes its evaluation
  (entry, FiftyOne fields, evaluation, run and view), and is refused while it is being evaluated. Evaluating
  polygons (with mAP: predicted polygons carry their score) needs `shapely`, now part of the `fiftyone` extra.

### Changed

- **Portuguese texts keep the technical terms in English (#222).** *frame*, *grid*, *batch*, *epoch*, *seed*,
  *fine-tuning*, *dataset* and *download* stay as they are in the English interface, so the same words appear in
  both languages. The Portuguese texts no longer say *quadro* or *época* for them. A test fails when a Portuguese
  text holds a common English word, or an English text a common Portuguese word.

## 0.1.0 - 2026-11-27

The first release.

### Added

- **Uniqueness and regrouping in the Curation step (#667).** *Compute uniqueness* (progress, Cancel, disabled with
  its reason until the similarity map exists) scores the images; the grid sorts by it (most unique or most repeated
  first), keeps only the images below a value to find near-repeated ones, shows the value on the card and in the
  image detail, and a notice says when the scores are outdated. *Regroup* on the map opens a dialog (*Automatic*,
  *Number of groups*, *Groups of any size* with the *Smallest group*) with progress and Cancel; the images HDBSCAN
  leaves without a group are drawn in a neutral colour and have their own *No group* filter. `GET
  /projects/{id}/images` gained `sort`, `unique_below` and `group`, and each scored image has `uniqueness`
  (README, docs/DESIGN.md, docs/GUIA.pt-BR.md).
- Regression projects: the Images screen shows the values of each image and lets you edit one by clicking it (#619).
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
  instead of classes, values imported from CSV or JSON and edited in ActiveLabel, shown in FiftyOne as a float
  field per target (view only), split into versions balanced by bins of the values (docs/IMPORT.md,
  docs/ANNOTATION.md, docs/DATASETS.md) and trained with the torchvision and timm networks and a regression
  head, the MSE, MAE or Huber loss and the RMSE, MAE and R² metrics, where the lowest error is the best
  (docs/TRAINING.md). Pre-annotation comes next.
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

- The segmentation colour is a raspberry rose, clearly apart from the violet of detection; a test keeps every pair of task colours 38 degrees or more apart in hue (#619).
- Regression projects no longer sync with CVAT (it keeps no number per image); in FiftyOne they are view-only, and values are edited in ActiveLabel (#619).
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

- **Regrouping refusals in the person's language (#667).** A refused regrouping has its own code
  (`cluster_method_unknown`, `cluster_groups_range`, `cluster_min_size_small`) with a pt/en text, instead of
  `cluster_invalid` with an English reason; *Compute uniqueness* is a secondary button, so Curation keeps one
  primary action (#340).

- **The steps of the project page keep the same spacing whatever their text.** Each card (data, model) is a grid of three equal columns, so a long summary (for example the annotations count) no longer pushes a step to a second row; the summary wraps inside its own cell (two lines, then an ellipsis, with the full text in a tooltip) and both cards have the same height. Under 900 px the two cards stack, and on a phone the steps stack in one column, with no sideways scroll.
- **The FiftyOne App opens an image ready to edit.** Clicking an image in the App started by ActiveLabel
  opened it in *Explore*, where boxes are only shown (hover shows class and id) and nothing can be dragged;
  editing needed the *Annotate* tab, which the App remembers per dataset and browser, so each new project
  started in *Explore* again. ActiveLabel's App plugin `@activelabel/annotate` now switches the image window
  to *Annotate* for every editable dataset: a click selects a box or polygon, and dragging moves it, its
  corners or its points (docs/ANNOTATION.md, *How to edit in the App*).
- **The App's edit panel shows only the class and the tags.** `confidence`, `index`, `mask_path` and the free
  `attributes` are left out of the label schema of `ground_truth`; their values stay on the labels (#706).
- **Delete several labels at once in the App, and undo it.** A trash button in the image window (*Apagar
  rótulos*) lists the boxes and polygons of the image with tick boxes and deletes the ticked ones; the button
  next to it (*Desfazer a remoção*) puts them back. Outside the image window it works on the labels selected in
  the App (#706).
- **After a deletion the App selects again.** The App kept its drawing tool on after a label was deleted, so the
  next click on the image drew a new box; the ActiveLabel plugin now picks the select tool again after *Delete*,
  *Backspace* or *Delete label* (#706).
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
- **The report of an annotation check (#651).** The report kept for a reload now lives beside the staging folder
  (`staging/<token>.report.json`), so a file sent with that name, in any folder, is no longer lost or taken for
  the report.
- **A pending report tells whether the project has annotations now (#651).** `GET /annotations/pending` reads
  `has_annotations` again instead of returning the value of the moment of the check.
- **A README sent with the images is not taken for annotations (#651).** In the images step, a `.txt` counts as
  an annotation file only when it has YOLO lines, and README, LICENSE and similar files never do, so the
  "looks like annotations" notice no longer appears for them.
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
