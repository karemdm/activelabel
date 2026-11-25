# ActiveLabel — training and models

Steps **5 · Training** (*Treinamento*) and **6 · Models** (*Modelos*) of a project. A training run uses one
dataset version (docs/DATASETS.md), so it is always known which images and labels it saw; its
configuration is saved as YAML so it can be reproduced, on this or any other machine.

## Principles

- **Works on any machine, nothing to configure.** The device is chosen when a run starts (`auto`:
  a CUDA GPU when there is one, else Apple MPS, else the CPU). No path, device or machine setting is stored in a
  configuration.
- **Install only what is needed, when it is needed.** Before a run, ActiveLabel checks that the
  packages of the chosen model are present in the training environment and installs the missing
  ones (a visible step of the run, with progress), at tested versions (see *Pinned package versions*).
  PyTorch is installed with GPU support when an NVIDIA GPU is found (`nvidia-smi`) whose driver has a
  PyTorch CUDA build of the pinned version, otherwise the CPU build. Ultralytics (licence AGPL-3.0) is meant to be
  installed only when a YOLO model is chosen; YOLO is not available yet (see *Status*).
- **Separate environment.** Training packages live in their own Python environment managed by
  ActiveLabel (`<workspace>/training/env`, or a short folder under `%LOCALAPPDATA%` on Windows when
  the path is long), so ActiveLabel itself stays light and free of version conflicts.
- **Downloads are temporary.** Pretrained weights, the `spec.json` of the run and
  any other download go to a temporary folder (`<workspace>/training/tmp`), removed when the run ends
  (finished, failed or cancelled). Folders left by a process that was killed are removed when
  ActiveLabel starts. Only the run's results stay.
- **One run at a time.** Runs are queued; a heavy process never runs in parallel with another. There
  is a single queue for the whole app, shared by training runs, ONNX exports and pre-annotation
  (docs/PREANNOTATE.md), for all projects.

## Model catalogue

| Type (project) | Family | Models | Packages |
| --- | --- | --- | --- |
| Classification | timm / torchvision | ResNet-18/50, EfficientNet-B0/B3, ConvNeXt-T, ViT-B/16, Swin-T, MobileNetV3 | torch, torchvision, timm |
| Detection | torchvision | Faster R-CNN (ResNet-50 FPN, MobileNetV3), RetinaNet, SSDlite, FCOS | torch, torchvision |
| Detection | ultralytics (**planned, not available**) | YOLOv8 / YOLO11 n s m l | ultralytics |
| Segmentation (semantic) | torchvision | DeepLabV3 (ResNet-50, MobileNetV3), FCN, LR-ASPP | torch, torchvision |
| Segmentation (instance) | torchvision / ultralytics (YOLO planned) | Mask R-CNN ResNet-50 FPN (trainable now); YOLOv8/11-seg (planned) | torch, torchvision / ultralytics |
| Keypoints | torchvision / ultralytics (YOLO planned) | Keypoint R-CNN; YOLOv8/11-pose | torch, torchvision / ultralytics |

YOLO/Ultralytics models are listed in the catalogue returned by the API, but the screen hides them
and the API refuses to create a run with one (400, "Training YOLO models is not available yet").

The catalogue is data (`training/catalog.py`): each model lists its family, packages, input size,
and its hyperparameters with type, default, limits, choices and a one-line help. The screen builds
its form from it, so adding a model does not change the interface.

Regression (a number per image) needs a numeric annotation type, which ActiveLabel does not have
yet; it is left for later.

## A run

Options chosen on the screen and saved in `config.yaml`:

| Group | Options |
| --- | --- |
| Data | dataset version; for K-fold versions, all folds or one fold |
| Model | model; start: **transfer learning** (pretrained, backbone frozen, only the head trains) / **fine-tuning** (pretrained, everything trains; backbone learning rate × factor) / **from scratch** |
| Mode | **single run**, **grid search** (lists of values for some hyperparameters: every combination is a run; at most 50) |
| Hyperparameters | from the catalogue: epochs, batch size, learning rate, optimizer, weight decay, scheduler, image size, augmentation level (see below), early stopping patience, seed, and model-specific ones |
| Selection | metric that chooses the best epoch / best run, per type: classification accuracy (default) or macro F1; detection, instance segmentation and keypoints mAP50-95 (default) or mAP50; semantic segmentation mIoU (default) or pixel accuracy |

**Data augmentation.** The `augmentation` level is a preset (`none`, `light`, `standard`, `strong`) or
`custom`. Each preset is a list of techniques for the model's kind of task (the same techniques the workers
applied before they could be chosen); `custom` takes the list from the `augmentations` option of the
configuration, `[{"name", "params"}]`, checked against the catalogue (`training/augmentations.py`,
`GET /api/training/augmentations?type=`). Techniques: classification — random resized crop, horizontal and
vertical flip, rotation, RandAugment or TrivialAugment, colour jitter, blur, greyscale, random erasing, mixup
and cutmix; detection, instance segmentation, keypoints and semantic segmentation — flips, rotation, zoom
(scale jitter), shift, colour jitter, blur, greyscale (and random resized crop for semantic segmentation).
Geometric techniques move boxes, masks and keypoints with the image; a horizontal flip of keypoints needs the
left/right `flip_pairs` (an empty list when no point has a mirror twin). The runner resolves the level into
`spec.json` (`augmentations`) and records it in `run.json`; the worker applies exactly that list, in catalogue
order, to the training images only, and writes it to the log. Configurations made before techniques could be
chosen have no list and keep their preset. In the *New run* dialog, the level fills a list of techniques with
their settings; ticking a technique or changing a setting makes the level *Custom*.

There is no label conversion step. The runner writes a `spec.json` in the run's temporary folder (absolute
image paths, class indexes and the labels of each part, plus the model and hyperparameters); the worker
process reads it and builds the targets in memory (class indexes, torchvision targets, masks drawn from
polygons for semantic segmentation). A version needs images in the training and validation parts that
have labels (docs/DATASETS.md).

Each kind of model trains on **one shape** of the labels: a box for detection, polygons (three or more
points) for instance segmentation, polygons for semantic segmentation (a box stands in for a polygon only
there), and points for keypoints, all with the same number of points. The geometry is the truth: the box of
an instance comes from its polygons (a stored box that disagrees is not used, so box and mask match), and a
pose box is the stored box grown to hold every visible point. A label of another class or without
that shape is left out of the spec; a version whose images have labels but none usable for the model is
refused when the run starts, with a text that asks for a new version (a new version leaves such labels out
and says so, docs/DATASETS.md). A run is also refused (400 `dataset_classes_removed`) when its version has
classes the project no longer has.

The *New run* dialog keeps the hyperparameters closed behind one line that says whether they are the
*default values* or how many differ (*+{n}*), and tells which packages the first run installs when the
training environment lacks them.

During the run: phases (`environment`, `preparing`, `training`, `evaluating`), epoch progress,
loss and validation metric per epoch (live curves), cancel. At the end the test part (if the
version has one) is evaluated with the best epoch.

Besides the selection metric, the results of a keypoint model carry the object-keypoint-similarity mAP
(`kp_map50`, `kp_map50_95`, as in COCO but with the same tolerance `1/K` for every one of the `K` points,
because COCO's per-point values exist only for the 17 body points) and PCK@0.1; those of an instance
segmentation model carry the mask mAP (`mask_map50`, `mask_map50_95`). The selection metric of both stays the
box mAP.

A grid search or a run over *all folds* of a K-fold version trains several **children**, one after
another in the same slot of the queue (never two at once) and the same temporary folder (pretrained
weights are downloaded once). Children are ordered by grid combination, then fold; at most 50
trainings per run (combinations × folds). A child that fails is recorded and the others go on; the
run fails only if every child fails. At the end the combinations are ranked by the mean of the
selection metric on validation over their folds (± standard deviation); the best fold of the best
combination is the run's result: its curves, results and weights are copied to the run's folder, and
the other children keep their curves and results but no weights. The ranking value (the mean over folds)
is the run's headline value; the weights of a model saved from such a run are those of the best fold,
not an average. The screen shows the ranking (or,
while running, every child with its status) and the curves of any child.

## Metrics

How every score is computed (`training/workers/`, `training/runner.py`). Validated against
scikit-learn (classification, segmentation) and pycocotools `COCOeval` (box, mask and keypoint AP) on random
cases, and against hand-computed examples (`tests/activelabel/training/test_metrics_validation.py`).

| Metric | Definition | Conventions |
|---|---|---|
| Accuracy (classification) | correct / images | all classes |
| Precision, recall, F1 per class | TP/(TP+FP), TP/(TP+FN), 2PR/(P+R) | 0 when the denominator is 0 (no error); `support` = true images of the class |
| Macro F1 | unweighted mean of the per-class F1 | over the classes present in the truth **or** the predictions (as scikit-learn); a class absent from both is left out |
| Confusion matrix | `matrix[true][predicted]` | image counts (classification), pixel counts (segmentation, index 0 = background) |
| IoU, Dice per class (segmentation) | TP/(TP+FP+FN), 2TP/(2TP+FP+FN) | from one confusion matrix summed over all images of the part (not a mean of per-image scores) |
| mIoU, mean Dice | mean over the object classes | the background is left out of the means (reported as `background_iou`); a class absent from truth and prediction is left out |
| Pixel accuracy | correct pixels / pixels | background included |
| AP, mAP50, mAP50-95 (box, mask) | COCO: detections ranked by score over the whole part; greedy match, per image, to the unmatched object of the class with the highest IoU ≥ t; precision made monotone; mean of the precision at 101 recall points (0, 0.01 … 1; 0 beyond the last recall reached) | mAP50 at IoU 0.50 (required among the thresholds), mAP50-95 the mean over 0.50:0.05:0.95; mean over the classes with ground truth in the part. Every detection the model returns is used (up to 100 per image): `score_thresh` only limits what a saved model shows or pre-annotates |
| Keypoint mAP (`kp_map50`, `kp_map50_95`) | AP as above with OKS instead of IoU: OKS = mean over the labelled points of exp(−d² / (2·s²·(2σ)²)) | s² = area of the object's box, σ = 1/K for every point; a pose without any labelled point is ignored (not counted, and a detection on it is neither a hit nor a miss), as COCO does |
| PCK@0.1 | labelled points within 0.1 × the longer side of the object's box / labelled points | predictions with score ≥ 0.5 are paired with objects of the same class at box IoU ≥ 0.5 (greedy by score); the points of an unpaired object count as wrong |
| K-fold mean ± std | mean of the per-fold validation values and their **population** standard deviation (ddof = 0) | not a pooled metric over all folds; folds have about the same size |

Segmentation metrics are measured on the images resized to the run's `image_size` (both sides), not at
their original resolution, so every image weighs the same number of pixels. No metric is rounded in the
results; the interface rounds only for display. A part without images gives 0 for every metric.

## Storage

`<project>/runs/r<N>/` (numbers never reused):

| File | What |
| --- | --- |
| `config.yaml` | everything needed to repeat the run (dataset version and its fingerprint, model, start, mode, hyperparameters, seed) — no device or path |
| `run.json` | status, times, device used, package versions, ActiveLabel version, child runs (grid / folds) |
| `metrics.jsonl` | one line per epoch: losses and validation metrics |
| `results.json` | best epoch, validation and test metrics (confusion matrix and per-class metrics when the type has them) |
| `best.pt`, `last.pt` | weights (of the best child for grid / all folds). When the run is saved as a model both leave the run: `best.pt` **moves** to the model and `last.pt` is removed |
| `log.txt` | output of the training process (every child, each after a `== c<k>: … ==` line) |
| `c<k>/` | a child of a grid / all-folds run: `config.yaml` (single mode, one fold), `metrics.jsonl`, `results.json` |

For a grid / all-folds run, `run.json` lists `children` (`id`, `combo`, `fold`, `values`, `status`,
`best`, `error`), `trainings`, `current` (the child training now) and `best_child`; `results.json`
adds `ranking` (`rank`, `combo`, `values`, `children`, `scores`, `mean`, `std`, `complete`),
`best_combo`, `best_child`, `best_value`, `folds` and `metric`.

*Repeat* (*Reproduzir*) starts a new run with the same `config.yaml` (409 when the file of the run is
missing or damaged). A run whose `config.yaml` cannot be read still appears in the list, with a
`config_error` instead of its configuration, and can be deleted. `config_error` is a stable code
(`config_missing` or `config_damaged`) with a plain `config_message`; the detail, which may name a path,
only goes to the server log.

**Cancel** kills the worker process: the run gets the status `cancelled` and keeps its partial curves
(`metrics.jsonl`) and its log, but not usable results, and it cannot be saved as a model (only a
finished run can). Deleting a run that is running answers **409** ("cancel it before deleting"). A run
that was running when ActiveLabel stopped ends as failed (`interrupted`); queued ones go back in line.

## Models (step 6)

A finished run is **saved as a model** (button *Save as model* (*Salvar como modelo*) on the run): `<project>/models/m<N>/`
holds its best weights (`best.pt`, **moved** from the run, so the weights exist once and a saved run
has no `best.pt` any more; the run keeps its curves, results and log), `config.yaml`, `results.json` and a card, `model.json` (run and
best child, model, dataset version and fingerprint, start, selection metric and value, the classes in
the order of the network's outputs, the number of points of a keypoint model, device and package
versions, size, note, exports). Numbers are never reused, a run is saved at most once, and saving removes
the run's `last.pt`. The save dialog asks which
weights to keep: the PyTorch weights always, and also ONNX (classification and semantic
segmentation), which is then queued right after saving.

- **Public models**: *Add a public model* (*Adicionar modelo público*) adds a pretrained torchvision
  model (COCO) without local training, with a mapping from its categories to the project's classes;
  only to pre-annotate. It has no weights stored here, so it has no download and no ONNX export (the
  API answers 404 and 400). See docs/PREANNOTATE.md (Public models).
- **Current model**: one model of the project is marked current (`models/current.json`, button *Make
  current*); the first
  saved model becomes current. Later steps use it to pre-annotate new tasks (see docs/PREANNOTATE.md), which are reviewed (in
  CVAT), go into a new dataset version and a new training: the loop that improves the model.
- **Model card** (step 6): short by design. It shows the number of classes (*{n} classes*, with a link to
  see them all) and the actions of the model; *Pre-annotate with this one* (*Pré-anotar com este*) opens
  the *Pre-annotate* tab of *Annotations* with that model chosen. The step does not need images in the
  project (a public model needs none), and an empty step offers its action: save a finished run or add a
  public model. In the dialog of a public model the mapping is summarised as *{n} of {total} project
  classes linked to a model category* (*Review* opens the choices).
- **Comparison**: the screen lists the models and compares them in a table (every validation and
  test metric, the main per-class metric, size), highlighting the best value of each row.
- A saved model whose architecture this version no longer offers is listed as *unavailable*
  (*indisponível*) and cannot pre-annotate (it can still be deleted).
- **Deleting** a model (after confirmation) is refused with 409 while an export of it or a
  pre-annotation that uses it is queued or running.
- **Downloads**: the PyTorch weights (`.pt`, a dict with `state_dict`, `classes`, `model`, `family`,
  `image_size` and, for classification and segmentation, `mean`/`std`).
- **ONNX export** (classification and semantic segmentation): queued with the training runs (one heavy
  process at a time); the `onnx` package is installed in the training environment on the first
  export; the network is rebuilt without downloading anything, the weights are loaded and the graph
  is checked (`onnx.checker`). Opset 17, dynamic batch size and the fixed image size of the training.
  Input `images` (N×3×S×S, normalised with the ImageNet mean and std), output `logits` (classes, or
  classes + background per pixel). Detection, instance and
  keypoint models (torchvision) are not exported yet: their traced graphs break in ONNX Runtime on
  images without candidate boxes; their PyTorch weights can be downloaded.

## Pinned package versions

A new release of PyTorch or timm must not change results or break a first install (#160), so the training
packages are installed at fixed, tested versions: `PINNED_VERSIONS` in
`src/activelabel/training/environment.py` (torch, torchvision, timm, onnx and ultralytics). A package that
is already in the training environment at another version is installed again at the pinned one, as the
same visible step of the run. A requirement written with its own specifier (`timm>=1.0`) is left as it is.
The versions actually used are recorded in each run's `run.json`.

`CUDA_TAGS` in the same module lists the PyTorch CUDA builds (`cu130`, `cu126`, ...) that have the pinned
torch and torchvision; a driver older than the oldest one gets the CPU build (update the NVIDIA driver to
train on the GPU).

To update the versions:

1. Pick a torch release and its matching torchvision (the pairs are listed in the torchvision README), and
   check that the pair exists for every supported Python on the CPU index, on each CUDA index you keep and
   on PyPI for macOS, for example
   `pip index versions torch --index-url https://download.pytorch.org/whl/cu126 --python-version 3.10`.
2. Edit `PINNED_VERSIONS` and, when the CUDA builds of that release changed, `CUDA_TAGS` (newest first).
   Pick timm, onnx and ultralytics versions released after that torch.
3. Open a pull request. The CI job *training workers* installs exactly the pinned versions (read from the
   module) and fails if a training test is skipped; also run one short training on a GPU machine.
4. Note the change in CHANGELOG.md: results of new runs may differ from older ones.

## HTTP API (summary)

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/training/catalog?type=detection` | models and hyperparameter specs |
| GET | `/api/training/environment` | installed packages, device found; with `?model=<id>` also `missing` (packages that model needs) and `install_needed`, which the *New run* dialog uses to say that the first run installs packages |
| GET/POST | `/api/projects/{id}/runs` | list / create (queued job) |
| GET | `/api/projects/{id}/runs/{n}` | config, status, metrics per epoch (of the child training now, while a grid runs), results |
| GET | `/api/projects/{id}/runs/{n}/children/{c}` | one child of a grid / all-folds run |
| GET | `/api/projects/{id}/runs/{n}/log?lines=200` | last lines of the log |
| POST | `/api/projects/{id}/runs/{n}/cancel`, `/reproduce` | cancel (a queued or running run) / new run with the same configuration |
| DELETE | `/api/projects/{id}/runs/{n}` | after confirmation |
| GET/POST | `/api/projects/{id}/models` | list (with results and the current model) / save a run (`{run, note}`) |
| GET | `/api/projects/{id}/models/public-catalog` | public pretrained models for the project type, with a suggested mapping |
| POST | `/api/projects/{id}/models/public` | add a public model (`{public, mapping, note}`) |
| GET/PATCH/DELETE | `/api/projects/{id}/models/{n}` | card and results / note, `current` / delete (after confirmation) |
| POST | `/api/projects/{id}/models/{n}/export` | queue an export (`{"format": "onnx"}`) |
| GET | `/api/projects/{id}/models/{n}/download/{pytorch\|onnx}` | the weights or the export as a file |

## Status

Done: catalogue, training environment, runner, classification, detection, semantic segmentation,
instance segmentation (Mask R-CNN) and keypoints (torchvision), grid search and all folds, the step 5 screen (Treinamento), and step 6
(Modelos) with save, current model, comparison, download and ONNX export.

Pre-annotation works with classification, detection, keypoint, instance segmentation and semantic
segmentation models (docs/PREANNOTATE.md).

Not yet: YOLO / Ultralytics models (training them is refused), ONNX for instance, detection and keypoint
models, and regression.

## Image orientation

The workers apply the EXIF orientation of each photo when they load it (training, evaluation and
pre-annotation), so a photo stored rotated is seen upright, as in the gallery and in the annotation tools.
