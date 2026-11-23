# ActiveLabel — training and models

Steps **5 · Treinamento** (training) and **6 · Modelos** (models) of a project. A training run uses one
dataset version (docs/DATASETS.md), so it is always known which images and labels it saw; its
configuration is saved as YAML so it can be reproduced, on this or any other machine.

## Principles

- **Works on any machine, nothing to configure.** The device is chosen when a run starts (`auto`:
  a CUDA GPU when there is one, else Apple MPS, else the CPU). No path, device or machine setting is stored in a
  configuration.
- **Install only what is needed, when it is needed.** Before a run, ActiveLabel checks that the
  packages of the chosen model are present in the training environment and installs the missing
  ones (a visible step of the run, with progress). PyTorch is installed with GPU support when an
  NVIDIA GPU is found (`nvidia-smi`), otherwise the CPU build. Ultralytics (licence AGPL-3.0) is meant to be
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
| Segmentation (instance) | torchvision / ultralytics (YOLO planned) | Mask R-CNN; YOLOv8/11-seg | torch, torchvision / ultralytics |
| Keypoints | torchvision / ultralytics (YOLO planned) | Keypoint R-CNN; YOLOv8/11-pose | torch, torchvision / ultralytics |

YOLO/Ultralytics models are listed in the catalogue, but the runner refuses to train them ("comes
later"); the screen is meant to hide them until they work.

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
| Hyperparameters | from the catalogue: epochs, batch size, learning rate, optimizer, weight decay, scheduler, image size, augmentation preset, early stopping patience, seed, and model-specific ones |
| Selection | metric that chooses the best epoch / best run, per type: classification accuracy (default) or macro F1; detection, instance segmentation and keypoints mAP50-95 (default) or mAP50; semantic segmentation mIoU (default) or pixel accuracy |

There is no label conversion step. The runner writes a `spec.json` in the run's temporary folder (absolute
image paths, class indexes and the labels of each part, plus the model and hyperparameters); the worker
process reads it and builds the targets in memory (class indexes, torchvision targets, masks drawn from
polygons for semantic segmentation). A version needs images in the training and validation parts that
have labels (docs/DATASETS.md).

During the run: phases (`environment`, `preparing`, `training`, `evaluating`), epoch progress,
loss and validation metric per epoch (live curves), cancel. At the end the test part (if the
version has one) is evaluated with the best epoch.

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

## Storage

`<project>/runs/r<N>/` (numbers never reused):

| File | What |
| --- | --- |
| `config.yaml` | everything needed to repeat the run (dataset version and its fingerprint, model, start, mode, hyperparameters, seed) — no device or path |
| `run.json` | status, times, device used, package versions, ActiveLabel version, child runs (grid / folds) |
| `metrics.jsonl` | one line per epoch: losses and validation metrics |
| `results.json` | best epoch, validation and test metrics (confusion matrix and per-class metrics when the type has them) |
| `best.pt`, `last.pt` | weights (of the best child for grid / all folds; `last.pt` removed when the run is saved as a model) |
| `log.txt` | output of the training process (every child, each after a `== c<k>: … ==` line) |
| `c<k>/` | a child of a grid / all-folds run: `config.yaml` (single mode, one fold), `metrics.jsonl`, `results.json` |

For a grid / all-folds run, `run.json` lists `children` (`id`, `combo`, `fold`, `values`, `status`,
`best`, `error`), `trainings`, `current` (the child training now) and `best_child`; `results.json`
adds `ranking` (`rank`, `combo`, `values`, `children`, `scores`, `mean`, `std`, `complete`),
`best_combo`, `best_child`, `best_value`, `folds` and `metric`.

*Reproduce* starts a new run with the same `config.yaml`.

**Cancel** kills the worker process: the run gets the status `cancelled` and keeps its partial curves
(`metrics.jsonl`) and its log, but not usable results, and it cannot be saved as a model (only a
finished run can). Deleting a run that is running answers **409** ("cancel it before deleting"). A run
that was running when ActiveLabel stopped ends as failed (`interrupted`); queued ones go back in line.

## Models (step 6)

A finished run is **saved as a model** (button *Salvar como modelo* on the run): `<project>/models/m<N>/`
holds its best weights (`best.pt`), `config.yaml`, `results.json` and a card, `model.json` (run and
best child, model, dataset version and fingerprint, start, selection metric and value, the classes in
the order of the network's outputs, device and package versions, size, note, exports). Numbers are
never reused, a run is saved at most once, and saving removes the run's `last.pt`. The save dialog asks which
weights to keep: the PyTorch weights always, and also ONNX (classification and semantic
segmentation), which is then queued right after saving.

- **Public models**: *Adicionar modelo público* adds a pretrained torchvision model (COCO) without
  local training, with a mapping from its categories to the project's classes; only to pre-annotate.
  See docs/PREANNOTATE.md (Public models).
- **Current model**: one model of the project is marked current (`models/current.json`); the first
  saved model becomes current. Later steps use it to pre-annotate new tasks (see docs/PREANNOTATE.md), which are reviewed (in
  CVAT), go into a new dataset version and a new training: the loop that improves the model.
- **Comparison**: the screen lists the models and compares them in a table (every validation and
  test metric, the main per-class metric, size), highlighting the best value of each row.
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

## HTTP API (summary)

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/training/catalog?type=detection` | models and hyperparameter specs |
| GET | `/api/training/environment` | installed packages, device found |
| GET/POST | `/api/projects/{id}/runs` | list / create (queued job) |
| GET | `/api/projects/{id}/runs/{n}` | config, status, metrics per epoch (of the child training now, while a grid runs), results |
| GET | `/api/projects/{id}/runs/{n}/children/{c}` | one child of a grid / all-folds run |
| GET | `/api/projects/{id}/runs/{n}/log?lines=200` | last lines of the log |
| POST | `/api/projects/{id}/runs/{n}/cancel`, `/reproduce` | |
| DELETE | `/api/projects/{id}/runs/{n}` | after confirmation |
| GET/POST | `/api/projects/{id}/models` | list (with results and the current model) / save a run (`{run, note}`) |
| GET | `/api/projects/{id}/models/public-catalog` | public pretrained models for the project type, with a suggested mapping |
| POST | `/api/projects/{id}/models/public` | add a public model (`{public, mapping, note}`) |
| GET/PATCH/DELETE | `/api/projects/{id}/models/{n}` | card and results / note, `current` / delete (after confirmation) |
| POST | `/api/projects/{id}/models/{n}/export` | queue an export (`{"format": "onnx"}`) |
| GET | `/api/projects/{id}/models/{n}/download/{pytorch\|onnx}` | the weights or the export as a file |

## Status

Done: catalogue, training environment, runner, classification, detection, semantic segmentation and
keypoints (torchvision), grid search and all folds, the step 5 screen (Treinamento), and step 6
(Modelos) with save, current model, comparison, download and ONNX export.

Not yet: YOLO / Ultralytics models (training them is refused), instance segmentation in pre-annotation,
and regression.
