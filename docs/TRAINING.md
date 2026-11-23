# ActiveLabel — training and models

Steps **5 · Treinos** (train) and **6 · Modelos** (models) of a project. A training run uses one
dataset version (docs/DATASETS.md), so it is always known which images and labels it saw; its
configuration is saved as YAML so it can be reproduced, on this or any other machine.

## Principles

- **Works on any machine, nothing to configure.** The device is chosen when a run starts (`auto`:
  a CUDA GPU when there is one, else the CPU). No path, device or machine setting is stored in a
  configuration.
- **Install only what is needed, when it is needed.** Before a run, ActiveLabel checks that the
  packages of the chosen model are present in the training environment and installs the missing
  ones (a visible step of the run, with progress). PyTorch is installed with GPU support when an
  NVIDIA GPU is found (`nvidia-smi`), otherwise the CPU build. Ultralytics is installed only when a
  YOLO model is chosen (its licence is AGPL-3.0).
- **Separate environment.** Training packages live in their own Python environment managed by
  ActiveLabel (`<workspace>/training/env`, or a short folder under `%LOCALAPPDATA%` on Windows when
  the path is long), so ActiveLabel itself stays light and free of version conflicts.
- **Downloads are temporary.** Pretrained weights, the dataset converted to the model's format and
  any other download go to a temporary folder of the run, removed when the run ends (finished,
  failed or cancelled). Only the run's results stay.
- **One run at a time.** Runs are queued; a heavy process never runs in parallel with another.

## Model catalogue

| Type (project) | Family | Models | Packages |
| --- | --- | --- | --- |
| Classification | timm / torchvision | ResNet-18/50, EfficientNet-B0/B3, ConvNeXt-T, ViT-B/16, Swin-T, MobileNetV3 | torch, torchvision, timm |
| Detection | torchvision | Faster R-CNN (ResNet-50 FPN, MobileNetV3), RetinaNet, SSDlite, FCOS | torch, torchvision |
| Detection | ultralytics | YOLOv8 / YOLO11 n s m l | ultralytics |
| Segmentation (semantic) | torchvision | DeepLabV3 (ResNet-50, MobileNetV3), FCN, LR-ASPP | torch, torchvision |
| Segmentation (instance) | torchvision / ultralytics | Mask R-CNN; YOLOv8/11-seg | torch, torchvision / ultralytics |
| Keypoints | torchvision / ultralytics | Keypoint R-CNN; YOLOv8/11-pose | torch, torchvision / ultralytics |

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
| Selection | metric that chooses the best epoch / best run (default per type: accuracy or macro F1; mAP50-95; mIoU) |

Label conversion is automatic: from the dataset version to the format the model needs (class
folders or index lists for classification, torchvision targets, YOLO txt + data.yaml, masks drawn
from polygons for semantic segmentation), written in the run's temporary folder.

During the run: phases (preparing environment, converting, training, evaluating), epoch progress,
loss and validation metric per epoch (live curves), cancel. At the end the test part (if the
version has one) is evaluated with the best epoch.

A grid search or a run over *all folds* of a K-fold version trains several **children**, one after
another in the same slot of the queue (never two at once) and the same temporary folder (pretrained
weights are downloaded once). Children are ordered by grid combination, then fold; at most 50
trainings per run (combinations × folds). A child that fails is recorded and the others go on; the
run fails only if every child fails. At the end the combinations are ranked by the mean of the
selection metric on validation over their folds (± standard deviation); the best fold of the best
combination is the run's result: its curves, results and weights are copied to the run's folder, and
the other children keep their curves and results but no weights. The screen shows the ranking (or,
while running, every child with its status) and the curves of any child.

## Storage

`<project>/runs/r<N>/` (numbers never reused):

| File | What |
| --- | --- |
| `config.yaml` | everything needed to repeat the run (dataset version and its fingerprint, model, start, mode, hyperparameters, seed) — no device or path |
| `run.json` | status, times, device used, package versions, ActiveLabel version, child runs (grid / folds) |
| `metrics.jsonl` | one line per epoch: losses and validation metrics |
| `results.json` | best epoch, validation and test metrics (confusion matrix, per-class metrics, PR curve points when the type has them) |
| `best.pt`, `last.pt` | weights (of the best child for grid / all folds; `last.pt` removed when the run is saved as a model) |
| `log.txt` | output of the training process (every child, each after a `== c<k>: … ==` line) |
| `c<k>/` | a child of a grid / all-folds run: `config.yaml` (single mode, one fold), `metrics.jsonl`, `results.json` |

For a grid / all-folds run, `run.json` lists `children` (`id`, `combo`, `fold`, `values`, `status`,
`best`, `error`), `trainings`, `current` (the child training now) and `best_child`; `results.json`
adds `ranking` (`rank`, `combo`, `values`, `children`, `scores`, `mean`, `std`, `complete`),
`best_combo`, `best_child`, `best_value`, `folds` and `metric`.

*Reproduce* starts a new run with the same `config.yaml`.

## Models (step 6)

A finished run can be **saved as a model**: `<project>/models/m<N>/` with the best weights, the
config, the results and a model card (dataset version, metrics, date, note). Models are versioned
(m1, m2…), compared in a table (metric per class, size, speed), one can be marked **current**, and
they can be exported (PyTorch, ONNX). Later steps use the current model to pre-annotate new tasks,
which are reviewed (in CVAT), go into a new dataset version and a new training: the loop that
improves the model with new data.

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
| GET/POST/PATCH/DELETE | `/api/projects/{id}/models[/{n}]` | save a run as model, list, mark current, delete |

## Delivery (small pull requests)

1. Catalogue, run configuration and run store (no training yet).
2. Training environment: detection of the GPU, on-demand install, package check.
3. Runner (separate process) + classification end to end, temporary folder, metrics, cancel.
4. Detection (torchvision). 5. Segmentation and keypoints (torchvision). 6. YOLO (ultralytics).
7. Grid search and K-fold orchestration.
8. Screen: step 5 Treinos. 9. Screen: step 6 Modelos.
