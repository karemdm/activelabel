# ActiveLabel — pre-annotation with a model

The loop that improves a model with new data: a saved model (step 6) suggests labels for new images,
a person reviews them in CVAT or FiftyOne, the reviewed images go into a new version of the data split
and a new training.

Everything happens in one place: **Annotations → Pre-annotate** (*Anotações → Pré-anotar*), in the order
of the work. The model must
be one of the project's models (step 6, docs/TRAINING.md): a training saved as a model, or a
[public pretrained model](#public-models) added without any local training.

1. **Pre-annotate with a model** (*Pré-anotar com um modelo*): model (the current one by default),
   task (the newest one with images without annotations by default), which images (*Without annotations*
   or *Pre-annotated, not reviewed*), minimum confidence (the screen accepts 0.05 to 0.95); one button.
2. **To review** (*Para revisar*): per task, how many images have pre-annotations that nobody reviewed
   yet, with *Review in CVAT* (opens the CVAT tab with that task chosen), *Review in FiftyOne* (opens
   the FiftyOne tab with "Pre-annotated, not reviewed" chosen) and *Accept as they are*
   (*Aceitar como estão*).
3. **History** (*Histórico*): every run with model, task, threshold and result.

## Rules

- A pre-annotation is a label with a `score` (the model's confidence) and the `model` that made it
  (`m1`…).
- It **never replaces a label made by a person**. *Without annotations* fills only images without labels;
  *Pre-annotated, not reviewed* redoes only images whose labels are all pre-annotations (for example with a better
  model). An image a person labelled while the run was going is left as it is.
- It **never trains a model by accident**: a dataset version ignores labels with a score, so an image
  with only pre-annotations counts as unlabelled until it is reviewed.
- **Reviewed** means: the labels came back from CVAT (sync) or from FiftyOne (edit session), or the
  person chose *Accept as they are* (labels kept, score and model removed). Sync compares labels by
  class and geometry only, so accepting does not send anything to CVAT by itself.
- FiftyOne shows the confidence of pre-annotations (`confidence` of each detection), to filter the
  doubtful ones first.

## Minimum confidence

Detections below it are left out; for classification, an image whose best class is below it stays
without a label. The screen limits it to 0.05–0.95 and proposes 0.5 for classification and 0.3 for detection and
keypoints (a reviewer corrects a few extra boxes faster than drawing missing ones). Models trained on few images give low
scores: when a run finds nothing, the screen says so and suggests a lower threshold.

## How it runs

A run is queued with the training runs and exports (one heavy process at a time) and uses the training
environment; model packages that are missing are installed first (a visible step of the run). The network is rebuilt from the model's configuration, the
weights are loaded, and each image is processed as in the model's evaluation (classification: resize
and ImageNet normalisation; detection: the image as it is, resized inside the network). Labels are
written only when the run ends; a cancelled or failed run writes nothing. An image that cannot be read
is skipped and counted (`counts.unreadable`). A run interrupted by a restart ends as failed with the
error `interrupted`; it can be started again. Logs:
`<project>/preannotations/p<N>.log`; runs: `<project>/preannotations.json`.

Supported now: classification, detection and keypoints (torchvision). Semantic and instance
segmentation need masks turned into polygons and come later; YOLO comes with the Ultralytics models.

## Public models

A project can pre-annotate before it has any training: **Models → Add a public model** (*Modelos →
Adicionar modelo público*) adds a
model already trained by others as the next model (`m<N>`), like a saved one. It is meant for a first
round of labels to review, not to replace a model trained on the project's own images.

- **Catalogue** (`activelabel.training.public_models`): torchvision models trained on COCO 2017.
  Detection projects: Faster R-CNN MobileNetV3, RetinaNet ResNet-50 v2 and Faster R-CNN ResNet-50
  v2 (80 categories); keypoint projects: Keypoint R-CNN ResNet-50 (person, 17 body points). The
  dialog shows the COCO box mAP and the download size of each. Classification models (ImageNet,
  1000 classes) come later.
- **Mapping**: for each class of the project, the category of the model that stands for it
  (suggested when the names match, ignoring case, `_` and `-`). Several categories can go to one
  class (`truck` and `car` → `car`); categories not mapped are ignored.
- **Weights**: nothing is stored in the project (`model.json` only, with `source: "public"`, the
  catalogue id and the mapping). Each pre-annotation run downloads the weights from the torchvision
  servers into the run's temporary folder, removed at the end; it needs `torch` and `torchvision`
  in the training environment (installed first when missing) and an internet connection.
- A public model has no training results, no downloads and no ONNX export, and is left out of the
  comparison table. It can be the current model, renamed (note) and deleted like any other.
- **Licence**: torchvision code is BSD-3; the weights were trained on COCO 2017. Check the terms of
  the weights and of COCO before using the labels in a product.

API: `GET /api/projects/{id}/models/public-catalog` (models for the project type, with the suggested
mapping, and the project's classes); `POST /api/projects/{id}/models/public` with
`{public, mapping, note}` (404 for an unknown model; 400 when it does not fit the project type, the
mapping is empty, or names a category or class that does not exist).

## HTTP API

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/projects/{id}/preannotations` | models that can pre-annotate, tasks with counts (images, unlabeled, unreviewed), runs |
| POST | `/api/projects/{id}/preannotations` | start: `{model, task, scope: "unlabeled"\|"unreviewed", threshold}` (threshold 0.5 when omitted, accepted from 0.001 to 0.999; the screen offers 0.05 to 0.95 and proposes 0.3 for detection) |
| POST | `/api/projects/{id}/preannotations/{n}/cancel` | cancel a queued or running run |
| POST | `/api/projects/{id}/preannotations/accept` | accept the pre-annotations of a task (`{task}`) or of every task |

Errors: 404 for an unknown model or run; 400 for a model whose kind is not supported yet or does not fit
the project type, a threshold outside 0.001–0.999, an unknown scope or task, or no image to fill; 409 when the
same task and scope are already being pre-annotated.

Edit sessions accept the image scope `preannotated` (images whose labels are all pre-annotations).

## Counts

Images whose labels are all pre-annotations are counted apart, as *a revisar*, in the step bar, the task
progress and the annotation summary; "anotadas" counts reviewed images only. An image listed with no
labels was reviewed as empty (a background image) and is never pre-annotated again.

## Writes

Every change to `annotations.json` (applying a review or a sync, pre-annotation, accept, deleting images,
imports) holds the same lock, so one never loses another's change. A cancel that arrives while the labels
are being written does not stop that last step; the run ends as done.
