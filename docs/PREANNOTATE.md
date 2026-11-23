# ActiveLabel — pre-annotation with a saved model

The loop that improves a model with new data: a saved model (step 6) suggests labels for new images,
a person reviews them in CVAT or FiftyOne, the reviewed images go into a new version of the data split
and a new training.

Everything happens in one place: **Anotações → Pré-anotar**, in the order of the work.

1. **Pré-anotar com um modelo**: model (the current one by default), task (the newest one with images
   without annotations by default), which images, minimum confidence; one button.
2. **Para revisar**: per task, how many images have pre-annotations that nobody reviewed yet, with
   *Revisar no CVAT* (opens the CVAT tab with that task chosen), *Revisar no FiftyOne* (opens the
   FiftyOne tab with "Pré-anotadas, não revisadas" chosen) and *Aceitar como estão*.
3. **Histórico**: every run with model, task, threshold and result.

## Rules

- A pre-annotation is a label with a `score` (the model's confidence) and the `model` that made it
  (`m1`…).
- It **never replaces a label made by a person**. *Sem anotação* fills only images without labels;
  *Não revisadas* redoes only images whose labels are all pre-annotations (for example with a better
  model). An image a person labelled while the run was going is left as it is.
- It **never trains a model by accident**: a dataset version ignores labels with a score, so an image
  with only pre-annotations counts as unlabelled until it is reviewed.
- **Reviewed** means: the labels came back from CVAT (sync) or from FiftyOne (edit session), or the
  person chose *Aceitar como estão* (labels kept, score and model removed). Sync compares labels by
  class and geometry only, so accepting does not send anything to CVAT by itself.
- FiftyOne shows the confidence of pre-annotations (`confidence` of each detection), to filter the
  doubtful ones first.

## Minimum confidence

Detections below it are left out; for classification, an image whose best class is below it stays
without a label. The default is 0.3 for detection and keypoints (a reviewer corrects a few extra boxes
faster than drawing missing ones) and 0.5 for classification. Models trained on few images give low
scores: when a run finds nothing, the screen says so and suggests a lower threshold.

## How it runs

A run is queued with the training runs and exports (one heavy process at a time) and uses the training
environment (nothing new is installed). The network is rebuilt from the model's configuration, the
weights are loaded, and each image is processed as in the model's evaluation (classification: resize
and ImageNet normalisation; detection: the image as it is, resized inside the network). Labels are
written only when the run ends; a cancelled or failed run writes nothing. Logs:
`<project>/preannotations/p<N>.log`; runs: `<project>/preannotations.json`.

Supported now: classification, detection and keypoints (torchvision). Semantic and instance
segmentation need masks turned into polygons and come later; YOLO comes with the Ultralytics models.

## HTTP API

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/projects/{id}/preannotations` | models that can pre-annotate, tasks with counts (images, unlabeled, unreviewed), runs |
| POST | `/api/projects/{id}/preannotations` | start: `{model, task, scope: "unlabeled"\|"unreviewed", threshold}` (threshold 0.5 when omitted; the screen proposes 0.3 for detection) |
| POST | `/api/projects/{id}/preannotations/{n}/cancel` | cancel a queued or running run |
| POST | `/api/projects/{id}/preannotations/accept` | accept the pre-annotations of a task (`{task}`) or of every task |

Errors: 404 for an unknown model or run; 400 for a model whose kind is not supported yet or does not fit
the project type, a threshold outside (0, 1), an unknown scope or task, or no image to fill; 409 when the
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
