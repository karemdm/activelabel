# ActiveLabel — pre-annotation with a model

The loop that improves a model with new data: a saved model (step 6) suggests labels for new images,
a person reviews them in CVAT or FiftyOne, the reviewed images go into a new version of the data split
and a new training.

Everything happens in one place: **Annotations → Pre-annotate** (*Anotações → Pré-anotar*), in the order
of the work. The model must
be one of the project's models (step 6, docs/TRAINING.md): a training saved as a model, or a
[public pretrained model](#public-models) added without any local training.

1. **Pre-annotate with a model** (*Pré-anotar com um modelo*): model (the current one by default; only the
   models that can pre-annotate this project are listed, and a model saved with an architecture this version no
   longer offers is shown disabled as *unavailable*), task (the newest one with images without annotations by
   default), which images (*Without annotations* or *Not reviewed*, that is, pre-annotated and not reviewed;
   each with its count, and a choice with no image is disabled), minimum confidence (the screen accepts 0.05 to
   0.95); one button, which says how many images it will pre-annotate. A run that is queued or running is shown
   on the same screen with its progress and a stop button. When it ends with nothing written because the model
   was never sure enough, the screen says so and suggests a lower confidence.
   If the project has no model at all, the tab says *To pre-annotate, save a trained model or add a public model
   in Models* with a button to the Models step. If it has models but none can pre-annotate this project (they
   make another kind of label, or are YOLO models, which are not supported yet), the tab says that it needs a
   model that makes the same kind of label as the project.
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

Detections (and segments) below it are left out; for classification, an image whose best class is below it stays
without a label. The screen limits it to 0.05–0.95 and proposes 0.5 for classification and 0.3 for detection and
keypoints (a reviewer corrects a few extra boxes faster than drawing missing ones); a segmentation model proposes
0.5. Models trained on few images give low scores: when a run finds nothing, the screen says so and suggests a
lower threshold. A detection model is allowed to return candidates down to this threshold even when its own
`score_thresh` hyperparameter (docs/TRAINING.md) is higher; the metrics of a training run do not depend on
either.

## How it runs

A run is queued with the training runs and exports (one heavy process at a time) and uses the training
environment; model packages that are missing are installed first (a visible step of the run). The network is rebuilt from the model's configuration, the
weights are loaded, and each image is processed as in the model's evaluation: the **resize policy** saved with
the weights (`stretch`, `letterbox` or `center_crop`, docs/TRAINING.md) brings the photo to the model's input size
(ImageNet normalisation for classification and segmentation, values 0 to 1 for detection), and boxes, keypoints
and masks are taken back to the photo with the same policy, so they come back in the photo's own coordinates. A
model saved before the policy was recorded is read as it was trained: stretched for classification and
segmentation, the photo as it is (resized inside the network) for detection, like a public model. Labels are
written only when the run ends; a cancelled or failed run writes nothing.

**Device.** The run works on the device it was asked for: `auto` (a CUDA GPU when there is one, else Apple MPS,
else the CPU), `cpu`, `cuda:N` or `mps`. The API takes it as `device` (default `auto`; a device the computer does
not have is refused with 400 `device_unavailable`, a name that is not a device with `device_invalid`) and the run
keeps it; the *Pre-annotate* screen does not offer the choice and always uses `auto`. An image that cannot be read
is skipped and counted (`counts.unreadable`). A run interrupted by a restart ends as failed with the
error `interrupted`; it can be started again. Logs:
`<project>/preannotations/p<N>.log`; runs: `<project>/preannotations.json`.

Supported: classification, detection, keypoints, **instance segmentation and semantic segmentation**
(torchvision). A segmentation model predicts masks; the worker (which has no OpenCV) sends each mask as a
run-length code of at most a fixed size on its longer side, and ActiveLabel turns it into polygons: the
outer outline of every region (holes are not kept), regions smaller than 0.02 % of the image are dropped
as noise, and each outline is simplified (Douglas-Peucker, tolerance 0.4 % of its perimeter, at least one
pixel). An instance gives one object with all its regions as rings; the mask of a whole class from a
semantic model becomes one object per region. Every polygon object also gets its bounding box. YOLO
(Ultralytics) models cannot pre-annotate yet.

What is written is checked when the run ends:

- Only labels of the project's **current classes** and with the **shape its type needs** are written; the
  others are counted, not written (`counts.dropped_class`, `counts.dropped_geometry`; the model's classes that
  the project no longer has are listed in `counts.removed_classes`). A model whose classes are all gone from the
  project is refused (the run fails with a plain text).
- An image in which the model finds nothing is counted (`counts.empty`) and left without labels; with *Not
  reviewed*, its old pre-annotations are removed and nothing replaces them.
- A keypoint model must find the same number of points per object as the labels of the project
  (`num_keypoints` on the model card); otherwise the run is refused with a text that names both numbers.
- If the project's type or classes changed while the model was running, **nothing is written** and the run
  fails with a text that says to run it again (changing them waits for runs that already started, but a
  run can be queued first).

## Public models

A project can pre-annotate before it has any training: **Models → Add a public model** (*Modelos →
Adicionar modelo público*) adds a
model already trained by others as the next model (`m<N>`), like a saved one. It is meant for a first
round of labels to review, not to replace a model trained on the project's own images.

- **Catalogue** (`activelabel.training.public_models`): torchvision models trained on COCO 2017.
  Detection projects: Faster R-CNN MobileNetV3, RetinaNet ResNet-50 v2 and Faster R-CNN ResNet-50
  v2 (80 categories); keypoint projects: Keypoint R-CNN ResNet-50 (person, 17 body points). The
  dialog shows the COCO box mAP and the download size of each. There are no public models for
  classification or segmentation projects yet: for them the *Add a public model* button is hidden, and an empty
  *Models* step says there is no public model for the project type (and offers training instead).
- **Mapping**: for each class of the project, the category of the model that stands for it
  (suggested when the names match, ignoring case, accents, `_` and `-`, or when a class is the Portuguese
  name of a COCO category, also in the plural: `gato` or `gatos` for `cat`, `pessoa` for `person`, `cachorro` or
  `cão` for `dog`, `carro` for `car`; a name that matches exactly wins over a translation). The dialog opens
  with the suggestion filled in and shows *N of M project classes linked to a model category*; *Review* opens the
  choices, one list of categories per class (*— not used —* leaves the class out). Categories not mapped are
  ignored. Through the API the mapping is `{category: class}` and several categories may name one class
  (`truck` and `car` → `car`); the dialog chooses one category per class.
- **Weights**: nothing is stored in the project (`model.json` only, with `source: "public"`, the
  catalogue id and the mapping). Each pre-annotation run downloads the weights from the torchvision
  servers into the run's temporary folder, removed at the end; it needs `torch` and `torchvision`
  in the training environment (installed first when missing) and an internet connection.
- In the mapping dialog, a category of the model can go to only one class (the dialog says which classes
  chose it twice and does not save until that is fixed), and at least one class must be linked.
- A public model has no training results, no downloads and no ONNX export, and is left out of the
  comparison table. It can be the current model, renamed (note) and deleted like any other.
- **Licence**: torchvision code is BSD-3; the weights were trained on COCO 2017. Check the terms of
  the weights and of COCO before using the labels in a product.

API: `GET /api/projects/{id}/models/public-catalog` (models for the project type, with the suggested
mapping, and the project's classes); `POST /api/projects/{id}/models/public` with
`{public, mapping, note}` (404 for an unknown model; 400 when it does not fit the project type, the
mapping is empty, or names a category or class that does not exist). For a project type without public
models the catalogue answer has an empty `models` list.

## HTTP API

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/projects/{id}/preannotations` | models that can pre-annotate, tasks with counts (images, unlabeled, unreviewed), runs |
| POST | `/api/projects/{id}/preannotations` | start: `{model, task, scope: "unlabeled"\|"unreviewed", threshold, device}` (threshold 0.5 when omitted, accepted from 0.001 to 0.999; the screen offers 0.05 to 0.95 and proposes 0.3 for detection and keypoints; `device` is `auto` when omitted, see *Device*) |
| POST | `/api/projects/{id}/preannotations/{n}/cancel` | cancel a queued or running run |
| POST | `/api/projects/{id}/preannotations/accept` | accept the pre-annotations of a task (`{task}`) or of every task |

Errors (each with a `code`, docs/DESIGN.md): 404 for an unknown model, run or task (`model_not_found`,
`preannotation_not_found`, `task_not_found`); 400 for a model that is not in this version's catalogue
(`model_not_in_catalogue`), whose labels the project does not use (`model_kind_mismatch`) or whose kind
cannot pre-annotate (`preannotation_kind_unsupported`: today only the YOLO family, whatever its task), a threshold outside
0.001–0.999 (`threshold_range`), a device that is not valid or not on this computer (`device_invalid`,
`device_unavailable`), an unknown scope (`scope_invalid`) or no image to fill
(`no_images_for_choice`); 409 when the same task and scope are already being pre-annotated
(`preannotation_running`). The result of a run has `counts` with `images`, `objects`, `skipped`,
`unreadable`, `empty`, `dropped_class`, `dropped_geometry` and `removed_classes`.

Edit sessions accept the image scope `preannotated` (images whose labels are all pre-annotations).

## Counts

Images whose labels are all pre-annotations are counted apart, as *a revisar*, in the step bar, the task
progress and the annotation summary; "anotadas" counts reviewed images only. An image listed with no
labels was reviewed as empty (a background image) and is never pre-annotated again.

## Writes

Every change to `annotations.json` (applying a review or a sync, pre-annotation, accept, deleting images,
imports) holds the same lock, so one never loses another's change. A cancel that arrives while the labels
are being written does not stop that last step; the run ends as done.
