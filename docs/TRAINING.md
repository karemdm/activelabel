# ActiveLabel — training and models

Steps **5 · Training** (*Treinamento*) and **6 · Models** (*Modelos*) of a project. A training run uses one
dataset version (docs/DATASETS.md), so it is always known which images and labels it saw; its
configuration is saved as YAML so it can be reproduced, on this or any other machine.

## Principles

- **Works on any machine, nothing to configure.** The device is chosen when a run starts, in the *Device*
  field of the *New run* dialog: *Automatic* (`auto`: a CUDA GPU when there is one, else Apple MPS, else the
  CPU), *CPU*, *GPU n* (`cuda:N`, one entry per NVIDIA GPU that `nvidia-smi` lists) or *Apple GPU (MPS)* (on a Mac
  with an Apple processor). The dialog only offers the devices of this computer, and the API refuses a device
  it does not have (400 `device_unavailable`, or `device_invalid` for a name that is not a device); the worker
  checks it once more with PyTorch before it trains. The choice is kept in `run.json` (`device_choice`), not in
  the configuration: no path, device or machine setting is stored there, so a configuration repeats on any
  computer. *Repeat* asks for the same device again.
- **Install only what is needed, when it is needed.** Before a run, ActiveLabel checks that the
  packages of the chosen model are present in the training environment and installs the missing
  ones (a visible step of the run, with progress), at tested versions (see *Pinned package versions*).
  PyTorch is installed with GPU support when an NVIDIA GPU is found (`nvidia-smi`) whose driver has a
  PyTorch CUDA build of the pinned version, otherwise the CPU build. Ultralytics (licence AGPL-3.0) is installed
  only when a YOLO model is chosen and the licence was accepted (see *Licences*).
- **Separate environment.** Training packages live in their own Python environment managed by
  ActiveLabel (`<workspace>/training/env`, or a short folder under `%LOCALAPPDATA%` on Windows when
  the path is long), so ActiveLabel itself stays light and free of version conflicts.
- **Downloads are temporary.** Pretrained weights, the `spec.json` of the run and
  any other download go to a temporary folder of the job (`<workspace>/training/tmp`): the worker's
  `TORCH_HOME` (torchvision), `HF_HOME` (timm, Hugging Face), `XDG_CACHE_HOME`, `YOLO_CONFIG_DIR` and its
  working folder (where Ultralytics saves its weights) all point inside it. The folder is removed when the
  training run or pre-annotation ends (finished, failed or cancelled), and the job records what it had
  downloaded in `downloads` (`run.json`, or the pre-annotation run): `{files: [{name, bytes}], bytes,
  deleted}`. The trained weights (`best.pt`, `last.pt`) are written to the run's own folder and stay;
  caches outside ActiveLabel are never read nor removed. The price is that every run, pre-annotation and
  resumed run (#321) downloads its pretrained weights again (from tens to a few hundred MB, depending on
  the model). Folders left by a process that was killed are removed when ActiveLabel starts. Only the
  run's results stay.
- **One heavy job per device.** Training runs, pre-annotations (docs/PREANNOTATE.md), the embeddings of
  the similarity map, its regrouping, uniqueness scores, label checks and evaluations with a saved model, ONNX exports, curation analyses and video extractions of every
  project wait in one global queue with a lane per
  device (`cpu`, `cuda:0`, `cuda:1`..., `mps`): by default one job at a time on each GPU and one on the
  CPU (`ACTIVELABEL_QUEUE_GPU`, `ACTIVELABEL_QUEUE_CPU`), first come first served inside a lane. A run
  on `cuda:0` and an export (CPU) may run together; two runs on `cuda:0` never do. A run with device
  `auto` takes the lane of the device the worker will pick (the first NVIDIA GPU, else the Apple GPU,
  else the CPU). The queue survives a restart (docs/DESIGN.md, *Work queue*).

## Model catalogue

| Type (project) | Family | Models | Packages |
| --- | --- | --- | --- |
| Classification | timm | 102 networks, by architecture: ResNet, ResNeXt, RegNetY, DenseNet, MobileNetV3/V4, EfficientNet B0–B7 and V2, GhostNet, MobileViT and V2, ConvNeXt and V2, NFNet, ViT, DeiT, Swin, BEiT, EVA-02, CaiT, CoAtNet, MaxViT, FastViT | torch, torchvision, timm |
| Classification | torchvision | every torchvision classification network (80), by architecture: ResNet 18–152, ResNeXt, Wide ResNet, DenseNet, VGG (with and without BN), MobileNet V2/V3, EfficientNet B0–B7 and V2 S/M/L, RegNetX/RegNetY, ConvNeXt T–L, ViT B/L/H, Swin and Swin V2 T/S/B, MaxViT-T, ShuffleNet V2, SqueezeNet, MNASNet, GoogLeNet, Inception V3, AlexNet | torch, torchvision |
| Detection | torchvision | Faster R-CNN (ResNet-50 FPN, MobileNetV3), RetinaNet, SSDlite, FCOS | torch, torchvision |
| Detection | ultralytics (AGPL-3.0, after accepting the licence) | YOLO26 (proposed first), YOLO11, YOLOv8; sizes n s m l x | ultralytics |
| Segmentation (semantic) | torchvision | DeepLabV3 (ResNet-50, MobileNetV3), FCN, LR-ASPP | torch, torchvision |
| Segmentation (semantic) | transformers (Hugging Face, Apache-2.0) | SegFormer (transformers) B0–B5 (nvidia/mit-b0 to b5) | torch, torchvision, transformers |
| Segmentation (semantic) | smp (Segmentation Models, segmentation_models_pytorch, MIT) | U-Net, U-Net++, DeepLabV3+, FPN, PSPNet, LinkNet, MA-Net, PAN, UPerNet on ResNet-34/50 or MobileNetV2; SegFormer (smp) on MiT-B0/B1/B2 | torch, torchvision, segmentation-models-pytorch |
| Detection | transformers (Hugging Face, Apache-2.0) | DETR ResNet-50/101; RT-DETR R18/R34/R50/R101 | torch, torchvision, transformers |
| Segmentation (instance) | transformers (Hugging Face, Apache-2.0) | Mask2Former Swin-T/S/B (COCO instance checkpoints) | torch, torchvision, transformers |
| Segmentation (instance) | torchvision / ultralytics | Mask R-CNN ResNet-50 FPN; YOLO26/11/v8-seg n–x | torch, torchvision / ultralytics |
| Keypoints | torchvision / ultralytics | Keypoint R-CNN; YOLO26/11/v8-pose n–x | torch, torchvision / ultralytics |

**segmentation_models_pytorch (#606).** The `smp` family puts a decoder (the architecture: U-Net, U-Net++,
DeepLabV3+, FPN, PSPNet, LinkNet, MA-Net, PAN, UPerNet, SegFormer) on an ImageNet encoder (the variant:
ResNet-34, ResNet-50, MobileNetV2; SegFormer on its Mix Transformers MiT-B0 to B2). Ids are
`smp_<decoder>_<encoder>` and smp builds them as `<decoder>:<encoder>`. The package is installed, at its pinned
version, by the first run that uses it. The semantic segmentation worker trains them like torchvision's: same
data, masks, losses (class weights, Dice), metrics, checkpoints, pre-annotation and ONNX; the encoder plays the
backbone (frozen in `transfer`, `backbone_lr_factor` in `finetune`) and the decoder with its segmentation head
the classifier. smp refuses sides that are not a multiple of the encoder's output stride (32 for these encoders;
16 for DeepLabV3+ and PAN, which dilate it), so the form and the server ask for a multiple of 32 (`multiples`,
#671), which suits all of them, and the worker checks it again;
the dialog keeps at least 128 pixels (an old limit of PAN, not checked by smp itself); 512 by default.
The dialog calls the family *Segmentation Models (smp)*, and its SegFormers *SegFormer (smp)*, apart from the
*SegFormer (transformers)* of the Hugging Face family (#671). Their size (`params_m`) is unknown: smp publishes
the size of its encoders, not of whole models.

**Hugging Face transformers (#606).** The `transformers` family has SegFormer B0 to B5 (semantic segmentation,
encoders `nvidia/mit-b0` to `b5`), DETR ResNet-50/101 and RT-DETR R18/R34/R50/R101 (detection) and Mask2Former
Swin-T/S/B (instance segmentation, COCO instance checkpoints; 47.5, 68.8 and 107 million parameters, from the
Hub's model pages and the Mask2Former paper). Ids are `hf_<architecture>_<checkpoint name>`
(for example `hf_detr_detr_resnet_50`) and are built as `<architecture>:<checkpoint>` from the Hugging Face
Hub; the checkpoints are public and not gated. The downloads go to the run's temporary folder (`HF_HOME`) and
the Hub's telemetry is off. They train with the workers that already exist, `segmentation.py` for SegFormer and
`detection.py` for DETR, RT-DETR and Mask2Former, so metrics, checkpoints and pre-annotation are the same;
ONNX export is not checked for them. Images are normalised with the ImageNet mean and standard deviation and
padded to a multiple of 32. In fine-tuning the whole encoder trains (`trainable_backbone_layers` is fixed at
5 for them). Defaults: AdamW, learning rate 1e-4 (detectors) or 6e-5 (SegFormer), batch 4. `transformers` is
pinned at 5.23.0 and installed by the first run that uses it.

**Architecture and variant (#618).** Each model of the catalogue has an `arch` (the architecture or
generation the dialog groups by: `ResNet`, `EfficientNet`, `Faster R-CNN`...) and a `variant` (its size inside
the architecture: `18`, `B3`, `ResNet-50 FPN`...); the ids do not change, so older runs stay valid. Two
families can have a network of the same name (timm's and torchvision's EfficientNet-B0): the family is shown
first, timm keeps its ids (`efficientnet_b0`) and torchvision's EfficientNet B0–B7 have ids `tv_efficientnet_b0`…
while the worker still builds them by their torchvision name (`library_name`). The catalogue
API (`GET /api/training/catalog`) also gives `params_m` (parameters in millions, of the network with its
original ImageNet or COCO head) and `file_mb` (the size of the weights file: the official one when known, else about four bytes per parameter) and
`experimental`. Each classification network has its own defaults: the input size of its pretrained weights
(EfficientNet B0 224 up to B7 600, EfficientNetV2 384/480, Inception V3 299, RegNetY-128GF 384, ViT-H/14 518,
Swin V2 256, the others 224), a fixed size where its weights only take one (ViT, MaxViT), a smallest size where
its pooling needs it (Inception V3 75, AlexNet 64), a lower learning rate for transformers and ConvNeXt, and a
smaller batch for the largest networks (4 above 100 M parameters, 2 above 300 M). The dialog lists the models of
a family grouped by architecture.

**timm (#617).** The timm classifiers come from the default pretrained configuration of each network in the
pinned timm (`_TIMM_CLASSIFIERS` in `catalog.py`): the weights are the first pretrained tag of the name, the
default input size is the one of those weights, and transformers (ViT, DeiT, Swin, BEiT, EVA-02, CaiT, CoAtNet,
MaxViT) keep it fixed. A network both libraries have keeps its torchvision id and gets `timm_<name>` in timm
(`timm_resnet50`); `efficientnet_b0` and `efficientnet_b3` stay timm's, as older runs used them. Images are
normalised with the mean and standard deviation the weights expect (timm's `pretrained_cfg`: ImageNet,
Inception or CLIP values; ImageNet for torchvision), and the saved model records them for pre-annotation and
ONNX.

The new-run dialog chooses the model in three fields (#632): the **family** (*Model family* / *Família do
modelo*), then its **architecture** and then its **variant**, named in the family's own words: Ultralytics
*Generation* and *Size* (YOLO26, YOLO11, YOLOv8; n, s, m, l, x), torchvision and timm *Architecture* and
*Variant* (ResNet 18-152, EfficientNet B0-B7...). The task is not a choice: it comes from the project type and
is said in the line that names the model (*Ultralytics YOLO26 m · Detection*, with the model id as its
tooltip). A short list of variants (up to eight, short names) is a row of buttons, a radio group moved with
the arrow keys (two rows of up to four on a phone), with the parameters and file size of the chosen one and
"faster ... more accurate" at its ends when the variants grow; a longer list is a select. The family proposes
its default architecture (YOLO26) in the size that suits the computer (`default_size` of
`/training/environment`: n without a GPU, s on a small GPU, m from 8 GB) until the person picks a model.
Another size of the same architecture keeps every value the person set and says nothing (unless a value has
to go back to its default); another architecture or family keeps the values that still apply and says, in
one line, what went back to its default (#312). Changing the generation keeps the size (YOLO11 m to YOLO26 m). Every screen names a catalogue model the same way, its architecture and variant with a
space ("ResNet 18", "EfficientNet B5", "YOLO26 m"), whatever its family. When the run would train on the CPU (the
device is CPU, or automatic on a computer without a GPU) and the model has 100 M parameters or more, the dialog
warns before it starts (#668). Only the families with a model for the project type are listed (timm, for instance,
has no segmentation model, so a segmentation project does not show it). YOLO/Ultralytics models train with their own worker (`workers/yolo.py`) once the
AGPL-3.0 licence is accepted (*Licences* below). Until then the family stays disabled in the dialog, and its hint
offers *Read and accept the AGPL-3.0 licence…*: a window says in plain words what the licence asks, links to its
text and accepts it only after an explicit checkbox; the family can then be chosen.

The catalogue is data (`training/catalog.py`): each model lists its family, packages, input size,
and its hyperparameters with type, default, limits, choices and a one-line help. The screen builds
its form from it, so adding a model does not change the interface.

A model can have **its own defaults and limits** for the common hyperparameters, and the form and the API
check a value against them: ViT-B/16 only takes an image size of 224 (its position embeddings are fixed) and
SSDlite320 only 320; the transformers (ConvNeXt-T, ViT-B/16, Swin-T) start at a learning rate of 0.0001; the
torchvision detectors, instance segmentation and keypoint models start with SGD, learning rate 0.005, batch size
4, weight decay 0.0005, a one-epoch warmup and the `letterbox` resize (SSDlite320 uses `stretch`); the
semantic segmentation models start with learning rate 0.0001 and batch size 4. A model can also ask for a **multiple** (`multiples`, for example an image
size that is a multiple of 32 for a network that halves it five times): the catalogue gives it to the form with
the parameter (`multiple`), and both the form and the server refuse other values (`param_not_multiple`).

**Regression (#619).** A regression project (one or more numbers per image, its *targets*) trains the
classification networks of torchvision and timm with one output per target instead of one per class. They have
ids of their own, `reg_<classifier id>` (for example `reg_resnet18`), because a catalogue entry has one kind of
task, which decides its worker, metrics and parameters; they are built and optimised like the classifiers (the
same start modes, learning rates and `backbone_lr_factor`). The worker is `regression.py`:

- the items of the spec are `{"path", "values"}`, one number per target, `null` where the image has no value of
  it (an image without any value is left out); a missing value does not count in the loss nor in the metrics;
- the targets are normalised with the mean and standard deviation of the training part, kept in the checkpoint
  (`target_mean`, `target_std`), so targets of very different scales weigh alike; the metrics use real values;
- the loss (`loss`) is `mse` (default), `mae` or `huber` (`huber_delta`, in standard deviations of the target);
- the augmentations keep the value of an image: flips, rotation, colour, blur and grey; no crop (it can cut
  what is counted), no mixup, cutmix or erasing;
- the selection metric is RMSE (default), MAE or R²: RMSE and MAE are better lower (see *Selection*).

## A run

Options chosen on the screen and saved in `config.yaml`:

| Group | Options |
| --- | --- |
| Data | dataset version; for K-fold versions, all folds or one fold |
| Model | model; start: **transfer learning** (pretrained, backbone frozen, only the head trains) / **fine-tuning** (pretrained, everything trains; backbone learning rate × factor) / **from scratch** |
| Mode | **single run**, **grid search** (lists of values for some hyperparameters: every combination is a run; at most 50). The values are separated by commas (`0.01, 0.001`); in Portuguese, where the comma is the decimal sign, by `;` or a comma followed by a space (`0,01; 0,001`), and a list of whole numbers by any comma. Thousands may be grouped the way the language writes them (`1.000` in Portuguese, `1,000` in English; `0.001` stays a decimal), and `0,1,0,2` in Portuguese is answered with the separator to use |
| Hyperparameters | from the catalogue: epochs, batch size, learning rate, optimizer (SGD, Adam, AdamW), weight decay, scheduler and warmup epochs, image size, resize policy, augmentation level (see below), early stopping patience, mixed precision, seed, the backbone rate factor (fine-tuning only), and model-specific ones (label smoothing, class weights, loss and focal gamma, trainable backbone layers, score threshold) |
| Device | *Automatic*, *CPU*, *GPU n* or *Apple GPU (MPS)*: where this run trains (see *Principles*); not part of the saved configuration |
| Selection | metric that chooses the best epoch / best run, per type: classification accuracy (default) or macro F1; detection, instance segmentation and keypoints mAP50-95 (default) or mAP50; semantic segmentation mIoU (default) or pixel accuracy. Higher is better, except for the error metrics (RMSE and MAE, `catalog.LOWER_IS_BETTER`): the best epoch, the plateau schedule, the ranking of a grid and the comparison then take the lowest value (the spec says `lower_is_better`) |

**Resize policy.** A photo becomes the square input of the network (`image_size`) by one of three rules, the
`resize` hyperparameter: **stretch** (both sides scaled to the square, the proportions change), **letterbox** (the
longer side fits, bands of a neutral grey fill the rest, nothing is cut) or **centre crop** (`center_crop`: the
shorter side fits, the edges are cut). Every model has the same rule in training, in evaluation on the validation
and test parts, in pre-annotation (docs/PREANNOTATE.md) and in the ONNX file, which keeps it in its metadata
(`resize`, `image_size`, `mean`, `std`) because the photo is resized outside the graph. Boxes, masks and
keypoints go through the same transformation into the network and back to the photo, so predictions come back
in the photo's own coordinates. The policy is saved with the weights (and on the model card); a model saved
before the policy existed is read as it was trained (stretched for classification and segmentation, the photo as
it is for detection). In semantic segmentation the letterbox bands are background in the mask, so they count
as background pixels in the metrics; objects that a centre crop cuts away entirely are dropped from the target.
The default is `stretch`, and `letterbox` for the torchvision detectors, instance segmentation and keypoint
models.

**Learning rate schedule and warmup.** The `scheduler` is one of: `none` (constant), `cosine` (half a cosine
down to 0), `step` (divided by 10 at every third of the epochs), `onecycle` (rises over the first 30 % of the
steps, then falls; it has its own ramp, so `warmup_epochs` is not used with it) or `plateau` (divided by 10 when
the selection metric on validation has not improved for 2 epochs). `warmup_epochs` (0 to 50, default 0; 1 for
the torchvision detectors) raises the rate linearly from almost 0 to the full value over that many epochs, which
steadies fine-tuning and detectors started from COCO weights. The warmup cannot be longer than the
training (`warmup_epochs` above `epochs` is refused with `warmup_exceeds_epochs`), and an early stopping
`patience` (0 = off) must be below `epochs` (else `patience_exceeds_epochs`: it could never stop the run); in a
grid search every combination counts. Runs saved before this rule still resume as they were. All workers (classification, detection,
instance segmentation, keypoints, semantic segmentation) apply the same schedule, set step by step; the rate at
the end of each epoch is kept as `lr` in `metrics.jsonl`.

**Class imbalance.** When one class has far more examples (or pixels) than the others, a plain cross-entropy can
settle for the common class: a semantic model that only predicts the background shows an mIoU of 0 and a high pixel
accuracy. Two model-specific hyperparameters, both off by default, counter it:

- `class_weights`: `none` (default) or `inverse`. With `inverse` each class weighs the inverse of its frequency in
  the **training** part (images per class in classification, pixels per class in semantic segmentation, the
  background included), scaled to a mean of 1 and capped so that the heaviest class weighs at most 10 times the
  lightest. The weights used are written to the training log.
- `loss`: classification takes `cross_entropy` (default) or `focal` (cross-entropy that counts easy examples less;
  `focal_gamma`, 0 to 5, default 2, sets how much, and is read only with `focal`); semantic segmentation takes
  `cross_entropy` (default), `dice` (one minus the mean Dice overlap of the classes) or `cross_entropy_dice` (the
  two added). The class weights also apply to the focal and Dice terms.

When a semantic model only predicts the background, try `class_weights=inverse` and `loss=cross_entropy_dice`;
when a classifier ignores its rare classes, try `class_weights=inverse` (and `focal` if the rare classes are also
the hard ones). Both change the loss values, so compare runs by the selection metric, not by the loss. Instance
segmentation, detection, keypoint and YOLO models do not have these options.

**Mixed precision and batch size.** `amp` (`auto`, `on`, `off`; default `auto`) trains in float16 with a gradient
scaler on a CUDA GPU, which is faster and lighter; `auto` and `on` use it on CUDA only, and the CPU and Apple
GPUs always train in full precision (asking for `on` there adds a warning, not an error). Evaluation passes stay
in full precision. A loss that is not finite stops the run with a text that suggests turning `amp` off or lowering
the learning rate. What was really used is recorded in `run.json` (`setup`: `device`, `amp`, `batch_size`) and shown in the
run's details (*What the training used* / *O que o treino usou*, with the batch asked for when it was halved), next
to the curves of the loss, the score and the learning rate of each epoch. On a
CUDA GPU the worker first tries one training step with the batch size asked for: when the GPU runs out of memory
the batch is **halved**, again and again, down to the smallest batch the model accepts (1 image, or 2 for semantic
segmentation), and each halving is a warning saying which batch size the training went on with (`warnings` in
`run.json`, and a `warning:` line in the log). If not even the smallest batch fits, the run fails with a text that
suggests a smaller image size or the CPU. The CPU and MPS are not probed.

**Resume, warnings and failures.** A failed or cancelled single run that kept a checkpoint (`checkpoint_epoch` in
`run.json` and its `last.pt`) shows **Resume** (*Retomar*): it goes on from that epoch with the same number and
settings (`POST /api/projects/{id}/runs/{n}/resume`); it is refused when the configuration or the data changed since,
and grid and all-folds runs cannot be resumed yet. The card and the details of a run show its warnings and its failure
in the interface language, by their codes (`tr.warn.*`, `error.*`); a run made before the codes keeps the text it has.

**Data augmentation.** The `augmentation` level (default `light`) is a preset (`none`, `light`, `standard`,
`strong`) or `custom`. Each preset is a list of techniques for the model's kind of task (the same techniques the
workers applied before they could be chosen); `custom` takes the list from the `augmentations` option of the
configuration, `[{"name", "params"}]`, checked against the catalogue (`training/augmentations.py`,
`GET /api/training/augmentations?type=`: levels, and per kind of task the techniques with their settings and the
list of each preset). A technique may be given once, and RandAugment and TrivialAugment cannot be used together.
Techniques: classification — random resized crop, horizontal and vertical flip, rotation, RandAugment or
TrivialAugment, colour jitter, blur, greyscale, random erasing, mixup and cutmix; detection and instance
segmentation — flips, rotation, zoom (scale jitter), shift, colour jitter, blur, greyscale; keypoints — the
same without the vertical flip; semantic segmentation — the same as detection plus random resized crop.
Geometric techniques move boxes, masks and keypoints with the image; a horizontal flip of keypoints needs the
left/right `flip_pairs` (an empty list when no point has a mirror twin), which is why the keypoint presets do
not include it (their `light` varies brightness and contrast slightly instead, so it is not the same as `none`). The runner resolves the level into `spec.json` (`augmentations`) and records it in `run.json`;
the worker applies exactly that list, in catalogue order, to the training images only, and writes it to the log.
Configurations made before techniques could be chosen have no list and keep their preset.

In the *New run* dialog the level is a choice and, under it, **Augmentation techniques** lists the techniques of
the chosen level with their settings (chance, angle, zoom range and so on, each with a hint) and a one-line
description; ticking or unticking a technique or changing a setting makes the level *Custom*, and choosing a
preset again reloads its list (for keypoints, a *Left-right pairs* field takes the points that swap places in a
mirror image, such as `0-1, 2-3`). In a grid search, *Custom* cannot be compared with other levels: choose it
alone, or compare only presets.

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

**Switching the model in the dialog.** The values the person has changed are kept when the new model has the
same hyperparameter and accepts the value (within its limits); a value it does not accept, or one the person did
not touch, takes the new model's default. A line under the model says what changed: *Switching to {model}:* then,
for example, *learning rate now 0.0001 (default)*, *image size back to 224 (640 does not suit this model)*,
*{p}: new parameter*, *{p}: does not apply*, *kept: epochs, seed*, and *same hyperparameters and defaults* when
nothing differs (at most four changes are listed, then *+N*). The selection metric and the start (transfer,
fine-tuning, from scratch) are reset in the same way when the new model does not have the chosen one.

During the run: phases (`environment`, `preparing`, `training`, `evaluating`), epoch progress,
loss and validation metric per epoch (live curves), cancel. At the end the test part (if the
version has one) is evaluated with the best epoch.

Besides the selection metric, the results of a keypoint model carry the object-keypoint-similarity mAP
(`kp_map50`, `kp_map50_95`, as in COCO but with the same tolerance `1/K` for every one of the `K` points,
because COCO's per-point values exist only for the 17 body points) and PCK@0.1; those of an instance
segmentation model carry the mask mAP (`mask_map50`, `mask_map50_95`). The selection metric of both stays the
box mAP.

A grid search or a run over *all folds* of a K-fold version trains several **children**, one after
another in the same place of the queue (never two at once, on the run's device) and the same temporary folder (pretrained
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
| MAE, RMSE, R² (regression) | per target: mean absolute error, root of the mean squared error, 1 − squared error / total variance, over the images with a value of the target | `mae`, `rmse` and `r2` are the means over the targets that have values; R² is 0 when the real values of a target do not vary; `per_target` has each target's scores and `count`; `points` keeps up to 2000 `[real, predicted]` pairs over the targets for the charts |
| K-fold mean ± std | mean of the per-fold validation values and their **population** standard deviation (ddof = 0) | not a pooled metric over all folds; folds have about the same size |

Segmentation metrics are measured on the images resized to the run's `image_size` by the run's resize policy
(both sides for `stretch`), not at their original resolution, so every image weighs the same number of pixels;
with `letterbox` the added bands are background. Box and mask IoU between an image's detections and its objects
is a matrix of size (detections × objects): an image with no objects or no detections gives an empty matrix, so
its detections are false positives and its objects misses, and a mask pair whose union is empty has IoU 0. For
instance segmentation the box of an object is the extent of its polygons, and for a pose the stored box grown to
hold every visible point, so box, mask and keypoints agree. No metric is rounded in the
results; the interface rounds only for display. A part without images gives 0 for every metric.

### Curves by confidence threshold

Detection, instance segmentation and keypoint runs (torchvision, transformers and YOLO) add `curves` to the
`val` and `test` parts of `results.json`, from the evaluation pass the run already does (#322):

```json
"curves": {
  "iou": 0.5, "match": "box",
  "confidence": [0.0, 0.01, 0.02, "...", 1.0],
  "classes": {"cat": {"precision": ["..."], "recall": ["..."], "f1": ["..."], "support": 12}},
  "all": {"precision": ["..."], "recall": ["..."], "f1": ["..."]}
}
```

For each class with ground truth in the part, `precision`, `recall` and `f1` are those of the detections whose
confidence is at least each value of `confidence` (101 values, 0 to 1, so at most 101 points per curve and
class; 4 decimals). A detection is right when it matches an object of its class at IoU 0.5 with the same greedy
matching as mAP50, by box IoU (`match: "box"`, detection and keypoints) or mask IoU (`"mask"`, instance
segmentation). Where no detection is that confident precision is 1 and recall 0, as in scikit-learn and
Ultralytics. `all` is the mean of the classes' curves. The detection worker computes them in
`training/workers/curves.py` (numpy only); for YOLO they are Ultralytics' own curves (`p_curve`, `r_curve`,
`f1_curve` of the box or mask metric, 1000 confidences) sampled at the same 101 confidences. The run detail
draws precision × recall and F1 × confidence with a class picker (*All classes (mean)* first, then the classes
with the most objects) and the same numbers as a table (*Show as a table*: every 0.1 of confidence). Runs made
before this have no `curves` and show no charts.

The validation part also has the **suggested confidence threshold** (#323): `suggested_threshold`, the
confidence where the F1 of `all` is highest, and `suggested_thresholds`, the same for each class (a class
whose F1 is 0 everywhere has none). F1 is flat between two detections' scores, so the middle of the first run
of the highest values is taken (2 decimals). The test part has no suggestion of its own: it is measured at the
validation's. The F1 chart marks it, and saving the run as a model copies both onto the card (`model.json`),
where pre-annotation takes `suggested_threshold` as its default confidence (docs/PREANNOTATE.md). Older runs and
models have none and keep the default of their kind.

## Quality alerts

When a run ends, `training/quality.py` reads its `results.json` and `metrics.jsonl` and stores `alerts` (a list of
`{code, params}`) in `run.json`; the run card shows a warning badge with their number and the run details list them
(texts `tr.alert.<code>` in `i18n/`). They are warnings only: saving the run as a model stays allowed. A finished
run made before the alerts gets them when it is listed (from its results; the validation-size check needs the
dataset version and is only done at the end of a run). For a grid or all-folds run the alerts are those of the
best child, whose results and curves are the run's own. Like every output of the tool, they are a hint for a person
to review, not a validated verdict. On the run's card the alerts are a badge ("2 quality alerts", a button): it opens the
run's *Details* at the alerts list, so the texts are reachable by click, touch and keyboard (they were only a tooltip).
The thresholds are named constants in the module.

| Code | When | What to try |
|---|---|---|
| `metric_zero` | the selection metric is 0, NaN or empty | check that the labels match the task, train more epochs, lower the learning rate, start from pretrained weights |
| `only_background` | segmentation: mIoU below 0.01 while the pixel accuracy is at least 0.5 | the model predicts only background: more epochs, a higher learning rate for the head, more or larger objects, class weights or Dice loss if the run offers them |
| `single_class` | classification: one class takes at least 95% of the predicted images | balance the classes (more images of the others), lower the learning rate, check the labels |
| `class_zero` | a class has score 0 (recall, AP50 or IoU) while the mean of the classes is above 0 (not with `single_class`, which already explains it) | add examples of that class and check its labels |
| `class_far_below` | a class scores under half the mean of the classes, and the mean is above 0.2 (at most 5 classes named) | same: more and cleaner examples of that class |
| `val_worse_than_train` | at the best epoch the validation loss is at least 2 times the training loss and at least 0.1 higher | overfitting: more data or augmentation, a smaller model, early stopping (patience) |
| `few_val_images` | fewer than 10 images in the validation part (or fold) | make a version with a larger validation part, or use K-fold |
| `few_val_per_class` | classification: a class with fewer than 2 validation images | the score of that class is not reliable: add images or change the split |

## Storage

`<project>/runs/r<N>/` (numbers never reused):

| File | What |
| --- | --- |
| `config.yaml` | everything needed to repeat the run (dataset version and its fingerprint, model, start, mode, hyperparameters including the resize policy, scheduler and mixed-precision choice, the list of augmentation techniques of a `custom` level, seed) — no device or path |
| `run.json` | status, times, the device asked for (`device_choice`) and the one found (`device`: PyTorch, CUDA and MPS availability, GPU name), `setup` (`device`, `amp`, `batch_size` the training really used), `warnings` (for example a halved batch), `resize`, the augmentation techniques applied, package versions, ActiveLabel version, child runs (grid / folds) |
| `metrics.jsonl` | one line per epoch: losses and validation metrics |
| `results.json` | best epoch, validation and test metrics (confusion matrix, per-class metrics and the curves by confidence threshold when the type has them) |
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
that was running when ActiveLabel stopped ends as failed (`interrupted`) and can be resumed from its
last checkpoint; queued runs, pre-annotations and exports go back in line in their old order.

## Models (step 6)

A finished run is **saved as a model** (button *Save as model* (*Salvar como modelo*) on the run; it is
disabled, with a note, when the run has no weights any more — see below): `<project>/models/m<N>/`
holds its best weights (`best.pt`, **moved** from the run, so the weights exist once and a saved run
has no `best.pt` any more; the run keeps its curves, results and log), `config.yaml`, `results.json` and a card, `model.json` (run and
best child, model, dataset version and fingerprint, start, selection metric and value, the classes in
the order of the network's outputs, the number of points of a keypoint model, device and package
versions, size, note, exports, and the `suggested_threshold` and `suggested_thresholds` of the
validation when the run has them). Numbers are never reused, a run is saved at most once, and saving removes
the run's `last.pt`. The save dialog asks which
weights to keep: the PyTorch weights always, and also ONNX (classification and semantic
segmentation), which is then queued right after saving.

- **Public models**: *Add a public model* (*Adicionar modelo público*) adds a pretrained torchvision
  model (COCO) without local training, with a mapping from its categories to the project's classes;
  only to pre-annotate. It has no weights stored here, so it has no download and no ONNX export (the
  API answers 404 and 400). Only detection and keypoint projects have public models today; for any other
  project type the *Add a public model* button is hidden and an empty step says to save a finished training run
  (there are no public models for this project type yet). See docs/PREANNOTATE.md (Public models).
- **Current model**: one model of the project is marked current (`models/current.json`, button *Make
  current*); the first
  saved model becomes current. Later steps use it to pre-annotate new tasks (see docs/PREANNOTATE.md), which are reviewed (in
  CVAT), go into a new dataset version and a new training: the loop that improves the model.
- **Renaming a class** (#521). The class list of a saved model is the names of its network's outputs, so when a
  class of the project is renamed (in CVAT, see docs/SYNC.md, or by its letter case in the project's classes) the
  name at that position of the model's `classes` is changed too and the weights keep working: the pre-annotation
  writes the new name. The card then keeps `trained_classes` (the names it was trained with, never changed
  again) and the model card shows *Classes renamed since training: gato → felino*. The `mapping` of a public
  model (its categories to project classes) follows the new name. The `classes` of the training runs and of the
  pre-annotation runs follow it as well. Dataset versions are frozen copies and keep the names they were made
  with, and so do the results of a finished run: a model trained from an old version after the rename comes out
  with the old name, which the pre-annotation screen marks as a class the project no longer has (make a new
  version). Evaluations and FiftyOne fields made earlier are not rewritten: run them again.
- **Model card** (step 6): short by design. It shows the number of classes (*{n} classes*, with a link to
  see them all) and the actions of the model; *Pre-annotate with this one* (*Pré-anotar com este*) opens
  the *Pre-annotate* tab of *Annotations* with that model chosen. The step does not need images in the
  project (a public model needs none), and an empty step offers its action: save a finished run or add a
  public model. In the dialog of a public model the mapping is summarised as *{n} of {total} project
  classes linked to a model category* (*Review* opens the choices).
- **Comparison**: tick the models to compare (the box on each card, or *Select all* in the bar above the
  list, which says how many are chosen); with two or more, a table shows one row per metric, grouped as
  validation, test, the main per-class metric and size, with the best value of each row highlighted. The
  classes with 0 in every chosen model are folded under one line. When the chosen models were trained on
  different dataset versions (#257) a warning names the versions: their validation split (and fold) differ, so
  the values are not directly comparable. The Training step compares finished runs the same way.
- A saved model whose architecture this version no longer offers is listed as *unavailable*
  (*indisponível*) and cannot pre-annotate (it can still be deleted).
- **Evaluation in FiftyOne** (#611): *Evaluate* (*Avaliar*) on a model card predicts the *test* part of the
  dataset version the model was trained on (the *validation* part when the version has no test part; the
  newest version when that one was deleted or the model is public) with the saved model, and FiftyOne
  compares the predictions with your labels. On the K-fold version it was trained on, a model of one fold is
  evaluated only on the validation part of its own fold (the only images it never saw;
  `evaluation_fold_only_own` otherwise), and a model trained on every fold only on a test part
  (`evaluation_needs_test_part` when the version has none). Public models can be evaluated too. It is a job of the global queue (device lane, cancellable, one
  at a time in a project) with three phases: *environment*, *predicting* (the prediction worker, under a
  name of its own so that cancelling an evaluation never stops a label check, with a low confidence of 0.05
  so that the doubtful predictions count) and *fiftyone*. Only the
  active images of the part that a person reviewed are evaluated (an image reviewed with no objects counts);
  the API can also choose the version, part and fold. The predictions go to the field `eval_m<N>` of the
  project's FiftyOne dataset (`eval_m<N>_pred` for a classification, because FiftyOne's own `eval_m<N>` is
  there the field that says whether the image was right), and FiftyOne runs `evaluate_detections` (with the
  mean average precision; polygons are compared as polylines by their IoU, which needs `shapely`) or
  `evaluate_classifications` under the evaluation key `eval_m<N>`, which also writes `eval_m<N>_tp`, `_fp`
  and `_fn` per image. The evaluation is registered as a FiftyOne run (the App's *Model Evaluation* panel
  lists it; ActiveLabel also records `activelabel_evaluation_m<N>`) and the saved view `evaluation m<N>`
  holds the images evaluated. The card then shows, across the row: the main metrics (detection and
  segmentation: precision, recall, F1, mAP; classification: accuracy and macro F1), the metrics of each
  class, the confusion matrix (the 20 most frequent classes; the others are summed in *(other classes)*; for
  objects the *(none)* row and column are the objects invented and the objects missed), *See the worst
  images* (up to 50: the most false positives and misses, or the wrong classifications the model was most
  sure of), which filters *Images* to them, and *Open in FiftyOne*, which opens the App on the evaluated
  images: there open the *Model Evaluation* panel (the **+** beside the sample tabs) and choose `eval_m<N>`
  to see the matrix and compare two models. Keypoints and regression models cannot be evaluated here
  (`evaluation_task_unsupported`); without FiftyOne the answer is 501 `fiftyone_not_installed`. The state of
  each model is kept in `evaluations.json` (a cancelled or failed evaluation leaves the previous result);
  the labels are never changed. The evaluation compares with the project's labels as they are now, which may
  differ from the version's snapshot if you edited them after freezing it.
- **Deleting** a model (after confirmation) is refused with 409 while an export of it, a
  pre-annotation or an evaluation that uses it is queued or running. Its evaluation goes with it: the entry
  of `evaluations.json` and, in FiftyOne, the fields `eval_m<N>*`, the evaluation, the run and the view. The weights live in the model (they were moved from the
  run), so the run it came from stays in Training with its curves and results but **cannot be saved as a model
  again**: the runs list says `has_weights: false` for it, and its *Save as model* button is disabled with the
  note *The weights of this run went with the deleted model... To have a model, train again* (also right after
  the deletion, before the list is reloaded). Saving such a run through the API answers `run_no_weights`. A
  saved run answers `run_already_saved`; an unfinished or cancelled one `run_not_finished`.
- **Downloads**: the PyTorch weights (`.pt`, a dict with `state_dict`, `classes`, `model`, `family`,
  `image_size`, `resize` and, for classification and segmentation, `mean`/`std`).
- **ONNX export** (classification and semantic segmentation): queued on the CPU lane of the global
  queue; the `onnx` package is installed in the training environment on the first
  export; the network is rebuilt without downloading anything, the weights are loaded and the graph
  is checked (`onnx.checker`). Opset 17, dynamic batch size and the fixed image size of the training.
  Input `images` (N×3×S×S, normalised with the ImageNet mean and std; the photo is resized to S×S outside the
  graph by the training's resize policy, recorded in the file's metadata), output `logits` (classes, or
  classes + background per pixel). Detection, instance and
  keypoint models (torchvision) are not exported yet: their traced graphs break in ONNX Runtime on
  images without candidate boxes; their PyTorch weights can be downloaded.

## Licences

Ultralytics is licensed AGPL-3.0, a copyleft licence that asks more than the other training packages
(PyTorch, torchvision and timm are BSD or Apache; transformers is Apache-2.0): software that uses Ultralytics and is distributed, or
offered to others as a network service, must be offered under the AGPL with its source. Using it privately to
train models does not by itself trigger that, but whoever accepts takes responsibility for how the trained
models and any service built on them are used. Read the licence text before accepting
(<https://www.gnu.org/licenses/agpl-3.0.html>) and ask your organisation when in doubt.

The weights of the `transformers` models have their own licences. DETR and RT-DETR are Apache-2.0; SegFormer
(NVIDIA) and Mask2Former (Meta) are listed as "other" on the Hub: read the licence before commercial use. The
catalogue `notes` of those models say so.

Because of this, ActiveLabel does not install or use Ultralytics until the licence is accepted:

- The acceptance is kept with its date in the workspace settings, `<workspace>/settings.json`, under
  `licences.ultralytics` (`accepted_at`, `name`). Accepting again keeps the first date.
- `GET /api/training/licences` lists the licences and whether each was accepted; `POST
  /api/training/licences/ultralytics/accept` with `{"accept": true}` records the acceptance (without
  `accept: true` it answers 400 `licence_not_confirmed`; an unknown licence 404 `licence_not_found`).
  `GET /api/training/environment?model=<id>` also tells `licence` and `licence_accepted` for that model.
- Creating a run with a YOLO model before the acceptance is refused with 400 `licence_not_accepted` (params
  `licence` and `name`). The environment never installs the package either: `TrainingEnvironment.ensure`
  raises the same error before anything is created or installed, so no other path can bypass the check.
  Models of the other families need no acceptance.

## YOLO runs

**Generations, sizes and tasks (#632).** YOLO26 (proposed first), YOLO11 and YOLOv8, each in sizes n, s, m, l and
x, for detection, instance segmentation (`-seg`), pose (`-pose`) and classification (`-cls`): every one of them has
official pretrained weights in the pinned Ultralytics (release `v8.4.0` of `ultralytics/assets`). YOLO12 is not
offered. Oriented boxes (`-obb`) exist too but wait for a project type of their own. In the catalogue `arch` is
the generation and `variant` the size; the task comes from the project type. Each size has its own batch (n and
s 16, m 8, l and x 4) and the input is 640 pixels (224 for classification). The size proposed first follows the
computer (`default_size` of `GET /api/training/environment`): `n` without a GPU, `s` on a GPU under 8 GB, `m` from
8 GB (the GPU memory is read with `nvidia-smi`). `params_m` is given for YOLO26, YOLO11 and YOLOv8 (Ultralytics'
tables).

A classification model gets one folder per class in each part (named by the class index, so Ultralytics numbers
the classes as the project does) and reports the scores of the classification worker (accuracy, macro F1, per
class, confusion matrix) from Ultralytics' confusion matrix; `mosaic` has no effect on it.

The worker `workers/yolo.py` runs in the training environment, reads the same `spec.json` and writes the same
events, metrics, `best.pt`, `last.pt` and `results.json` as the other workers, so the screens, the metrics
and *Save as model* work as they do for them. The pinned `ultralytics` version is the one of
`PINNED_VERSIONS`. How the run options become Ultralytics options:

- **Data.** The images of the version are linked (or copied) into a temporary Ultralytics dataset with one
  label file each (boxes; polygons for `-seg`; boxes and points for `-pose`), removed when the training ends.
- **Resize.** `letterbox` is what Ultralytics does itself; `stretch` and `center_crop` fit the photos and their
  labels to the square input first, as the other workers do.
- **Start.** `transfer` loads the COCO weights and freezes the backbone (10 layers), `finetune` trains all,
  `scratch` builds the network from its `.yaml`. Ultralytics downloads the pretrained weights to the run's
  temporary folder, and its own telemetry (`sync`) is turned off.
- **Device and precision.** The device the run asked for (`cuda:N` becomes the GPU index); mixed precision on
  CUDA only (`amp`), with the same warning on the CPU. Ultralytics does not halve the batch on its own: a GPU
  out of memory fails with `out_of_memory`.
- **Optimizer, schedule, warmup.** `sgd`, `adam` and `adamw`, `lr`, `weight_decay` and `warmup_epochs` map
  directly. The `none` and `cosine` schedules map to `lrf` and `cos_lr`; `step`, `onecycle` and `plateau` do
  not exist in Ultralytics, so the YOLO models offer only `none` and `cosine` (the catalogue narrows their
  choices; any other value is refused with `param_not_choice`). A run stored before that falls back to the
  linear decay of Ultralytics (warning `yolo_scheduler_unsupported`).
  `backbone_lr_factor` has no counterpart (one learning rate for the whole network), so the YOLO models do not
  offer it (the catalogue's `without`): an older configuration that still has it is accepted and the factor is
  dropped (the worker's `yolo_backbone_lr_ignored` warning remains for a spec written before this).
- **Augmentation.** The resolved techniques become Ultralytics hyperparameters: `hflip` and `vflip`
  (`fliplr`, `flipud`), `rotation` (`degrees`), `scale_jitter` (`scale`), `translate` and `color_jitter`
  (`hsv_h`, `hsv_s`, `hsv_v`; contrast has no counterpart); every other augmentation of Ultralytics is off.
  `blur` and `grayscale` cannot be applied (warning `yolo_technique_unsupported`). Mosaic is the model's own
  `mosaic` and `close_mosaic` options (four images in one; off for the last epochs), not a technique of the
  list, because only Ultralytics can apply it.
- **Selection and patience.** The best epoch is chosen by the run's selection metric (`map50_95` or `map50`
  of the boxes), not by the fitness of Ultralytics: `best.pt` is a copy of the checkpoint of that epoch, and
  patience stops the training by the same metric. Instance runs also report the mask mAP, pose runs the
  keypoint mAP.
- **Resume.** `last.pt` is the checkpoint of Ultralytics (optimizer, EMA, learning rate schedule);
  `yolo_state.json` keeps the best epoch and the digest of the run. A failed or cancelled run resumes after its
  last epoch.
- **Not yet.** Pre-annotation and ONNX export refuse YOLO models; their PyTorch checkpoints can be
  downloaded and opened with Ultralytics.

A smoke test with the real package is `tests/smoke/yolo_smoke.py` (not run by pytest): it trains a tiny model
from scratch and checks the events and files.

## Pinned package versions

A new release of PyTorch or timm must not change results or break a first install (#160), so the training
packages are installed at fixed, tested versions: `PINNED_VERSIONS` in
`src/activelabel/training/environment.py` (torch, torchvision, timm, segmentation-models-pytorch, transformers, onnx and
ultralytics). A package that
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
   `pip index versions torch --index-url https://download.pytorch.org/whl/cu126 --python-version 3.12`.
2. Edit `PINNED_VERSIONS` and, when the CUDA builds of that release changed, `CUDA_TAGS` (newest first).
   Pick timm, onnx and ultralytics versions released after that torch.
3. Open a pull request. The CI job *training workers* installs exactly the pinned versions (read from the
   module) and fails if a training test is skipped; also run one short training on a GPU machine.
4. Note the change in CHANGELOG.md: results of new runs may differ from older ones.

## HTTP API (summary)

| Method | Path | What |
| --- | --- | --- |
| GET | `/api/queue?project=` | the global work queue: running and waiting jobs of every project (or of one), with lane, position and estimates (docs/DESIGN.md, *Work queue*) |
| POST | `/api/queue/{job_id}/cancel` | cancel a waiting or running job of any kind; answers the queue |
| GET | `/api/training/catalog?type=detection` | models and hyperparameter specs |
| GET | `/api/training/augmentations?type=` | augmentation levels and, per kind of task, the techniques with their settings and the list of each preset |
| GET | `/api/training/licences` | the licences to accept before a package is installed (`ultralytics`: AGPL-3.0) and whether each was accepted, with its date |
| POST | `/api/training/licences/{id}/accept` | `{"accept": true}`: record the acceptance in the workspace settings |
| GET | `/api/training/environment` | installed packages, the GPU found and `devices` (the device choices of this computer: `auto`, `cpu`, `cuda:N`, `mps`); with `?model=<id>` also `missing` (packages that model needs) and `install_needed`, which the *New run* dialog uses to say that the first run installs packages |
| GET/POST | `/api/projects/{id}/runs` | list (each run has `model_number`, the saved model made from it, and `has_weights`) / create (queued job): the configuration plus an optional `device` (default `auto`) |
| GET | `/api/projects/{id}/runs/{n}` | config, status, metrics per epoch (of the child training now, while a grid runs), results |
| GET | `/api/projects/{id}/runs/{n}/children/{c}` | one child of a grid / all-folds run |
| GET | `/api/projects/{id}/runs/{n}/log?lines=200` | last lines of the log |
| POST | `/api/projects/{id}/runs/{n}/cancel`, `/reproduce`, `/resume` | cancel (a queued or running run) / new run with the same configuration / resume a failed or cancelled single run from its checkpoint |
| DELETE | `/api/projects/{id}/runs/{n}` | after confirmation |
| GET/POST | `/api/projects/{id}/models` | list (with results and the current model) / save a run (`{run, note}`) |
| GET | `/api/projects/{id}/models/public-catalog` | public pretrained models for the project type, with a suggested mapping |
| POST | `/api/projects/{id}/models/public` | add a public model (`{public, mapping, note}`) |
| GET/PATCH/DELETE | `/api/projects/{id}/models/{n}` | card and results / note, `current` / delete (after confirmation) |
| POST | `/api/projects/{id}/models/{n}/export` | queue an export (`{"format": "onnx"}`) |
| GET/POST | `/api/projects/{id}/models/{n}/evaluation`, `.../evaluation/cancel` | the evaluation in FiftyOne: state and result / queue it (`{device, version?, part?, fold?}`) / cancel (docs/DESIGN.md has the answers and errors) |
| GET | `/api/projects/{id}/models/{n}/download/{pytorch\|onnx}` | the weights or the export as a file |

## Status

Done: catalogue, training environment, runner, YOLO models in the training dialog after the licence is accepted, resume of a run, classification, detection, semantic segmentation,
instance segmentation (Mask R-CNN) and keypoints (torchvision), grid search and all folds, the step 5 screen (Treinamento), and step 6
(Modelos) with save, current model, comparison, download and ONNX export.

Pre-annotation works with classification, detection, keypoint, instance segmentation and semantic
segmentation models (docs/PREANNOTATE.md).

Regression models train (#619); their pre-annotation and ONNX export come next.

Not yet: pre-annotation and ONNX export of YOLO models, ONNX for instance, detection and keypoint models, and
pre-annotation and ONNX export of regression models.

## Image orientation

The workers apply the EXIF orientation of each photo when they load it (training, evaluation and
pre-annotation), so a photo stored rotated is seen upright, as in the gallery and in the annotation tools.
