# ActiveLabel — dataset versions (split)

Step **4 · Dataset** of a project. A *dataset version* is a frozen choice of images and labels split
into train / validation / test (or K folds plus a test part). Every training (a later step) points to
one version, so it is always known exactly which images and labels it used.

## Creating a version

The person chooses:

| Option | Default | Meaning |
| --- | --- | --- |
| Tasks | all | which tasks of the project go in (active images only) |
| Split | train 70 / val 20 / test 10 | percentages of images, summing to 100; val or test may be 0 |
| K-fold | off | instead of one train/val split: K folds (2–10) of the non-test part; the test part stays apart |
| Background images | off | include active images without annotations (negatives for detection / segmentation); classification never has them |
| Seed | 42 | the same options and seed always give the same split |
| Note | — | free text kept with the version |

Before creating, a **preview** shows per part: images, background images, objects per class, and
warnings. Nothing is saved until the person confirms.

## Rules of the split

- **Groups stay together** (no leakage): frames of the same video (`ImageRecord.video`) and images
  that curation found to be near duplicates (`duplicate_of` in `curation.json`, linked transitively)
  form one group, and a group goes whole to one part. Without curation analysis the near-duplicate
  rule cannot apply: the preview says so.
- **Stratified by class**: groups are assigned one by one, rarest classes first, to the part that is
  furthest below its target for that class (iterative stratification for multi-label data), then to
  the part furthest below its image target; ties are broken with the seeded random generator.
  Classification uses the image class; detection/segmentation/keypoints the classes of its objects.
- **Warnings** (never errors): a class missing from a part that should have images; a part with
  very few images (< 5); groups larger than a part's target (a big video can make the real
  percentages drift: the preview shows the real ones).
- Excluded images and images of tasks not chosen never enter.

## Storage

`<project>/datasets/v<N>.json` (N = 1, 2, …, never reused), immutable once written:

```json
{"version": 3, "created": "<iso>", "note": "…", "type": "detection", "classes": ["car", "person"],
 "options": {"tasks": [1, 2], "split": {"train": 70, "val": 20, "test": 10}, "kfold": null,
             "background": false, "seed": 42},
 "parts": {"train": ["<image id>", …], "val": […], "test": […]},
 "folds": null,
 "labels": {"<image id>": [<label>, …]},
 "stats": {"train": {"images": 140, "background": 0, "objects": {"car": 300, "person": 120}}, …},
 "warnings": [{"code": "class_missing", "part": "test", "class": "bike"}],
 "fingerprint": "<sha256 of the parts and labels>"}
```

With K-fold, `parts` has only `test`, and `folds` is a list of K lists of image ids (fold *i* is the
validation part of run *i*; the other folds are its training part).

`labels` is a copy of the annotations at creation time: editing annotations later does not change a
version. The images themselves are not copied (they never change; an image deleted from the project
makes the version *incomplete*, shown on its card).

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/datasets` | | `{versions: [summary]}` (newest first) |
| POST | `/api/projects/{id}/datasets/preview` | options | `{parts/folds sizes, stats, warnings}` (nothing saved) |
| POST | `/api/projects/{id}/datasets` | options + `note` | the new version (summary) |
| GET | `/api/projects/{id}/datasets/{n}` | | the version without the label copy |
| DELETE | `/api/projects/{id}/datasets/{n}` | `{confirm: true}` | `{deleted: true}` |
| GET | `/api/projects/{id}/images?dataset=n&part=train` | | images of a part (also `fold=i`) |

Summary: `{version, created, note, type, options, sizes: {train, val, test} or {test, folds: [...]},
stats, warnings, incomplete}`.

## Interface

Step 4 shows the versions as cards (number, date, note, sizes per part as a bar, warnings) and a
*New version* form (tasks, split sliders/inputs or K-fold, background switch, seed, note) with the
live preview. A version card opens its images per part in the image grid (filter) and can be
deleted after confirmation. Label conversion to a model's format (YOLO, COCO, class folders…) is done
by the training step from a version.
