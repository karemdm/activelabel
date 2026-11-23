# ActiveLabel — dataset versions (split)

Step **4 · Divisão dos dados** (data split) of a project. A *dataset version* is a frozen choice of images and labels split
into train / validation / test (or K folds plus a test part). Every training (step 5, docs/TRAINING.md) points to
one version, so it is always known exactly which images and labels it used.

## Creating a version

The person chooses:

| Option | Default | Meaning |
| --- | --- | --- |
| Tasks | all | which tasks of the project go in (active images only) |
| Split | train 70 / val 20 / test 10 | percentages of images, summing to 100; val or test may be 0 |
| K-fold | off | instead of one train/val split: K folds (2–10) of the non-test part; the test part stays apart |
| Background images | off | include active images without annotations (negatives for detection / segmentation); classification never has them |
| Videos | auto | `auto`: a video stays whole unless that makes the proportions unreachable, then it is cut into segments (see below); `whole`: a video always stays in one part |
| Safety gap | 0.5 s | `gap_seconds` (0 or more): train frames this close (in time) to a val/test segment of the same video are left out of the version; 0 keeps them |
| Seed | 42 | the same options and seed always give the same split |
| Note | — | free text kept with the version |

Before creating, a **preview** shows per part: images, background images, objects per class, and
warnings. Nothing is saved until the person confirms.

## Rules of the split

- **Groups stay together** (no leakage): frames of the same video (`ImageRecord.video`) and images
  that curation found to be near duplicates (`duplicate_of` in `curation.json`, linked transitively)
  form one group, and a group goes whole to one part. Without curation analysis the near-duplicate
  rule cannot apply: the preview says so.
- **Videos stay together, unless that breaks the proportions** (`videos: "auto"`, the default). The
  split is first tried with every video whole. If a requested part (val/test, or a fold) ends up
  empty or with less than half of its target share, the videos bigger than the smallest such target
  (and with at least 4 frames) are cut into **contiguous segments** in frame order (`ImageRecord.frame`,
  then `time`): length = frames / 10 rounded up, at least 2 and at most 30 frames, so a video gives
  about 10 segments (more when long). Each segment then behaves like a group (class stratification
  applies). For every cut video, each part other than train gets as many segments as it still needs
  (proportional to its missing images); those are taken **evenly spread over the timeline** (the
  centre of each of the q equal cells of the video, so never adjacent unless q is more than half of
  the segments) and train gets the rest. With K-fold all segments are shared among the folds (and
  test) the same way, neighbouring segments going to different parts. Near-duplicate links between
  frames of the *same* cut video are ignored (consecutive frames are similar by nature); links to
  other videos or images still group. `videos: "whole"` never cuts.
- **Safety gap**: for a cut video, train frames (with K-fold: fold frames) within `gap_seconds` of a
  frame of a val/test (with K-fold: test) segment are left out of the version, so near-identical
  frames are not on both sides. Time is `ImageRecord.time`, or `frame / fps` of the video; if a
  frame of the video has neither, that video has no gap. The distance is inclusive. Between folds
  there is no gap. Left-out frames are counted per video and in `stats.gap`. A long gap on a
  densely sampled video can leave little for train: the preview shows it.
- **Stratified by class**: groups are assigned one by one, rarest classes first, to the part that is
  furthest below its target for that class (iterative stratification for multi-label data), then to
  the part furthest below its image target; ties are broken with the seeded random generator.
  Everything is deterministic for the same options and seed.
  Classification uses the image class; detection/segmentation/keypoints the classes of its objects.
- **Information** `video_split` (not a problem): names a video that was cut, the number of
  segments, the parts that use spread segments and the frames left out as a gap.
- **Warnings** (never errors), each with a `code`:
  - `class_missing`: a class absent from a part that should have images (`part`, `class`);
  - `few_images`: a part with very few images (fewer than 5);
  - `ratio_drift`: a part whose real share is far from the requested one (`expected`, `actual`, in %);
  - `group_too_big`: a group larger than a part's target that moved the proportions (a big video can
    make the real percentages drift: the preview shows the real ones);
  - `no_curation`: curation was not analysed, so the near-duplicate rule could not apply;
  - `video_split`: information, not a problem (see above).
- Excluded images and images of tasks not chosen never enter.
- **Pre-annotations are ignored**: labels with a `score` (made by a model, docs/PREANNOTATE.md) are
  not copied into a version, so an image that only has pre-annotations counts as having no labels
  until a person reviews it.
- **Training needs a validation part**: a version with validation at 0 can be created, but a training
  run on it fails ("make a version with validation"); with K-fold, each fold is a validation part.

Known limitation: with K-fold on split videos, the safety gap only protects the test part. Frames of
neighbouring segments can land in different folds, and no frames are left out between folds.

## Storage

`<project>/datasets/v<N>.json` (N = 1, 2, …, never reused), immutable once written:

```json
{"version": 3, "created": "<iso>", "note": "…", "type": "detection", "classes": ["car", "person"],
 "options": {"tasks": [1, 2], "split": {"train": 70, "val": 20, "test": 10}, "kfold": null,
             "background": false, "seed": 42, "videos": "auto", "gap_seconds": 0.5},
 "parts": {"train": ["<image id>", …], "val": […], "test": […]},
 "folds": null,
 "labels": {"<image id>": [<label>, …]},
 "stats": {"train": {"images": 140, "background": 0, "objects": {"car": 300, "person": 120}}, …},
 "warnings": [{"code": "class_missing", "part": "test", "class": "bike"},
              {"code": "video_split", "video": "<id>", "name": "clip.mp4", "segments": 10,
               "parts": ["val", "test"], "gap": 6}],
 "videos": [{"video": "<id>", "name": "clip.mp4", "frames": 50, "split": true, "gap": 6,
             "segments": [{"part": "train", "first_frame": 0, "last_frame": 4},
                          {"part": "val", "first_frame": 5, "last_frame": 9}, …]}],
 "fingerprint": "<sha256 of the parts, folds and labels>"}
```

With K-fold, `parts` has only `test`, and `folds` is a list of K lists of image ids (fold *i* is the
validation part of run *i*; the other folds are its training part). Its `stats` is `{test, folds: [...]}`
(one entry per fold, as for a part), plus `gap` (frames left out) when a video was split.

`labels` is a copy of the annotations at creation time: editing annotations later does not change a
version. The images themselves are not copied (they never change; an image deleted from the project
makes the version *incomplete*, shown on its card).

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/datasets` | | `{versions: [summary]}` (newest first) |
| POST | `/api/projects/{id}/datasets/preview` | options | `{type, options, sizes, stats, warnings, duplicates_analysed, videos}` (nothing saved) |
| POST | `/api/projects/{id}/datasets` | options + `note` | the new version (summary) |
| GET | `/api/projects/{id}/datasets/{n}` | | the version without the label copy |
| DELETE | `/api/projects/{id}/datasets/{n}` | `{confirm: true}` | `{deleted: true}` |
| GET | `/api/projects/{id}/images?dataset=n&part=train` | | images of a part (also `fold=i`) |

Summary: `{version, created, note, type, classes, options, sizes: {train, val, test} or {test, folds: [...]},
stats, warnings, videos, incomplete}`. Options accepted:
`tasks, split, kfold, background, seed, videos, gap_seconds` (plus `note` on create). Segments of a
folded version carry `{"part": "fold", "fold": n}`.

## Interface

Step 4 shows the versions as cards (number, date, note, sizes per part as a bar, warnings) and a
*New version* form (tasks, split percentages as number inputs (no sliders) or K-fold, background switch, note; under *Advanced*:
seed, videos mode and safety gap) with the live preview, which shows the `video_split` information in
a neutral style (a small neutral badge "video in N segments" on the version card). A version card opens its images per part in the image grid (filter) and can be
deleted after confirmation.

A version holds only image ids and a copy of the labels in the internal format. It is not converted to
any model format: the training runner writes a `spec.json` (image paths and targets) for the worker,
which builds the model's targets in memory (docs/TRAINING.md).
