# ActiveLabel — dataset versions (split)

Step **4 · Data split** (*Divisão dos dados*) of a project. A *dataset version* is a frozen choice of images and labels split
into train / validation / test (or K folds plus a test part). Every training (step 5, docs/TRAINING.md) points to
one version, so it is always known exactly which images and labels it used.

## Creating a version

The person chooses:

| Option | Default | Meaning |
| --- | --- | --- |
| Tasks | all | which tasks of the project go in (active images only) |
| Split | train 70 / val 20 / test 10 | percentages of images, summing to 100; val or test may be 0 |
| K-fold | off | instead of one train/val split: K folds (2–10) of the non-test part; the test part stays apart (its percentage is the only one asked; the rest is shared by the folds). K cannot be larger than the number of groups the folds can hold: the preview says the largest K the data allow (`max_kfold`) and the form limits the field to it |
| Background images | off | include active images without annotations (negatives for detection / segmentation); classification never has them |
| Videos | auto | `auto`: videos stay whole when there are enough of them for the proportions and for every class; otherwise the videos that block it are cut into spaced segments (see below); `whole`: a video always stays in one part |
| Safety gap | on, 0.5 s | `gap_seconds` (0 to 3600): the least gap around the val/test segments of a cut video; it grows per video (see below); 0 keeps every frame. In the form it is a switch (on by default) with a field for the seconds; off sends 0 |
| Seed | 42 | a whole number from 0 to 4294967295; the same options and seed always give the same split |
| Note | — | free text kept with the version |

Before creating, a **preview** shows per part: images, background images, objects per class, and
warnings. Nothing is saved until the person confirms.

## Rules of the split

- **Groups stay together** (no leakage): frames of the same video (`ImageRecord.video`) and images
  that curation found to be near duplicates (`duplicate_of` in `curation.json`, linked transitively)
  form one group, and a group goes whole to one part. Without curation analysis the near-duplicate
  rule cannot apply: the preview says so.
- **Videos: whole when possible, spaced segments otherwise** (`videos: "auto"`, the default). The
  split is first tried with every video whole. It is tried again with some videos cut when a requested
  part (val/test, or a fold) ends up empty or with less than half of its share (the videos bigger than
  the smallest such target are cut), or when a class is missing from a requested part (the videos
  holding such a class are cut); the cut split is kept only if it is better: first no part too small,
  then fewer (part, class) pairs missing. So with many videos they stay whole, with one or two videos
  (or a class found in too few videos) those videos are cut, and short clips that are not needed stay
  whole. Videos with fewer than 4 frames never cut. A cut video is split into **contiguous segments**
  in frame order (`ImageRecord.frame`, then `time`): about `classes / smallest share` segments (so the
  smallest part can hold one segment per class of the video), at least 10 and at most 20, of 2 to 30
  frames. For every cut video, each part other than train gets its share of the video's segments (no
  more than it still needs); slot *k* of *q* aims at the centre of the *k*-th of *q* equal cells of
  the timeline (beginning, middle and end, never one contiguous end). A segment holding the last frames
  of a class train lacks is never taken; a segment holding a class a part still lacks is preferred over
  the aimed position; then segments not next to one already taken. Each chosen segment's part follows
  the class stratification, never the part of a neighbour. If a part still lacks a class that at least
  as many segments as parts hold, it gets one more segment with that class (its share of the video
  grows a little). Train gets the rest. With K-fold all segments are shared among the folds (and test),
  neighbouring segments going to different parts. Near-duplicate links between frames of the *same*
  cut video are ignored for grouping (they size the gap instead); links to other videos or images still
  group. `videos: "whole"` never cuts.
- **Safety gap**: for a cut video, train frames (with K-fold: fold frames) within the video's gap of a
  frame of a val/test (with K-fold: test) segment are left out of the version, and so are val frames
  within the gap of a test segment. The gap of each cut video (`videos[].gap_seconds`) is the largest
  of: `gap_seconds`; one frame interval (median time between consecutive frames, so a sparse video
  still loses its direct neighbours); the longest time between two frames of the video that curation
  found to be near duplicates; but no more than a tenth of the video's duration (unless `gap_seconds`
  asks for more). `gap_seconds: 0` turns the gap off. Time is `ImageRecord.time`, or `frame / fps`; if
  a frame of the video has neither, that video has no gap. The distance is inclusive. Between folds
  there is no gap. Left-out frames are counted per video and in `stats.gap`.
- **Residual leakage**: segments of one video are in training and in validation/test even with the
  gap: the scene, the objects and the light are the same, so the scores measure frames of videos the
  model has seen, not new videos, and tend to be optimistic. Whenever a video is cut the warning
  `video_parts_shared` says so ("Scores may be optimistic: parts of the same video are in training and in validation/test"). To measure
  generalisation to new videos, record more videos and use `videos: "whole"`.
- **Stratified by class**: groups are assigned one by one, rarest classes first. A group goes first
  to a part that has none of its rarest class yet, so a class with at least as many groups as parts is
  in every part (3 images of a class give 1/1/1 with 70/20/10, not 2/1/0: its share then differs from
  the requested one); then to the part that is
  furthest below its target for that class (iterative stratification for multi-label data), then to
  the part furthest below its image target; ties are broken with the seeded random generator.
  Everything is deterministic for the same options and seed.
  Classification uses the image class; detection/segmentation/keypoints the classes of its objects.
- **Information** `video_split` (not a problem): a video that was cut and whose segments went to two or
  more parts is reported with its name, the number of segments, the parts that use spread segments and the
  frames left out as a gap. It is shown in a neutral style, not as a warning.
- **Warnings** (never errors), each with a `code`:
  - `class_missing`: a class absent from a part that should have images (`part`, `class`, plus `available`:
    how many images of the whole split have the class, and `avoidable`: whether that is at least the number of
    parts asked to have images, so the class could have been in every part);
  - `video_parts_shared`: some videos were cut, their segments are on both sides (`videos`: how many);
  - `few_images`: a part with very few images (fewer than 5);
  - `ratio_drift`: a part whose real share is more than 5 percentage points from the requested one
    (`expected`, `actual`, in %);
  - `group_too_big`: a group larger than a part's target when the part's share is also more than 5 points off
    (it only matters when it moved the proportions); `cause` says why the
    group is big: `video` (frames of one video that stay whole) or `duplicates` (near-identical images
    kept together), and the interface words the warning accordingly ("Whole videos stay together: the
    real split is ..." or "Near-identical images stay together: the real split is ...");
  - `labels_without_geometry`, `keypoints_count_mismatch` and `multiple_classes`: labels left out of the
    version, see *Labels that are left out*;
  - `no_curation`: curation was not analysed, so the near-duplicate rule could not apply;
  - `video_split`: information, not a problem (see above).
- Excluded images and images of tasks not chosen never enter.
- **Pre-annotations are ignored**: labels with a `score` (made by a model, docs/PREANNOTATE.md) are
  not copied into a version, so an image that only has pre-annotations counts as having no labels
  until a person reviews it.
- **Labels that are left out.** The version keeps, in its label copy, only labels a model of the project's
  type can use (the same check as the import of annotation files): labels **without the shape of the
  type** (a box project does not take a label with no box, a segmentation project one with no polygon, and
  so on) are left out and counted (`labels_without_geometry`); in a keypoints project the labels whose
  number of points differs from the most common one (`expected`) are left out
  (`keypoints_count_mismatch`); in a classification project an image with more than one class keeps only
  the first (`multiple_classes`). Each warning carries `count` and up to five `examples` (image ids). If
  nothing is usable at all the preview and the creation answer 400 `no_usable_labels`.
- **A version that cannot train is refused.** A hold-out split whose training part or whose validation
  part (when it was asked to have images) ended up empty, because there are too few images or groups of
  images for the proportions, is refused with **409 `dataset_part_empty`** (`params.parts`); send
  `allow_empty: true` with the options to save it anyway. A training percentage of 0 is a 400
  (`split_train_empty`). A version with validation at 0 can still be created on purpose, but a training
  run on it fails ("make a version with validation"); with K-fold, each fold is a validation part. An empty
  test part or fold is not refused and is listed in the preview (`empty_parts`).

Known limitation: with K-fold on split videos, the safety gap only protects the test part. Frames of
neighbouring segments can land in different folds, and no frames are left out between folds.

## Reading the preview

The class table of the preview has one row per class (*Images per class* in classification, *Objects per class*
otherwise) and one column per part or fold. The rarest classes come first: a class missing from a part that
has images goes before all the others, then classes by total count (smallest first), so the classes at risk
are never hidden behind *Show all classes* (the table shows 8 rows until it is opened). Each cell gives the
count and, small beside it, the share of that class that sits in the part (the cells of a row add up to
about 100 %); a 0 in a part that has images is marked with a warning icon and a hidden text for screen readers
("class missing from this part"), not only with a colour. A part that has no images at all is said once as
*{part} ended up with no images* instead of one "class missing" per class.

A "class missing" warning also says how many images have the class and whether it could have been in every
part: *Only N images have it, but it could be in every part: try another seed or other proportions* when
`avoidable` is true, *Only N images have it: it cannot be in every part* when it is false (`available` is
below the number of parts asked to have images). When several classes miss from the same part they are listed
in one warning, with a summary for the group. The drift warnings (`ratio_drift` and `group_too_big`) are shown
as one line with the real split, for example 64/27/9.

## Safety gap switch

In the new-version form (under *Advanced*), dropping the frames at the border between parts of the same video
is a switch, on by default (recommended: **Descartar quadros na fronteira entre partes do mesmo vídeo**), with
a field for the minimum seconds (the gap grows by itself to the frame interval and to the reach of the video's
near-duplicates). With the switch on, the field must be more than 0 and at most 3600. Off sends
`gap_seconds: 0` (no frame is dropped) and the form warns that validation and test may then contain frames
almost identical to training ones, so the metrics will be optimistic. *New version from this one* restores the
switch from the version's options (0 = off).

When parts of a cut video share training and validation/test, the warning `video_parts_shared` ("scores may be
optimistic: parts of the same video are in training and in validation/test", in Portuguese *Métricas podem ser
otimistas...*) is shown prominently
(an alert on the preview and on the version card, not among the small badges), and the preview's video line
says the gap used (`videos[].gap_seconds`, *Gap used: N s*).

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
 "videos": [{"video": "<id>", "name": "clip.mp4", "frames": 50, "split": true, "gap": 6, "gap_seconds": 0.5,
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

### Labels changed after a version

A version keeps its labels on purpose, so training stays reproducible. To make sure nobody trains on old
labels without knowing, the summary of every version carries `stale: {changed_images, added, removed}`:
the usable human labels of the version's images today, compared per image (by hash) with the copy in the
version. `changed_images` counts images whose labels differ, `added` images that had none and have some now,
`removed` images that had some and have none (or left the project). Predictions (labels with a score) do
not count. All zeros means the version is current. The result is cached per version and recomputed only when
`annotations.json` changes.

The version card shows a warning (icon and text, not only colour) with the number of images (the sum of the
three counts): *Labels changed after this version (N images) — create a new version to train with them*
(*Rótulos mudaram depois desta versão...*), and a *New version from this one* button that opens the
new-version form with this version's options (tasks, split or K-fold, background, seed, videos mode and gap).
The *New run* dialog shows a similar warning under the version chosen ("training uses the old labels"). Neither
blocks training. A version that was just created has all zeros.

## HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/projects/{id}/datasets` | | `{versions: [summary]}` (newest first) |
| POST | `/api/projects/{id}/datasets/preview` | options | `{type, options, sizes, stats, warnings, duplicates_analysed, videos, empty_parts, groups, max_kfold}` (nothing saved); `empty_parts`: parts asked to have images that have none (`train`, `val`, `test`, or `fold`); `max_kfold`: the largest K the images allow (at most 10; below 2 K-fold is not possible) |
| POST | `/api/projects/{id}/datasets` | options + `note` + `allow_empty` (default false) | the new version (summary); 409 `dataset_part_empty` when the training or validation part would be empty |
| GET | `/api/projects/{id}/datasets/{n}` | | the version without the label copy |
| DELETE | `/api/projects/{id}/datasets/{n}` | `{confirm: true, confirm_runs?}` | `{deleted: true}`; 400 `confirm_required` without `confirm`; 409 `dataset_in_use` while a queued or running training run uses the version (cancel it first); 409 `dataset_used_by_runs` (with `params.runs`) when finished runs used it, because they could not be repeated afterwards (saved models keep working): send `confirm_runs: true` to delete anyway |
| GET | `/api/projects/{id}/images?dataset=n&part=train` | | images of a part (also `fold=i`) |

Summary: `{version, created, note, type, classes, options, sizes: {train, val, test} or {test, folds: [...]},
stats, warnings, videos, incomplete, stale}`. Options accepted:
`tasks, split, kfold, background, seed, videos, gap_seconds` (plus `note` on create). Segments of a
folded version carry `{"part": "fold", "fold": n}`.

## Errors

Split options are checked with a code the interface translates (400): `not_whole_number` (a decimal or a
boolean where a whole number is needed), `split_percentage_range` (outside 0–100), `split_sum` (without
K-fold the parts must add up to 100), `split_incomplete` (some parts given, others left out so the total is
not 100; `params.parts` names the missing ones), `split_unknown_part`, `split_train_empty`, `kfold_range`
(K outside 2–10), `kfold_test_too_big` (test 100 % with K-fold), `kfold_too_few_groups` (fewer groups of
images than folds), `seed_range` (outside 0–4294967295), `background_not_bool`, `videos_mode`,
`gap_not_number`, `gap_range` (outside 0–3600 s), `tasks_not_list`, `task_not_found`, `nothing_to_split`,
`no_usable_labels`, `note_too_long`; when reading a version's images: `part_unknown`, `fold_required`,
`fold_not_found`, `version_has_no_folds`. Every answer has `{detail, code, params}`. 404 `dataset_not_found`;
409 `dataset_part_empty`, `dataset_in_use` and `dataset_used_by_runs` are described above.

## Interface

Step 4 shows the versions as cards (number, date, project type, note, the requested split such as
"Requested: 70 / 20 / 10 %" or "5 folds + test", the parts as a bar with the real sizes, warnings). The
warnings of a card are **grouped** under one line such as "5 warnings: missing classes (2) · balance (1) · few
images (1) · labels (1)" (groups: missing classes, balance, few images, labels, other), that opens the
list; the badges name the cause (for example "{n} labels without shape", "{n} images with several
classes", "{n} labels with another number of points", "Real split 64/27/9"). `video_parts_shared`, the stale
warning and an *Incomplete* badge (some image of the version was deleted from the project) are shown apart
from the badges.

Each card has buttons to see the images of each part in the image grid (as a filter), *Train with this version*
(*Treinar com esta versão*), which opens the *New run* dialog with the version chosen (it is disabled for an
incomplete version and for one whose training part has no images, with the reason written on the card), and a
delete button.

The *New version* form has: tasks, split percentages as number inputs (no sliders; editing one adjusts the
others so the total stays 100) or K-fold with its test percentage, a background switch (not offered for
classification), a note, and under *Advanced*: seed, videos mode (**Dividir em trechos distantes quando
necessário**, "cut into distant segments when needed" = `auto`; **Manter sempre inteiros**, "always keep whole" =
`whole`) and the safety gap. The live preview below it
shows the part bar, the class table, the warnings and the `video_split` information in a neutral style (a small
neutral badge "video in N segments" on the version card). A form that cannot be sent says why (no task chosen,
a percentage outside 0–100, a total different from 100, a seed that is not a whole number, a gap of 0 with the
switch on, K above the largest the data allow).

When a hold-out version would have no training or no validation images, the preview says so and offers a
checkbox, *Create it anyway (this version cannot be used to train)*, which sends `allow_empty: true`.

Deleting a version asks for confirmation. When training runs that have not finished use it, the server refuses
(`dataset_in_use`) and the interface shows its text; when finished runs used it, the interface asks again
(*Delete version vN anyway?*, naming the runs, "Saved models keep working") and only then deletes it.

A version holds only image ids and a copy of the labels in the internal format. It is not converted to
any model format: the training runner writes a `spec.json` (image paths and targets) for the worker,
which builds the model's targets in memory (docs/TRAINING.md).
