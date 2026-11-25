# ActiveLabel — tasks from a video

A task can be created from a **video** instead of images, the way CVAT does it: the video is kept,
and the task uses a selection of its frames.

## Creating a task from a video

In the **Images** step, tab *Tasks and jobs* (*Tasks e jobs*): *New task* → *Video*:

| Field | Meaning |
| --- | --- |
| Video | one file (MP4, M4V, AVI, MOV, MKV, WebM; the drop zone lists the most common ones) |
| Start / end | first and last frame used, or a time such as `00:01:30` (the interface turns times into frame numbers; the API takes frame numbers); default: whole video |
| Take one frame every | **N frames** or **X seconds** (seconds are turned into frames with the video's FPS and the result is shown: "every 2 s = every 60 frames at 30 fps") |
| Images per job | as for images (CVAT `segment_size`) |

A task takes at most **20,000 frames** from a video, and the disk must have about twice the estimated
size of the frames free (they are written to a temporary folder first); otherwise the request is refused
(400) with a text that says to take fewer frames. The form shows, before creating: duration, FPS, resolution, total frames, **how many frames the
task will have**, and a strip of thumbnails at the chosen step.

## Storage

- The original video: `<project>/videos/<video id>.<ext>` and its metadata (FPS, frame count,
  duration, size) in `videos.json`.
- The selected frames are extracted once (JPEG, quality 95) and become the images of the task, in
  frame order, named `<video file name without extension>/frame_000120.jpg`. Each image record keeps `video`, `frame`
  (index in the video) and `time` (seconds).
- The task records its source: `{"video": id, "start": 0, "stop": 8999, "step": 30}`.

Frames are images for everything else: annotations, curation, jobs, sync, FiftyOne, dataset versions and
training. Nothing has to know about video formats except the extraction.

## HTTP API

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| POST | `/api/projects/{id}/videos` | multipart `file` (one video; extensions `.mp4 .m4v .avi .mov .mkv .webm`, 400 otherwise) | the video: `{id, name, fps, frames, duration, size, ...}` |
| GET | `/api/projects/{id}/videos` | | `{videos: [...]}` |
| DELETE | `/api/projects/{id}/videos/{video_id}` | | `{deleted}`; 409 while a task uses the video or one is being made |
| GET | `/api/projects/{id}/videos/{video_id}/frame` | `n` (frame number, required), `w` (width in pixels, 16 to 4096, default 320) | JPEG of that frame |
| POST | `/api/projects/{id}/videos/{video_id}/plan` | `{start, stop, step, every_seconds}` | how many frames the selection gives (no extraction) |
| POST | `/api/projects/{id}/tasks/from-video` | `{video, name?, job_size?, start, stop, step, every_seconds}` | job `video_task`; result `{task, frames, planned, stopped_early, step}`; 400 when the selection has more than 20,000 frames or does not fit on the disk |

`start`, `stop` and `step` are **frame numbers** (`stop` empty: the last frame; `step` default 1);
`every_seconds`, when given, replaces `step` (rounded to whole frames with the video's FPS). The task is
created closed.

## CVAT

The video task is created in CVAT **from the original video** with the same selection
(`start_frame`, `stop_frame`, `frame_filter = "step=N"`, `segment_size`), so CVAT shows it as a
video (playback, tracks, interpolation) and frame *i* of the CVAT task is image *i* of the local
task. The sync works as for image tasks (keys `"<task>:<frame>"`). Tracks drawn in CVAT come back
as per-frame objects that keep the track id, and edits made here go back into the track as
keyframes (docs/SYNC.md, *Tracks*).

A CVAT task made from a video and imported into ActiveLabel (or added in CVAT and brought by a
sync) comes as its frames, downloaded as images in frame order (deleted frames kept as excluded
images, so the jobs still match). The original video is also downloaded, from the task backup
(`POST /api/tasks/{id}/backup/export`, the file is `data/<video>` in the zip), kept in the
project's videos and set as the task's source, so the local task is a task from a video: each
image gets its video, frame (`start_frame + i * step`) and time. The frames are not extracted
again. If the video cannot be downloaded, read, or does not fit the task's selection, the task
stays as images and the sync result carries a warning (`warnings`); the import never fails
because of it.

Details found against CVAT: a video task lists one entry in `/data/meta` (the video) while it has
`size` frames, numbered 0..size-1 in the annotations and jobs; jobs of a video overlap by default,
so tasks are created with `overlap = 0`; a task is made of the video with `start_frame`,
`stop_frame` and `frame_filter = "step=N"`, and CVAT works on it in interpolation mode (tracks).

## FiftyOne

The project's FiftyOne dataset stays an image dataset (one sample per frame). Every sample has
`task`, `task_name`, `job` and `position` (1-based, in the task); frames of a video task also have
`video` (id), `video_name`, `frame_number` and `time` (seconds). Each task has a saved view
`task-<id>` (description `Task: <name>`; the id keeps the URL stable when the task is renamed, and
views of deleted tasks are removed on refresh). For a video task the view **groups the frames by
video, ordered by frame number** (`group_by("video", order_by="frame_number")`, FiftyOne dynamic
groups, saved with `save_view`), so the App shows each video as a sequence you can play through,
with the objects and their track ids (`index`). Curation (blur, darkness, near duplicates) works
per frame; near-duplicate detection is especially useful to drop almost identical consecutive frames.

## Loading annotations of a video

Annotation files made for the **whole video** can be loaded into a video task. In the Annotations
step, tab *Import annotations* (*Importar anotações*), with video tasks in the project, choose *For task* (the only video task is preselected;
*Project images* keeps the matching by file name used for images) and upload the files. They go
through the same review (format, matched images, classes, problems) and the same *Import*, with
*Merge* or *Replace*. Labels are matched by **frame number of the video**: a label at frame *f*
goes to the task image whose `frame` is *f*. Labels on frames the task does not use (for example
the files cover every frame but the task takes one in 10) are counted and reported as
"annotations on frames not used by the task" with examples, never as an error. *Replace* only
replaces the images of that task; the labels of other tasks stay. Coordinates are scaled with the
video size.

| Format | Detected as | Frame number | Notes |
| --- | --- | --- | --- |
| CVAT for video 1.1 XML | `cvatvideo` (has `<track>`) | `frame` of each `box`, `polygon`, `points` of a track; `id` of non-tracked `<image>` | see below |
| MOT `gt.txt` / `det.txt` | `mot` (comma separated numbers) | column 1, **1-based** (video frame = number - 1) | `frame,id,left,top,width,height,conf,class,...` |
| YOLO, one `.txt` per frame | `yolo` | trailing integer of the file name: `frame_000120.txt`, `000120.txt`, `video_120.txt` | `data.yaml` or `classes.txt` for names |
| COCO / COCO-VID / TAO JSON | `coco` | `frame_id`, `frame_index` or `frame` of each image, else the trailing integer of `file_name` | one `video_id` is chosen by the video name when the file has several |

- **CVAT for video**: every frame entry of a track is a label at that frame, with the track id
  kept in `Label.track` (it goes back to CVAT as a track, docs/SYNC.md). Entries with
  `outside="1"` are not visible and are skipped. `polyline`, `ellipse`, `mask`, `cuboid` and
  `skeleton` tracks are skipped as unsupported shapes. **Frame numbers**: when `<meta><task>` has
  `start_frame` or `frame_filter`, they are relative to the CVAT task (0 to size-1) and are
  converted to video frames as `start_frame + i * step` (`frame_filter = "step=N"`); without
  them they are taken as video frame indexes. `original_size` in the meta gives the size for the
  coordinates. A CVAT task made from the already extracted frames (not from the video) has frames
  0..size-1 and no start or step in the meta: they are read as video frames, so use the
  *Project images* option (names) or export from the video task instead.
- **MOT**: `conf` and visibility are ignored; the id is the track. Class ids are 1-based: with a
  labels file (`labels.txt` / `classes.txt`, one name per line, line 1 = class 1; or a `data.yaml`
  with `names`) the class column gives the name, an id without a name is skipped; **without** a
  labels file every box gets one class: the `class_name` sent with the upload, or `object`.
- **YOLO** and **COCO** work as for images except for the matching. COCO `track_id` or
  `instance_id` becomes the track id. A COCO image without any frame number is matched by file
  name. YOLO files without a number at the end of the name are reported and skipped.
- Images between the first and the last annotated frame (CVAT: the task range `start_frame`..`stop_frame`
  in the meta, whether the file numbers its frames by task or by video) that
  have no label count as annotated without objects; YOLO and COCO only count the frames they list.
- The project's only video task is used by default for CVAT video and MOT files; when there are
  several, one must be chosen (`task` in `POST /projects/{id}/annotations`). COCO and YOLO use
  the frame matching only when a task is chosen.

## Not in this step

Overlap between jobs (CVAT `overlap`), audio, and annotating directly on the video timeline in
ActiveLabel (that is what CVAT is for).
