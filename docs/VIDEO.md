# ActiveLabel — tasks from a video

A task can be created from a **video** instead of images, the way CVAT does it: the video is kept,
and the task uses a selection of its frames.

## Creating a task from a video

*New task* → *Video*:

| Field | Meaning |
| --- | --- |
| Video | one file (MP4, AVI, MOV, MKV, WebM) |
| Start / end | first and last frame used (or time, e.g. `00:01:30`); default: whole video |
| Take one frame every | **N frames** or **X seconds** (seconds are turned into frames with the video's FPS and the result is shown: "every 2 s = every 60 frames at 30 fps") |
| Images per job | as for images (CVAT `segment_size`) |

The form shows, before creating: duration, FPS, resolution, total frames, **how many frames the
task will have**, and a strip of thumbnails at the chosen step.

## Storage

- The original video: `<project>/videos/<video id>.<ext>` and its metadata (FPS, frame count,
  duration, size) in `videos.json`.
- The selected frames are extracted once (JPEG, quality 95) and become the images of the task, in
  frame order, named `<video name>/frame_000120.jpg`. Each image record keeps `video`, `frame`
  (index in the video) and `time` (seconds).
- The task records its source: `{"video": id, "start": 0, "stop": 8999, "step": 30}`.

Frames are images for everything else: annotations, curation, jobs, sync, FiftyOne and, later,
training. Nothing has to know about video formats except the extraction.

## CVAT

The video task is created in CVAT **from the original video** with the same selection
(`start_frame`, `stop_frame`, `frame_filter = "step=N"`, `segment_size`), so CVAT shows it as a
video (playback, tracks, interpolation) and frame *i* of the CVAT task is image *i* of the local
task. The sync works as for image tasks (keys `"<task>:<frame>"`). Tracks drawn in CVAT come back
as per-frame objects that keep the track id, and edits made here go back into the track as
keyframes (docs/SYNC.md, *Tracks*).

A CVAT task made from a video and imported into ActiveLabel (or added in CVAT and brought by a
sync) comes as its frames, downloaded as images in frame order (deleted frames kept as excluded
images, so the jobs still match); the original video is not downloaded.

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

## Not in this step

Overlap between jobs (CVAT `overlap`), audio, and annotating directly on the video timeline in
ActiveLabel (that is what CVAT is for).
