"""The video task that annotation files of a video are matched to (by frame number)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from activelabel.images.manifest import ACTIVE, ImageRecord, Manifest
from activelabel.images.video import VideoStore
from activelabel.tasks_store import ImageTask, TaskNotFound, TaskStore

DEFAULT_CLASS = "object"


@dataclass
class VideoTarget:
    """A video task seen as frames of its video.

    Attributes:
        task: task id.
        video: video id.
        name: name of the video file.
        width: video width in pixels.
        height: video height in pixels.
        start: first video frame of the task.
        stop: last video frame of the task.
        step: the task takes one frame every ``step`` frames.
        frames: active images of the task by video frame index.
        class_name: class for formats without class names (MOT without a labels file).
    """

    task: int
    video: str
    name: str
    width: int
    height: int
    start: int = 0
    stop: int = 0
    step: int = 1
    frames: dict[int, ImageRecord] = field(default_factory=dict)
    class_name: str = DEFAULT_CLASS

    def images(self) -> list[ImageRecord]:
        """Images of the task in frame order."""
        return [self.frames[f] for f in sorted(self.frames)]


def video_tasks(project_dir: Path) -> list[ImageTask]:
    """Tasks made from a video."""
    return [t for t in TaskStore(project_dir).load() if t.source and t.source.get("video")]


def build_target(
    project_dir: Path, manifest: Manifest, task_id: int | None, class_name: str = ""
) -> VideoTarget | None:
    """Target for a video task; with ``task_id`` None, the only video task of the project (else None).

    Raises:
        ValueError: the task does not exist or is not made from a video.
    """
    if task_id is None:
        tasks = video_tasks(project_dir)
        if len(tasks) != 1:
            return None
        task = tasks[0]
    else:
        try:
            task = TaskStore(project_dir).get(task_id)
        except TaskNotFound:
            raise ValueError(f"Task {task_id} does not exist.") from None
        if not task.source or not task.source.get("video"):
            raise ValueError(f"Task {task_id} was not made from a video.")
    source = task.source or {}
    video = str(source["video"])
    try:
        info = VideoStore(project_dir).get(video)
        name, width, height = info.name, info.width, info.height
    except KeyError:
        name, width, height = video, 0, 0
    records = manifest.load()
    frames = {
        r.frame: r
        for r in (records.get(i) for i in task.images)
        if r is not None and r.status == ACTIVE and r.frame is not None
    }
    if not width and frames:
        first = next(iter(frames.values()))
        width, height = first.width, first.height
    return VideoTarget(
        task=task.id,
        video=video,
        name=name,
        width=width,
        height=height,
        start=int(source.get("start", 0)),
        stop=int(source.get("stop", 0)),
        step=int(source.get("step", 1)) or 1,
        frames=frames,
        class_name=class_name.strip() or DEFAULT_CLASS,
    )
