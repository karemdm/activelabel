import pytest
from ann_helpers import make_manifest

from activelabel.annotations.target import DEFAULT_CLASS, VideoTarget, build_target, video_tasks
from activelabel.images.manifest import ImageRecord


def test_images_come_in_frame_order():
    frames = {
        f: ImageRecord(f"i{f}", f"{f}.png", f"images/{f}.png", 10, 10, 1, "t", video="v", frame=f)
        for f in (30, 10, 20)
    }
    target = VideoTarget(1, "v", "clip.mp4", 10, 10, frames=frames)
    assert [r.frame for r in target.images()] == [10, 20, 30]
    assert target.class_name == DEFAULT_CLASS


def test_a_project_without_video_tasks_has_no_target(tmp_path):
    manifest = make_manifest(tmp_path, ["a.png"])
    assert video_tasks(tmp_path) == []
    assert build_target(tmp_path, manifest, None) is None
    with pytest.raises(ValueError, match="does not exist"):
        build_target(tmp_path, manifest, 99)
