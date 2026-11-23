import pytest
from make_video import make_video

from activelabel.images.video import VideoInfo, plan, probe, resolve_selection


def _info(frames=60, fps=30.0):
    return VideoInfo("x", "v.mp4", fps, frames, frames / fps, 64, 48, 1)


def test_plan_counts():
    assert plan(0, 59, 1) == {"frames": 60, "step": 1, "first": 0, "last": 59}
    assert plan(0, 59, 7) == {"frames": 9, "step": 7, "first": 0, "last": 56}
    assert plan(10, 10, 5) == {"frames": 1, "step": 5, "first": 10, "last": 10}


def test_selection_every_seconds_and_validation():
    assert resolve_selection(_info(), 0, None, 1, 0.5) == (0, 59, 15)
    assert resolve_selection(_info(), 0, None, 1, 0.001)[2] == 1
    for bad in [(-1, None, 1), (5, 4, 1), (0, 60, 1), (0, None, 0)]:
        with pytest.raises(ValueError):
            resolve_selection(_info(), *bad)
    with pytest.raises(ValueError):
        resolve_selection(_info(), 0, None, 1, 0)


def test_probe_synthetic_video(tmp_path):
    fps, frames, width, height = probe(make_video(tmp_path / "a.mp4"))
    assert (round(fps), frames, width, height) == (30, 60, 64, 48)


def test_probe_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe(bad)
