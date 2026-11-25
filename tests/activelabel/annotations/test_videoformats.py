import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import VideoTarget, detect_format, parse
from activelabel.images.manifest import ImageRecord

W, H = 200, 100
FRAMES = [10, 20, 30, 40]  # frames of the video that the task uses (every 10th)


@pytest.fixture
def target():
    frames = {
        f: ImageRecord(
            f"id{f}", f"clip/frame_{f:06d}.jpg", f"images/{f}.jpg", W, H, 1, "t", video="v1", frame=f
        )
        for f in FRAMES
    }
    return VideoTarget(1, "v1", "clip.mp4", W, H, 10, 40, 10, frames)


def run(files, fmt, target):
    return parse(files, fmt, target.images(), target)


def by_frame(result, target):
    return {f: result.labels.get(rec.id) for f, rec in target.frames.items()}


CVAT = """<annotations><version>1.1</version>
<meta><task><mode>interpolation</mode><start_frame>{start}</start_frame><stop_frame>{stop}</stop_frame>
<frame_filter>{filter}</frame_filter><original_size><width>200</width><height>100</height></original_size>
<labels><label><name>car</name></label><label><name>person</name></label></labels></task></meta>
<track id="7" label="car">
 <box frame="{f0}" outside="0" occluded="0" keyframe="1" xtl="20" ytl="10" xbr="120" ybr="60"/>
 <box frame="{f1}" outside="0" occluded="0" keyframe="0" xtl="40" ytl="10" xbr="140" ybr="60"/>
 <box frame="{f2}" outside="1" occluded="0" keyframe="1" xtl="40" ytl="10" xbr="140" ybr="60"/>
 <box frame="{f9}" outside="0" occluded="0" keyframe="0" xtl="40" ytl="10" xbr="140" ybr="60"/>
</track>
<track id="8" label="person">
 <polygon frame="{f0}" outside="0" occluded="0" keyframe="1" points="0,0;100,0;100,50"/></track>
<image id="{f1}" name="x"><points label="person" occluded="0" points="100,50"/></image>
</annotations>"""


def cvat_file(tmp_path, **kw):
    values = {"start": 10, "stop": 40, "filter": "step=10", "f0": 0, "f1": 1, "f2": 2, "f9": 7}
    values.update(kw)
    return write(tmp_path, "annotations.xml", CVAT.format(**values))


def test_cvat_video_relative_frames_and_tracks(tmp_path, target):
    path = cvat_file(tmp_path)
    assert detect_format([path]) == "cvatvideo"
    res = run([path], "cvatvideo", target)
    got = by_frame(res, target)
    # task frame i -> video frame start + i * step: 0 -> 10, 1 -> 20, 2 -> 30 (outside), 7 -> 80 (not used)
    assert [(lb.label, lb.track) for lb in got[10]] == [("car", 7), ("person", 8)]
    assert got[10][0].bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert got[10][1].polygons is not None
    assert [(lb.label, lb.track) for lb in got[20]] == [("car", 7), ("person", None)]
    assert got[20][1].keypoints is not None
    assert got[30] == [] and got[40] == []  # inside the task range, nothing visible
    problem = next(p for p in res.problems if p.code == "frames_not_in_task")
    assert problem.count == 1 and problem.examples == ["frame 80"]
    assert res.classes[:2] == ["car", "person"] and res.unmatched_count == 0


def test_cvat_video_frames_without_meta_are_video_frames(tmp_path, target):
    xml = (
        '<annotations><track id="1" label="car"><box frame="20" outside="0" occluded="0" keyframe="1" '
        'xtl="0" ytl="0" xbr="100" ybr="50"/><box frame="25" outside="0" occluded="0" keyframe="1" '
        'xtl="0" ytl="0" xbr="100" ybr="50"/></track></annotations>'
    )
    path = write(tmp_path, "v.xml", xml)
    assert detect_format([path]) == "cvatvideo"
    res = run([path], "cvatvideo", target)
    got = by_frame(res, target)
    assert len(got[20]) == 1 and got[20][0].track == 1 and got[10] is None
    assert next(p for p in res.problems if p.code == "frames_not_in_task").examples == ["frame 25"]


def test_cvat_images_xml_is_still_cvat(tmp_path):
    path = write(
        tmp_path, "i.xml", '<annotations><image id="0" name="a.png" width="2" height="2"/></annotations>'
    )
    assert detect_format([path]) == "cvat"


def test_video_format_needs_a_task(tmp_path):
    path = cvat_file(tmp_path)
    with pytest.raises(ValueError, match="video"):
        parse([path], "cvatvideo", [])


MOT = "1,3,20,10,100,50,1,1,1\n2,3,40,10,100,50,1,1,1\n3,4,0,0,50,25,1,2,1\n9,3,0,0,50,25,1,1,1\n"


def test_mot_is_one_based_with_classes_and_tracks(tmp_path, target):
    # task frames 10, 20, 30, 40 are MOT frames 11, 21, 31, 41
    rows = "11,3,20,10,100,50,1,1,1\n21,3,40,10,100,50,1,1,1\n31,4,0,0,50,25,1,2,1\n12,9,0,0,50,25,1,1,1\n"
    gt = write(tmp_path / "gt", "gt.txt", rows)
    names = write(tmp_path / "gt", "labels.txt", "pedestrian\ncar\n")
    assert detect_format([gt, names]) == "mot"
    res = run([gt, names], "mot", target)
    got = by_frame(res, target)
    assert [(lb.label, lb.track) for lb in got[10]] == [("pedestrian", 3)]
    assert got[20][0].bbox == pytest.approx([0.2, 0.1, 0.5, 0.5])
    assert [(lb.label, lb.track) for lb in got[30]] == [("car", 4)]
    assert got[40] is None  # after the last frame of the file: unknown, not background
    problem = next(p for p in res.problems if p.code == "frames_not_in_task")
    assert problem.count == 1 and problem.examples == ["frame 12 (MOT)"]
    assert res.classes == ["pedestrian", "car"]


def test_mot_without_labels_uses_one_class(tmp_path, target):
    gt = write(tmp_path, "det.txt", "11,-1,20,10,100,50,0.9,-1,-1\n")
    res = run([gt], "mot", target)
    assert by_frame(res, target)[10][0].label == "object" and by_frame(res, target)[10][0].track == -1
    target.class_name = "player"
    assert run([gt], "mot", target).classes == ["player"]


def test_mot_is_not_yolo(tmp_path):
    gt = write(tmp_path, "gt.txt", MOT)
    yolo = write(tmp_path, "f.txt", "0 0.5 0.5 0.2 0.2\n")
    assert detect_format([gt]) == "mot"
    assert detect_format([yolo]) == "yolo"
    assert detect_format([gt, yolo]) is None


def test_yolo_per_frame_by_trailing_number(tmp_path, target):
    files = [
        write(tmp_path, "labels/frame_000010.txt", "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.1 0.1\n"),
        write(tmp_path, "labels/000020.txt", "1 0.5 0.5 0.2 0.2\n"),
        write(tmp_path, "labels/video_30.txt", ""),
        write(tmp_path, "labels/frame_000011.txt", "0 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.2 0.2\n"),
        write(tmp_path, "labels/noframe.txt", "0 0.5 0.5 0.2 0.2\n"),
        write(tmp_path, "classes.txt", "car\nperson\n"),
    ]
    assert detect_format(files) == "yolo"
    res = run(files, "yolo", target)
    got = by_frame(res, target)
    assert [lb.label for lb in got[10]] == ["car", "person"]
    assert got[20][0].bbox == pytest.approx([0.4, 0.4, 0.2, 0.2])
    assert got[30] == [] and got[40] is None
    codes = {p.code: p for p in res.problems}
    assert codes["frames_not_in_task"].count == 2  # two objects in frame 11
    assert codes["frame_number_missing"].examples == ["noframe.txt"]


COCO = {
    "images": [
        {"id": 1, "file_name": "a/frame_000010.jpg", "width": W, "height": H},
        {"id": 2, "file_name": "x.jpg", "frame_id": 20},
        {"id": 3, "file_name": "frame_000015.jpg"},
    ],
    "categories": [{"id": 1, "name": "car"}],
    "annotations": [
        {"image_id": 1, "category_id": 1, "bbox": [20, 10, 100, 50], "track_id": 5},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 100, 50], "instance_id": 5},
        {"image_id": 3, "category_id": 1, "bbox": [0, 0, 100, 50]},
        {"image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 50]},
    ],
}


def test_coco_with_frames(tmp_path, target):
    path = write(tmp_path, "vid.json", COCO)
    assert detect_format([path]) == "coco"
    res = run([path], "coco", target)
    got = by_frame(res, target)
    assert got[10][0].track == 5 and got[10][0].bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert got[20][0].track == 5
    assert got[30] is None
    problem = next(p for p in res.problems if p.code == "frames_not_in_task")
    assert problem.count == 2 and problem.examples == ["frame_000015.jpg (frame 15)"]


def test_coco_vid_picks_the_video_by_name(tmp_path, target):
    data = {
        "videos": [{"id": 1, "name": "other.mp4"}, {"id": 2, "name": "clip.mp4"}],
        "images": [
            {"id": 1, "file_name": "o/1.jpg", "frame_id": 10, "video_id": 1},
            {"id": 2, "file_name": "c/1.jpg", "frame_id": 20, "video_id": 2},
        ],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 20, 20]},
        ],
    }
    res = run([write(tmp_path, "vid.json", data)], "coco", target)
    got = by_frame(res, target)
    assert got[10] is None and len(got[20]) == 1


def test_coco_without_target_still_matches_names(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a/frame_000010.jpg"], size=(W, H))
    path = write(tmp_path / "f", "c.json", COCO)
    res = parse([path], "coco", manifest.all())
    assert res.images_matched == 1 and res.unmatched_count == 2


def test_cvat_export_uses_video_frame_numbers(tmp_path, target):
    """CVAT's own export (checked on app.cvat.ai) numbers frames as in the video: 10, 20, 30..."""
    path = cvat_file(tmp_path, f0=10, f1=20, f2=30, f9=40)
    got = by_frame(run([path], "cvatvideo", target), target)
    assert [lb.track for lb in got[10]] == [7, 8] and [lb.label for lb in got[20]] == ["car", "person"]
    assert [lb.label for lb in got[40]] == ["car"]


def test_cvat_video_numbered_frames_cover_the_whole_task_range(tmp_path, target):
    """The last frames without objects are annotated empty, as with task-relative numbers."""
    path = cvat_file(tmp_path, f0=10, f1=20, f2=30, f9=20)  # nothing visible after frame 20
    res = run([path], "cvatvideo", target)
    assert set(res.labels) == {target.frames[f].id for f in FRAMES}
