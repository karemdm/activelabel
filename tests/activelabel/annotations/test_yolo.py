import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, parse
from activelabel.annotations.yolo import is_names_file
from activelabel.labels import task_of
from activelabel.tasks import Task


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


def test_yolo_detection_names_and_folders(env):
    folder, images, recs = env
    files = [
        write(folder, "data.yaml", "names:\n  0: cat\n  1: dog\n"),
        write(folder, "labels/a.txt", "0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.2 0.2\n"),
        write(folder, "labels/sub/c.txt", "1 0.5 0.5 1.4 0.4\n"),
        write(folder, "labels/b.txt", ""),
        write(folder, "labels/d.txt", "0 0.5 0.5 0.2 0.4\n"),
        write(folder, "train.txt", "images/a.png\n"),
    ]
    assert detect_format(files) == "yolo"
    res = parse(files, "yolo", images)
    assert res.classes == ["cat", "dog"]
    assert res.labels[recs["a.png"].id][0].bbox == pytest.approx([0.4, 0.3, 0.2, 0.4])
    assert res.labels[recs["b.png"].id] == []
    assert res.labels[recs["sub/c.png"].id][0].bbox == pytest.approx([0.0, 0.3, 1.0, 0.4])  # clipped
    assert res.unmatched == ["labels/d.txt"]
    assert {"box_clipped", "unmatched_entries"} <= {p.code for p in res.problems}


def test_yolo_segmentation_without_names(env):
    folder, images, recs = env
    files = [write(folder, "labels/a.txt", "3 0.1 0.1 0.5 0.1 0.5 0.5\n2 0.1 0.1 0.5\n")]
    res = parse(files, "yolo", images)
    lb = res.labels[recs["a.png"].id][0]
    assert lb.label == "class_3" and lb.polygons is not None and len(lb.polygons[0]) == 3
    codes = {p.code for p in res.problems}
    assert "missing_class_names" in codes and "polygon_too_few_points" in codes


def test_yolo_pose(env):
    folder, images, recs = env
    files = [
        write(folder, "data.yaml", "names: [person]\nkpt_shape: [2, 3]\n"),
        write(folder, "labels/a.txt", "0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.6 0.6 1\n0 0.5 0.5 0.4 0.4 0.4 0.4 2\n"),
    ]
    res = parse(files, "yolo", images)
    labels = res.labels[recs["a.png"].id]
    assert labels[0].keypoints == [[0.4, 0.4, 2.0], [0.6, 0.6, 1.0]]
    assert task_of(res.all_labels()) is Task.KEYPOINTS
    assert "keypoint_count_mismatch" in {p.code for p in res.problems}


def test_yolo_bad_lines_are_reported_with_file_and_line(env):
    folder, images, _ = env
    write(folder, "data.yaml", "names: [cat, dog]\n")
    write(folder, "a.txt", "0 0.5 0.5 0.2 0.2\nabc\n5 0.5 0.5 0.2 0.2\n1 0.5 0.5 2 2\n")
    files = sorted(folder.iterdir())
    assert detect_format(files) == "yolo"
    result = parse(files, "yolo", images)
    labels = result.labels[next(r.id for r in images if r.name == "a.png")]
    assert len(labels) == 2  # the good line and the clipped box
    problems = {p.code: p for p in result.problems}
    assert problems["invalid_annotation"].examples[0].startswith("a.txt:2: abc")
    assert problems["unknown_class_id"].examples[0].startswith("a.txt:3:")
    assert "a.txt:4:" in problems["box_clipped"].examples[0]


def test_a_label_file_called_labels_txt_is_not_a_names_file(env):
    folder, images, _ = env
    write(folder, "data.yaml", "names: [cat]\n")
    write(folder, "classes.txt", "cat\n")
    write(folder, "a.txt", "0 0.5 0.5 0.2 0.2\n")

    assert is_names_file(folder / "classes.txt")
    write(folder, "labels.txt", "0 0.5 0.5 0.2 0.2\n")
    assert not is_names_file(folder / "labels.txt")
    write(folder, "names.txt", "cat\ndog\n")
    assert is_names_file(folder / "names.txt")


def _pose_line(points: int, visibility: float = 2) -> str:
    values = ["0", "0.5", "0.5", "0.4", "0.4"]
    for i in range(points):
        values += [str(0.3 + i / 100), str(0.3 + i / 100), str(visibility)]
    return " ".join(values) + "\n"


@pytest.mark.parametrize("points", [2, 4, 6])
def test_yolo_pose_without_kpt_shape_with_an_even_point_count(env, points):
    folder, images, recs = env
    files = [write(folder, "data.yaml", "names: [p]\n"), write(folder, "a.txt", _pose_line(points))]
    res = parse(files, "yolo", images)
    (label,) = res.labels[recs["a.png"].id]
    assert label.keypoints is not None and len(label.keypoints) == points and label.polygons is None
    assert "yolo_pose_ambiguous" not in {p.code for p in res.problems}


def test_yolo_ambiguous_line_follows_the_other_lines_and_is_reported(env):
    folder, images, recs = env
    ambiguous = _pose_line(2, 1)  # 11 values, every visibility 0 or 1: a pose or a 5-point polygon
    pose = write(folder, "a.txt", ambiguous + _pose_line(1))  # the 8-value line is clearly a pose
    res = parse([write(folder, "data.yaml", "names: [p]\n"), pose], "yolo", images)
    assert all(lb.keypoints is not None for lb in res.labels[recs["a.png"].id])
    problem = next(p for p in res.problems if p.code == "yolo_pose_ambiguous")
    assert problem.count == 1
    polygon = write(folder, "b.txt", ambiguous + "0 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.95 0.1 0.9 0.2 0.5\n")
    res = parse([write(folder, "data.yaml", "names: [p]\n"), polygon], "yolo", images)
    assert all(lb.polygons is not None for lb in res.labels[recs["b.png"].id])
