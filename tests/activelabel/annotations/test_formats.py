import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, folder_classes, parse
from activelabel.labels import task_of
from activelabel.tasks import Task


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


def test_coco_detection_polygon_rle(env):
    folder, images, recs = env
    data = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 50},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
            {
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            },
            {
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 50, 25],
                "segmentation": {"counts": "x", "size": [50, 100]},
            },
            {"image_id": 2, "category_id": 9, "bbox": [0, 0, 5, 5]},
        ],
    }
    path = write(folder, "inst.json", data)
    assert detect_format([path]) == "coco"
    res = parse([path], "coco", images)
    assert res.classes == ["cat", "dog"]
    first = res.labels[recs["a.png"].id]
    assert first[0].bbox == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert first[1].polygons is not None and first[1].bbox == pytest.approx([0, 0, 0.1, 0.2])
    second = res.labels[recs["b.png"].id]
    assert len(second) == 1 and second[0].polygons is None
    codes = {p.code for p in res.problems}
    assert {"rle_bbox_only", "unknown_class_id"} <= codes
    assert task_of(res.all_labels()) is Task.SEGMENTATION


def test_coco_keypoints_and_size_mismatch(env):
    folder, images, recs = env
    data = {
        "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose", "eye", "ear"]}],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 10, 100, 50],
                "keypoints": [100, 50, 2, 0, 0, 0, 40, 20, 1],
            }
        ],
    }
    res = parse([write(folder, "k.json", data)], "coco", images)
    lb = res.labels[recs["a.png"].id][0]
    assert lb.keypoints == [[0.5, 0.5, 2.0], [0.0, 0.0, 0.0], [0.2, 0.2, 1.0]]  # scaled by the file size
    assert lb.bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert task_of(res.all_labels()) is Task.KEYPOINTS
    assert "size_mismatch" in {p.code for p in res.problems}
    data["categories"][0]["keypoints"] = ["nose", "eye"]
    res = parse([write(folder, "k2.json", data)], "coco", images)
    assert "keypoint_count_mismatch" in {p.code for p in res.problems}


def test_coco_invalid_file(env):
    folder, images, _ = env
    res = parse([write(folder, "bad.json", "{not json")], "coco", images)
    assert res.problems[0].code == "invalid_file" and res.problems[0].level == "error"


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


def test_voc(env):
    folder, images, recs = env
    xml = """<annotation><filename>B.PNG</filename><size><width>100</width><height>50</height></size>
    <object><name>cat</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>15</ymax></bndbox></object>
    <object><name>dog</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>10</xmax><ymax>15</ymax></bndbox></object>
    </annotation>"""
    path = write(folder, "b.xml", xml)
    assert detect_format([path]) == "voc"
    res = parse([path], "voc", images)
    assert res.labels[recs["b.png"].id][0].bbox == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert len(res.labels[recs["b.png"].id]) == 1 and "box_degenerate" in {p.code for p in res.problems}


def test_cvat(env):
    folder, images, recs = env
    xml = """<annotations><meta><task><labels><label><name>cat</name></label><label><name>dog</name></label>
    </labels></task></meta>
    <image id="0" name="a.png" width="100" height="50">
      <box label="cat" xtl="10" ytl="5" xbr="30" ybr="15"/>
      <polygon label="dog" points="0,0;50,0;50,25"/>
      <points label="cat" points="10,10;20,20" occluded="0"/>
      <tag label="dog"/>
      <polyline label="dog" points="0,0;5,5"/>
    </image>
    <image id="1" name="zzz.png" width="100" height="50"><tag label="cat"/></image></annotations>"""
    path = write(folder, "annotations.xml", xml)
    assert detect_format([path]) == "cvat"
    res = parse([path], "cvat", images)
    labels = res.labels[recs["a.png"].id]
    assert [lb.label for lb in labels] == ["cat", "dog", "cat", "dog"]
    assert labels[1].polygons == [[[0, 0], [0.5, 0], [0.5, 0.5]]]
    assert labels[2].keypoints == [[0.1, 0.2, 2.0], [0.2, 0.4, 2.0]]
    assert labels[3].bbox is None
    assert res.unmatched == ["zzz.png"] and "unsupported_shape" in {p.code for p in res.problems}


def test_label_studio_json_is_no_longer_recognised(env):
    folder, images, _ = env
    tasks = [{"id": 1, "data": {"image": "/data/upload/1/1a2b3c4d-a.png"}, "annotations": [{"result": []}]}]
    path = write(folder, "export.json", tasks)
    assert detect_format([path]) is None
    with pytest.raises(ValueError, match="Unknown annotation format"):
        parse([path], "labelstudio", images)


def test_folder_tree(tmp_path):
    manifest = make_manifest(tmp_path, ["cat/1.png", "cat/2.png", "dog/3.png", "4.png"])
    assert folder_classes([r.name for r in manifest.all()]) is None  # 1 of 4 images outside a folder
    manifest = make_manifest(tmp_path / "x", ["cat/1.png", "cat/2.png", "dog/3.png"])
    assert folder_classes([r.name for r in manifest.all()]) == {
        "cat/1.png": "cat",
        "cat/2.png": "cat",
        "dog/3.png": "dog",
    }
    assert folder_classes(["a/1.png", "a/2.png"]) is None
    res = parse([], "folders", manifest.all())
    assert res.classes == ["cat", "dog"] and task_of(res.all_labels()) is Task.CLASSIFICATION


def test_matching_case_subfolders_and_ambiguity(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["x/a.png", "y/a.png", "Photo.PNG"])
    recs = {r.name: r for r in manifest.all()}
    data = {
        "images": [
            {"id": 1, "file_name": "x/a.png"},  # exact relative name
            {"id": 2, "file_name": "a.png"},  # ambiguous basename
            {"id": 3, "file_name": "photo.png"},  # case-insensitive
            {"id": 4, "file_name": "x/a.png"},  # duplicate entry
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10]} for i in (1, 2, 3, 4)],
    }
    res = parse([write(tmp_path / "f", "c.json", data)], "coco", manifest.all())
    assert set(res.labels) == {recs["x/a.png"].id, recs["Photo.PNG"].id}
    codes = {p.code: p for p in res.problems}
    assert codes["ambiguous_image_name"].examples == ["a.png"]
    assert "duplicate_image_entry" in codes


def test_detect_none_and_mixed(env):
    folder, _, _ = env
    readme = write(folder, "readme.txt", "hello there")
    coco = write(folder, "c.json", {"images": [], "annotations": []})
    voc = write(folder, "v.xml", "<annotation/>")
    assert detect_format([readme]) is None
    assert detect_format([coco, voc]) is None
    assert detect_format([coco, readme]) == "coco"
    assert detect_format([write(folder, "x.json", {"a": 1})]) is None


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
    from activelabel.annotations.yolo import is_names_file

    assert is_names_file(folder / "classes.txt")
    write(folder, "labels.txt", "0 0.5 0.5 0.2 0.2\n")
    assert not is_names_file(folder / "labels.txt")
    write(folder, "names.txt", "cat\ndog\n")
    assert is_names_file(folder / "names.txt")


def test_cvat_shape_without_label_is_a_problem_not_an_empty_class(env):
    folder, images, recs = env
    xml = """<annotations><image id="0" name="a.png" width="100" height="50">
      <box xtl="10" ytl="5" xbr="30" ybr="15"/><box label="cat" xtl="1" ytl="1" xbr="9" ybr="9"/>
    </image></annotations>"""
    res = parse([write(folder, "a.xml", xml)], "cvat", images)
    assert [lb.label for lb in res.labels[recs["a.png"].id]] == ["cat"] and res.classes == ["cat"]
    problem = next(p for p in res.problems if p.code == "invalid_annotation")
    assert "without a label" in problem.examples[0]
