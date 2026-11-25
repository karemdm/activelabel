import pytest
from ann_helpers import make_manifest, write

from activelabel.annotations import detect_format, parse


@pytest.fixture
def env(tmp_path):
    manifest = make_manifest(tmp_path / "p", ["a.png", "b.png", "sub/c.png"])
    recs = {r.name: r for r in manifest.all()}
    return tmp_path / "files", manifest.all(), recs


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


def test_cvat_shape_without_label_is_a_problem_not_an_empty_class(env):
    folder, images, recs = env
    xml = """<annotations><image id="0" name="a.png" width="100" height="50">
      <box xtl="10" ytl="5" xbr="30" ybr="15"/><box label="cat" xtl="1" ytl="1" xbr="9" ybr="9"/>
    </image></annotations>"""
    res = parse([write(folder, "a.xml", xml)], "cvat", images)
    assert [lb.label for lb in res.labels[recs["a.png"].id]] == ["cat"] and res.classes == ["cat"]
    problem = next(p for p in res.problems if p.code == "invalid_annotation")
    assert "without a label" in problem.examples[0]


def test_voc_without_filename_uses_the_xml_name(env):
    folder, images, recs = env
    xml = (
        "<annotation><object><name>cat</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax>"
        "<ymax>15</ymax></bndbox></object></annotation>"
    )
    res = parse([write(folder, "b.xml", xml)], "voc", images)
    assert len(res.labels[recs["b.png"].id]) == 1 and res.unmatched_count == 0
