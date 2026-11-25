from sync_fakes import box

from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.sync import links
from activelabel.sync.links import match_by_name, same_labels
from activelabel.sync.remote import RemoteItem


def record(image_id, name):
    return ImageRecord(image_id, name, f"images/{image_id}.png", 64, 48, 1, "t")


def test_hash_ignores_order_noise_and_score():
    a = [box("cat", 0.1, 0.1, 0.3, 0.3), box("dog", 0.5, 0.5, 0.2, 0.2)]
    b = [box("dog", 0.500004, 0.5, 0.2, 0.2), box("cat", 0.1, 0.100002, 0.3, 0.3)]
    scored = [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3], score=0.9), a[1]]
    assert links.label_hash(a) == links.label_hash(b) == links.label_hash(scored)
    assert len(links.label_hash(a)) == 64


def test_hash_detects_real_changes():
    base = links.label_hash([box("cat")])
    assert links.label_hash([box("dog")]) != base
    assert links.label_hash([box("cat", 0.11)]) != base
    assert links.label_hash([box("cat"), box("cat")]) != base
    assert links.label_hash([]) != base
    assert links.label_hash([]) == links.label_hash([])


def test_hash_covers_polygons_and_keypoints():
    ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]
    poly = Label("cat", polygons=[ring])
    kp = Label("cat", keypoints=[[0.1, 0.2, 2], [0.3, 0.4, 1]])
    assert len({links.label_hash([poly]), links.label_hash([kp]), links.label_hash([Label("cat")])}) == 3
    moved = Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.7]]])
    assert links.label_hash([poly]) != links.label_hash([moved])


def test_store_roundtrip(tmp_path):
    assert links.load_links(tmp_path) == {}
    assert links.get_link(tmp_path, "cvat") is None
    link = links.new_link("http://cvat", "7", "Pets", {"a": {"key": "1:0", "base": "h"}})
    links.save_link(tmp_path, "cvat", link)
    links.save_link(tmp_path, "labelstudio", links.new_link("http://ls", "3", "Pets LS"))
    assert links.get_link(tmp_path, "cvat")["items"]["a"]["key"] == "1:0"
    view = links.public_view("cvat", links.get_link(tmp_path, "cvat"))
    assert view["tool"] == "cvat" and view["remote_id"] == "7" and view["images"] == 1
    assert set(view) == {"tool", "url", "remote_id", "remote_name", "last_sync", "images"}
    assert links.remove_link(tmp_path, "cvat") is True
    assert links.remove_link(tmp_path, "cvat") is False
    assert set(links.load_links(tmp_path)) == {"labelstudio"}


def test_match_by_name_exact_then_base_name():
    records = [record("a", "x/one.png"), record("b", "two.png"), record("c", "three.png")]
    items = [
        RemoteItem("1", "x/one.png", 1, 1),
        RemoteItem("2", "sub/TWO.png", 1, 1),
        RemoteItem("3", "other.png", 1, 1),
    ]
    pairs = links.match_by_name(records, items)
    assert {i: it.key for i, it in pairs.items()} == {"a": "1", "b": "2"}


def test_match_by_name_ambiguous_base_name_is_left_out():
    records = [record("a", "x/one.png"), record("b", "y/one.png")]
    assert links.match_by_name(records, [RemoteItem("1", "z/one.png", 1, 1)]) == {}


def test_match_by_name_accepts_the_stored_file_name():
    record = ImageRecord("6875cde5e28f5618", "000002.jpg", "images/6875cde5e28f5618.jpg", 10, 10, 1, "t")
    item = RemoteItem("1:0", "6875cde5e28f5618.jpg", 10, 10)
    assert match_by_name([record], [item]) == {"6875cde5e28f5618": item}


def test_same_labels_tolerates_round_trip_noise():
    a = [Label("cat", bbox=[0.1, 0.2, 0.3, 0.4]), Label("dog", bbox=[0.5, 0.5, 0.1, 0.1])]
    noisy = [Label("dog", bbox=[0.5008, 0.4995, 0.1, 0.1]), Label("cat", bbox=[0.1001, 0.2, 0.3, 0.4])]
    assert same_labels(a, noisy)
    assert not same_labels(a, [Label("cat", bbox=[0.12, 0.2, 0.3, 0.4]), a[1]])
    assert not same_labels(a, [Label("bird", bbox=[0.1, 0.2, 0.3, 0.4]), a[1]])
    assert not same_labels(a, a[:1])


def test_same_labels_ignores_the_derived_box_of_a_polygon():
    """#589: a COCO box is one pixel wider than the box of its polygon; the polygon decides."""
    ring = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]
    coco = [Label("pessoa", bbox=[0.1, 0.1, 0.303, 0.403], polygons=[ring])]
    from_tool = [Label("pessoa", bbox=[0.1, 0.1, 0.3, 0.4], polygons=[ring])]
    assert same_labels(coco, from_tool)
    moved = [
        Label(
            "pessoa", bbox=[0.1, 0.1, 0.3, 0.4], polygons=[[[0.1, 0.1], [0.4, 0.1], [0.4, 0.6], [0.1, 0.6]]]
        )
    ]
    assert not same_labels(coco, moved)


def test_match_by_name_duplicate_names_are_not_collapsed():
    twin_a, twin_b, other = record("a", "same.png"), record("b", "same.png"), record("c", "other.png")
    items = [RemoteItem("1", "same.png", 1, 1), RemoteItem("2", "other.png", 1, 1)]
    assert {i: it.key for i, it in links.match_by_name([twin_a, twin_b, other], items).items()} == {"c": "2"}
    # two items with one name and one image: nobody is paired by guessing
    twins = [RemoteItem("1", "same.png", 1, 1), RemoteItem("2", "same.png", 1, 1)]
    assert links.match_by_name([record("a", "same.png")], twins) == {}
    # a name that is unique on both sides is still paired
    assert {i: it.key for i, it in links.match_by_name([record("a", "same.png")], twins[:1]).items()} == {
        "a": "1"
    }


def test_save_link_counts_revisions(tmp_path):
    links.save_link(tmp_path, "cvat", links.new_link("u", "1", "n"))
    assert links.get_link(tmp_path, "cvat")["revision"] == 1
    links.save_link(tmp_path, "cvat", links.new_link("u", "1", "n"))
    assert links.get_link(tmp_path, "cvat")["revision"] == 2


def test_carry_predictions_only_for_the_same_untouched_label():
    scored = Label("cat", bbox=[0.1, 0.1, 0.2, 0.2], score=0.9, model="m1")
    other = Label("cat", bbox=[0.6, 0.6, 0.2, 0.2], score=0.7, model="m1")
    new = [
        Label("cat", bbox=[0.1005, 0.1, 0.2, 0.2], track=4),  # same place (track id ignored)
        Label("dog", bbox=[0.1, 0.1, 0.2, 0.2]),  # another class: a person's label
        Label("cat", bbox=[0.3, 0.3, 0.2, 0.2]),  # moved: reviewed
    ]
    out = links.carry_predictions([scored, other], new)
    assert [(lb.score, lb.model) for lb in out] == [(0.9, "m1"), (None, None), (None, None)]
    assert out[0].track == 4 and new[0].score is None  # a copy: the input is not modified
    assert links.carry_predictions([scored], [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])] * 2)[1].score is None


def test_carry_pose_boxes_keeps_the_box_of_a_pose_whose_points_did_not_move() -> None:
    kps = [[0.1, 0.2, 2.0], [0.5, 0.25, 1.0]]
    coco = Label("cat", bbox=[0.05, 0.1, 0.9, 0.8], keypoints=kps)
    read = Label("cat", bbox=[0.1, 0.2, 0.4, 0.05], keypoints=[list(k) for k in kps])
    moved = Label("cat", bbox=[0.1, 0.2, 0.5, 0.05], keypoints=[[0.1, 0.2, 2.0], [0.6, 0.25, 1.0]])
    box = Label("dog", bbox=[0.1, 0.1, 0.2, 0.2])
    out = links.carry_pose_boxes([coco], [read, moved, box])
    assert out[0].bbox == coco.bbox and out[1].bbox == moved.bbox and out[2] is box
    assert links.carry_pose_boxes([coco], [read, read])[1].bbox == read.bbox  # each old pose used once
