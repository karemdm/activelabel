import fiftyone as fo
import numpy as np
import pytest
from fiftyone import core  # noqa: F401

from activelabel.annotate.fo_labels import fingerprint, labels_and_notes, labels_from_fo
from activelabel.fiftyone_bridge import _ground_truth
from activelabel.labels import Label
from activelabel.sync.links import label_hash
from activelabel.tasks import Task


def test_classification_and_classifications():
    assert [x.label for x in labels_from_fo(fo.Classification(label="cat"), Task.CLASSIFICATION)] == ["cat"]
    many = fo.Classifications(classifications=[fo.Classification(label="a"), fo.Classification(label="b")])
    assert [x.label for x in labels_from_fo(many, Task.CLASSIFICATION)] == ["a"]
    assert labels_from_fo(None, Task.CLASSIFICATION) == []
    assert labels_from_fo(fo.Classifications(), Task.CLASSIFICATION) == []


def test_detections_round_and_mask_note():
    dets = fo.Detections(
        detections=[
            fo.Detection(label="cat", bounding_box=[0.1234567891, 0.2, 0.3, 0.4]),
            fo.Detection(label="dog", bounding_box=[0, 0, 0.5, 0.5], mask=np.ones((2, 2), dtype=bool)),
        ]
    )
    labels, notes = labels_and_notes(dets, Task.SEGMENTATION)
    assert labels[0].bbox == [0.123457, 0.2, 0.3, 0.4]
    assert [x.label for x in labels] == ["cat", "dog"] and notes == {"mask_only": 1}


def test_polylines_to_polygons_with_bbox():
    poly = fo.Polyline(label="a", points=[[(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)]], filled=True, closed=True)
    (label,) = labels_from_fo(fo.Polylines(polylines=[poly]), Task.SEGMENTATION)
    assert label.polygons == [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]]
    assert label.bbox == [0.1, 0.2, 0.4, 0.4]


def test_keypoints_nan_is_not_visible():
    nan = float("nan")
    kp = fo.Keypoint(label="p", points=[(0.2, 0.3), (nan, nan), (0.6, 0.7)])
    (label,) = labels_from_fo(fo.Keypoints(keypoints=[kp]), Task.KEYPOINTS)
    assert label.keypoints == [[0.2, 0.3, 2.0], [0.0, 0.0, 0.0], [0.6, 0.7, 2.0]]
    assert label.bbox == [0.2, 0.3, 0.4, 0.4]


def test_unsupported_type():
    with pytest.raises(ValueError):
        labels_from_fo(fo.Segmentation(), Task.DETECTION)


def test_track_is_the_fiftyone_index():
    dets = fo.Detections(
        detections=[
            fo.Detection(label="a", bounding_box=[0, 0, 1, 1], index=3),
            fo.Detection(label="b", bounding_box=[0, 0, 1, 1]),
        ]
    )
    assert [x.track for x in labels_from_fo(dets, Task.DETECTION)] == [3, None]
    poly = fo.Polyline(label="a", points=[[(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)]], index=4, filled=True)
    assert labels_from_fo(fo.Polylines(polylines=[poly]), Task.SEGMENTATION)[0].track == 4
    kp = fo.Keypoint(label="p", points=[(0.2, 0.3)], index=5)
    assert labels_from_fo(fo.Keypoints(keypoints=[kp]), Task.KEYPOINTS)[0].track == 5


def test_track_round_trip_through_fiftyone():
    labels = [Label("a", bbox=[0.1, 0.1, 0.2, 0.2], track=9)]
    back = labels_from_fo(_ground_truth(Task.DETECTION, labels), Task.DETECTION)
    assert back[0].track == 9


def test_keypoint_visibility_and_pose_box_survive_a_round_trip():
    pose = Label(
        "person",
        bbox=[0.1, 0.1, 0.6, 0.8],  # the pose box is larger than the points
        keypoints=[[0.2, 0.3, 2], [0.4, 0.5, 1], [0.0, 0.0, 0], [0.5, 0.6, 2]],
    )
    sent = _ground_truth(Task.KEYPOINTS, [pose])
    (back,) = labels_from_fo(sent, Task.KEYPOINTS)
    assert [p[2] for p in back.keypoints] == [2.0, 1.0, 0.0, 2.0]  # the occluded point stays occluded
    assert back.bbox == pose.bbox

    sent.keypoints[0].points[3] = (0.95, 0.95)  # an edit moves a point outside the old box
    (moved,) = labels_from_fo(sent, Task.KEYPOINTS)
    assert moved.bbox == [0.2, 0.3, 0.75, 0.65]  # the box follows the points again
    plain = fo.Keypoints(keypoints=[fo.Keypoint(label="p", points=[(0.2, 0.3)])])  # older datasets
    assert labels_from_fo(plain, Task.KEYPOINTS)[0].keypoints == [[0.2, 0.3, 2.0]]


def test_fingerprint_ignores_float_noise_and_scores_but_not_edits():
    written = _ground_truth(Task.DETECTION, [Label("cat", bbox=[0.1, 0.2, 0.3, 0.4])])
    same = fo.Detections(
        detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.3, 0.4000001], confidence=0.5)]
    )
    moved = fo.Detections(detections=[fo.Detection(label="cat", bounding_box=[0.15, 0.2, 0.3, 0.4])])
    renamed = fo.Detections(detections=[fo.Detection(label="dog", bounding_box=[0.1, 0.2, 0.3, 0.4])])
    assert fingerprint(written, Task.DETECTION) == fingerprint(same, Task.DETECTION)
    assert fingerprint(written, Task.DETECTION) != fingerprint(moved, Task.DETECTION)
    assert fingerprint(written, Task.DETECTION) != fingerprint(renamed, Task.DETECTION)
    assert fingerprint(None, Task.DETECTION) == fingerprint(fo.Detections(), Task.DETECTION)


def test_polyline_with_the_apps_defaults_is_a_polygon():
    """#457: the App makes new polylines closed=False, filled=False; with 3 points it is a polygon."""
    drawn = fo.Polyline(label="a", points=[[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]], closed=False, filled=False)
    labels, notes = labels_and_notes(fo.Polylines(polylines=[drawn]), Task.SEGMENTATION)
    assert [(lb.label, lb.polygons) for lb in labels] == [("a", [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])]
    assert notes == {}


def test_open_line_without_a_polygon_ring_is_left_out_and_noted():
    line = fo.Polyline(label="a", points=[[(0.1, 0.1), (0.5, 0.1)]], closed=False, filled=False)
    closed = fo.Polyline(label="a", points=[[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]], closed=True)
    labels, notes = labels_and_notes(fo.Polylines(polylines=[line, closed]), Task.SEGMENTATION)
    assert len(labels) == 1 and notes == {"open_polyline": 1}


def test_ring_of_fewer_than_three_points_is_left_out_and_noted():
    short = fo.Polyline(label="a", points=[[(0.2, 0.2), (0.5, 0.5)]], filled=True)
    mixed = fo.Polyline(label="b", points=[[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], [(0.5, 0.5)]], filled=True)
    labels, notes = labels_and_notes(fo.Polylines(polylines=[short, mixed]), Task.SEGMENTATION)
    assert [(lb.label, len(lb.polygons or [])) for lb in labels] == [("b", 1)]
    assert notes == {"polygon_too_few_points": 2}


def test_regression_values_are_read_from_a_dict_and_missing_ones_left_out():
    found = labels_from_fo({"age": 31, "height": float("nan"), "weight": None}, Task.REGRESSION)
    assert found == [Label("age", value=31.0)]
    assert fingerprint({"age": 31.0}, Task.REGRESSION) != fingerprint({"age": 32.0}, Task.REGRESSION)


def test_labels_carry_their_tags_in_every_field_type():
    (cls,) = labels_from_fo(fo.Classification(label="cat", tags=["check"]), Task.CLASSIFICATION)
    assert cls.tags == ["check"]
    many = fo.Classifications(classifications=[fo.Classification(label="a", tags=["x", "x", " "])])
    assert labels_from_fo(many, Task.CLASSIFICATION)[0].tags == ["x"]
    det = fo.Detection(label="cat", bounding_box=[0, 0, 0.5, 0.5], tags=["b", "a"])
    assert labels_from_fo(fo.Detections(detections=[det]), Task.DETECTION)[0].tags == ["b", "a"]
    poly = fo.Polyline(label="a", points=[[(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)]], filled=True, tags=["t"])
    assert labels_from_fo(fo.Polylines(polylines=[poly]), Task.SEGMENTATION)[0].tags == ["t"]
    kp = fo.Keypoint(label="p", points=[(0.2, 0.3)], tags=["k"])
    assert labels_from_fo(fo.Keypoints(keypoints=[kp]), Task.KEYPOINTS)[0].tags == ["k"]


def test_a_label_without_tags_has_none():
    det = fo.Detections(detections=[fo.Detection(label="cat", bounding_box=[0, 0, 0.5, 0.5])])
    assert labels_from_fo(det, Task.DETECTION)[0].tags is None
    assert labels_from_fo(fo.Classification(label="cat"), Task.CLASSIFICATION)[0].tags is None


def test_label_tags_round_trip_through_fiftyone():
    labels = [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2], tags=["check"]), Label("dog", bbox=[0, 0, 0.5, 0.5])]
    assert labels_from_fo(_ground_truth(Task.DETECTION, labels), Task.DETECTION) == labels
    assert labels_from_fo(_ground_truth(Task.CLASSIFICATION, [Label("a", tags=["t"])]), Task.CLASSIFICATION)[
        0
    ].tags == ["t"]


def test_fingerprint_without_tags_is_the_label_hash_and_tags_change_it():
    plain = fo.Detections(detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.3, 0.4])])
    tagged = fo.Detections(
        detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.3, 0.4], tags=["check"])]
    )
    other = fo.Detections(
        detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.3, 0.4], tags=["hard"])]
    )
    assert fingerprint(plain, Task.DETECTION) == label_hash(labels_from_fo(plain, Task.DETECTION))
    assert fingerprint(tagged, Task.DETECTION) != fingerprint(plain, Task.DETECTION)
    assert fingerprint(tagged, Task.DETECTION) != fingerprint(other, Task.DETECTION)
    again = fo.Detections(
        detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.3, 0.4], tags=["check"])]
    )
    assert fingerprint(tagged, Task.DETECTION) == fingerprint(again, Task.DETECTION)
