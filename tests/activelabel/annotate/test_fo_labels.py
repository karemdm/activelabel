import fiftyone as fo
import numpy as np
import pytest
from fiftyone import core  # noqa: F401

from activelabel.annotate.fo_labels import fingerprint, labels_and_notes, labels_from_fo
from activelabel.fiftyone_bridge import _ground_truth
from activelabel.labels import Label
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
    poly = fo.Polyline(label="a", points=[[(0.1, 0.2), (0.5, 0.2), (0.5, 0.6)]], index=4)
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
