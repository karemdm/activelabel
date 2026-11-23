from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations, task_of
from activelabel.tasks import Task


def test_task_of():
    assert task_of([]) is None
    assert task_of([Label("cat")]) is Task.CLASSIFICATION
    assert task_of([Label("cat", bbox=[0, 0, 1, 1])]) is Task.DETECTION
    assert (
        task_of([Label("cat", bbox=[0, 0, 1, 1], polygons=[[[0, 0], [1, 0], [1, 1]]])]) is Task.SEGMENTATION
    )
    assert task_of([Label("p", bbox=[0, 0, 1, 1], keypoints=[[0.5, 0.5, 2]])]) is Task.KEYPOINTS


def test_round_trip(tmp_path):
    assert load_annotations(tmp_path) is None
    data = AnnotationSet(
        Task.DETECTION, ["cat"], {"a": [Label("cat", bbox=[0.1, 0.2, 0.3, 0.4])]}, {"format": "coco"}
    )
    save_annotations(tmp_path, data)
    loaded = load_annotations(tmp_path)
    assert loaded is not None
    assert loaded.images["a"][0].bbox == [0.1, 0.2, 0.3, 0.4]
    assert loaded.source["format"] == "coco" and "saved" in loaded.source
    assert loaded.images["a"][0].as_dict() == {"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4]}


def test_track_in_dict_round_trip():
    label = Label("cat", bbox=[0, 0, 1, 1], track=7)
    assert label.as_dict()["track"] == 7 and Label.from_dict(label.as_dict()) == label
    assert "track" not in Label("cat").as_dict() and Label.from_dict({"label": "cat"}).track is None
