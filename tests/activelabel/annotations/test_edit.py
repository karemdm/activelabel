import pytest
from sample_images import make_image

from activelabel.annotations import edit
from activelabel.errors import CodedError
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.projects import ProjectStore
from activelabel.sync import links
from activelabel.tasks import Task

BOX = [0.1, 0.1, 0.3, 0.3]
POSE = [[0.2, 0.2, 2], [0.4, 0.4, 1]]
THREE = [[0.1, 0.1, 2], [0.2, 0.2, 2], [0.3, 0.3, 2]]
RING = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]]


def make(tmp_path, task="detection", classes=("cat", "dog"), count=2):
    """A project of ``task`` with ``count`` images; returns ``(store, project, ids)``."""
    store = ProjectStore(tmp_path / "projects")
    project = store.create("P", "", task, list(classes))
    folder = tmp_path / "src"
    for i in range(count):
        make_image(folder / f"img{i}.png", seed=i + 1, fmt="PNG")
    manifest = Manifest(store.dir(project.id))
    ingest_folder(manifest, folder, folder)
    by_name = {r.name: r.id for r in manifest.all()}
    return store, project, [by_name[f"img{i}.png"] for i in range(count)]


def put(store, project_id, ids, index, labels, add_classes=False, expected=None):
    if expected is None:
        project = store.get(project_id)
        expected = edit.image_labels(project, store.dir(project_id), ids[index])["revision"]
    return edit.save_image_labels(store, project_id, ids[index], labels, expected, add_classes)


def error(store, project_id, ids, labels, index=0, **kw):
    with pytest.raises(CodedError) as caught:
        put(store, project_id, ids, index, labels, **kw)
    return caught.value


def test_revision_changes_with_any_field():
    base = [Label("cat", bbox=BOX)]
    assert edit.revision(base) == edit.revision([Label("cat", bbox=list(BOX))])
    assert len(edit.revision(base)) == 16
    for other in (
        [],
        [Label("dog", bbox=BOX)],
        [Label("cat", bbox=[0.1, 0.1, 0.3, 0.31])],
        [Label("cat", bbox=BOX, score=0.5)],
        [Label("cat", bbox=BOX, track=3)],
    ):
        assert edit.revision(other) != edit.revision(base)


def test_image_labels_with_and_without_labels(tmp_path):
    store, project, ids = make(tmp_path)
    folder = store.dir(project.id)
    empty = edit.image_labels(project, folder, ids[0])
    assert empty == {"labels": [], "classes": ["cat", "dog"], "revision": edit.revision([]), "tags": []}
    saved = put(store, project.id, ids, 0, [Label("cat", bbox=BOX)])
    got = edit.image_labels(project, folder, ids[0])
    assert got["labels"][0]["label"] == "cat" and got["revision"] == saved.revision
    assert edit.image_labels(project, folder, ids[1])["labels"] == []


def test_save_detection_keeps_track_score_model_and_project_spelling(tmp_path):
    store, project, ids = make(tmp_path)
    saved = put(store, project.id, ids, 0, [Label("  CAT ", bbox=BOX, track=4, score=0.7, model="m1")])
    assert saved.classes_added == []
    only = saved.labels[0]
    assert (only.label, only.track, only.score, only.model) == ("cat", 4, 0.7, "m1")
    stored = load_annotations(store.dir(project.id)).images[ids[0]][0]
    assert stored.label == "cat" and stored.track == 4
    assert saved.revision == edit.revision(saved.labels)


def test_save_empty_list_keeps_the_image_listed(tmp_path):
    store, project, ids = make(tmp_path)
    before = {ids[0]: [Label("cat", bbox=BOX)]}
    save_annotations(store.dir(project.id), AnnotationSet(Task.DETECTION, ["cat", "dog"], before, {}))
    saved = put(store, project.id, ids, 0, [])
    assert saved.labels == []
    assert load_annotations(store.dir(project.id)).images[ids[0]] == []


def test_save_segmentation(tmp_path):
    store, project, ids = make(tmp_path, "segmentation")
    assert put(store, project.id, ids, 0, [Label("cat", polygons=RING)]).labels[0].polygons == RING


def test_save_keypoints(tmp_path):
    store, project, ids = make(tmp_path, "keypoints")
    saved = put(store, project.id, ids, 0, [Label("cat", bbox=BOX, keypoints=POSE)])
    assert saved.labels[0].keypoints == POSE


def test_save_classification(tmp_path):
    store, project, ids = make(tmp_path, "classification")
    assert put(store, project.id, ids, 0, [Label("dog")]).labels[0].label == "dog"


def test_stale_revision_is_refused_with_the_current_one(tmp_path):
    store, project, ids = make(tmp_path)
    first = put(store, project.id, ids, 0, [Label("cat", bbox=BOX)])
    exc = error(store, project.id, ids, [Label("dog", bbox=BOX)], expected=edit.revision([]))
    assert exc.code == "labels_changed" and exc.params["revision"] == first.revision


def test_new_class_is_refused_then_added(tmp_path):
    store, project, ids = make(tmp_path)
    news = [Label("bird", bbox=BOX), Label("Fish", bbox=BOX), Label("BIRD", bbox=BOX)]
    exc = error(store, project.id, ids, news)
    assert exc.code == "classes_new" and exc.params == {"classes": ["bird", "Fish"], "count": 2}
    assert store.get(project.id).classes == ["cat", "dog"]
    saved = put(store, project.id, ids, 0, [Label("bird", bbox=BOX), Label("BIRD", bbox=BOX)], True)
    assert saved.classes_added == ["bird"] and [lb.label for lb in saved.labels] == ["bird", "bird"]
    assert store.get(project.id).classes == ["cat", "dog", "bird"]
    assert load_annotations(store.dir(project.id)).classes == ["cat", "dog", "bird"]


def test_invalid_labels_report_count_index_problem(tmp_path):
    store, project, ids = make(tmp_path)
    labels = [Label("cat", bbox=BOX), Label("cat", bbox=[0.1, 0.1, 0, 0.3]), Label(" ", bbox=BOX)]
    exc = error(store, project.id, ids, labels)
    assert exc.code == "labels_invalid"
    assert exc.params["count"] == 2 and exc.params["index"] == 2 and exc.params["problem"] == "box_invalid"
    assert exc.params["problems"] == [
        {"index": 1, "problem": "box_invalid"},
        {"index": 2, "problem": "shape_invalid"},
    ]
    assert load_annotations(store.dir(project.id)) is None


def test_problems_are_capped_at_twenty(tmp_path):
    store, project, ids = make(tmp_path)
    exc = error(store, project.id, ids, [Label("cat", bbox=[0, 0, 0, 0])] * 25)
    assert exc.params["count"] == 25 and len(exc.params["problems"]) == edit.MAX_LISTED


@pytest.mark.parametrize(
    ("task", "label", "problem"),
    [
        (Task.DETECTION, Label("cat", bbox=[0.8, 0.1, 0.4, 0.3]), "outside_image"),
        (Task.DETECTION, Label("cat", bbox=[0.1, 0.1, 0.0, 0.3]), "box_invalid"),
        (Task.DETECTION, Label("cat"), "no_geometry"),
        (Task.DETECTION, Label("cat", bbox=[0.1, 0.1, 0.3]), "shape_invalid"),
        (Task.DETECTION, Label("cat", bbox=[0.1, 0.1, 0.3, float("nan")]), "shape_invalid"),
        (Task.DETECTION, Label("cat", bbox=[0.1, True, 0.3, 0.3]), "shape_invalid"),
        (Task.DETECTION, Label("cat", bbox=BOX, polygons=RING), "geometry_not_for_task"),
        (Task.CLASSIFICATION, Label("cat", bbox=BOX), "geometry_not_for_task"),
        (Task.SEGMENTATION, Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1]]]), "polygon_too_few_points"),
        (Task.SEGMENTATION, Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 1.5]]]), "outside_image"),
        (
            Task.SEGMENTATION,
            Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1, 0.2], [0.5, 0.5]]]),
            "shape_invalid",
        ),
        (Task.SEGMENTATION, Label("cat", polygons="x"), "shape_invalid"),
        (Task.KEYPOINTS, Label("cat", keypoints=[[0.2, 0.2, 3]]), "shape_invalid"),
        (Task.KEYPOINTS, Label("cat", keypoints=[[0.2, 0.2]]), "shape_invalid"),
        (Task.KEYPOINTS, Label("cat", keypoints="x"), "shape_invalid"),
        (Task.KEYPOINTS, Label("cat", keypoints=POSE), "keypoints_count_mismatch"),
        (
            Task.KEYPOINTS,
            Label("cat", keypoints=[[1.5, 0.2, 1], [0.4, 0.4, 1], [0.1, 0.1, 1]]),
            "outside_image",
        ),
        (Task.KEYPOINTS, Label("cat", keypoints=[[1.5, 0.2, 0], [0.4, 0.4, 1], [0.1, 0.1, 1]]), None),
        (Task.DETECTION, Label("cat", bbox=[0.5, 0.5, 0.5 + 1e-9, 0.5]), None),
    ],
)
def test_label_problem(task, label, problem):
    points = 3 if task is Task.KEYPOINTS else None
    assert edit.label_problem(task, label, points) == problem


def test_check_labels_classification_and_empty_name():
    found = edit.check_labels(Task.CLASSIFICATION, [Label("a"), Label("b")], {})
    assert found == [{"index": 1, "problem": "multiple_classes"}]
    assert edit.check_labels(Task.CLASSIFICATION, [Label("")], {}) == [
        {"index": 0, "problem": "shape_invalid"}
    ]
    assert edit.check_labels(Task.DETECTION, [], {}) == []


def test_classification_two_labels_is_refused(tmp_path):
    store, project, ids = make(tmp_path, "classification")
    exc = error(store, project.id, ids, [Label("cat"), Label("dog")])
    assert exc.code == "labels_invalid" and exc.params["problem"] == "multiple_classes"


def test_keypoint_count_follows_the_other_images(tmp_path):
    store, project, ids = make(tmp_path, "keypoints", count=3)
    put(store, project.id, ids, 0, [Label("cat", keypoints=THREE)])
    put(store, project.id, ids, 1, [Label("cat", keypoints=THREE)])
    exc = error(store, project.id, ids, [Label("CAT", keypoints=POSE)], index=2)
    assert exc.params["problem"] == "keypoints_count_mismatch"
    put(
        store,
        project.id,
        ids,
        1,
        [Label("cat", keypoints=THREE)],
        expected=edit.revision([Label("cat", keypoints=THREE)]),
    )
    assert put(store, project.id, ids, 2, [Label("dog", keypoints=POSE)]).labels[0].keypoints == POSE


def test_expected_points_uses_the_new_labels_when_no_other_image_has_the_class():
    mine = [Label("Cat", keypoints=POSE), Label("cat", keypoints=POSE), Label("dog", keypoints=[[0, 0, 1]])]
    images = {"a": [Label("dog", keypoints=THREE)], "me": [Label("cat", keypoints=[[0, 0, 1]])]}
    assert edit._expected_points(images, "me", mine) == {"cat": 2, "dog": 3}


def test_a_save_marks_the_image_in_the_links(tmp_path):
    store, project, ids = make(tmp_path)
    folder = store.dir(project.id)
    link = links.new_link("http://cvat", "7", "P", {ids[0]: {"key": "1:0", "base": links.label_hash([])}})
    links.save_link(folder, "cvat", link)
    saved = put(store, project.id, ids, 0, [Label("cat", bbox=BOX)])
    entry = links.get_link(folder, "cvat")["items"][ids[0]]
    assert entry["edited"] == links.label_hash(saved.labels)
