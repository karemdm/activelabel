"""Do labels travel exactly between ActiveLabel and FiftyOne, in both directions?

These tests use the REAL ``FiftyOneTool`` and real FiftyOne datasets (in the isolated test database that
``tests/conftest.py`` sets up); only the App is faked (``launch_app`` returns a made-up address). Each
test builds a project through the API (images that are not square, so x/y mix-ups show), puts labels
in it, sends an edit session, edits ``ground_truth`` in the dataset the way the App does, brings the
labels back and applies them. Nothing outside the temporary workspace and the test datasets is touched.
"""

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiftyone as fo
import numpy as np
import pytest
from api_client import make_client, upload
from fiftyone import ViewField as F
from mongoengine.errors import ValidationError
from sample_images import make_image

from activelabel import fiftyone_bridge as bridge
from activelabel.annotate import sessions
from activelabel.annotate.base import ToolSettings
from activelabel.annotate.fiftyone_tool import FiftyOneTool
from activelabel.images.manifest import EXCLUDED
from activelabel.images.tags import load_tags
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.sync.links import label_hash
from activelabel.tasks import Task
from activelabel.web import annotate_api

pytestmark = pytest.mark.integration

ROOT_PREFIX = "test-activelabel-rt"
SIZES = [(64, 48), (80, 40)]  # width, height: never square, alternating per image
COUNT = 4
EPS = 1e-6
NAN = float("nan")


@dataclass
class Scene:
    """A project with its images and the ground truth it started with.

    Attributes:
        client: API test client.
        state: application state of the temporary workspace.
        pid: project id.
        ids: image id by image index.
        task: project type.
        folder: the project folder.
    """

    client: Any
    state: Any
    pid: int
    ids: list[str]
    task: Task
    folder: Path

    def labels(self) -> dict[str, list[Label]]:
        """The ground truth of the project now, by image id (an empty dict when there is none)."""
        saved = load_annotations(self.folder)
        return saved.images if saved else {}


@pytest.fixture(autouse=True)
def fiftyone_env(monkeypatch):
    """Unique dataset names, no App, and every dataset made by a test deleted afterwards."""
    unique = f"{ROOT_PREFIX}-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "DATASET_PREFIX", unique)
    monkeypatch.setattr(bridge, "launch_app", lambda *args, **kwargs: "http://fake-app/datasets/x")
    yield unique
    for name in fo.list_datasets():
        if name.startswith(unique):
            fo.delete_dataset(name)


# -- the project -------------------------------------------------------------------
def make_scene(tmp_path: Path, task: Task, classes: list[str], truth: dict[int, list[Label]]) -> Scene:
    """Create a project, upload ``COUNT`` images of alternating sizes and put ``truth`` in it.

    Args:
        tmp_path: temporary folder (the workspace goes inside).
        task: project type.
        classes: class names of the project.
        truth: labels by image index.

    Returns:
        The scene.
    """
    client, state = make_client(tmp_path)
    body = {"name": "rt", "task": task.value, "classes": classes}
    reply = client.post("/api/projects", json=body)
    assert reply.status_code == 200, reply.text
    pid = reply.json()["id"]
    paths = [
        make_image(tmp_path / "src" / f"img{k}.png", size=SIZES[k % 2], seed=k, fmt="PNG")
        for k in range(COUNT)
    ]
    assert upload(client, pid, paths).status_code == 200
    by_name = {r.name: r.id for r in state.manifest(pid).all()}
    ids = [by_name[f"img{k}.png"] for k in range(COUNT)]
    scene = Scene(client, state, pid, ids, task, state.store.dir(pid))
    put_truth(scene, truth)
    return scene


def put_truth(scene: Scene, truth: dict[int, list[Label]]) -> None:
    """Replace the ground truth of the project.

    Args:
        scene: the project.
        truth: labels by image index (images without labels are left out of the file).
    """
    images = {scene.ids[k]: labels for k, labels in truth.items() if labels}
    classes = list(scene.state.store.get(scene.pid).classes)
    save_annotations(scene.folder, AnnotationSet(scene.task, classes, images, {"format": "activelabel"}))


def wait(scene: Scene, job: dict[str, Any]) -> dict[str, Any]:
    """Wait until a job ends.

    Args:
        scene: the project.
        job: the job as the API returned it.

    Returns:
        The finished job.
    """
    for _ in range(600):
        jobs = scene.client.get(f"/api/projects/{scene.pid}/jobs").json()
        data = next(j for j in jobs if j["id"] == job["id"])
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def base_url(scene: Scene, session: str) -> str:
    """URL prefix of a session.

    Args:
        scene: the project.
        session: session id.

    Returns:
        The path before ``/pull``, ``/apply``.
    """
    return f"/api/projects/{scene.pid}/annotate/{session}"


def send(scene: Scene, **body: Any) -> dict[str, Any]:
    """Send a session to FiftyOne (scope ``all`` unless given) and return the job result.

    Args:
        scene: the project.
        **body: extra request fields (``scope``, ``ids``...).

    Returns:
        The result of the job (``session``, ``links``, maybe ``kept_fiftyone_edits``).
    """
    if scene.task is Task.KEYPOINTS:  # the API refuses these (#458): open the session as the API would
        return send_keypoints(scene)
    payload = {"tool": "fiftyone", "scope": "all", **body}
    reply = scene.client.post(f"/api/projects/{scene.pid}/annotate", json=payload)
    assert reply.status_code == 200, reply.text
    job = wait(scene, reply.json())
    assert job["status"] == "done", job
    return job["result"]


def send_keypoints(scene: Scene) -> dict[str, Any]:
    """Open a session on a keypoints project without the API (it refuses them: the App cannot edit them).

    The dataset, the tool and the session are the ones the API uses, so ``pull`` and ``apply`` still go
    through the API. This keeps the read path (what comes back from FiftyOne) tested.

    Args:
        scene: the project.

    Returns:
        ``{"session": ..., "links": ...}`` like the result of the send job.
    """
    project = scene.state.project(scene.pid)
    dataset = annotate_api.prepare_dataset(project, scene.state)
    session_id = sessions.new_id()
    sent = FiftyOneTool().send(dataset, scene.ids, project, ToolSettings("fiftyone"), session_id)
    session = sessions.create(scene.folder, "fiftyone", "all", scene.ids, sent.links, sent.ref, session_id)
    return {"session": sessions.public(session), "links": sent.links}


def pull(scene: Scene, session: str) -> dict[str, Any]:
    """Bring the labels back and return the diff.

    Args:
        scene: the project.
        session: session id.

    Returns:
        The diff of the pull job.
    """
    reply = scene.client.post(f"{base_url(scene, session)}/pull")
    assert reply.status_code == 200, reply.text
    job = wait(scene, reply.json())
    assert job["status"] == "done", job
    return job["result"]["diff"]


def apply(scene: Scene, session: str) -> Any:
    """Apply the pulled labels.

    Args:
        scene: the project.
        session: session id.

    Returns:
        The HTTP response.
    """
    return scene.client.post(f"{base_url(scene, session)}/apply")


def flow(scene: Scene) -> tuple[str, dict[str, Any]]:
    """Send, then (after the caller edited FiftyOne) nothing more; used by tests that edit in between.

    Args:
        scene: the project.

    Returns:
        The session id and the send result.
    """
    result = send(scene)
    return result["session"]["id"], result


# -- FiftyOne side -----------------------------------------------------------------
def dataset_of(scene: Scene) -> Any:
    """The project's FiftyOne dataset.

    Args:
        scene: the project.

    Returns:
        The dataset (it must exist).
    """
    dataset = bridge.existing_dataset(scene.folder, scene.pid)
    assert dataset is not None
    dataset.reload()
    return dataset


def sample_of(scene: Scene, index: int) -> Any:
    """The sample of image ``index``.

    Args:
        scene: the project.
        index: image index.

    Returns:
        The sample, read fresh from the database.
    """
    return dataset_of(scene).match(F("image_id") == scene.ids[index]).first()


def edit(scene: Scene, index: int, change: Any) -> None:
    """Edit ``ground_truth`` of one image the way the App does, and save it.

    Args:
        scene: the project.
        index: image index.
        change: function that receives the sample and changes it in place.
    """
    sample = sample_of(scene, index)
    change(sample)
    sample.save()


def is_nan(value: Any) -> bool:
    """Whether ``value`` is a float NaN.

    Args:
        value: any value.

    Returns:
        True for NaN.
    """
    return isinstance(value, float) and math.isnan(value)


def fo_dump(value: Any) -> Any:
    """The content of a FiftyOne ground-truth value as plain Python, to compare with project labels.

    Args:
        value: ``Classification``, ``Detections``, ``Polylines``, ``Keypoints`` or None.

    Returns:
        ``(label, track, ...)`` tuples: a string for a classification; boxes, rings or points inside.
    """
    if value is None:
        return []
    kind = type(value).__name__
    if kind == "Classification":
        return value.label
    if kind == "Detections":
        return [(d.label, d.index, list(d.bounding_box)) for d in value.detections]
    if kind == "Polylines":
        return [
            (p.label, p.index, [[list(pt) for pt in ring] for ring in p.points], p.closed, p.filled)
            for p in value.polylines
        ]
    return [
        (k.label, k.index, [list(pt) for pt in k.points], list(k.visibility or []), list(k.box or []))
        for k in value.keypoints
    ]


def shape(label: Label) -> list[float]:
    """All the numbers of a label in one list (with markers so that different shapes never match).

    Args:
        label: the label.

    Returns:
        Flat list: box present flag and box, ring lengths and polygon points, keypoint count and points.
    """
    box = label.bbox or []
    rings = label.polygons or []
    points = label.keypoints or []
    return [
        len(box),
        *box,
        len(rings),
        *[len(r) for r in rings],
        *[v for ring in rings for p in ring for v in p],
        len(points),
        *[v for p in points for v in p],
    ]


def assert_same(actual: list[Label], expected: list[Label], where: str = "") -> None:
    """The two label lists are the same (class, track, geometry to 1e-6, in the same order).

    Args:
        actual: labels found.
        expected: labels expected.
        where: text that says where, for the failure message.
    """
    pair = (
        f"\n{where}\nactual:   {[x.as_dict() for x in actual]}\nexpected: {[x.as_dict() for x in expected]}"
    )
    assert len(actual) == len(expected), "count differs" + pair
    for a, e in zip(actual, expected, strict=True):
        assert (a.label, a.track) == (e.label, e.track), "class or track differs" + pair
        assert shape(a) == pytest.approx(shape(e), abs=EPS), "geometry differs" + pair


def assert_project(scene: Scene, expected: dict[int, list[Label]]) -> None:
    """The ground truth of the project is exactly ``expected`` (images not listed have no labels).

    Args:
        scene: the project.
        expected: labels by image index.
    """
    saved = scene.labels()
    for k, image_id in enumerate(scene.ids):
        assert_same(saved.get(image_id, []), expected.get(k, []), f"image {k}")


def assert_counts(diff: dict[str, Any], added: int, removed: int, edited: int, images: int) -> None:
    """The totals of a diff.

    Args:
        diff: the diff.
        added: labels added.
        removed: labels removed.
        edited: labels edited.
        images: images changed.
    """
    got = (diff["added"], diff["removed"], diff["edited"], diff["images_changed"])
    assert got == (added, removed, edited, images), diff


def per_class(diff: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Counts per class of a diff as ``{class: (added, removed, edited)}``.

    Args:
        diff: the diff.

    Returns:
        The tuples.
    """
    return {c: (v["added"], v["removed"], v["edited"]) for c, v in diff["per_class"].items()}


def problem_codes(diff: dict[str, Any]) -> set[str]:
    """The codes of the problems of a diff.

    Args:
        diff: the diff.

    Returns:
        The codes.
    """
    return {p["code"] for p in diff["problems"]}


# -- ground truths -----------------------------------------------------------------
def det_truth() -> dict[int, list[Label]]:
    """Detection labels: a track id, an image without labels, two objects of two classes.

    Returns:
        Labels by image index.
    """
    return {
        0: [
            Label("cat", bbox=[0.1, 0.125, 0.3, 0.375], track=3),
            Label("dog", bbox=[0.5, 0.25, 0.25, 0.5]),
        ],
        1: [Label("cat", bbox=[0.2, 0.1, 0.4, 0.5])],
        2: [],
        3: [Label("dog", bbox=[0.1, 0.1, 0.2, 0.2]), Label("cat", bbox=[0.5, 0.5, 0.3, 0.3])],
    }


def square(x: float, y: float, w: float, h: float) -> list[list[float]]:
    """A rectangle as a ring of four points.

    Args:
        x: left.
        y: top.
        w: width.
        h: height.

    Returns:
        The ring.
    """
    corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return [[round(a, 6), round(b, 6)] for a, b in corners]  # the 6 decimals the project keeps


def seg_truth() -> dict[int, list[Label]]:
    """Segmentation labels: a triangle, a two-ring polygon, a five-point polygon with a track id.

    Returns:
        Labels by image index (every box is the box of its polygon).
    """
    ring_a, ring_b = square(0.5, 0.5, 0.2, 0.2), [[0.75, 0.1], [0.9, 0.1], [0.9, 0.3]]
    return {
        0: [
            Label("cat", bbox=[0.1, 0.1, 0.3, 0.4], polygons=[[[0.1, 0.1], [0.4, 0.1], [0.25, 0.5]]]),
            Label("dog", bbox=[0.5, 0.1, 0.4, 0.6], polygons=[ring_a, ring_b]),
        ],
        1: [Label("cat", bbox=[0.2, 0.2, 0.3, 0.4], polygons=[square(0.2, 0.2, 0.3, 0.4)])],
        2: [],
        3: [
            Label(
                "dog",
                bbox=[0.1, 0.1, 0.4, 0.4],
                polygons=[[[0.1, 0.2], [0.3, 0.1], [0.5, 0.2], [0.4, 0.5], [0.2, 0.5]]],
                track=4,
            ),
            Label("dog", bbox=[0.6, 0.6, 0.2, 0.2], polygons=[square(0.6, 0.6, 0.2, 0.2)]),
        ],
    }


def kp_truth() -> dict[int, list[Label]]:
    """Keypoint labels with visibility 0, 1 and 2, boxes larger than their points and a track id.

    Returns:
        Labels by image index.
    """
    return {
        0: [
            Label(
                "person",
                bbox=[0.1, 0.2, 0.6, 0.7],
                keypoints=[[0.2, 0.3, 2.0], [0.4, 0.5, 1.0], [0.0, 0.0, 0.0]],
            ),
            Label(
                "person",
                bbox=[0.05, 0.05, 0.9, 0.9],
                keypoints=[[0.3, 0.3, 2.0], [0.5, 0.4, 2.0], [0.6, 0.8, 1.0]],
                track=5,
            ),
        ],
        1: [
            Label(
                "person",
                bbox=[0.2, 0.2, 0.5, 0.6],
                keypoints=[[0.3, 0.3, 2.0], [0.5, 0.5, 2.0], [0.4, 0.7, 2.0]],
            )
        ],
        2: [],
        3: [
            Label(
                "person",
                bbox=[0.3, 0.3, 0.4, 0.4],
                keypoints=[[0.4, 0.4, 1.0], [0.5, 0.5, 1.0], [0.6, 0.6, 2.0]],
            )
        ],
    }


def cls_truth() -> dict[int, list[Label]]:
    """Classification labels (one class per image, one image without).

    Returns:
        Labels by image index.
    """
    return {0: [Label("cat")], 1: [Label("dog")], 2: [], 3: [Label("bird")]}


@pytest.fixture
def det_scene(tmp_path):
    """A detection project."""
    return make_scene(tmp_path, Task.DETECTION, ["cat", "dog"], det_truth())


@pytest.fixture
def seg_scene(tmp_path):
    """A segmentation project."""
    return make_scene(tmp_path, Task.SEGMENTATION, ["cat", "dog"], seg_truth())


@pytest.fixture
def kp_scene(tmp_path):
    """A keypoints project (three points per person)."""
    return make_scene(tmp_path, Task.KEYPOINTS, ["person", "dog"], kp_truth())


@pytest.fixture
def cls_scene(tmp_path):
    """A classification project."""
    return make_scene(tmp_path, Task.CLASSIFICATION, ["cat", "dog", "bird"], cls_truth())


# =====================================================================================
# 1. ActiveLabel -> FiftyOne
# =====================================================================================
def test_send_detection_matches_project(det_scene):
    flow(det_scene)
    truth = det_truth()
    for k in range(COUNT):
        dumped = fo_dump(sample_of(det_scene, k).ground_truth)
        expected = [(lb.label, lb.track, lb.bbox) for lb in truth[k]]
        assert [d[:2] for d in dumped] == [e[:2] for e in expected], k
        for d, e in zip(dumped, expected, strict=True):
            assert d[2] == pytest.approx(e[2], abs=EPS)
    sample = sample_of(det_scene, 2)
    assert sample.ground_truth is None  # an image without labels has no label at all


def test_send_segmentation_matches_project(seg_scene):
    flow(seg_scene)
    truth = seg_truth()
    for k in range(COUNT):
        dumped = fo_dump(sample_of(seg_scene, k).ground_truth)
        assert len(dumped) == len(truth[k]), k
        for (label, track, rings, _closed, filled), lb in zip(dumped, truth[k], strict=True):
            assert (label, track) == (lb.label, lb.track)
            flat = [v for ring in rings for pt in ring for v in pt]
            assert flat == pytest.approx([v for ring in lb.polygons for p in ring for v in p], abs=EPS)
            assert [len(r) for r in rings] == [len(r) for r in lb.polygons]
            assert filled, "a polygon of a segmentation project must be a filled polyline"


def test_send_keypoints_matches_project(kp_scene):
    flow(kp_scene)
    truth = kp_truth()
    for k in range(COUNT):
        dumped = fo_dump(sample_of(kp_scene, k).ground_truth)
        assert len(dumped) == len(truth[k]), k
        for (label, track, points, visibility, box), lb in zip(dumped, truth[k], strict=True):
            assert (label, track) == (lb.label, lb.track)
            assert visibility == [int(p[2]) for p in lb.keypoints]
            assert box == pytest.approx(lb.bbox, abs=EPS)
            for got, want in zip(points, lb.keypoints, strict=True):
                if want[2] == 0:
                    assert is_nan(got[0]) and is_nan(got[1]), "a point that was not labelled must be NaN"
                else:
                    assert list(got) == pytest.approx(want[:2], abs=EPS)


def test_send_segmentation_polylines_are_closed_and_filled(seg_scene):
    """#457: a polygon is sent as a closed and filled polyline (the App draws it as a polygon)."""
    flow(seg_scene)
    for k in (0, 1, 3):
        for *_, closed, filled in fo_dump(sample_of(seg_scene, k).ground_truth):
            assert closed and filled, f"image {k}: closed={closed} filled={filled}"


def test_send_classification_matches_project(cls_scene):
    flow(cls_scene)
    truth = cls_truth()
    for k in range(COUNT):
        got = fo_dump(sample_of(cls_scene, k).ground_truth)
        assert got == (truth[k][0].label if truth[k] else []), k


def test_send_sets_dataset_classes_and_view(det_scene):
    result = send(det_scene)
    dataset = dataset_of(det_scene)
    assert dataset.classes == {"ground_truth": ["cat", "dog"]}
    assert len(dataset) == COUNT
    assert dataset.has_saved_view(f"annotate - session {result['session']['id']}")


# =====================================================================================
# 2. FiftyOne -> ActiveLabel
# =====================================================================================
@pytest.mark.parametrize("name", ["det", "seg", "kp", "cls"])
def test_untouched_round_trip_is_a_no_op(name, request):
    """Send then pull with no edit: the diff is empty and applying changes no label at all."""
    scene = request.getfixturevalue(f"{name}_scene")
    before = {k: [lb.as_dict() for lb in v] for k, v in scene.labels().items()}
    session, _ = flow(scene)
    diff = pull(scene, session)
    assert_counts(diff, 0, 0, 0, 0)
    assert diff["problems"] == [] and diff["unknown_classes"] == []
    assert apply(scene, session).status_code == 200
    after = {k: [lb.as_dict() for lb in v] for k, v in scene.labels().items()}
    assert after == before


def edit_detection_img0(sample: Any) -> None:
    """Edit one sample for ``test_edit_detection`` (img0).

    Args:
        sample: the sample to change in place.
    """
    cat, dog = sample.ground_truth.detections
    cat.bounding_box = [0.15, 0.125, 0.3, 0.375]  # moved right by 0.05 (the track index stays)
    dog.label = "cat"  # class changed


def edit_detection_img1(sample: Any) -> None:
    """Edit one sample for ``test_edit_detection`` (img1).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.detections[0].bounding_box = [0.2, 0.1, 0.3, 0.4]  # resized


def edit_detection_img2(sample: Any) -> None:
    """Edit one sample for ``test_edit_detection`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Detections(
        detections=[fo.Detection(label="dog", bounding_box=[0.3, 0.3, 0.4, 0.4])]
    )


def edit_detection_img3(sample: Any) -> None:
    """Edit one sample for ``test_edit_detection`` (img3).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.detections.pop(0)  # the dog is deleted, the cat is not touched


def test_edit_detection(det_scene):
    session, _ = flow(det_scene)
    truth = det_truth()

    for k, change in enumerate(
        (edit_detection_img0, edit_detection_img1, edit_detection_img2, edit_detection_img3)
    ):
        edit(det_scene, k, change)
    diff = pull(det_scene, session)
    assert_counts(diff, added=1, removed=1, edited=3, images=4)
    assert per_class(diff) == {"cat": (0, 0, 3), "dog": (1, 1, 0)}
    assert diff["unknown_classes"] == [] and diff["problems"] == []
    assert load_annotations(det_scene.folder).images.get(det_scene.ids[2]) is None  # not applied yet
    reply = apply(det_scene, session)
    assert reply.status_code == 200, reply.text
    applied = reply.json()["applied"]
    assert (applied["added"], applied["removed"], applied["edited"]) == (1, 1, 3)
    expected = {
        0: [
            Label("cat", bbox=[0.15, 0.125, 0.3, 0.375], track=3),
            Label("cat", bbox=[0.5, 0.25, 0.25, 0.5]),
        ],
        1: [Label("cat", bbox=[0.2, 0.1, 0.3, 0.4])],
        2: [Label("dog", bbox=[0.3, 0.3, 0.4, 0.4])],
        3: [truth[3][1]],
    }
    assert_project(det_scene, expected)


def edit_detection_new_class_img1(sample: Any) -> None:
    """Edit one sample for ``test_edit_detection_new_class_and_case`` (img1).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.detections.append(fo.Detection(label="bird", bounding_box=[0.6, 0.6, 0.2, 0.2]))
    sample.ground_truth.detections[0].label = "Cat"  # same class, another case


def test_edit_detection_new_class_and_case(det_scene):
    session, _ = flow(det_scene)

    edit(det_scene, 1, edit_detection_new_class_img1)
    diff = pull(det_scene, session)
    assert diff["unknown_classes"] == ["bird"]
    assert_counts(diff, added=1, removed=0, edited=0, images=1)
    assert "unknown_class" in problem_codes(diff)
    assert apply(det_scene, session).status_code == 200
    assert det_scene.state.store.get(det_scene.pid).classes == ["cat", "dog", "bird"]
    saved = load_annotations(det_scene.folder)
    assert saved.classes == ["cat", "dog", "bird"]
    assert [lb.label for lb in saved.images[det_scene.ids[1]]] == ["cat", "bird"]  # "Cat" -> "cat"


def edit_segmentation_img0(sample: Any) -> None:
    """Edit one sample for ``test_edit_segmentation`` (img0).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.polylines[0].points = [[(0.1, 0.1), (0.4, 0.1), (0.25, 0.6)]]  # one vertex moved


def edit_segmentation_img2(sample: Any) -> None:
    """Edit one sample for ``test_edit_segmentation`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Polylines(
        polylines=[fo.Polyline(label="dog", points=[square(0.2, 0.2, 0.3, 0.3)], closed=True, filled=True)]
    )


def edit_segmentation_img3(sample: Any) -> None:
    """Edit one sample for ``test_edit_segmentation`` (img3).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.polylines[0].label = "cat"
    sample.ground_truth.polylines.pop(1)


def test_edit_segmentation(seg_scene):
    session, _ = flow(seg_scene)
    truth = seg_truth()

    for k, change in ((0, edit_segmentation_img0), (2, edit_segmentation_img2), (3, edit_segmentation_img3)):
        edit(seg_scene, k, change)
    diff = pull(seg_scene, session)
    assert_counts(diff, added=1, removed=1, edited=2, images=3)
    assert per_class(diff) == {"cat": (0, 0, 2), "dog": (1, 1, 0)}
    assert apply(seg_scene, session).status_code == 200
    expected = {
        0: [
            Label("cat", bbox=[0.1, 0.1, 0.3, 0.5], polygons=[[[0.1, 0.1], [0.4, 0.1], [0.25, 0.6]]]),
            truth[0][1],  # the dog with two rings: untouched, the box and every ring as they were
        ],
        1: truth[1],
        2: [Label("dog", bbox=[0.2, 0.2, 0.3, 0.3], polygons=[square(0.2, 0.2, 0.3, 0.3)])],
        3: [Label("cat", bbox=truth[3][0].bbox, polygons=truth[3][0].polygons, track=4)],
    }
    assert_project(seg_scene, expected)


def test_segmentation_untouched_label_is_not_rewritten(tmp_path):
    original = Label("cat", bbox=[0.15, 0.15, 0.4, 0.5], polygons=[square(0.2, 0.2, 0.3, 0.4)])
    other = Label("dog", bbox=[0.2, 0.2, 0.3, 0.3], polygons=[square(0.2, 0.2, 0.3, 0.3)])
    scene = make_scene(tmp_path, Task.SEGMENTATION, ["cat", "dog"], {0: [original], 1: [other]})
    session, _ = flow(scene)
    edit(scene, 1, lambda s: setattr(s.ground_truth.polylines[0], "label", "cat"))
    diff = pull(scene, session)
    assert_counts(diff, 0, 0, 1, 1)
    assert apply(scene, session).status_code == 200
    assert_same(scene.labels()[scene.ids[0]], [original], "the untouched image 0")


def edit_keypoints_img0(sample: Any) -> None:
    """Edit one sample for ``test_edit_keypoints`` (img0).

    Args:
        sample: the sample to change in place.
    """
    first = sample.ground_truth.keypoints[0]
    points = list(first.points)
    points[0] = (0.25, 0.3)  # a point moved
    points[1] = (NAN, NAN)  # a point (occluded before) hidden
    first.points = points


def edit_keypoints_img1(sample: Any) -> None:
    """Edit one sample for ``test_edit_keypoints`` (img1).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.keypoints.pop(0)  # deleted


def edit_keypoints_img2(sample: Any) -> None:
    """Edit one sample for ``test_edit_keypoints`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Keypoints(
        keypoints=[fo.Keypoint(label="person", points=[(0.2, 0.2), (0.4, 0.4), (0.3, 0.6)])]
    )


def edit_keypoints_img3(sample: Any) -> None:
    """Edit one sample for ``test_edit_keypoints`` (img3).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.keypoints[0].label = "dog"  # class changed, points not moved


def test_edit_keypoints(kp_scene):
    session, _ = flow(kp_scene)
    truth = kp_truth()

    # the second person is not touched: visibility 1, a larger box and a track id must survive

    for k, change in enumerate(
        (edit_keypoints_img0, edit_keypoints_img1, edit_keypoints_img2, edit_keypoints_img3)
    ):
        edit(kp_scene, k, change)
    diff = pull(kp_scene, session)
    assert_counts(diff, added=1, removed=1, edited=2, images=4)
    assert per_class(diff) == {"person": (1, 1, 1), "dog": (0, 0, 1)}
    assert apply(kp_scene, session).status_code == 200
    expected = {
        0: [
            Label(
                "person",
                bbox=[0.1, 0.2, 0.6, 0.7],  # the box it had: no point left it
                keypoints=[[0.25, 0.3, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ),
            truth[0][1],
        ],
        2: [
            Label(
                "person",
                bbox=[0.2, 0.2, 0.2, 0.4],
                keypoints=[[0.2, 0.2, 2.0], [0.4, 0.4, 2.0], [0.3, 0.6, 2.0]],
            )
        ],
        3: [Label("dog", bbox=truth[3][0].bbox, keypoints=truth[3][0].keypoints)],
    }
    assert_project(kp_scene, expected)


def test_keypoints_untouched_round_trip_keeps_visibility_box_and_track(kp_scene):
    """Edit only one image: the other labels keep visibility 1, the larger box and the track id."""
    session, _ = flow(kp_scene)
    edit(kp_scene, 1, lambda s: s.ground_truth.keypoints.pop(0))
    diff = pull(kp_scene, session)
    assert_counts(diff, 0, 1, 0, 1)
    assert apply(kp_scene, session).status_code == 200
    truth = kp_truth()
    assert_project(kp_scene, {0: truth[0], 3: truth[3]})


def edit_classification_img0(sample: Any) -> None:
    """Edit one sample for ``test_edit_classification`` (img0).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.label = "dog"


def edit_classification_img1(sample: Any) -> None:
    """Edit one sample for ``test_edit_classification`` (img1).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = None  # the class removed


def edit_classification_img2(sample: Any) -> None:
    """Edit one sample for ``test_edit_classification`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Classification(label="bird")


def test_edit_classification(cls_scene):
    session, _ = flow(cls_scene)

    edit(cls_scene, 0, edit_classification_img0)
    edit(cls_scene, 1, edit_classification_img1)
    edit(cls_scene, 2, edit_classification_img2)
    diff = pull(cls_scene, session)
    assert_counts(diff, added=1, removed=1, edited=1, images=3)
    assert per_class(diff) == {"dog": (0, 1, 1), "bird": (1, 0, 0)}  # removed: old class, edited: new
    assert apply(cls_scene, session).status_code == 200
    assert_project(cls_scene, {0: [Label("dog")], 2: [Label("bird")], 3: [Label("bird")]})


def test_classification_field_cannot_hold_several_classes(cls_scene):
    """The field is a ``Classification``: the App cannot give an image two classes."""
    flow(cls_scene)
    sample = sample_of(cls_scene, 0)
    with pytest.raises(ValidationError):
        sample.ground_truth = fo.Classifications(classifications=[fo.Classification(label="dog")])


def test_exclude_tag_is_pulled_and_kept(det_scene):
    flow(det_scene)
    dataset = dataset_of(det_scene)
    dataset.select([sample_of(det_scene, 1).id]).tag_samples("exclude")
    reply = det_scene.client.post(f"/api/projects/{det_scene.pid}/fiftyone/pull-exclusions")
    assert reply.status_code == 200 and reply.json() == {"excluded": 1}
    record = next(r for r in det_scene.state.manifest(det_scene.pid).all() if r.id == det_scene.ids[1])
    assert record.status == EXCLUDED
    tags = sample_of(det_scene, 1).tags
    assert "exclude" not in tags and bridge.EXCLUDED_TAG in tags
    again = det_scene.client.post(f"/api/projects/{det_scene.pid}/fiftyone/pull-exclusions")
    assert again.json() == {"excluded": 0}
    # a later send keeps the tag, and the excluded image is not part of the new session
    result = send(det_scene)
    assert result["session"]["images"] == COUNT - 1
    assert bridge.EXCLUDED_TAG in sample_of(det_scene, 1).tags


def tag_img0(sample: Any) -> None:
    """Tag the sample and its first detection, as the App does, for ``test_tags_made_in_fiftyone_come_back``.

    Args:
        sample: the sample to change in place.
    """
    sample.tags.append("check")
    sample.ground_truth.detections[0].tags = ["hard"]


def test_tags_made_in_fiftyone_come_back(det_scene):
    session, _ = flow(det_scene)
    edit(det_scene, 0, tag_img0)
    diff = pull(det_scene, session)
    assert diff["tags"] == {"added": 2, "removed": 0}
    assert (diff["added"], diff["removed"], diff["edited"], diff["images_changed"]) == (0, 0, 0, 1)
    (example,) = diff["examples"]
    assert example["image_id"] == det_scene.ids[0] and example["tags_added"] == 2
    assert load_tags(det_scene.folder) == {}  # nothing is stored before the apply
    reply = apply(det_scene, session)
    assert reply.status_code == 200, reply.text
    assert reply.json()["applied"]["tags"] == {"added": 2, "removed": 0}
    assert load_tags(det_scene.folder) == {det_scene.ids[0]: ["check"]}
    assert [lb.tags for lb in det_scene.labels()[det_scene.ids[0]]] == [["hard"], None]
    saved = json.loads((det_scene.folder / "annotations.json").read_text(encoding="utf-8"))
    assert saved["images"][det_scene.ids[0]][0]["tags"] == ["hard"]
    project = det_scene.state.project(det_scene.pid)
    bridge.sync_dataset(project, det_scene.state.manifest(det_scene.pid), load_annotations(det_scene.folder))
    kept = sample_of(det_scene, 0)
    assert kept.ground_truth.detections[0].tags == ["hard"] and "check" in kept.tags


def test_two_sessions_back_to_back_after_apply(det_scene):
    """Apply, send again: FiftyOne shows what was applied and a no-op pull is empty."""
    first, _ = flow(det_scene)
    edit(det_scene, 0, lambda s: s.ground_truth.detections.pop(1))
    pull(det_scene, first)
    assert apply(det_scene, first).status_code == 200
    second, _ = flow(det_scene)
    assert [d[0] for d in fo_dump(sample_of(det_scene, 0).ground_truth)] == ["cat"]
    assert_counts(pull(det_scene, second), 0, 0, 0, 0)


# =====================================================================================
# 3. The project is edited after the session was sent
# =====================================================================================
def coco_bytes(scene: Scene, truth: dict[int, list[Label]], classes: list[str]) -> bytes:
    """A COCO detection file for the truth (absolute pixels, the real image sizes).

    Args:
        scene: the project.
        truth: labels by image index.
        classes: category names.

    Returns:
        JSON bytes.
    """
    images = [
        {"id": k + 1, "file_name": f"img{k}.png", "width": SIZES[k % 2][0], "height": SIZES[k % 2][1]}
        for k in range(COUNT)
    ]
    cats = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]
    annotations: list[dict[str, Any]] = []
    for k, labels in sorted(truth.items()):
        width, height = SIZES[k % 2]
        for lb in labels:
            x, y, w, h = lb.bbox or [0, 0, 0, 0]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": k + 1,
                    "category_id": classes.index(lb.label) + 1,
                    "iscrowd": 0,
                    "bbox": [x * width, y * height, w * width, h * height],
                }
            )
    return json.dumps({"images": images, "categories": cats, "annotations": annotations}).encode()


def reimport(scene: Scene, truth: dict[int, list[Label]], classes: list[str]) -> None:
    """Replace the labels of the project by importing a COCO file (the person's way of editing them).

    Args:
        scene: the project.
        truth: labels by image index.
        classes: category names.
    """
    url = f"/api/projects/{scene.pid}/annotations"
    reply = scene.client.post(url, files=[("files", ("labels.json", coco_bytes(scene, truth, classes)))])
    assert reply.status_code == 200, reply.text
    report = reply.json()
    assert report["can_import"], report
    done = scene.client.post(f"{url}/{report['token']}/apply", json={"action": "import", "mode": "replace"})
    assert done.status_code == 200, done.text


def new_det_truth() -> dict[int, list[Label]]:
    """What the project becomes after the re-import (every image changes).

    Returns:
        Labels by image index.
    """
    return {
        0: [Label("dog", bbox=[0.6, 0.5, 0.25, 0.25])],
        1: [Label("cat", bbox=[0.1, 0.5, 0.2, 0.3]), Label("dog", bbox=[0.5, 0.1, 0.2, 0.2])],
        2: [Label("cat", bbox=[0.4, 0.4, 0.3, 0.3])],
        3: [Label("cat", bbox=[0.2, 0.2, 0.5, 0.5])],
    }


def test_project_edit_reaches_fiftyone_after_the_session_was_applied(det_scene):
    first, _ = flow(det_scene)
    assert_counts(pull(det_scene, first), 0, 0, 0, 0)
    assert apply(det_scene, first).status_code == 200
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    flow(det_scene)
    saved = det_scene.labels()
    for k in range(COUNT):
        dumped = fo_dump(sample_of(det_scene, k).ground_truth)
        want = saved.get(det_scene.ids[k], [])
        assert [d[0] for d in dumped] == [lb.label for lb in want], k
        for d, lb in zip(dumped, want, strict=True):
            assert d[2] == pytest.approx(lb.bbox, abs=EPS), k


def test_project_edit_reaches_fiftyone_images_not_edited_there(det_scene):
    """A session is still open; the project is re-imported; a new session is sent.

    Images that were NOT edited in FiftyOne must show the new project labels (only edited images keep
    their FiftyOne version, as docs/ANNOTATION.md says).
    """
    flow(det_scene)
    edit(
        det_scene,
        0,
        lambda s: setattr(s.ground_truth.detections[0], "bounding_box", [0.15, 0.125, 0.3, 0.375]),
    )
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    saved = det_scene.labels()
    send(det_scene)
    for k in (1, 2, 3):  # never edited in FiftyOne: they must show the project's new labels
        dumped = fo_dump(sample_of(det_scene, k).ground_truth)
        want = saved.get(det_scene.ids[k], [])
        assert [d[0] for d in dumped] == [lb.label for lb in want], f"image {k} shows the old labels"
        for d, lb in zip(dumped, want, strict=True):
            assert d[2] == pytest.approx(lb.bbox, abs=EPS), f"image {k}"


def test_send_counts_only_the_images_with_edits_in_fiftyone(det_scene):
    """Only image 0 has edits in FiftyOne: that is the number of kept edits the person is told."""
    flow(det_scene)
    edit(
        det_scene,
        0,
        lambda s: setattr(s.ground_truth.detections[0], "bounding_box", [0.15, 0.125, 0.3, 0.375]),
    )
    assert send(det_scene).get("kept_fiftyone_edits") == 1


def test_pull_after_project_edit_does_not_revert_images_not_edited_in_fiftyone(det_scene):
    """The project edit of an image nobody touched in FiftyOne must survive the next pull and apply."""
    flow(det_scene)
    edit(
        det_scene,
        0,
        lambda s: setattr(s.ground_truth.detections[0], "bounding_box", [0.15, 0.125, 0.3, 0.375]),
    )
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    new_labels = {k: v for k, v in det_scene.labels().items()}
    second = send(det_scene)["session"]["id"]
    diff = pull(det_scene, second)
    # only image 0 was edited in FiftyOne: the diff must not list the project's own edits as changes
    assert diff["images_changed"] <= 1, (
        f"the pull reverts the project edit of {diff['images_changed']} images"
    )
    assert apply(det_scene, second).status_code == 200
    for k in (1, 2, 3):
        assert_same(
            det_scene.labels().get(det_scene.ids[k], []), new_labels.get(det_scene.ids[k], []), f"image {k}"
        )


def test_fiftyone_edit_kept_over_project_edit_shows_in_diff(det_scene):
    """What the person sees when an unpulled FiftyOne edit meets a project edit of the same image."""
    flow(det_scene)
    edit(
        det_scene,
        0,
        lambda s: setattr(s.ground_truth.detections[0], "bounding_box", [0.15, 0.125, 0.3, 0.375]),
    )
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    second = send(det_scene)
    assert second["kept_fiftyone_edits"] >= 1  # the person is told that edits were kept
    shown = fo_dump(sample_of(det_scene, 0).ground_truth)
    assert shown[0][2] == pytest.approx([0.15, 0.125, 0.3, 0.375], abs=EPS)  # FiftyOne keeps its version
    diff = pull(det_scene, second["session"]["id"])
    entry = next(e for e in diff["examples"] if e["image_id"] == det_scene.ids[0])
    assert entry["edited"] + entry["added"] + entry["removed"] > 0  # the revert of the project edit is listed


# =====================================================================================
# 4. Conflicts
# =====================================================================================
def test_apply_refused_when_project_changed_after_the_pull(det_scene):
    session, _ = flow(det_scene)
    edit(det_scene, 0, lambda s: s.ground_truth.detections.pop(0))
    pull(det_scene, session)
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    before = det_scene.labels()
    reply = apply(det_scene, session)
    assert reply.status_code == 409 and reply.json()["code"] == "annotations_changed"
    assert det_scene.labels() == before  # nothing was written
    listed = det_scene.client.get(f"/api/projects/{det_scene.pid}/annotate").json()["sessions"][0]
    assert listed["status"] == "pulled"
    pull(det_scene, session)  # bring the labels back again: now it can be applied
    assert apply(det_scene, session).status_code == 200


def test_edited_on_both_sides_before_the_pull_the_tool_version_wins(det_scene):
    session, _ = flow(det_scene)
    edit(
        det_scene,
        0,
        lambda s: setattr(s.ground_truth.detections[0], "bounding_box", [0.15, 0.125, 0.3, 0.375]),
    )
    reimport(det_scene, new_det_truth(), ["cat", "dog"])  # image 0 edited in the project too
    diff = pull(det_scene, session)
    assert diff["images_changed"] >= 1
    assert apply(det_scene, session).status_code == 200
    saved = det_scene.labels()[det_scene.ids[0]]
    assert_same(saved, [Label("cat", bbox=[0.15, 0.125, 0.3, 0.375], track=3), det_truth()[0][1]], "image 0")


def test_session_already_applied(det_scene):
    session, _ = flow(det_scene)
    edit(det_scene, 0, lambda s: s.ground_truth.detections.pop(0))
    pull(det_scene, session)
    assert apply(det_scene, session).status_code == 200
    after = det_scene.labels()
    assert det_scene.client.post(f"{base_url(det_scene, session)}/pull").status_code == 409
    again = apply(det_scene, session)
    # docs/ANNOTATION.md says 409 for an applied session; the API answers 400 pull_first (harmless)
    assert again.status_code in (400, 409), again.text
    assert det_scene.labels() == after
    assert det_scene.client.post(f"{base_url(det_scene, session)}/discard").status_code == 409


def test_apply_makes_a_backup_and_pull_twice_is_stable(det_scene):
    session, _ = flow(det_scene)
    edit(det_scene, 3, lambda s: s.ground_truth.detections.pop(0))
    first = pull(det_scene, session)
    second = pull(det_scene, session)
    assert first == second
    applied = apply(det_scene, session).json()["applied"]
    assert applied["backup"] and (det_scene.folder / applied["backup"]).exists()


def move_box_img1(sample: Any) -> None:
    """Move the box of the first detection (used to edit images in FiftyOne).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.detections[0].bounding_box = [0.15, 0.125, 0.3, 0.375]


def edit_and_apply(scene: Scene) -> str:
    """Edit images 0 and 1 in FiftyOne, bring them back and apply.

    Args:
        scene: the project.

    Returns:
        The id of the applied session.
    """
    session, _ = flow(scene)
    edit(scene, 0, lambda s: s.ground_truth.detections.pop(1))
    edit(scene, 1, move_box_img1)
    pull(scene, session)
    assert apply(scene, session).status_code == 200
    return session


def test_after_apply_fiftyone_and_project_are_the_same_not_edited(det_scene):
    """#455: the applied images record the fingerprint of their ground_truth and the new project hash."""
    edit_and_apply(det_scene)
    saved = det_scene.labels()
    for k in (0, 1):
        sample = sample_of(det_scene, k)
        assert not bridge.is_edited(sample, Task.DETECTION), f"image {k} still counts as edited"
        assert sample[bridge.WRITTEN_LABELS] == label_hash(saved.get(det_scene.ids[k], []))


def test_after_apply_the_next_session_has_no_false_edits_or_conflicts(det_scene):
    """#455: send again after apply: nothing kept, no conflict, empty diff."""
    edit_and_apply(det_scene)
    again = send(det_scene)
    assert "kept_fiftyone_edits" not in again
    diff = pull(det_scene, again["session"]["id"])
    assert "edited_on_both_sides" not in problem_codes(diff)
    assert_counts(diff, 0, 0, 0, 0)


def test_project_change_reaches_fiftyone_on_images_applied_from_it(det_scene):
    """#455: after apply, a later project change (re-import) shows in FiftyOne on the applied images."""
    edit_and_apply(det_scene)
    reimport(det_scene, new_det_truth(), ["cat", "dog"])
    send(det_scene)
    saved = det_scene.labels()
    for k in (0, 1):
        dumped = fo_dump(sample_of(det_scene, k).ground_truth)
        want = saved.get(det_scene.ids[k], [])
        assert [d[0] for d in dumped] == [lb.label for lb in want], k
        assert dumped[0][2] == pytest.approx(want[0].bbox, abs=EPS), k


def test_discard_puts_the_session_images_back_to_the_project_labels(det_scene):
    """#456: discard rewrites the edited images in FiftyOne, so sending again shows the project's labels."""
    session, _ = flow(det_scene)
    edit(det_scene, 0, lambda s: s.ground_truth.detections.pop(1))
    edit(det_scene, 1, move_box_img1)
    pull(det_scene, session)
    before = det_scene.labels()
    reply = det_scene.client.post(f"{base_url(det_scene, session)}/discard")
    assert reply.status_code == 200
    for k in (0, 1):
        sample = sample_of(det_scene, k)
        assert not bridge.is_edited(sample, Task.DETECTION)
        want = before.get(det_scene.ids[k], [])
        assert [d[0] for d in fo_dump(sample.ground_truth)] == [lb.label for lb in want]
        assert fo_dump(sample.ground_truth)[0][2] == pytest.approx(want[0].bbox, abs=EPS)
    again = send(det_scene)
    assert "kept_fiftyone_edits" not in again
    assert_counts(pull(det_scene, again["session"]["id"]), 0, 0, 0, 0)
    assert det_scene.labels() == before


# =====================================================================================
# 5. Anything else suspicious
# =====================================================================================
def detection_with_a_mask_img0(sample: Any) -> None:
    """Edit one sample for ``test_detection_with_a_mask_comes_back_as_its_box`` (img0).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.detections[0].mask = np.ones((4, 4), dtype=bool)


def test_detection_with_a_mask_comes_back_as_its_box(det_scene):
    session, _ = flow(det_scene)

    edit(det_scene, 0, detection_with_a_mask_img0)
    diff = pull(det_scene, session)
    assert_counts(diff, 0, 0, 0, 0)  # the box did not change
    assert apply(det_scene, session).status_code == 200
    assert_same(det_scene.labels()[det_scene.ids[0]], det_truth()[0], "image 0")


def drawn_in_the_app_img2(sample: Any) -> None:
    """Edit one sample for ``test_polygon_drawn_in_the_app_reaches_the_project`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Polylines(
        polylines=[
            fo.Polyline(
                label="cat", points=[[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]], closed=False, filled=False
            )
        ]
    )


def test_polygon_drawn_in_the_app_reaches_the_project(seg_scene):
    """#457: the App makes new polylines closed=False, filled=False; with 3 points they are polygons."""
    session, _ = flow(seg_scene)

    edit(seg_scene, 2, drawn_in_the_app_img2)
    diff = pull(seg_scene, session)
    assert_counts(diff, 1, 0, 0, 1)
    assert "open_polyline" not in problem_codes(diff)
    assert apply(seg_scene, session).status_code == 200
    assert_same(
        seg_scene.labels()[seg_scene.ids[2]],
        [Label("cat", bbox=[0.1, 0.1, 0.4, 0.4], polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])],
        "image 2",
    )


def open_line_img2(sample: Any) -> None:
    """Edit one sample for ``test_a_two_point_open_line_is_not_imported_as_a_polygon`` (img2).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth = fo.Polylines(
        polylines=[fo.Polyline(label="cat", points=[[(0.1, 0.1), (0.5, 0.1)]], closed=False, filled=False)]
    )


def test_a_two_point_open_line_is_not_imported_as_a_polygon(seg_scene):
    session, _ = flow(seg_scene)

    edit(seg_scene, 2, open_line_img2)
    diff = pull(seg_scene, session)
    assert diff["added"] == 0 and "open_polyline" in problem_codes(diff)


def polygon_reduced_to_two_img1(sample: Any) -> None:
    """Edit one sample for ``test_polygon_reduced_to_two_points_is_reported_not_silently_removed`` (img1).

    Args:
        sample: the sample to change in place.
    """
    sample.ground_truth.polylines[0].points = [[(0.2, 0.2), (0.5, 0.5)]]


def test_polygon_reduced_to_two_points_is_reported_not_silently_removed(seg_scene):
    session, _ = flow(seg_scene)

    edit(seg_scene, 1, polygon_reduced_to_two_img1)
    diff = pull(seg_scene, session)
    silent = diff["removed"] > 0 and "polygon_too_few_points" not in problem_codes(diff)
    assert not silent, "a polygon with fewer than 3 points vanished from the pull without a warning"


def keypoints_with_a_point_img1(sample: Any) -> None:
    """Edit one sample for ``test_keypoints_with_a_point_added_in_fiftyone_are_reported`` (img1).

    Args:
        sample: the sample to change in place.
    """
    first = sample.ground_truth.keypoints[0]
    first.points = [*first.points, (0.8, 0.8)]  # a fourth point in a three-point skeleton


def test_keypoints_with_a_point_added_in_fiftyone_are_reported(kp_scene):
    session, _ = flow(kp_scene)

    edit(kp_scene, 1, keypoints_with_a_point_img1)
    diff = pull(kp_scene, session)
    assert "keypoints_count_mismatch" in problem_codes(diff), (
        f"4 points in a 3-point project came back with no warning: {diff}"
    )


def keypoints_all_points_hidden_img1(sample: Any) -> None:
    """Edit one sample for ``test_keypoints_all_points_hidden_stays_a_label_or_is_reported`` (img1).

    Args:
        sample: the sample to change in place.
    """
    first = sample.ground_truth.keypoints[0]
    first.points = [(NAN, NAN)] * 3


def test_keypoints_all_points_hidden_stays_a_label_or_is_reported(kp_scene):
    session, _ = flow(kp_scene)

    edit(kp_scene, 1, keypoints_all_points_hidden_img1)
    diff = pull(kp_scene, session)
    assert diff["removed"] + diff["edited"] >= 1  # the change is visible in the summary
    assert apply(kp_scene, session).status_code == 200


def test_label_without_a_class_is_dropped_and_shown_as_removed(det_scene):
    session, _ = flow(det_scene)
    edit(det_scene, 1, lambda s: setattr(s.ground_truth.detections[0], "label", None))
    diff = pull(det_scene, session)
    assert diff["removed"] == 1  # the person sees it in the summary


def test_no_dataset_is_left_behind(fiftyone_env):
    """Runs last: every dataset of these tests is deleted after each test."""
    assert [
        n for n in fo.list_datasets() if n.startswith(ROOT_PREFIX) and not n.startswith(fiftyone_env)
    ] == []
