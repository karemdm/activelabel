import os
import socket
import subprocess
import sys
import threading
import uuid
from types import SimpleNamespace

import fiftyone as fo
import pytest
from fiftyone_isolation import free_port
from sample_images import make_image

from activelabel import fiftyone_bridge as bridge
from activelabel.annotate import sessions
from activelabel.fiftyone_bridge import resolve_view
from activelabel.fiftyone_names import recorded_name
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.sync.links import label_hash
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore

pytestmark = pytest.mark.integration


@pytest.fixture
def project(monkeypatch):
    unique = f"test-activelabel-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "dataset_name", lambda project_dir, pid: unique)
    yield Project(1, "t", "", Task.DETECTION, ["cat", "dog"])
    if fo.dataset_exists(unique):
        fo.delete_dataset(unique)


def build(tmp_path, count=3):
    manifest = Manifest(tmp_path)
    records = []
    for i in range(count):
        make_image(tmp_path / "images" / f"i{i}.png", seed=i)
        records.append(ImageRecord(f"i{i}", f"i{i}.png", f"images/i{i}.png", 64, 48, 1, "2026"))
    manifest.add(records)
    metrics = {"brightness": 10.0, "contrast": 5.0, "sharpness": 2.0}
    manifest.update({"i0": {"metrics": metrics, "issues": ["dark"]}})
    return manifest


@pytest.fixture
def prefix(monkeypatch):
    """A unique dataset prefix: the tests never touch (or delete) a dataset they did not create."""
    unique = f"test-activelabel-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "DATASET_PREFIX", unique)
    yield unique
    for name in fo.list_datasets():
        if name.startswith(unique):
            fo.delete_dataset(name)


def plain_dataset(name, manifest):
    """A dataset as earlier versions created it: the plain name, samples with the project's images."""
    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples([fo.Sample(filepath=str(manifest.file_path(r).resolve())) for r in manifest.all()])
    return dataset


def test_dataset_names_differ_between_workspaces(tmp_path, prefix):
    one, two = tmp_path / "ws1" / "1", tmp_path / "ws2" / "1"  # the same project id in two workspaces
    one.mkdir(parents=True), two.mkdir(parents=True)
    name_one, name_two = bridge.dataset_name(one, 1), bridge.dataset_name(two, 1)
    assert name_one != name_two and name_one.startswith(f"{prefix}-1-") and name_two.startswith(prefix)
    assert bridge.dataset_name(one, 1) == name_one  # chosen once, recorded in the project folder
    assert recorded_name(one) == name_one


def test_two_workspaces_do_not_overwrite_each_other(tmp_path, prefix):
    datasets = []
    for workspace, count in (("ws1", 3), ("ws2", 1)):
        folder = tmp_path / workspace
        folder.mkdir()
        datasets.append(
            bridge.sync_dataset(Project(1, "t", "", Task.DETECTION, ["cat"]), build(folder, count), None)
        )
    assert datasets[0].name != datasets[1].name
    assert len(fo.load_dataset(datasets[0].name)) == 3 and len(fo.load_dataset(datasets[1].name)) == 1


def test_dataset_of_earlier_versions_keeps_its_name(tmp_path, prefix):
    manifest = build(tmp_path)
    legacy = plain_dataset(f"{prefix}-1", manifest)
    assert bridge.dataset_name(tmp_path, 1) == legacy.name
    dataset = bridge.sync_dataset(Project(1, "t", "", Task.DETECTION, ["cat"]), manifest, None)
    assert dataset.name == legacy.name and len(dataset) == 3


def test_dataset_of_another_workspace_is_not_adopted(tmp_path, prefix):
    mine, other = tmp_path / "mine", tmp_path / "other"
    mine.mkdir()
    manifest = build(other)
    foreign = plain_dataset(f"{prefix}-1", manifest)
    name = bridge.dataset_name(mine, 1)
    assert name != foreign.name and len(fo.load_dataset(foreign.name)) == 3
    empty = fo.Dataset(f"{prefix}-2", persistent=True)  # empty: whose it is cannot be told
    assert bridge.dataset_name(mine, 2) != empty.name


def test_delete_dataset_removes_only_the_projects_own(tmp_path, prefix):
    mine, other = tmp_path / "mine", tmp_path / "other"
    mine.mkdir()
    foreign_manifest = build(other)
    foreign = fo.Dataset(f"{prefix}-1", persistent=True)
    foreign.add_samples(
        [fo.Sample(filepath=str(foreign_manifest.file_path(r).resolve())) for r in foreign_manifest.all()]
    )
    dataset = bridge.sync_dataset(Project(1, "t", "", Task.DETECTION, ["cat"]), build(mine), None)
    assert bridge.delete_dataset(mine, 1, recorded_name(mine)) is True
    assert not fo.dataset_exists(dataset.name) and fo.dataset_exists(foreign.name)
    assert bridge.delete_dataset(mine, 1, None) is False  # the plain name holds another workspace's images
    assert fo.dataset_exists(foreign.name)


def test_delete_dataset_of_earlier_versions_after_the_folder_is_gone(tmp_path, prefix):
    manifest = build(tmp_path)
    legacy = plain_dataset(f"{prefix}-1", manifest)
    assert bridge.delete_dataset(tmp_path, 1, None) is True and not fo.dataset_exists(legacy.name)
    assert bridge.delete_dataset(tmp_path, 1, None) is False  # nothing left; never raises


def test_sync_is_idempotent_and_removes(tmp_path, project):
    manifest = build(tmp_path)
    labels = {"i0": [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]}
    annotations = AnnotationSet(Task.DETECTION, ["cat", "dog"], labels, {})
    dataset = bridge.sync_dataset(project, manifest, annotations)
    assert dataset.persistent and len(dataset) == 3
    again = bridge.sync_dataset(project, manifest, annotations)
    assert len(again) == 3
    sample = again.match(fo.ViewField("image_id") == "i0").first()
    assert sample.quality_issues == ["dark"] and sample.brightness == 10.0 and sample.image_size == [64, 48]
    assert sample.ground_truth.detections[0].label == "cat"
    assert "curation - quality issues" in again.list_saved_views()
    assert "curation - near duplicates" in again.list_saved_views()
    assert len(again.load_saved_view("curation - dark")) == 1
    manifest.set_status(["i1"], EXCLUDED, "x")
    manifest.delete(["i2"])
    again = bridge.sync_dataset(project, manifest, annotations)
    assert len(again) == 2
    assert len(again.match_tags("excluded")) == 1
    assert len(again.load_saved_view("curation - excluded")) == 1


def test_pull_exclusions(tmp_path, project):
    manifest = build(tmp_path)
    dataset = bridge.sync_dataset(project, manifest, None)
    dataset.match(fo.ViewField("image_id") == "i1").tag_samples("exclude")
    assert bridge.pull_exclusions(dataset, manifest) == 1
    assert manifest.get("i1").status == EXCLUDED
    assert manifest.get("i1").excluded_reason == "tagged in FiftyOne"
    assert len(dataset.match_tags("exclude")) == 0
    assert bridge.pull_exclusions(dataset, manifest) == 0


def test_other_tasks_declare_fields(tmp_path, project):
    manifest = build(tmp_path, 1)
    project.task = Task.SEGMENTATION
    labels = {"i0": [Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.SEGMENTATION, ["cat"], labels, {}))
    assert dataset.first().ground_truth.polylines[0].filled
    project.task = Task.KEYPOINTS
    kp = {"i0": [Label("p", keypoints=[[0.1, 0.2, 2], [0.0, 0.0, 0]])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.KEYPOINTS, ["p"], kp, {}))
    assert len(dataset.first().ground_truth.keypoints) == 1
    project.task = Task.CLASSIFICATION
    dataset = bridge.sync_dataset(project, manifest, None)
    assert dataset.get_field("ground_truth").document_type is fo.Classification


def test_launch_app_returns_url(tmp_path, project):
    """A free port of its own (#525): a fixed one may hold the App of another test run or a leftover one."""
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    port = free_port()
    try:
        url = bridge.launch_app(dataset, "curation - dark", port=port)
        slug = "curation-dark"  # FiftyOne's slug of the view name
        assert url == f"http://localhost:{port}/datasets/{dataset.name}?view={slug}"
        assert (
            bridge.launch_app(dataset, None, port=port) == f"http://localhost:{port}/datasets/{dataset.name}"
        )
    finally:
        if bridge._session is not None:
            bridge._drop_session(bridge._session)
        bridge._session = None


# -- time limits of the App (#525) ------------------------------------------------------------


class Stuck:
    """A call that never answers until released (a FiftyOne server that accepts and never replies)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.release.wait(10)


def _boom():
    raise ValueError("boom")


def test_within_returns_raises_and_times_out():
    assert bridge.within(1, lambda: 3, "x") == 3
    with pytest.raises(ValueError, match="boom"):
        bridge.within(1, _boom, "x")
    stuck = Stuck()
    with pytest.raises(bridge.AppTimeout, match="did not answer within 0 s"):
        bridge.within(0.1, stuck, "starting")
    stuck.release.set()


def test_a_hung_launch_fails_instead_of_waiting_for_ever(tmp_path, project, monkeypatch):
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    stuck = Stuck()
    monkeypatch.setattr(bridge.fo, "launch_app", stuck)
    monkeypatch.setattr(bridge, "_session", None)
    monkeypatch.setattr(bridge, "port_answers", lambda port: False)
    monkeypatch.setattr(bridge, "LAUNCH_TIMEOUT", 0.2)
    with pytest.raises(bridge.AppTimeout):
        bridge.launch_app(dataset)
    assert stuck.calls == 1  # the next ports are not tried: they would hang the same way
    stuck.release.set()


class FakeSession:
    """An App session: records what it is asked to show; ``hang`` makes switching block."""

    def __init__(self, port, hang=None):
        self.port, self.hang, self.closed, self.shown = port, hang, False, []

    @property
    def dataset(self):
        return None

    @dataset.setter
    def dataset(self, value):
        if self.hang is not None:
            self.hang.wait(10)
        self.shown.append(value)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("problem", ["stopped", "hung"])
def test_a_dead_or_hung_session_is_dropped_and_a_new_app_started(tmp_path, project, monkeypatch, problem):
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    hang = threading.Event() if problem == "hung" else None
    old = FakeSession(5161, hang)
    monkeypatch.setattr(bridge, "_session", old)
    monkeypatch.setattr(bridge, "SWITCH_TIMEOUT", 0.2)
    monkeypatch.setattr(bridge, "port_answers", lambda port: problem == "hung" and port == 5161)
    monkeypatch.setattr(bridge, "_launched", {5161})
    monkeypatch.setattr(bridge.fo, "launch_app", lambda target, port, remote: FakeSession(port))
    url = bridge.launch_app(dataset)
    assert old.closed and bridge._session is not old
    assert url.startswith(f"http://localhost:{bridge._session.port}/datasets/")
    if hang is not None:
        hang.set()
    bridge._session = None


def test_default_port_is_dedicated():
    assert bridge.APP_PORT == 5161


def test_launch_app_tries_next_port_when_busy(tmp_path, project, monkeypatch):
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    tried = []

    class Fake:
        def __init__(self, port):
            self.port = port

    def fake_launch(target, port, remote):
        tried.append(port)
        if port < 5163:
            raise OSError("busy")
        return Fake(port)

    monkeypatch.setattr(bridge.fo, "launch_app", fake_launch)
    monkeypatch.setattr(bridge, "_session", None)
    monkeypatch.setattr(bridge, "port_answers", lambda port: False)
    url = bridge.launch_app(dataset)
    assert tried == [5161, 5162, 5163] and url == f"http://localhost:5163/datasets/{dataset.name}"
    bridge._session = None


def test_launch_app_never_joins_the_app_of_another_program(tmp_path, project, monkeypatch):
    """#410: fo.launch_app connects to an App that already answers on the port (another database)."""
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    tried = []
    monkeypatch.setattr(
        bridge.fo, "launch_app", lambda target, port, remote: tried.append(port) or SimpleNamespace(port=port)
    )
    monkeypatch.setattr(bridge, "_session", None)
    monkeypatch.setattr(bridge, "port_answers", lambda port: port == 5161)  # another program's App
    url = bridge.launch_app(dataset)
    assert tried == [5162] and url == f"http://localhost:5162/datasets/{dataset.name}"
    bridge._session = None


def test_port_answers_sees_a_listening_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        assert bridge.port_answers(server.getsockname()[1])


@pytest.mark.integration
def test_resolve_view_accepts_issue_names(tmp_path):
    name = f"vl-test-views-{tmp_path.name}"
    dataset = fo.Dataset(name)
    try:
        dataset.save_view("curation - quality issues", dataset.view())
        dataset.save_view("curation - near duplicates", dataset.view())
        dataset.save_view("curation - low contrast", dataset.view())
        assert resolve_view(dataset, None) is None
        assert resolve_view(dataset, "curation - near_duplicate") == "curation - near duplicates"
        assert resolve_view(dataset, "low_contrast") == "curation - low contrast"
        assert resolve_view(dataset, "curation - unknown") == "curation - quality issues"
    finally:
        dataset.delete()


def test_classification_ground_truth_is_one_class(tmp_path, project):
    manifest = build(tmp_path, 2)
    project.task = Task.CLASSIFICATION
    labels = {"i0": [Label("cat"), Label("dog")]}
    gt = AnnotationSet(Task.CLASSIFICATION, ["cat", "dog"], labels, {})
    dataset = bridge.sync_dataset(project, manifest, gt)
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    assert sample.ground_truth.label == "cat"
    assert dataset.match(fo.ViewField("image_id") == "i1").first().ground_truth is None


def _video_manifest(tmp_path):
    manifest = Manifest(tmp_path)
    records = []
    for i in range(4):  # two videos, frames added out of order
        make_image(tmp_path / "images" / f"v{i}.png", seed=i)
        video, frame = ("va", "vb")[i % 2], (3 - i) * 10
        records.append(
            ImageRecord(
                f"v{i}", f"v{i}.png", f"images/v{i}.png", 64, 48, 1, "2026",
                video=video, frame=frame, time=frame / 10,
            )
        )  # fmt: skip
    manifest.add(records)
    return manifest


def test_task_fields_and_views(tmp_path, project):
    manifest = build(tmp_path, 3)
    store = TaskStore(tmp_path)
    tasks = store.load()  # migration: every image in "Task 1"
    tasks[0].name, tasks[0].job_size = "Mine", 2
    store._write(2, tasks)  # noqa: SLF001
    dataset = bridge.sync_dataset(project, manifest, None)
    sample = dataset.match(fo.ViewField("image_id") == "i2").first()
    assert (sample.task, sample.task_name, sample.job, sample.position) == (1, "Mine", 2, 3)
    assert sample.video is None and sample.frame_number is None and sample.time is None
    assert bridge.task_view_name(1) in dataset.list_saved_views()
    assert dataset.load_saved_view("task-1").values("image_id") == ["i0", "i1", "i2"]
    assert bridge.resolve_view(dataset, "task-1") == "task-1"
    info = next(i for i in dataset.list_saved_views(info=True) if i["name"] == "task-1")
    assert info["description"] == "Task: Mine"
    assert bridge.task_view(dataset, 1, 2).values("image_id") == ["i2"]


def test_video_task_view_groups_frames(tmp_path, project):
    manifest = _video_manifest(tmp_path)
    store = TaskStore(tmp_path)
    store.load()  # migration: every image in "Task 1"
    store.delete(1)
    store.create("Clips", images=["v0", "v1", "v2", "v3"], source={"video": "va"})
    dataset = bridge.sync_dataset(project, manifest, None)
    sample = dataset.match(fo.ViewField("image_id") == "v0").first()
    assert (sample.video, sample.frame_number, sample.time) == ("va", 30, 3.0)
    view = dataset.load_saved_view("task-2")
    assert view._is_dynamic_groups  # noqa: SLF001
    assert len(view) == 2  # two videos, two groups
    assert view.get_dynamic_group("va").values("frame_number") == [10, 30]


def test_refresh_after_task_deleted(tmp_path, project):
    manifest = build(tmp_path, 3)
    store = TaskStore(tmp_path)
    store.load()  # migration: every image in "Task 1"
    store.remove_images(["i2"])
    store.create("B", images=["i2"])
    dataset = bridge.sync_dataset(project, manifest, None)
    assert {"task-1", "task-2"} <= set(dataset.list_saved_views())
    manifest.delete(store.delete(1))  # the images of the deleted task go away too
    dataset = bridge.sync_dataset(project, manifest, None)
    assert "task-1" not in dataset.list_saved_views() and "task-2" in dataset.list_saved_views()
    assert dataset.distinct("task") == [2]


def test_open_fiftyone_session_keeps_its_ground_truth(tmp_path, project):
    manifest = build(tmp_path)
    cat = Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.DETECTION, ["cat"], {}, {}))
    sessions.create(tmp_path, "fiftyone", "all", ["i0"], [], {})
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    sample[bridge.GROUND_TRUTH] = bridge._ground_truth(Task.DETECTION, [cat])  # edited in the App
    sample.save()
    annotations = AnnotationSet(Task.DETECTION, ["cat"], {"i0": [], "i1": [cat]}, {})
    again = bridge.sync_dataset(project, manifest, annotations)
    kept = again.match(fo.ViewField("image_id") == "i0").first()[bridge.GROUND_TRUTH]
    assert len(kept.detections) == 1  # not overwritten by the (empty) project labels
    other = again.match(fo.ViewField("image_id") == "i1").first()[bridge.GROUND_TRUTH]
    assert len(other.detections) == 1  # images outside the session are refreshed


def test_a_keypoint_without_visibility_counts_as_visible():
    label = Label("p", keypoints=[[0.1, 0.2], [0.3, 0.4, 2], [0.5, 0.6, 0]])  # [x, y] has no third value
    pose = bridge._ground_truth(Task.KEYPOINTS, [label]).keypoints[0]
    assert list(pose.points[0]) == [0.1, 0.2] and list(pose.points[1]) == [0.3, 0.4]
    assert pose.points[2][0] != pose.points[2][0]  # not labeled: NaN
    assert pose.visibility == [2, 2, 0]


def test_open_session_keeps_only_the_images_edited_in_fiftyone(tmp_path, project):
    manifest = build(tmp_path)
    cat, dog = [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])], [Label("dog", bbox=[0.5, 0.5, 0.2, 0.2])]
    old = AnnotationSet(Task.DETECTION, ["cat", "dog"], {"i0": cat, "i1": cat}, {})
    dataset = bridge.sync_dataset(project, manifest, old)
    assert bridge.edited_images(dataset, ["i0", "i1", "i2"], Task.DETECTION) == set()
    sessions.create(tmp_path, "fiftyone", "all", ["i0", "i1"], [], {})
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    sample[bridge.GROUND_TRUTH].detections[0].bounding_box = [0.2, 0.2, 0.2, 0.2]  # edited in the App
    sample.save()
    assert bridge.edited_images(dataset, ["i0", "i1"], Task.DETECTION) == {"i0"}

    new = AnnotationSet(Task.DETECTION, ["cat", "dog"], {"i0": dog, "i1": dog}, {})
    again = bridge.sync_dataset(project, manifest, new)
    view = again.match(fo.ViewField("image_id").is_in(["i0", "i1"]))
    shown = {s["image_id"]: s[bridge.GROUND_TRUTH].detections[0] for s in view}
    assert shown["i0"].label == "cat" and shown["i0"].bounding_box == [0.2, 0.2, 0.2, 0.2]  # edit kept
    assert shown["i1"].label == "dog"  # not edited in FiftyOne: shows the new project labels
    assert bridge.written_labels(again, ["i0", "i1"]) == {"i0": label_hash(cat), "i1": label_hash(dog)}


def test_a_sample_without_a_fingerprint_counts_as_edited(tmp_path, project):
    """A dataset written by an earlier version: its edits cannot be told apart, so they are kept."""
    manifest = build(tmp_path, count=1)
    dataset = bridge.sync_dataset(project, manifest, None)
    sample = dataset.first()
    assert not bridge.is_edited(sample, Task.DETECTION)
    sample[bridge.WRITTEN] = None
    sample.save()
    assert bridge.is_edited(sample, Task.DETECTION)


def test_app_url_is_localhost():
    assert bridge.app_url(SimpleNamespace(name="d"), None, 5161) == "http://localhost:5161/datasets/d"


@pytest.mark.parametrize(
    ("task", "labels"),
    [
        (Task.DETECTION, {"i0": [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]}),
        (Task.SEGMENTATION, {"i0": [Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])]}),
        (Task.CLASSIFICATION, {"i0": [Label("cat")]}),
    ],
)
def test_ground_truth_can_be_edited_in_the_app_with_the_projects_classes(tmp_path, project, task, labels):
    """#392: the App's Annotate mode only edits a field with an active label schema."""
    project.task = task
    manifest = build(tmp_path, 2)
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(task, ["cat", "dog"], labels, {}))
    assert dataset.active_label_schemas == [bridge.GROUND_TRUTH]
    assert dataset.label_schemas[bridge.GROUND_TRUTH]["classes"] == ["cat", "dog"]
    project.classes = ["cat", "dog", "bird"]
    again = bridge.sync_dataset(project, manifest, AnnotationSet(task, project.classes, labels, {}))
    assert again.label_schemas[bridge.GROUND_TRUTH]["classes"] == ["cat", "dog", "bird"]


def test_the_app_shows_only_ground_truth_and_hides_the_internal_fields(tmp_path, project):
    """#699: only ground_truth is drawn; the fields ActiveLabel keeps go to a collapsed group."""
    manifest = build(tmp_path, 2)
    labels = {"i0": [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.DETECTION, ["cat"], labels, {}))
    assert list(dataset.app_config.active_fields.paths) == [bridge.GROUND_TRUTH]
    groups = {g.name: g for g in dataset.app_config.sidebar_groups}
    internal = groups[bridge.INTERNAL_GROUP]
    assert bridge.WRITTEN in internal.paths and internal.expanded is False
    assert all(bridge.WRITTEN not in g.paths for g in groups.values() if g.name != bridge.INTERNAL_GROUP)
    before = [(g.name, list(g.paths)) for g in dataset.app_config.sidebar_groups]
    bridge.tidy_app(dataset)  # a second call changes nothing
    assert [(g.name, list(g.paths)) for g in dataset.app_config.sidebar_groups] == before


def test_a_refreshed_dataset_is_reloaded_in_the_open_app(tmp_path, project, monkeypatch):
    """#699: after a refresh the App that shows the dataset reloads it, so the person does not press F5."""
    manifest = build(tmp_path, 1)
    labels = AnnotationSet(Task.DETECTION, ["cat"], {}, {})
    dataset = bridge.sync_dataset(project, manifest, labels)
    calls = []
    app = SimpleNamespace(dataset=dataset, server_port=1, refresh=lambda: calls.append("refresh"))
    monkeypatch.setattr(bridge, "_session", app)
    monkeypatch.setattr(bridge, "port_answers", lambda port: True)
    bridge.sync_dataset(project, manifest, labels)
    assert calls == ["refresh"]
    app.dataset = SimpleNamespace(name="another dataset")  # the App shows another dataset: left alone
    assert bridge.refresh_app(dataset) is False and calls == ["refresh"]
    app.dataset = dataset
    monkeypatch.setattr(bridge, "port_answers", lambda port: False)  # the App is gone
    assert bridge.refresh_app(dataset) is False and calls == ["refresh"]
    monkeypatch.setattr(bridge, "port_answers", lambda port: True)
    app.refresh = lambda: 1 / 0  # a failing App never fails the refresh of the dataset
    assert bridge.refresh_app(dataset) is False
    monkeypatch.setattr(bridge, "_session", None)
    assert bridge.refresh_app(dataset) is False


def test_keypoints_are_shown_but_not_editable_in_the_app(tmp_path, project):
    """The FiftyOne App (1.22) cannot edit keypoints: no label schema, the labels are only shown."""
    project.task = Task.KEYPOINTS
    manifest = build(tmp_path, 1)
    kp = {"i0": [Label("cat", keypoints=[[0.1, 0.2, 2], [0.3, 0.4, 2]])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.KEYPOINTS, ["cat"], kp, {}))
    assert bridge.enable_editing(dataset, project) is False
    assert bridge.GROUND_TRUTH not in dataset.active_label_schemas


def test_an_edit_made_outside_a_session_is_kept(tmp_path, project):
    """A task opened for a look can be edited in the App too: the next refresh must not overwrite it."""
    manifest = build(tmp_path, 2)
    cat = [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]
    dataset = bridge.sync_dataset(
        project, manifest, AnnotationSet(Task.DETECTION, ["cat", "dog"], {"i0": cat}, {})
    )
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    sample[bridge.GROUND_TRUTH].detections[0].label = "dog"  # edited in the App, no session open
    sample.save()
    dog = [Label("dog", bbox=[0.5, 0.5, 0.2, 0.2])]
    again = bridge.sync_dataset(
        project, manifest, AnnotationSet(Task.DETECTION, ["cat", "dog"], {"i1": dog}, {})
    )
    shown = {s["image_id"]: s[bridge.GROUND_TRUTH] for s in again}
    assert shown["i0"].detections[0].label == "dog" and shown["i0"].detections[0].bounding_box == [
        0.1,
        0.1,
        0.2,
        0.2,
    ]
    assert shown["i1"].detections[0].label == "dog"  # not edited: refreshed from the project


def test_polygons_are_written_closed_and_filled_and_new_ones_default_to_that(tmp_path, project):
    """#457: closed+filled polygons; the label schema makes polylines drawn in the App closed and filled."""
    project.task = Task.SEGMENTATION
    manifest = build(tmp_path, 1)
    labels = {"i0": [Label("cat", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.SEGMENTATION, ["cat"], labels, {}))
    polyline = dataset.first().ground_truth.polylines[0]
    assert polyline.closed and polyline.filled
    attributes = {a["name"]: a for a in dataset.label_schemas[bridge.GROUND_TRUTH]["attributes"]}
    assert attributes["closed"]["default"] is True and attributes["filled"]["default"] is True


def test_sync_reports_how_many_images_it_has_handled(tmp_path, project):
    manifest = build(tmp_path, 3)
    seen = []
    bridge.sync_dataset(project, manifest, None, lambda done, total: seen.append((done, total)))
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)  # from none to all, in the order the work is done
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    seen.clear()
    bridge.sync_dataset(project, manifest, None, lambda done, total: seen.append((done, total)))  # refresh
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)


def test_sync_stops_when_the_progress_callback_raises(tmp_path, project):
    manifest = build(tmp_path, 3)

    def stop(done, total):
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        bridge.sync_dataset(project, manifest, None, stop)


def test_the_app_port_is_5161_unless_the_environment_names_another():
    code = "from activelabel import fiftyone_bridge as b; print(b.APP_PORT)"

    def port(value):
        env = {k: v for k, v in os.environ.items() if k != "ACTIVELABEL_FIFTYONE_PORT"}
        if value is not None:
            env["ACTIVELABEL_FIFTYONE_PORT"] = value
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        return int(out.stdout.strip())

    assert (port(None), port(""), port("5183")) == (5161, 5161, 5183)


def test_the_port_of_a_session_is_its_server_port():
    """A FiftyOne session has ``server_port``, not ``port``: reading ``port`` dropped the App every time."""
    assert bridge._app_port(SimpleNamespace(server_port=5171)) == 5171
    assert bridge._app_port(SimpleNamespace(port="5172")) == 5172
