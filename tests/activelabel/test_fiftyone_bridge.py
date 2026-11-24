import uuid

import fiftyone as fo
import pytest
from sample_images import make_image

from activelabel import fiftyone_bridge as bridge
from activelabel.fiftyone_names import recorded_name
from activelabel.images.manifest import EXCLUDED, ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
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
    manifest = build(tmp_path, 1)
    dataset = bridge.sync_dataset(project, manifest, None)
    url = bridge.launch_app(dataset, "curation - dark", port=5199)
    slug = "curation-dark"  # FiftyOne's slug of the view name
    assert url == f"http://localhost:5199/datasets/{dataset.name}?view={slug}"
    assert bridge.launch_app(dataset, None, port=5199) == f"http://localhost:5199/datasets/{dataset.name}"
    bridge._session.close()
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
    url = bridge.launch_app(dataset)
    assert tried == [5161, 5162, 5163] and url == f"http://localhost:5163/datasets/{dataset.name}"
    bridge._session = None


@pytest.mark.integration
def test_resolve_view_accepts_issue_names(tmp_path):
    import fiftyone as fo

    from activelabel.fiftyone_bridge import resolve_view

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
    from activelabel.annotate import sessions

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
