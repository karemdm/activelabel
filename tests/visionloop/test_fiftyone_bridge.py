import uuid

import fiftyone as fo
import pytest
from sample_images import make_image

from visionloop import fiftyone_bridge as bridge
from visionloop.images.manifest import EXCLUDED, ImageRecord, Manifest
from visionloop.labels import AnnotationSet, Label
from visionloop.projects import Project
from visionloop.tasks import Task

pytestmark = pytest.mark.integration


@pytest.fixture
def project(monkeypatch):
    unique = f"test-visionloop-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "dataset_name", lambda pid: unique)
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


def test_dataset_name():
    assert bridge.dataset_name(7) == "visionloop-7"


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
    assert url.startswith("http") and bridge.launch_app(dataset, None, port=5199) == url
    bridge._session.close()
    bridge._session = None


@pytest.mark.integration
def test_resolve_view_accepts_issue_names(tmp_path):
    import fiftyone as fo

    from visionloop.fiftyone_bridge import resolve_view

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
