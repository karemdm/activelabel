import uuid

import fiftyone as fo
import pytest
from sample_images import make_image

from activelabel import fiftyone_bridge as bridge
from activelabel.annotate import fiftyone_tool as ft
from activelabel.annotate.base import ToolSettings
from activelabel.annotate.registry import get_tool
from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.tasks import Task

pytestmark = pytest.mark.integration


@pytest.fixture
def project(monkeypatch):
    unique = f"test-activelabel-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "dataset_name", lambda project_dir, pid: unique)
    monkeypatch.setattr(bridge, "launch_app", lambda dataset, view=None, port=0: f"http://x/{view}")
    yield Project(1, "t", "", Task.DETECTION, ["cat", "dog"])
    if fo.dataset_exists(unique):
        fo.delete_dataset(unique)


def test_send_edit_pull_close(tmp_path, project):
    manifest = Manifest(tmp_path)
    records = []
    for i in range(3):
        make_image(tmp_path / "images" / f"i{i}.png", seed=i)
        records.append(ImageRecord(f"i{i}", f"i{i}.png", f"images/i{i}.png", 64, 48, 1, "2026"))
    manifest.add(records)
    old = {"i0": [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]}
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.DETECTION, ["cat", "dog"], old, {}))
    tool, settings = get_tool("fiftyone"), ToolSettings("fiftyone")
    assert "login" in tool.check(settings)

    sent = tool.send(dataset, ["i0", "i1"], project, settings, "abc")
    assert sent.ref == {"view": ft.view_name("abc")} and sent.links[0]["label"] == "FiftyOne"
    assert len(dataset.load_saved_view(sent.ref["view"])) == 2

    # simulate editing in the App: move the box of i0, draw one on i1
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    sample["ground_truth"].detections[0].bounding_box = [0.3, 0.3, 0.2, 0.2]
    sample.save()
    other = dataset.match(fo.ViewField("image_id") == "i1").first()
    other["ground_truth"] = fo.Detections(
        detections=[fo.Detection(label="dog", bounding_box=[0, 0, 0.5, 0.5])]
    )
    other.save()

    pulled = tool.pull(dataset, ["i0", "i1"], project, settings, sent.ref)
    assert set(pulled) == {"i0", "i1"}
    assert pulled["i0"] == [Label("cat", bbox=[0.3, 0.3, 0.2, 0.2])]
    assert pulled["i1"] == [Label("dog", bbox=[0.0, 0.0, 0.5, 0.5])]

    tool.close(dataset, settings, sent.ref, False)
    assert not dataset.has_saved_view(sent.ref["view"])
    tool.close(dataset, settings, sent.ref, False)  # no error when already gone
