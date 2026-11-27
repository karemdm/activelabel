import uuid

import fiftyone as fo
import pytest
from sample_images import make_image

from activelabel import fiftyone_bridge as bridge
from activelabel.annotate import fiftyone_tool as ft
from activelabel.annotate import sessions
from activelabel.annotate.base import Pulled, ToolError, ToolSettings
from activelabel.annotate.registry import get_tool
from activelabel.images.manifest import ImageRecord, Manifest
from activelabel.labels import AnnotationSet, Label
from activelabel.projects import Project
from activelabel.sync.links import label_hash
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
    assert "id=" not in sent.links[0]["url"]  # several images: the App opens on the grid
    one = tool.send(dataset, ["i1"], project, settings, "one")
    sample_id = dataset.match(fo.ViewField("image_id") == "i1").first().id
    assert one.links[0]["url"].endswith(f"?id={sample_id}")  # one image: the App opens on it
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

    tool.close(dataset, settings, sent.ref)
    assert not dataset.has_saved_view(sent.ref["view"])
    tool.close(dataset, settings, sent.ref)  # no error when already gone


def test_tools_say_so_when_the_project_was_never_opened_in_fiftyone():
    tool, settings = get_tool("fiftyone"), ToolSettings("fiftyone")
    project = Project(1, "t", "", Task.DETECTION, ["cat"])
    with pytest.raises(ToolError, match="Open the project in FiftyOne"):
        tool.send(None, ["a"], project, settings, "s1")
    with pytest.raises(ToolError, match="Open the project in FiftyOne"):
        tool.pull(None, ["a"], project, settings, {})


def test_pull_tells_which_images_were_edited_and_send_counts_the_kept_ones(tmp_path, project):
    manifest = Manifest(tmp_path)
    for i in range(2):
        make_image(tmp_path / "images" / f"i{i}.png", seed=i)
    manifest.add([ImageRecord(f"i{i}", f"i{i}.png", f"images/i{i}.png", 64, 48, 1, "2026") for i in range(2)])
    cat = [Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])]
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.DETECTION, ["cat"], {"i0": cat}, {}))
    tool, settings = get_tool("fiftyone"), ToolSettings("fiftyone")
    assert tool.send(dataset, ["i0", "i1"], project, settings, "s1").kept == 0
    sample = dataset.match(fo.ViewField("image_id") == "i0").first()
    sample["ground_truth"].detections[0].bounding_box = [0.3, 0.3, 0.2, 0.2]
    sample.save()

    pulled = tool.pull(dataset, ["i0", "i1"], project, settings, {})
    assert isinstance(pulled, Pulled)
    assert pulled.edited == {"i0"} and pulled.written == {"i0": label_hash(cat), "i1": label_hash([])}
    # not brought back yet: a refresh keeps the edit while a session is open, and send counts it
    sessions.create(tmp_path, "fiftyone", "all", ["i0", "i1"], [], {})
    dataset = bridge.sync_dataset(project, manifest, AnnotationSet(Task.DETECTION, ["cat"], {"i0": cat}, {}))
    assert tool.send(dataset, ["i0", "i1"], project, settings, "s2").kept == 1


def test_pull_reads_the_values_of_a_regression_project(tmp_path, project):
    project.task, project.classes = Task.REGRESSION, ["age"]
    manifest = Manifest(tmp_path)
    make_image(tmp_path / "images" / "i0.png", seed=0)
    manifest.add([ImageRecord("i0", "i0.png", "images/i0.png", 64, 48, 1, "2026")])
    truth = AnnotationSet(Task.REGRESSION, ["age"], {"i0": [Label("age", value=31.0)]}, {})
    dataset = bridge.sync_dataset(project, manifest, truth)
    tool, settings = get_tool("fiftyone"), ToolSettings("fiftyone")
    sent = tool.send(dataset, ["i0"], project, settings, "reg")
    sample = dataset.first()
    sample["ground_truth_age"] = 29.5  # edited in the App
    sample.save()
    pulled = tool.pull(dataset, ["i0"], project, settings, sent.ref)
    assert pulled["i0"] == [Label("age", value=29.5)] and pulled.edited == {"i0"}
