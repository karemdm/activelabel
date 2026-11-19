import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from activelabel.annotate import cvat_tool
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.annotate.cvat_tool import CvatTool
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.tasks import Task

SETTINGS = ToolSettings("cvat", "https://cvat.example/", "ana", "s3cret-pw")


def project(task=Task.DETECTION, classes=("cat", "dog")):
    return Project(id=3, name="Pets", description="", task=task, classes=list(classes))


def test_anno_key_is_clean():
    assert cvat_tool.anno_key_of("2026-11-19-ab") == "al_2026_11_19_ab"


@pytest.mark.parametrize(
    ("task", "kind"),
    [
        (Task.CLASSIFICATION, "classification"),
        (Task.DETECTION, "detections"),
        (Task.SEGMENTATION, "polylines"),
        (Task.KEYPOINTS, "keypoints"),
    ],
)
def test_label_schema_per_task(task, kind):
    schema = cvat_tool.label_schema(project(task))["ground_truth"]
    assert schema["type"] == kind
    assert schema["classes"] == ["cat", "dog"]
    assert schema["allow_additions"] and schema["allow_deletions"] and schema["allow_label_edits"]


def test_schema_needs_classes():
    with pytest.raises(ToolError, match="no classes"):
        cvat_tool.label_schema(project(classes=()))


def test_send_arguments_and_links():
    dataset, view = MagicMock(), MagicMock()
    dataset.select_by.return_value = view
    view.annotate.return_value = SimpleNamespace(task_ids=[11, 12])
    sent = CvatTool().send(dataset, ["a", "b"], project(), SETTINGS, "s-1")
    dataset.select_by.assert_called_once_with("image_id", ["a", "b"])
    args, kw = view.annotate.call_args
    assert args == ("al_s_1",)
    assert kw["backend"] == "cvat" and kw["task_size"] == 50 and kw["launch_editor"] is False
    assert kw["project_name"] == "ActiveLabel 3 - Pets"
    assert kw["url"] == "https://cvat.example" and kw["password"] == "s3cret-pw"
    assert kw["label_schema"]["ground_truth"]["type"] == "detections"
    assert sent.ref == {"anno_key": "al_s_1"}
    assert [x["url"] for x in sent.links] == [
        "https://cvat.example/tasks/11",
        "https://cvat.example/tasks/12",
    ]


def test_send_failure_is_readable_and_secret_free():
    dataset, view = MagicMock(), MagicMock()
    dataset.select_by.return_value = view
    dataset.list_annotation_runs.return_value = ["al_s"]
    view.annotate.side_effect = RuntimeError("Request to https://cvat.example/api/auth/login failed: 401")
    with pytest.raises(ToolError, match="login failed") as exc:
        CvatTool().send(dataset, ["a"], project(), SETTINGS, "s")
    assert "s3cret-pw" not in str(exc.value)
    dataset.delete_annotation_run.assert_called_once_with("al_s")


def test_send_missing_credentials():
    with pytest.raises(ToolError, match="needs"):
        CvatTool().send(MagicMock(), ["a"], project(), ToolSettings("cvat", "https://x"), "s")


def test_check_ok_and_errors(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(cvat_tool, "_login", lambda s: api)
    assert CvatTool().check(SETTINGS) == "Connected to https://cvat.example as ana"
    api.close.assert_called_once()

    def boom(s):
        raise RuntimeError("401 Client Error: Unauthorized for url: .../api/auth/login")

    monkeypatch.setattr(cvat_tool, "_login", boom)
    with pytest.raises(ToolError, match="username and password"):
        CvatTool().check(SETTINGS)

    import requests

    def down(s):
        raise requests.ConnectionError("Max retries exceeded")

    monkeypatch.setattr(cvat_tool, "_login", down)
    with pytest.raises(ToolError, match="Cannot reach"):
        CvatTool().check(SETTINGS)


def test_pull_converts_and_cleans(monkeypatch):
    dataset, view = MagicMock(), MagicMock()
    dataset.select_by.return_value = view
    kp = SimpleNamespace(points=[(0.0, 0.0), (0.5, 0.5)])
    value = SimpleNamespace(keypoints=[kp])
    view.values.side_effect = lambda f: ["a", "b"] if f == "image_id" else [value, None]
    dataset.get_field_schema.return_value = {"_al_pull_al_s": object()}
    seen = []

    def convert(v, task):
        seen.append(list(v.keypoints[0].points))
        return [Label("hand")]

    monkeypatch.setattr(cvat_tool, "_labels_from_fo", convert)
    out = CvatTool().pull(dataset, ["a", "b", "c"], project(Task.KEYPOINTS), SETTINGS, {"anno_key": "al_s"})
    _, kw = dataset.load_annotations.call_args
    assert kw["dest_field"] == "_al_pull_al_s" and kw["unexpected"] == "keep" and kw["cleanup"] is False
    assert out == {"a": [Label("hand")], "b": [], "c": []}
    assert math.isnan(seen[0][0][0]) and seen[0][1] == (0.5, 0.5)  # corner -> missing
    dataset.delete_sample_field.assert_called_once_with("_al_pull_al_s")


def test_pull_error_still_deletes_field():
    dataset = MagicMock()
    dataset.load_annotations.side_effect = RuntimeError("404 not found")
    dataset.get_field_schema.return_value = {"_al_pull_k": 1}
    with pytest.raises(ToolError, match="not found"):
        CvatTool().pull(dataset, ["a"], project(), SETTINGS, {"anno_key": "k"})
    dataset.delete_sample_field.assert_called_once()


def test_close_cleans_remote_and_ignores_missing():
    dataset, results = MagicMock(), MagicMock()
    dataset.list_annotation_runs.return_value = ["k"]
    dataset.load_annotation_results.return_value = results
    CvatTool().close(dataset, SETTINGS, {"anno_key": "k"}, True)
    results.cleanup.assert_called_once()
    dataset.delete_annotation_run.assert_called_once_with("k")

    gone = MagicMock()
    gone.list_annotation_runs.return_value = ["k"]
    gone.load_annotation_results.side_effect = ValueError("task 404 not found")
    CvatTool().close(gone, SETTINGS, {"anno_key": "k"}, True)
    gone.delete_annotation_run.assert_called_once_with("k")

    none = MagicMock()
    none.list_annotation_runs.return_value = []
    CvatTool().close(none, SETTINGS, {"anno_key": "k"}, True)
    none.delete_annotation_run.assert_not_called()


def test_close_keeps_remote_when_not_asked():
    dataset = MagicMock()
    dataset.list_annotation_runs.return_value = ["k"]
    CvatTool().close(dataset, SETTINGS, {"anno_key": "k"}, False)
    dataset.load_annotation_results.assert_not_called()
    dataset.delete_annotation_run.assert_called_once_with("k")


@pytest.mark.integration
def test_pull_with_real_labels():
    import fiftyone as fo

    dataset = MagicMock()
    view = MagicMock()
    dataset.select_by.return_value = view
    value = fo.Detections(detections=[fo.Detection(label="cat", bounding_box=[0.1, 0.1, 0.2, 0.2])])
    view.values.side_effect = lambda f: ["a"] if f == "image_id" else [value]
    dataset.get_field_schema.return_value = {}
    out = CvatTool().pull(dataset, ["a"], project(), SETTINGS, {"anno_key": "k"})
    assert out["a"][0].label == "cat"
