from unittest.mock import MagicMock

import pytest
import requests
from PIL import Image

from activelabel.annotate import labelstudio_tool as ls
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.annotate.labelstudio_tool import LabelStudioTool, from_results, labeling_config, to_results
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.tasks import Task

SETTINGS = ToolSettings("labelstudio", "http://ls.local:8080/", "", "pat-token")


def project(task=Task.DETECTION):
    return Project(id=2, name="Pets", description="", task=task, classes=["cat", "dog"])


def resp(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body if body is not None else {}
    return r


class FakeServer:
    """Replaces requests.Session: records calls and answers from a route table."""

    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def request(self, method, url, **kw):
        path = url.split("8080", 1)[1]
        self.calls.append((method, path, kw))
        for (m, prefix), reply in self.routes.items():
            if m == method and path.startswith(prefix):
                return reply
        return resp(404)


@pytest.fixture
def server(monkeypatch):
    def install(routes):
        fake = FakeServer({("POST", "/api/token/refresh/"): resp(200, {"access": "ACCESS"}), **routes})
        monkeypatch.setattr(ls.requests, "Session", lambda: fake)
        return fake

    return install


def test_config_per_task():
    assert "<Choices" in labeling_config(Task.CLASSIFICATION, ["a"])
    assert "<RectangleLabels" in labeling_config(Task.DETECTION, ["a"])
    assert "<PolygonLabels" in labeling_config(Task.SEGMENTATION, ["a"])
    cfg = labeling_config(Task.KEYPOINTS, ["hand"], 2)
    assert "<KeyPointLabels" in cfg and 'value="hand 2"' in cfg
    assert "value='a\"b'" in labeling_config(Task.DETECTION, ['a"b'])


def test_roundtrip_all_tasks():
    det = [Label("cat", bbox=[0.1, 0.2, 0.3, 0.4])]
    back = from_results(to_results(det, Task.DETECTION, (100, 50)), Task.DETECTION)
    assert back[0].label == "cat" and back[0].bbox == pytest.approx([0.1, 0.2, 0.3, 0.4])
    poly = [Label("dog", polygons=[[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]])]
    back = from_results(to_results(poly, Task.SEGMENTATION, (10, 10)), Task.SEGMENTATION)
    assert back[0].polygons[0][1] == pytest.approx([0.5, 0.1])
    cls = from_results(to_results([Label("cat")], Task.CLASSIFICATION, (1, 1)), Task.CLASSIFICATION)
    assert cls == [Label("cat")]
    kp = [Label("hand", keypoints=[[0.1, 0.1, 2], [0.0, 0.0, 0], [0.7, 0.7, 2]])]
    res = to_results(kp, Task.KEYPOINTS, (10, 10))
    assert len(res) == 2  # the not-labeled point is skipped
    back = from_results(res, Task.KEYPOINTS, keypoints=3)
    assert back[0].keypoints[1] == [0.0, 0.0, 0.0]
    assert back[0].keypoints[2][:2] == pytest.approx([0.7, 0.7])


def test_two_objects_same_class_keypoints():
    kp = [Label("hand", keypoints=[[0.1, 0.1, 2]]), Label("hand", keypoints=[[0.9, 0.9, 2]])]
    back = from_results(to_results(kp, Task.KEYPOINTS, (10, 10)), Task.KEYPOINTS)
    assert len(back) == 2


def test_rotated_box_becomes_enclosing_box():
    value = {"x": 10, "y": 10, "width": 20, "height": 10, "rotation": 90, "rectanglelabels": ["cat"]}
    box = from_results([{"type": "rectanglelabels", "value": value}], Task.DETECTION)[0].bbox
    assert box[2] == pytest.approx(0.1) and box[3] == pytest.approx(0.2)


def test_check_uses_bearer_and_reports_email(server):
    fake = server({("GET", "/api/current-user/whoami"): resp(200, {"email": "a@b.c"})})
    assert LabelStudioTool().check(SETTINGS) == "Connected to http://ls.local:8080 as a@b.c"
    assert fake.headers["Authorization"] == "Bearer ACCESS"
    assert fake.calls[0][2]["json"] == {"refresh": "pat-token"}


def test_check_legacy_token_fallback(server):
    fake = server(
        {
            ("POST", "/api/token/refresh/"): resp(401),
            ("GET", "/api/current-user/whoami"): resp(200, {"email": "x"}),
        }
    )
    LabelStudioTool().check(SETTINGS)
    assert fake.headers["Authorization"] == "Token pat-token"


def test_check_401(server):
    server({("GET", "/api/current-user/whoami"): resp(401)})
    with pytest.raises(ToolError, match="refused the token") as exc:
        LabelStudioTool().check(SETTINGS)
    assert "pat-token" not in str(exc.value)


def test_check_unreachable(monkeypatch):
    class Down:
        headers: dict = {}

        def post(self, *a, **k):
            raise requests.ConnectionError("boom")

    monkeypatch.setattr(ls.requests, "Session", Down)
    with pytest.raises(ToolError, match="Cannot reach"):
        LabelStudioTool().check(SETTINGS)


def test_check_validates_settings():
    with pytest.raises(ToolError, match="http"):
        LabelStudioTool().check(ToolSettings("labelstudio", "localhost:8080", "", "t"))
    with pytest.raises(ToolError, match="token"):
        LabelStudioTool().check(ToolSettings("labelstudio", "http://x", "", ""))


def make_dataset(tmp_path, size=(40, 20)):
    img = tmp_path / "x.jpg"
    Image.new("RGB", size).save(img)
    dataset, view = MagicMock(), MagicMock()
    dataset.select_by.return_value = view
    data = {"image_id": ["a"], "filepath": [str(img)], "ground_truth": [object()]}
    view.values.side_effect = lambda f: data[f]
    return dataset


def test_send_creates_project_uploads_and_annotates(server, monkeypatch, tmp_path):
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(ls, "_labels_from_fo", lambda v, t: [Label("cat", bbox=[0.5, 0.5, 0.25, 0.25])])
    tasks = {"tasks": [{"id": 7, "data": {"image": "/data/upload/2/1a2b3c4d-a.jpg"}}]}
    fake = server(
        {
            ("POST", "/api/projects/9/import"): resp(201),
            ("POST", "/api/projects"): resp(201, {"id": 9}),
            ("GET", "/api/tasks"): resp(200, tasks),
            ("POST", "/api/tasks/7/annotations"): resp(201),
        }
    )
    sent = LabelStudioTool().send(dataset, ["a"], project(), SETTINGS, "sess")
    assert sent.ref["project_id"] == 9
    assert sent.links == [{"label": "Label Studio project", "url": "http://ls.local:8080/projects/9/data"}]
    created = next(k for m, p, k in fake.calls if (m, p) == ("POST", "/api/projects"))["json"]
    assert "RectangleLabels" in created["label_config"] and "[sess]" in created["title"]
    upload = next(k for m, p, k in fake.calls if p.startswith("/api/projects/9/import"))
    assert upload["files"][0][1][0] == "a.jpg"
    ann = next(k for m, p, k in fake.calls if p == "/api/tasks/7/annotations")["json"]["result"][0]
    assert ann["value"]["x"] == 50 and ann["original_width"] == 40


def test_send_failure_deletes_project(server, monkeypatch, tmp_path):
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(ls, "_labels_from_fo", lambda v, t: [])
    fake = server(
        {
            ("POST", "/api/projects/9/import"): resp(500),
            ("POST", "/api/projects"): resp(201, {"id": 9}),
            ("DELETE", "/api/projects/9"): resp(204),
        }
    )
    with pytest.raises(ToolError, match="500"):
        LabelStudioTool().send(dataset, ["a"], project(), SETTINGS, "s")
    assert ("DELETE", "/api/projects/9") in [(m, p) for m, p, _ in fake.calls]


def test_pull_uses_latest_annotation(server):
    old = {"updated_at": "2026-01-01", "result": []}
    new = {"updated_at": "2026-02-01", "result": [{"type": "choices", "value": {"choices": ["dog"]}}]}
    cancelled = {"updated_at": "2026-03-01", "was_cancelled": True, "result": []}
    export = [
        {"data": {"image": "/data/upload/2/1a2b3c4d-a.jpg"}, "annotations": [old, new, cancelled]},
        {"data": {"image": "/data/upload/2/1a2b3c4d-b.jpg"}, "annotations": []},
        {"data": {"image": "/data/upload/2/1a2b3c4d-zz.jpg"}, "annotations": [new]},
    ]
    server({("GET", "/api/projects/9/export"): resp(200, export)})
    task = project(Task.CLASSIFICATION)
    out = LabelStudioTool().pull(MagicMock(), ["a", "b", "c"], task, SETTINGS, {"project_id": 9})
    assert out == {"a": [Label("dog")], "b": [], "c": []}


def test_close(server):
    fake = server({("DELETE", "/api/projects/9"): resp(204)})
    LabelStudioTool().close(MagicMock(), SETTINGS, {"project_id": 9}, False)
    assert not [c for c in fake.calls if c[0] == "DELETE"]
    LabelStudioTool().close(MagicMock(), SETTINGS, {"project_id": 9}, True)
    assert ("DELETE", "/api/projects/9") in [(m, p) for m, p, _ in fake.calls]
    server({("DELETE", "/api/projects/9"): resp(404)})
    LabelStudioTool().close(MagicMock(), SETTINGS, {"project_id": 9}, True)  # already gone: no error


def test_image_id_parsing():
    assert ls._image_id({"data": {"image": "/data/upload/1/1a2b3c4d-my.img.jpg?x=1"}}) == "my.img"


def test_rotated_box_on_a_wide_image_matches_the_import():
    from activelabel.annotate.labelstudio_tool import from_results
    from activelabel.annotations.labelstudio import _rect_box as pixel_box
    from activelabel.tasks import Task

    value = {"x": 40, "y": 10, "width": 20, "height": 40, "rotation": 30, "rectanglelabels": ["c"]}
    result = {"type": "rectanglelabels", "original_width": 2000, "original_height": 1000, "value": value}
    (label,) = from_results([result], Task.DETECTION)
    x, y, w, h = pixel_box(value, 2000, 1000)
    assert label.bbox == pytest.approx([x / 2000, y / 1000, w / 2000, h / 1000], abs=1e-6)
    assert label.bbox == pytest.approx([0.3, 0.1, 0.2732, 0.5464], abs=1e-3)
