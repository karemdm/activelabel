"""LabelStudioRemote against a small in-memory fake of the Label Studio REST API."""

from __future__ import annotations

import io
import re
from typing import Any

import pytest
import requests
from PIL import Image

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import ImageRecord
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.sync import labelstudio_remote as lsr
from activelabel.sync.labelstudio_remote import LabelStudioRemote
from activelabel.tasks import Task

SETTINGS = ToolSettings("labelstudio", "http://ls.test:8080/", "", "pat-secret")
CONFIG = (
    '<View><Image name="img" value="$image"/>'
    '<RectangleLabels name="box" toName="img"><Label value="cat"/></RectangleLabels></View>'
)


class Resp:
    def __init__(self, status: int = 200, body: Any = None, content: bytes = b"") -> None:
        self.status_code, self._body, self.content = status, body, content

    def json(self) -> Any:
        return self._body


class FakeLs:
    """Stateful fake of the Label Studio endpoints the adapter uses."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.projects: dict[int, dict[str, Any]] = {}
        self.tasks: dict[int, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.next_id = 10
        self.fail: dict[tuple[str, str], int] = {}
        self.image = b""

    def post(self, url: str, **kw: Any) -> Resp:
        return self.request("POST", url, **kw)

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def request(self, method: str, url: str, **kw: Any) -> Resp:
        path = url.split("8080", 1)[1]
        if path.startswith("/api/token/refresh"):
            return Resp(200, {"access": "ACCESS"})
        self.calls.append((method, path, kw))
        path = path.split("?", 1)[0]
        for (m, prefix), status in self.fail.items():
            if m == method and path.startswith(prefix):
                return Resp(status, {})
        routes = [
            ("GET", r"/api/projects", self.list_projects),
            ("POST", r"/api/projects", self.post_project),
            ("GET", r"/api/projects/(\d+)", self.get_project),
            ("PATCH", r"/api/projects/(\d+)", self.patch_project),
            ("GET", r"/api/tasks", self.list_tasks),
            ("GET", r"/api/tasks/(\d+)", self.get_task),
            ("POST", r"/api/projects/(\d+)/import", self.do_import),
            ("POST", r"/api/tasks/(\d+)/annotations", self.post_ann),
            ("PATCH", r"/api/annotations/(\d+)", self.patch_ann),
            ("GET", r"/data/.*", lambda kw, *g: Resp(200, None, self.image)),
        ]
        for m, pattern, handler in routes:
            hit = re.fullmatch(pattern, path) if m == method else None
            if hit:
                return handler(kw, *hit.groups())
        return Resp(404, {})

    def add_project(self, title: str, config: str = CONFIG) -> dict[str, Any]:
        pid = self.new_id()
        self.projects[pid] = {"id": pid, "title": title, "label_config": config, "task_number": 0}
        return self.projects[pid]

    def add_task(self, pid: int, image: str, annotations: list[dict[str, Any]] | None = None) -> int:
        tid = self.new_id()
        self.tasks[tid] = {
            "id": tid,
            "project": pid,
            "data": {"image": image},
            "annotations": annotations or [],
        }
        self.projects[pid]["task_number"] += 1
        return tid

    def list_projects(self, kw: dict[str, Any]) -> Resp:
        params = kw["params"]
        rows = [p for p in self.projects.values() if params["title"] in p["title"]]
        size, page = params["page_size"], params["page"]
        chunk = rows[(page - 1) * size : page * size]
        return Resp(
            200, {"count": len(rows), "next": "x" if page * size < len(rows) else None, "results": chunk}
        )

    def post_project(self, kw: dict[str, Any]) -> Resp:
        p = self.add_project(kw["json"]["title"], kw["json"]["label_config"])
        return Resp(201, p)

    def get_project(self, kw: dict[str, Any], pid: str) -> Resp:
        return Resp(200, self.projects[int(pid)]) if int(pid) in self.projects else Resp(404, {})

    def patch_project(self, kw: dict[str, Any], pid: str) -> Resp:
        self.projects[int(pid)].update(kw["json"])
        return Resp(200, self.projects[int(pid)])

    def list_tasks(self, kw: dict[str, Any]) -> Resp:
        params = kw["params"]
        rows = [t for t in self.tasks.values() if t["project"] == int(params["project"])]
        size, page = params["page_size"], params["page"]
        if (page - 1) * size >= len(rows) and page > 1:
            return Resp(404, {})  # like DRF: invalid page
        return Resp(200, {"tasks": rows[(page - 1) * size : page * size], "total": len(rows)})

    def get_task(self, kw: dict[str, Any], tid: str) -> Resp:
        return Resp(200, self.tasks[int(tid)])

    def do_import(self, kw: dict[str, Any], pid: str) -> Resp:
        for _, (name, _fh, _mime) in kw["files"]:
            self.add_task(int(pid), f"/data/upload/1/abcdef12-{name}")
        return Resp(201, {"task_count": len(kw["files"])})

    def post_ann(self, kw: dict[str, Any], tid: str) -> Resp:
        ann = {"id": self.new_id(), "result": kw["json"]["result"], "was_cancelled": False, "updated_at": "z"}
        self.tasks[int(tid)]["annotations"].append(ann)
        return Resp(201, ann)

    def patch_ann(self, kw: dict[str, Any], aid: str) -> Resp:
        for t in self.tasks.values():
            for a in t["annotations"]:
                if a["id"] == int(aid):
                    a.update(kw["json"])
                    return Resp(200, a)
        return Resp(404, {})


@pytest.fixture
def ls(monkeypatch: pytest.MonkeyPatch) -> FakeLs:
    fake = FakeLs()
    monkeypatch.setattr(requests, "Session", lambda: fake)
    return fake


def project(task: Task = Task.DETECTION) -> Project:
    return Project(id=5, name="Pets", description="", task=task, classes=["cat", "dog"])


def rect_result(
    label: str, x: float, y: float, w: float, h: float, size: tuple[int, int] = (200, 100)
) -> dict[str, Any]:
    return {
        "id": "r1",
        "type": "rectanglelabels",
        "from_name": "box",
        "to_name": "img",
        "original_width": size[0],
        "original_height": size[1],
        "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "rectanglelabels": [label]},
    }


def ann(aid: int, result: list[dict[str, Any]], updated: str, cancelled: bool = False) -> dict[str, Any]:
    return {"id": aid, "result": result, "updated_at": updated, "was_cancelled": cancelled}


def test_find_get_and_auth(ls: FakeLs) -> None:
    p = ls.add_project("ActiveLabel 5 - Pets")
    ls.add_project("ActiveLabel 5 - Pets 2")
    ls.add_task(p["id"], "/data/upload/1/a.jpg")
    info = LabelStudioRemote().find(SETTINGS, "ActiveLabel 5 - Pets")
    assert info is not None and info.id == str(p["id"]) and info.images == 1
    assert info.url == f"http://ls.test:8080/projects/{p['id']}/data"
    assert ls.headers["Authorization"] == "Bearer ACCESS"
    assert LabelStudioRemote().find(SETTINGS, "missing") is None
    assert LabelStudioRemote().get(SETTINGS, str(p["id"])).name == "ActiveLabel 5 - Pets"


def test_find_paginates(ls: FakeLs, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lsr, "PAGE", 1)
    ls.add_project("ActiveLabel 5 - Pets 2")
    target = ls.add_project("ActiveLabel 5 - Pets")
    info = LabelStudioRemote().find(SETTINGS, "ActiveLabel 5 - Pets")
    assert info is not None and info.id == str(target["id"])
    pages = [c[2]["params"]["page"] for c in ls.calls if c[1] == "/api/projects"]
    assert pages == [1, 2]


def test_create_uses_task_config(ls: FakeLs) -> None:
    info = LabelStudioRemote().create(SETTINGS, project(Task.SEGMENTATION))
    post = next(c for c in ls.calls if c[0] == "POST" and c[1] == "/api/projects")[2]["json"]
    assert post["title"] == "ActiveLabel 5 - Pets"
    assert "<PolygonLabels" in post["label_config"] and 'value="dog"' in post["label_config"]
    assert info.name == "ActiveLabel 5 - Pets"


def test_items_latest_annotation_pixels_and_names(ls: FakeLs) -> None:
    p = ls.add_project("x")
    old = ann(1, [rect_result("cat", 0, 0, 10, 10)], "2024-01-01")
    new = ann(2, [rect_result("cat", 10, 20, 50, 50)], "2024-02-01")
    cancelled = ann(3, [rect_result("cat", 90, 90, 5, 5)], "2024-03-01", cancelled=True)
    t1 = ls.add_task(p["id"], "/data/upload/1/abcdef12-a.jpg", [old, new, cancelled])
    t2 = ls.add_task(p["id"], "/data/upload/1/b.jpg")
    items = LabelStudioRemote().items(SETTINGS, str(p["id"]), Task.DETECTION)
    assert [i.key for i in items] == [str(t1), str(t2)]
    assert items[0].file_name == "a.jpg" and (items[0].width, items[0].height) == (200, 100)
    assert items[0].labels == [Label("cat", bbox=pytest.approx([0.1, 0.2, 0.5, 0.5]))]
    assert items[1].labels == [] and items[1].file_name == "b.jpg"


def test_items_paginates(ls: FakeLs, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lsr, "PAGE", 2)
    p = ls.add_project("x")
    for n in range(4):  # an exact multiple of the page size must not ask for a missing page
        ls.add_task(p["id"], f"/data/upload/1/{n}.jpg")
    assert len(LabelStudioRemote().items(SETTINGS, str(p["id"]), Task.DETECTION)) == 4
    ls.add_task(p["id"], "/data/upload/1/4.jpg")
    assert len(LabelStudioRemote().items(SETTINGS, str(p["id"]), Task.DETECTION)) == 5


def test_write_patches_latest_and_creates_when_none(ls: FakeLs) -> None:
    p = ls.add_project("x")
    t1 = ls.add_task(p["id"], "/data/upload/1/a.jpg", [ann(1, [rect_result("cat", 0, 0, 10, 10)], "1")])
    t2 = ls.add_task(p["id"], "/data/upload/1/b.jpg")
    ls.image = _png(300, 150)
    labels = [Label("cat", bbox=[0.5, 0.5, 0.25, 0.25])]
    LabelStudioRemote().write(SETTINGS, str(p["id"]), project(), {str(t1): labels, str(t2): labels})
    patched = ls.tasks[t1]["annotations"][0]
    assert len(ls.tasks[t1]["annotations"]) == 1
    value = patched["result"][0]["value"]
    assert (value["x"], value["y"], value["width"], value["height"]) == (50.0, 50.0, 25.0, 25.0)
    assert patched["result"][0]["from_name"] == "box" and patched["result"][0]["to_name"] == "img"
    created = ls.tasks[t2]["annotations"][0]["result"][0]
    assert (created["original_width"], created["original_height"]) == (300, 150)  # image fetched


def test_write_empty_labels_clears_existing_annotation(ls: FakeLs) -> None:
    p = ls.add_project("x")
    t = ls.add_task(p["id"], "/data/upload/1/a.jpg", [ann(1, [rect_result("cat", 0, 0, 10, 10)], "1")])
    LabelStudioRemote().write(SETTINGS, str(p["id"]), project(), {str(t): []})
    assert ls.tasks[t]["annotations"][0]["result"] == []


def test_write_adds_new_class_and_rejects_bad_key(ls: FakeLs) -> None:
    p = ls.add_project("x")
    t = ls.add_task(p["id"], "/data/upload/1/a.jpg", [ann(1, [rect_result("cat", 0, 0, 1, 1)], "1")])
    LabelStudioRemote().write(SETTINGS, str(p["id"]), project(), {str(t): [Label("dog", bbox=[0, 0, 1, 1])]})
    assert 'value="dog"' in ls.projects[p["id"]]["label_config"]
    assert 'value="cat"' in ls.projects[p["id"]]["label_config"]
    with pytest.raises(ToolError, match="not a Label Studio item key"):
        LabelStudioRemote().write(SETTINGS, str(p["id"]), project(), {"x": []})


def test_ensure_classes_only_missing_and_keypoints(ls: FakeLs) -> None:
    p = ls.add_project("x")
    LabelStudioRemote().ensure_classes(SETTINGS, str(p["id"]), ["cat", "dog"])
    config = ls.projects[p["id"]]["label_config"]
    assert config.count('value="cat"') == 1 and 'value="dog"' in config
    patches = len([c for c in ls.calls if c[0] == "PATCH"])
    LabelStudioRemote().ensure_classes(SETTINGS, str(p["id"]), ["cat"])
    assert len([c for c in ls.calls if c[0] == "PATCH"]) == patches  # nothing to add: no request
    kp_config = (
        '<View><Image name="img" value="$image"/>'
        '<KeyPointLabels name="k" toName="img"><Label value="hand 1"/></KeyPointLabels></View>'
    )
    kp = ls.add_project("kp", kp_config)
    LabelStudioRemote().ensure_classes(SETTINGS, str(kp["id"]), ["hand", "foot"])
    assert 'value="foot 1"' in kp["label_config"] and kp["label_config"].count("hand") == 1


def test_add_imports_maps_by_suffix_and_creates_annotations(ls: FakeLs, tmp_path: Any) -> None:
    p = ls.add_project("x")
    ls.add_task(p["id"], "/data/upload/1/aaaaaaaa-one.jpg")  # an existing task with the same name
    images = []
    for n, name in enumerate(["sub/one.jpg", "two.jpg"]):
        f = tmp_path / f"{n}.jpg"
        f.write_bytes(b"jpg")
        rec = ImageRecord(f"id{n}", name, f"images/{n}.jpg", 200, 100, 3, "")
        images.append((rec, f, [Label("cat", bbox=[0.1, 0.2, 0.5, 0.5])] if n == 0 else []))
    keys = LabelStudioRemote().add(SETTINGS, str(p["id"]), project(), images)
    imp = next(c for c in ls.calls if c[1].startswith(f"/api/projects/{p['id']}/import"))[2]
    assert [f[1][0] for f in imp["files"]] == ["one.jpg", "two.jpg"]
    assert [f[0] for f in imp["files"]] == ["file0", "file1"]  # Label Studio keeps one file per field name
    assert set(keys) == {"id0", "id1"}
    assert int(keys["id0"]) != min(ls.tasks)  # not the pre-existing task
    result = ls.tasks[int(keys["id0"])]["annotations"][0]["result"][0]
    assert result["value"]["x"] == pytest.approx(10.0) and result["from_name"] == "box"
    assert result["original_width"] == 200
    assert ls.tasks[int(keys["id1"])]["annotations"] == []


def test_add_missing_task_is_an_error(ls: FakeLs, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    p = ls.add_project("x")
    f = tmp_path / "a.jpg"
    f.write_bytes(b"jpg")
    monkeypatch.setattr(FakeLs, "do_import", lambda self, kw, pid: Resp(201, {}))
    rec = ImageRecord("id0", "a.jpg", "images/a.jpg", 10, 10, 3, "")
    with pytest.raises(ToolError, match="did not create a task"):
        LabelStudioRemote().add(SETTINGS, str(p["id"]), project(), [(rec, f, [])])


@pytest.mark.parametrize(
    ("status", "text"),
    [(401, "refused the token"), (403, "refused the token"), (404, "not found"), (500, "error 500")],
)
def test_http_errors_are_readable_and_secret_free(ls: FakeLs, status: int, text: str) -> None:
    ls.fail[("GET", "/api/projects")] = status
    with pytest.raises(ToolError, match=text) as err:
        LabelStudioRemote().find(SETTINGS, "x")
    assert "pat-secret" not in str(err.value)


def test_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    class Down:
        headers: dict[str, str] = {}

        def post(self, *a: Any, **k: Any) -> Any:
            raise requests.ConnectionError("boom pat-secret")

    monkeypatch.setattr(requests, "Session", lambda: Down())
    with pytest.raises(ToolError, match="Cannot reach Label Studio") as err:
        LabelStudioRemote().get(SETTINGS, "1")
    assert "pat-secret" not in str(err.value)


def test_config_without_known_control(ls: FakeLs) -> None:
    p = ls.add_project("x", "<View><Text name='t' value='$text'/></View>")
    with pytest.raises(ToolError, match="no label control"):
        LabelStudioRemote().ensure_classes(SETTINGS, str(p["id"]), ["cat"])


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "PNG")
    return buf.getvalue()
