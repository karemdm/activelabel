import threading
from pathlib import Path

import pytest
import requests
from import_fakes import FakeResp, FakeSession, image_resp, make_images

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import Manifest
from activelabel.importing import labelstudio_import as lsi
from activelabel.importing.common import RemoteNotFound, read_origin
from activelabel.jobs import Cancelled, Job, Progress
from activelabel.labels import load_annotations
from activelabel.projects import ProjectStore
from activelabel.tasks import Task

BASE = "http://ls.local:8080"
SETTINGS = ToolSettings("labelstudio", BASE, "", "pat")
CONFIG = (
    '<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image">'
    '<Label value="cat"/><Label value="dog"/><Label value="bird"/></RectangleLabels></View>'
)


def rect(name="cat"):
    value = {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0, "rectanglelabels": [name]}
    return {"type": "rectanglelabels", "original_width": 64, "original_height": 48, "value": value}


def polygon(name="dog"):
    value = {"points": [[10, 10], [50, 10], [50, 60]], "polygonlabels": [name]}
    return {"type": "polygonlabels", "original_width": 64, "original_height": 48, "value": value}


def task(tid, image, *results):
    ann = [{"was_cancelled": False, "result": list(results)}] if results else []
    return {"id": tid, "data": {"image": image}, "annotations": ann}


@pytest.fixture
def server(monkeypatch, tmp_path):
    images = make_images(tmp_path / "remote", 5)

    def install(tasks, extra=None, title="Pets"):
        routes = {
            ("POST", "/api/token/refresh/"): FakeResp(200, {"access": "ACCESS"}),
            ("GET", "/api/projects/7/export"): FakeResp(200, tasks),
            ("GET", "/api/projects/7"): FakeResp(200, {"id": 7, "title": title, "label_config": CONFIG}),
            ("GET", "/data/upload/1/aaaa1111-img0.png"): image_resp(images[0]),
            ("GET", "/data/upload/1/bbbb2222-img1.png"): image_resp(images[1]),
            ("GET", "/data/local-files/?d=folder/img2.png"): image_resp(images[2]),
            **(extra or {}),
        }
        fake = FakeSession(routes, BASE)
        monkeypatch.setattr(requests, "Session", lambda: fake)
        return fake

    install.images = images
    return install


def run(store, tmp_path, progress=None, **kw):
    progress = progress or Progress(Job("j", 0, "import_project"), threading.Event())
    args = {"kind": "project", "remote_id": "7", "name": "", "description": "", "task": Task.DETECTION}
    return lsi.import_project(store, SETTINGS, tmp_path / "ws", progress=progress, **{**args, **kw})


def test_label_names():
    assert lsi.label_names(CONFIG) == ["cat", "dog", "bird"]
    choices = '<View><Choices name="c" toName="i"><Choice value="a"/><Choice value="b"/></Choices></View>'
    assert lsi.label_names(choices) == ["a", "b"]
    kp = '<View><KeyPointLabels name="k" toName="i"><Label value="nose"/></KeyPointLabels></View>'
    assert lsi.label_names(kp) == ["nose"] and lsi.label_names(kp, for_import=True) == []
    assert lsi.label_names("not xml <<") == []


def test_list_remote(server):
    page = {
        "count": 3,
        "next": None,
        "results": [
            {"id": 1, "title": "Old", "task_number": 4, "created_at": "2026-01-01", "label_config": CONFIG},
            {"id": 2, "title": "Newer pets", "task_number": 9, "created_at": "2026-05-01"},
            {"id": 3, "title": "Cars", "task_number": 1, "created_at": "2026-03-01"},
        ],
    }
    fake = server([], {("GET", "/api/projects?"): FakeResp(200, page)})
    items = lsi.list_remote(SETTINGS)
    assert [i["name"] for i in items] == ["Newer pets", "Cars", "Old"]
    assert items[2] == {
        "id": 1,
        "kind": "project",
        "name": "Old",
        "images": 4,
        "created": "2026-01-01",
        "classes": ["cat", "dog", "bird"],
    }
    assert [i["id"] for i in lsi.list_remote(SETTINGS, "PETS")] == [2]
    assert all("pat" not in str(c) for c in fake.calls if c[0] != "POST")


def test_list_remote_errors(server):
    server([], {("GET", "/api/projects?"): FakeResp(401)})
    with pytest.raises(ToolError, match="refused the token"):
        lsi.list_remote(SETTINGS)


def test_check_remote(server):
    server([])
    assert lsi.check_remote(SETTINGS, "project", "7")["name"] == "Pets"
    with pytest.raises(RemoteNotFound):
        lsi.check_remote(SETTINGS, "project", "99")
    with pytest.raises(RemoteNotFound):
        lsi.check_remote(SETTINGS, "project", "abc")
    with pytest.raises(ToolError):
        lsi.check_remote(SETTINGS, "task", "7")


def test_import_end_to_end(server, tmp_path, monkeypatch):
    tasks = [
        task(1, "/data/upload/1/aaaa1111-img0.png", rect("cat"), rect("dog")),
        task(2, "/data/upload/1/bbbb2222-img1.png", rect("cat")),
        task(3, "/data/local-files/?d=folder/img2.png", polygon("dog")),
        task(4, "/data/upload/1/missing0-gone.png", rect("cat")),  # 404: reported, skipped
        task(5, "http://cdn.example/img3.png"),  # absolute URL, no annotation
    ]
    fake = server(tasks)
    cdn = []

    def cdn_get(url, **kw):
        cdn.append((url, kw))
        return image_resp(server.images[3])

    monkeypatch.setattr(requests, "get", cdn_get)
    store = ProjectStore(tmp_path / "projects")
    store.create("Pets", "", "detection")
    result = run(store, tmp_path)
    assert result["name"] == "Pets (2)" and result["task"] == "detection"
    assert (result["images_added"], result["images_rejected"], result["labels"]) == (4, 0, 4)
    assert result["classes"] == ["cat", "dog", "bird"]
    problems = {p["code"]: p for p in result["problems"]}
    assert problems["image_download_failed"]["count"] == 1
    assert problems["converted_to_box"]["count"] == 1
    assert "unmatched_entries" not in problems
    assert cdn and "Authorization" not in cdn[0][1].get("headers", {})  # the token stays on the server
    project = store.get(result["project_id"])
    assert project.description == ""
    names = sorted(r.name for r in Manifest(store.dir(project.id)).all())
    assert names == ["folder/img2.png", "img0.png", "img1.png", "img3.png"]
    ann = load_annotations(store.dir(project.id))
    assert ann is not None and ann.task is Task.DETECTION
    assert sorted(len(v) for v in ann.images.values()) == [1, 1, 2]
    origin = read_origin(store.dir(project.id))
    assert origin is not None and origin["tool"] == "labelstudio" and origin["kind"] == "project"
    assert origin["remote_id"] == "7" and origin["name"] == "Pets" and "imported_at" in origin
    assert "pat" not in str(origin)
    assert list((tmp_path / "ws" / "tmp").glob("import-*")) == []  # temporary folder removed
    assert any(c[1].startswith("/data/local-files/?d=folder") for c in fake.calls)


def test_no_annotations_uses_given_task(server, tmp_path):
    server([task(1, "/data/upload/1/aaaa1111-img0.png")])
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, task=Task.SEGMENTATION, name="Mine", description="text")
    assert result["task"] == "segmentation" and result["labels"] == 0
    assert result["classes"] == ["cat", "dog", "bird"]  # declared classes are kept
    assert load_annotations(store.dir(result["project_id"])) is None
    project = store.get(result["project_id"])
    assert (project.name, project.description) == ("Mine", "text")


def test_all_downloads_fail_creates_nothing(server, tmp_path):
    server([task(1, "/data/upload/9/zzzz9999-x.png", rect())])
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(ToolError, match="No image"):
        run(store, tmp_path)
    assert store.list() == []


def test_rollback_when_parsing_fails(server, tmp_path, monkeypatch):
    server([task(1, "/data/upload/1/aaaa1111-img0.png", rect())])
    monkeypatch.setattr(lsi, "parse_labelstudio", lambda ctx, files: (_ for _ in ()).throw(ValueError("x")))
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(ValueError):
        run(store, tmp_path)
    assert store.list() == [] and not any(Path(tmp_path / "projects").glob("*/images"))


def test_cancel_removes_half_created_project(server, tmp_path):
    tasks = [
        task(1, "/data/upload/1/aaaa1111-img0.png", rect()),
        task(2, "/data/upload/1/bbbb2222-img1.png", rect()),
    ]
    cancel = threading.Event()

    def second_image(kw):
        cancel.set()  # the user cancels while the second image is downloading
        return image_resp(server.images[1])

    server(tasks, {("GET", "/data/upload/1/bbbb2222-img1.png"): second_image})
    store = ProjectStore(tmp_path / "projects")
    progress = Progress(Job("j", 0, "import_project"), cancel)
    with pytest.raises(Cancelled):
        run(store, tmp_path, progress)
    assert store.list() == []


def test_cancel_during_project_creation(server, tmp_path, monkeypatch):
    server([task(1, "/data/upload/1/aaaa1111-img0.png", rect())])
    cancel = threading.Event()
    orig = lsi.ingest_folder

    def ingest_then_cancel(*a, **kw):
        report = orig(*a, **kw)
        cancel.set()
        return report

    monkeypatch.setattr(lsi, "ingest_folder", ingest_then_cancel)
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(Cancelled):
        run(store, tmp_path, Progress(Job("j", 0, "import_project"), cancel))
    assert store.list() == []


def test_token_refused_stops_import(server, tmp_path):
    server(
        [task(1, "/data/upload/1/aaaa1111-img0.png", rect())],
        {("GET", "/data/upload/1/aaaa1111-img0.png"): FakeResp(403)},
    )
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(ToolError, match="refused the token"):
        run(store, tmp_path)
    assert store.list() == []


def test_task_data_cannot_make_the_server_fetch_internal_addresses(monkeypatch):
    from activelabel.importing import labelstudio_import as imp

    for host in (
        "127.0.0.1",
        "localhost",
        "169.254.169.254",
        "10.1.2.3",
        "192.168.0.7",
        "[::1]",
        "[::ffff:127.0.0.1]",
    ):
        with pytest.raises(ToolError, match="not a public"):
            imp._refuse_non_public(f"http://{host}:8080/x.png", "ls.example")
    imp._refuse_non_public("http://ls.example/x.png", "ls.example")  # the Label Studio server itself
    imp._refuse_non_public("http://no-such-host.invalid/x.png", "ls.example")  # fails later by itself
    monkeypatch.setattr(imp.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
    imp._refuse_non_public("http://cdn.example/x.png", "ls.example")


def test_a_redirect_to_an_internal_address_is_refused(monkeypatch):
    from activelabel.importing import labelstudio_import as imp

    monkeypatch.setattr(imp.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])

    class Redirect:
        status_code = 302
        headers = {"Location": "http://169.254.169.254/latest/meta-data"}

    seen = []
    monkeypatch.setattr(requests, "get", lambda url, **kw: seen.append(url) or Redirect())
    monkeypatch.setattr(
        imp.socket,
        "getaddrinfo",
        lambda host, port: [
            (2, 1, 6, "", ("169.254.169.254" if host.startswith("169") else "93.184.216.34", 0))
        ],
    )
    with pytest.raises(ToolError, match="not a public"):
        imp._get_public("http://cdn.example/a.png", "ls.example")
    assert seen == ["http://cdn.example/a.png"]  # the internal address was never requested
