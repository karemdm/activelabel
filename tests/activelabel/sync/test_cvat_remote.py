"""CvatRemote against a small in-memory fake of the CVAT REST API."""

from __future__ import annotations

import copy
import io
import re
import zipfile
from typing import Any

import pytest
import requests

from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import ImageRecord
from activelabel.importing import cvat_import
from activelabel.labels import Label
from activelabel.projects import Project
from activelabel.sync import cvat_remote, links
from activelabel.sync.cvat_remote import CvatRemote, _meta_frames
from activelabel.tasks import Task

SETTINGS = ToolSettings("cvat", "https://cvat.test/", "ana", "s3cret-pw")


class Resp:
    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else b"x"

    def json(self) -> Any:
        return self._body


class FakeCvat:
    """Stateful fake of the parts of CVAT API v2 the adapter uses."""

    def __init__(self) -> None:
        self.projects: dict[int, dict[str, Any]] = {}
        self.labels: list[dict[str, Any]] = []  # {id, name, project_id}
        self.tasks: dict[int, dict[str, Any]] = {}  # id -> {name, project_id, frames, ann}
        self.next_id = 100
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.auth: Any = None
        self.polls = 0
        self.fail: dict[tuple[str, str], int] = {}
        self.backup_polls = 0
        self.backup_zip = b""
        self.status_only = False  # old CVAT without /api/requests

    # requests.Session surface
    def request(self, method: str, url: str, **kw: Any) -> Resp:
        path = url.split("cvat.test", 1)[1]
        self.auth = kw.get("auth")
        self.calls.append((method, path, kw))
        for (m, prefix), status in self.fail.items():
            if m == method and path.startswith(prefix):
                return Resp(status, {})
        for (m, pattern), handler in self.routes():
            hit = re.fullmatch(pattern, path) if m == method else None
            if hit:
                return handler(kw, *hit.groups())
        return Resp(404, {})

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def page(self, rows: list[Any], kw: dict[str, Any]) -> Resp:
        params = kw.get("params") or {}
        size, page = int(params.get("page_size", 100)), int(params.get("page", 1))
        chunk = rows[(page - 1) * size : page * size]
        more = page * size < len(rows)
        return Resp(200, {"count": len(rows), "next": "x" if more else None, "results": chunk})

    def add_project(self, name: str, labels: list[str]) -> dict[str, Any]:
        pid = self.new_id()
        self.projects[pid] = {"id": pid, "name": name}
        for n in labels:
            self.labels.append({"id": self.new_id(), "name": n, "project_id": pid})
        return self.projects[pid]

    def add_task(
        self, pid: int | None, frames: list[tuple[str, int, int]], ann: dict[str, Any] | None = None
    ) -> int:
        tid = self.new_id()
        self.tasks[tid] = {
            "id": tid,
            "name": f"t{tid}",
            "project_id": pid,
            "frames": [{"name": n, "width": w, "height": h} for n, w, h in frames],
            "ann": ann or {"version": 0, "tags": [], "shapes": [], "tracks": []},
        }
        return tid

    def label_id(self, pid: int, name: str) -> int:
        return next(x["id"] for x in self.labels if x["project_id"] == pid and x["name"] == name)

    def routes(self) -> list[tuple[tuple[str, str], Any]]:
        def meta(kw: dict[str, Any], t: str) -> Resp:
            task = self.tasks[int(t)]
            return Resp(
                200,
                {"frames": task["frames"], "deleted_frames": task.get("deleted", []), **task.get("meta", {})},
            )

        def backup_export(kw: dict[str, Any], t: str) -> Resp:
            self.backup_polls = 0
            return Resp(202, {"rq_id": f"bk{t}"})

        def backup_request(kw: dict[str, Any], rq: str) -> Resp:
            self.backup_polls += 1
            if self.backup_polls < 2:
                return Resp(200, {"status": "started"})
            return Resp(200, {"status": "finished", "result_url": f"https://cvat.test/api/backup/{rq}.zip"})

        def backup_file(kw: dict[str, Any], rq: str) -> Resp:
            r = Resp(200, {})
            r.content = self.backup_zip
            return r

        def ann(kw: dict[str, Any], t: str) -> Resp:
            stored = self.tasks[int(t)]["ann"]
            for kind in ("tags", "shapes", "tracks"):  # CVAT gives every stored object an id
                for item in stored.get(kind) or []:
                    item.setdefault("id", self.new_id())
            return Resp(200, copy.deepcopy(stored))

        return [
            (("GET", r"/api/projects"), self.list_projects),
            (("POST", r"/api/projects"), self.post_project),
            (("GET", r"/api/projects/(\d+)"), lambda kw, i: self.get_project(int(i))),
            (("PATCH", r"/api/projects/(\d+)"), self.patch_project),
            (("GET", r"/api/labels"), self.list_labels),
            (("GET", r"/api/tasks"), self.list_tasks),
            (("POST", r"/api/tasks"), self.post_task),
            (("DELETE", r"/api/tasks/(\d+)"), self.delete_task),
            (("GET", r"/api/users/self"), lambda kw: Resp(200, {"username": "ana"})),
            (("GET", r"/api/server/about"), lambda kw: Resp(200, {"name": "CVAT"})),
            (("GET", r"/api/tasks/(\d+)/data/meta"), meta),
            (("GET", r"/api/tasks/(\d+)/annotations"), ann),
            (("PUT", r"/api/tasks/(\d+)/annotations"), self.put_ann),
            (("PATCH", r"/api/tasks/(\d+)/annotations"), self.patch_ann),
            (("POST", r"/api/tasks/(\d+)/data"), self.post_data),
            (("POST", r"/api/tasks/(\d+)/backup/export"), backup_export),
            (("GET", r"/api/requests/(bk\w+)"), backup_request),
            (("GET", r"/api/backup/(\w+)\.zip"), backup_file),
            (("POST", r"/api/tasks/(\d+)/backup/export"), backup_export),
            (("GET", r"/api/requests/(bk\w+)"), backup_request),
            (("GET", r"/api/backup/(\w+)\.zip"), backup_file),
            (("GET", r"/api/requests/(\w+)"), self.get_request),
            (("GET", r"/api/tasks/(\d+)/status"), self.get_status),
        ]

    def list_projects(self, kw: dict[str, Any]) -> Resp:
        name = (kw.get("params") or {}).get("name")
        rows = [p for p in self.projects.values() if name is None or name in p["name"]]
        return self.page(rows, kw)

    def post_project(self, kw: dict[str, Any]) -> Resp:
        p = self.add_project(kw["json"]["name"], [x["name"] for x in kw["json"]["labels"]])
        return Resp(201, p)

    def get_project(self, pid: int) -> Resp:
        return Resp(200, self.projects[pid]) if pid in self.projects else Resp(404, {})

    def patch_project(self, kw: dict[str, Any], pid: str) -> Resp:
        for lb in kw["json"]["labels"]:
            self.labels.append({"id": self.new_id(), "name": lb["name"], "project_id": int(pid)})
        return Resp(200, self.projects[int(pid)])

    def list_labels(self, kw: dict[str, Any]) -> Resp:
        pid = int(kw["params"]["project_id"])
        return self.page([x for x in self.labels if x["project_id"] == pid], kw)

    def list_tasks(self, kw: dict[str, Any]) -> Resp:
        wanted = kw["params"].get("project_id")  # absent: every task of the server
        rows = [
            {"id": t["id"], "name": t["name"], "size": len(t["frames"]), "project_id": t["project_id"]}
            for t in self.tasks.values()
            if wanted is None or t["project_id"] == int(wanted)
        ]
        return self.page(rows, kw)

    def post_task(self, kw: dict[str, Any]) -> Resp:
        tid = self.add_task(kw["json"]["project_id"], [])
        self.tasks[tid]["name"] = kw["json"]["name"]
        return Resp(201, {"id": tid})

    def delete_task(self, kw: dict[str, Any], t: str) -> Resp:
        self.tasks.pop(int(t), None)
        return Resp(204)

    def put_ann(self, kw: dict[str, Any], t: str) -> Resp:
        self.tasks[int(t)]["ann"] = copy.deepcopy(kw["json"])
        return Resp(200, kw["json"])

    def patch_ann(self, kw: dict[str, Any], t: str) -> Resp:
        """``PATCH ?action=create|update|delete``: objects by id, new ones get an id."""
        stored = self.tasks[int(t)]["ann"]
        action = kw["params"]["action"]
        for kind in ("tags", "shapes", "tracks"):
            items = copy.deepcopy(kw["json"].get(kind) or [])
            by_id = {x.get("id"): n for n, x in enumerate(stored[kind])}
            if action == "create":
                stored[kind] += [{**x, "id": self.new_id()} for x in items]
                continue
            assert all(x.get("id") in by_id for x in items), "unknown id"
            if action == "delete":
                gone = {x["id"] for x in items}
                stored[kind] = [x for x in stored[kind] if x.get("id") not in gone]
            else:
                for x in items:
                    stored[kind][by_id[x["id"]]] = x
        return Resp(200, kw["json"])

    def post_data(self, kw: dict[str, Any], t: str) -> Resp:
        names = [f[1][0] for f in kw["files"]]
        # CVAT keeps the upload order (predefined sorting)
        self.tasks[int(t)]["frames"] = [{"name": n, "width": 200, "height": 100} for n in names]
        return Resp(202, {} if self.status_only else {"rq_id": f"rq{t}"})

    def get_request(self, kw: dict[str, Any], rq: str) -> Resp:
        if self.status_only:
            return Resp(404, {})
        self.polls += 1
        return Resp(200, {"status": "finished" if self.polls >= 2 else "started"})

    def get_status(self, kw: dict[str, Any], t: str) -> Resp:
        self.polls += 1
        return Resp(200, {"state": "Finished" if self.polls >= 2 else "Started"})


@pytest.fixture
def cvat(monkeypatch: pytest.MonkeyPatch) -> FakeCvat:
    fake = FakeCvat()
    monkeypatch.setattr(requests, "Session", lambda: fake)
    monkeypatch.setattr(cvat_remote, "_sleep", lambda s: None)
    return fake


def project(task: Task = Task.DETECTION) -> Project:
    return Project(id=7, name="Pets", description="", task=task, classes=["cat", "dog"])


def rect(frame: int, lid: int, pts: list[float]) -> dict[str, Any]:
    return {
        "type": "rectangle",
        "frame": frame,
        "label_id": lid,
        "points": pts,
        "occluded": False,
        "attributes": [],
    }


def seeded(cvat: FakeCvat) -> tuple[int, int]:
    p = cvat.add_project("ActiveLabel 7 - Pets", ["cat", "dog"])
    cat, dog = cvat.label_id(p["id"], "cat"), cvat.label_id(p["id"], "dog")
    t = cvat.add_task(
        p["id"],
        [("a.jpg", 200, 100), ("b.jpg", 400, 200)],
        {
            "version": 3,
            "tags": [],
            "shapes": [rect(0, cat, [20, 10, 120, 60]), rect(1, dog, [0, 0, 400, 200])],
            "tracks": [{"label_id": cat, "frame": 0, "shapes": [{"frame": 0, "points": [1, 1, 5, 5]}]}],
        },
    )
    return p["id"], t


def test_find_exact_name_and_get(cvat: FakeCvat) -> None:
    pid, _ = seeded(cvat)
    cvat.add_project("ActiveLabel 7 - Pets 2", [])
    info = CvatRemote().find(SETTINGS, "ActiveLabel 7 - Pets")
    assert info is not None and info.id == str(pid) and info.images == 2
    assert info.url == f"https://cvat.test/projects/{pid}"
    assert CvatRemote().find(SETTINGS, "nothing") is None
    assert CvatRemote().get(SETTINGS, str(pid)).name == "ActiveLabel 7 - Pets"
    assert cvat.auth == ("ana", "s3cret-pw")


def test_check_logs_in_and_names_the_user(cvat: FakeCvat) -> None:
    assert CvatRemote().check(SETTINGS) == "Connected to https://cvat.test as ana"
    cvat.fail[("GET", "/api/users/self")] = 401
    with pytest.raises(ToolError, match="username and password"):
        CvatRemote().check(SETTINGS)


def named_task(cvat: FakeCvat, pid: int | None, name: str, frames: int = 1) -> int:
    tid = cvat.add_task(pid, [(f"{i}.jpg", 10, 10) for i in range(frames)])
    cvat.tasks[tid]["name"] = name
    return tid


def test_audit_lists_tasks_per_project_and_groups_the_loose_ones(cvat: FakeCvat) -> None:
    a = cvat.add_project("ActiveLabel 7 - Pets", ["cat"])["id"]
    b = cvat.add_project("Empty", [])["id"]
    t1 = named_task(cvat, a, "first", 2)
    for n in (1, 2, 12):
        named_task(cvat, None, f"pets_round_{n}")
    named_task(cvat, None, "ActiveLabel 7 added 1 (2026-11-20)")
    named_task(cvat, None, "ActiveLabel 7 added 2 (2026-11-21)")
    named_task(cvat, None, "something else")
    cvat.calls.clear()
    found = CvatRemote().audit(SETTINGS)
    assert [p["name"] for p in found["projects"]] == ["ActiveLabel 7 - Pets", "Empty"]
    pets = found["projects"][0]
    assert pets["tasks"] == [
        {"id": str(t1), "name": "first", "size": 2, "url": f"https://cvat.test/tasks/{t1}"}
    ]
    assert pets["url"] == f"https://cvat.test/projects/{a}" and found["projects"][1]["tasks"] == [] and b
    loose = {g["pattern"]: [t["name"] for t in g["tasks"]] for g in found["loose"]}
    assert loose == {
        "pets_round_#": ["pets_round_1", "pets_round_2", "pets_round_12"],
        "ActiveLabel # added # (#-#-#)": [
            "ActiveLabel 7 added 1 (2026-11-20)",
            "ActiveLabel 7 added 2 (2026-11-21)",
        ],
        "something else": ["something else"],
    }
    assert found["loose"][0]["pattern"] == "pets_round_#"  # the biggest group first
    assert {c[0] for c in cvat.calls} == {"GET"}  # read-only: nothing was created, changed or deleted


def test_audit_shows_the_tasks_of_a_project_the_account_cannot_list(cvat: FakeCvat) -> None:
    named_task(cvat, 999, "hidden project task")
    (only,) = CvatRemote().audit(SETTINGS)["projects"]
    assert only["id"] == "999" and only["name"] == "" and only["tasks"][0]["name"] == "hidden project task"


def test_get_missing_is_readable(cvat: FakeCvat) -> None:
    with pytest.raises(ToolError, match="not found"):
        CvatRemote().get(SETTINGS, "999")


def test_create_sends_name_and_labels(cvat: FakeCvat) -> None:
    info = CvatRemote().create(SETTINGS, project())
    post = next(c for c in cvat.calls if c[0] == "POST" and c[1] == "/api/projects")
    assert post[2]["json"]["name"] == "ActiveLabel 7 - Pets"
    assert [x["name"] for x in post[2]["json"]["labels"]] == ["cat", "dog"]
    assert post[2]["json"]["labels"][0]["type"] == "any"
    assert info.name == "ActiveLabel 7 - Pets" and info.images == 0


def test_items_convert_pixels_to_relative_and_skip_tracks(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    items = CvatRemote().items(SETTINGS, str(pid), Task.DETECTION)
    assert [i.key for i in items] == [f"{tid}:0", f"{tid}:1"]
    assert [(i.file_name, i.width, i.height) for i in items] == [("a.jpg", 200, 100), ("b.jpg", 400, 200)]
    assert items[0].labels == [Label("cat", bbox=[0.1, 0.1, 0.5, 0.5])]
    assert items[1].labels == [Label("dog", bbox=[0.0, 0.0, 1.0, 1.0])]


def test_items_polygon_points_and_tags(cvat: FakeCvat) -> None:
    p = cvat.add_project("x", ["cat"])
    cat = cvat.label_id(p["id"], "cat")
    shapes = [
        {"type": "polygon", "frame": 0, "label_id": cat, "points": [20, 10, 120, 10, 70, 60]},
        {"type": "points", "frame": 0, "label_id": cat, "points": [100, 50, 0, 0, 50, 25]},
        {"type": "polyline", "frame": 0, "label_id": cat, "points": [1, 1, 2, 2]},
    ]
    ann = {"version": 0, "tags": [{"frame": 0, "label_id": cat}], "shapes": shapes, "tracks": []}
    cvat.add_task(p["id"], [("a.jpg", 200, 100)], ann)
    labels = CvatRemote().items(SETTINGS, str(p["id"]), Task.SEGMENTATION)[0].labels
    assert labels[0].polygons == [[[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]]]
    assert labels[0].bbox == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert labels[1].keypoints == [[0.5, 0.5, 2.0], [0.0, 0.0, 0.0], [0.25, 0.25, 2.0]]
    assert labels[1].bbox == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert len(labels) == 2  # polyline skipped, tag ignored outside classification
    cls = CvatRemote().items(SETTINGS, str(p["id"]), Task.CLASSIFICATION)[0].labels
    assert cls == [Label("cat")]


def test_items_paginates_tasks(cvat: FakeCvat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cvat_import, "PAGE", 2)
    p = cvat.add_project("x", ["a", "b", "c"])
    tids = [cvat.add_task(p["id"], [(f"{n}.jpg", 10, 10)]) for n in range(5)]
    items = CvatRemote().items(SETTINGS, str(p["id"]), Task.DETECTION)
    assert [i.key for i in items] == [f"{t}:0" for t in tids]
    pages = [c[2]["params"]["page"] for c in cvat.calls if c[1] == "/api/tasks" and c[0] == "GET"]
    assert pages == [1, 2, 3]


def test_write_replaces_only_updated_frames_and_keeps_tracks(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    new = [Label("dog", bbox=[0.25, 0.5, 0.5, 0.25])]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:0": new})
    ann = cvat.tasks[tid]["ann"]
    assert ann["version"] == 3 and len(ann["tracks"]) == 1
    by_frame = {s["frame"]: s for s in ann["shapes"]}
    assert by_frame[1]["points"] == [0, 0, 400, 200]  # untouched frame kept
    assert by_frame[0]["points"] == [50.0, 50.0, 150.0, 75.0]
    assert by_frame[0]["label_id"] == cvat.label_id(pid, "dog") and by_frame[0]["attributes"] == []
    assert len(ann["shapes"]) == 2


def test_write_polygon_keypoints_tag_and_new_class(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    labels = [
        Label("bird", bbox=[0, 0, 1, 1], polygons=[[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]]),
        Label("cat", bbox=[0, 0, 1, 1], keypoints=[[0.5, 0.5, 2.0], [0.0, 0.0, 0.0]]),
        Label("dog"),
    ]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:1": labels})
    ann = cvat.tasks[tid]["ann"]
    bird = cvat.label_id(pid, "bird")  # created on demand
    shapes = {s["type"]: s for s in ann["shapes"] if s["frame"] == 1}
    assert shapes["polygon"]["points"] == [0.0, 0.0, 200.0, 0.0, 200.0, 100.0]
    assert shapes["polygon"]["label_id"] == bird
    assert shapes["points"]["points"] == [200.0, 100.0, 0.0, 0.0]
    dog = cvat.label_id(pid, "dog")
    tags = [{k: v for k, v in t.items() if k != "id"} for t in ann["tags"]]
    assert tags == [{"frame": 1, "label_id": dog, "group": 0, "source": "manual", "attributes": []}]


def test_write_bad_key_and_frame(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    with pytest.raises(ToolError, match="not a CVAT item key"):
        CvatRemote().write(SETTINGS, str(pid), project(), {"abc": []})
    with pytest.raises(ToolError, match="no frame 9"):
        CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:9": []})


def test_ensure_classes_adds_only_missing(cvat: FakeCvat) -> None:
    pid, _ = seeded(cvat)
    CvatRemote().ensure_classes(SETTINGS, str(pid), ["cat", "bird", "fish"])
    patches = [c for c in cvat.calls if c[0] == "PATCH"]
    assert len(patches) == 1
    assert [x["name"] for x in patches[0][2]["json"]["labels"]] == ["bird", "fish"]
    CvatRemote().ensure_classes(SETTINGS, str(pid), ["bird"])
    assert len([c for c in cvat.calls if c[0] == "PATCH"]) == 1


def make_images(tmp_path: Any, names: list[str]) -> list[tuple[ImageRecord, Any, list[Label]]]:
    out = []
    for n, name in enumerate(names):
        f = tmp_path / f"{n}.jpg"
        f.write_bytes(b"jpg")
        rec = ImageRecord(f"id{n}", name, f"images/{n}.jpg", 200, 100, 3, "")
        out.append((rec, f, [Label("cat", bbox=[0.0, 0.0, 0.5, 1.0])] if n % 2 == 0 else []))
    return out


def test_add_uploads_waits_and_maps_keys(cvat: FakeCvat, tmp_path: Any) -> None:
    pid, _ = seeded(cvat)
    images = make_images(tmp_path, ["x/one.jpg", "two.jpg", "three.jpg"])
    keys = CvatRemote().add(SETTINGS, str(pid), project(), images)
    tid = int(keys["id0"].split(":")[0])
    assert keys == {"id0": f"{tid}:0", "id1": f"{tid}:1", "id2": f"{tid}:2"}
    data = next(c for c in cvat.calls if c[1] == f"/api/tasks/{tid}/data")[2]
    assert data["data"] == {"image_quality": 95, "sorting_method": "predefined"}
    assert [f[0] for f in data["files"]] == ["client_files[0]", "client_files[1]", "client_files[2]"]
    assert data["files"][0][1][0] == "x/one.jpg"
    assert cvat.polls >= 2  # waited for "finished"
    shapes = cvat.tasks[tid]["ann"]["shapes"]
    assert [s["frame"] for s in shapes] == [0, 2]
    assert shapes[0]["points"] == [0.0, 0.0, 100.0, 100.0]  # 0.5 * 200, 1.0 * 100 (frame size from meta)
    assert cvat.tasks[tid]["project_id"] == pid


def test_add_batches_and_duplicate_names(
    cvat: FakeCvat, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cvat_remote, "BATCH", 2)
    pid, _ = seeded(cvat)
    images = make_images(tmp_path, ["same.jpg", "same.jpg", "c.jpg"])
    keys = CvatRemote().add(SETTINGS, str(pid), project(), images)
    assert len({k.split(":")[0] for k in keys.values()}) == 2  # two tasks
    assert keys["id0"].endswith(":0") and keys["id1"].endswith(":1") and keys["id2"].endswith(":0")
    first = next(c for c in cvat.calls if c[0] == "POST" and c[1].endswith("/data"))[2]
    assert [f[1][0] for f in first["files"]] == ["same.jpg", "id1_same.jpg"]


def test_add_uses_task_status_on_old_cvat(cvat: FakeCvat, tmp_path: Any) -> None:
    cvat.status_only = True
    pid, _ = seeded(cvat)
    keys = CvatRemote().add(SETTINGS, str(pid), project(), make_images(tmp_path, ["a.jpg"]))
    assert keys["id0"].endswith(":0")


def test_add_times_out_and_reports_failure(
    cvat: FakeCvat, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = seeded(cvat)
    monkeypatch.setattr(cvat_remote, "MAX_POLLS", 3)
    monkeypatch.setattr(FakeCvat, "get_request", lambda self, kw, rq: Resp(200, {"status": "started"}))
    with pytest.raises(ToolError, match="too long"):
        CvatRemote().add(SETTINGS, str(pid), project(), make_images(tmp_path, ["a.jpg"]))
    failed = Resp(200, {"status": "failed", "message": "bad image"})
    monkeypatch.setattr(FakeCvat, "get_request", lambda self, kw, rq: failed)
    with pytest.raises(ToolError, match="bad image"):
        CvatRemote().add(SETTINGS, str(pid), project(), make_images(tmp_path, ["a.jpg"]))


@pytest.mark.parametrize(
    ("status", "text"),
    [
        (401, "login failed"),
        (403, "refused"),
        (404, "not found"),
        (429, "slow down"),
        (500, "error 500"),
    ],
)
def test_http_errors_are_readable_and_secret_free(cvat: FakeCvat, status: int, text: str) -> None:
    cvat.fail[("GET", "/api/projects")] = status
    with pytest.raises(ToolError, match=text) as err:
        CvatRemote().find(SETTINGS, "x")
    assert "s3cret-pw" not in str(err.value)


def test_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    class Down:
        def request(self, *a: Any, **k: Any) -> Any:
            raise requests.ConnectionError("boom s3cret-pw")

    monkeypatch.setattr(requests, "Session", lambda: Down())
    with pytest.raises(ToolError, match="Cannot reach CVAT") as err:
        CvatRemote().get(SETTINGS, "1")
    assert "s3cret-pw" not in str(err.value)


def test_missing_credentials() -> None:
    with pytest.raises(ToolError, match="needs the server address"):
        CvatRemote().get(ToolSettings("cvat", "https://cvat.test", "", ""), "1")


# -- tracks -----------------------------------------------------------------------------------------
def kf(frame: int, pts: list[float], kind: str = "rectangle", outside: bool = False) -> dict[str, Any]:
    return {
        "type": kind,
        "frame": frame,
        "points": pts,
        "occluded": False,
        "outside": outside,
        "z_order": 0,
        "rotation": 0,
        "attributes": [],
    }


def tracked(
    cvat: FakeCvat, shapes: list[dict[str, Any]], nframes: int = 6, deleted: list[int] | None = None
) -> tuple[int, int]:
    p = cvat.add_project("ActiveLabel 7 - Pets", ["cat", "dog"])
    cat = cvat.label_id(p["id"], "cat")
    ann = {
        "version": 1,
        "tags": [],
        "shapes": [],
        "tracks": [{"id": 42, "label_id": cat, "frame": shapes[0]["frame"], "shapes": shapes}],
    }
    tid = cvat.add_task(p["id"], [(f"{n}.jpg", 100, 100) for n in range(nframes)], ann)
    if deleted:
        cvat.tasks[tid]["deleted"] = deleted
    return p["id"], tid


def read(pid: int, task: Task = Task.DETECTION) -> list[list[Label]]:
    return [i.labels for i in CvatRemote().items(SETTINGS, str(pid), task)]


def test_track_rectangle_is_interpolated_between_keyframes_and_continues(cvat: FakeCvat) -> None:
    pid, _ = tracked(cvat, [kf(0, [0, 0, 10, 10]), kf(4, [40, 0, 50, 10])])
    labels = read(pid)
    assert all(len(x) == 1 and x[0].label == "cat" and x[0].track == 42 for x in labels)
    assert [lb[0].bbox[0] for lb in labels] == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.4])


def test_track_outside_hides_and_next_keyframe_shows_again(cvat: FakeCvat) -> None:
    pid, _ = tracked(
        cvat,
        [kf(0, [0, 0, 10, 10]), kf(2, [20, 0, 30, 10], outside=True), kf(4, [40, 0, 50, 10])],
    )
    labels = read(pid)
    assert [bool(x) for x in labels] == [True, True, False, False, True, True]
    assert labels[1][0].bbox[0] == 0  # held until the outside keyframe, not interpolated


def test_track_ending_outside_stays_hidden(cvat: FakeCvat) -> None:
    pid, _ = tracked(cvat, [kf(1, [0, 0, 10, 10]), kf(3, [0, 0, 10, 10], outside=True)])
    assert [bool(x) for x in read(pid)] == [False, True, True, False, False, False]


def test_track_polygon_uses_previous_keyframe(cvat: FakeCvat) -> None:
    a = kf(0, [0, 0, 50, 0, 50, 50], "polygon")
    b = kf(3, [10, 10, 60, 10, 60, 60], "polygon")
    pid, _ = tracked(cvat, [a, b], nframes=4)
    polys = [x[0].polygons for x in read(pid, Task.SEGMENTATION)]
    assert polys[1] == polys[2] == polys[0] != polys[3]


def test_track_skips_deleted_frames_classification_and_skeletons(cvat: FakeCvat) -> None:
    pid, tid = tracked(cvat, [kf(0, [0, 0, 10, 10])], nframes=3, deleted=[1])
    items = CvatRemote().items(SETTINGS, str(pid), Task.DETECTION)
    assert [i.key for i in items] == [f"{tid}:0", f"{tid}:2"]
    assert read(pid, Task.CLASSIFICATION) == [[], []]
    cvat.tasks[tid]["ann"]["tracks"][0]["elements"] = [{"label_id": 1, "shapes": []}]
    assert read(pid) == [[], []]


def test_write_tracked_edit_on_interpolated_frame_inserts_keyframe(cvat: FakeCvat) -> None:
    pid, tid = tracked(cvat, [kf(0, [0, 0, 10, 10]), kf(4, [40, 0, 50, 10])], nframes=5)
    assert read(pid)[2][0].bbox[0] == pytest.approx(0.2)
    edited = [Label("cat", bbox=[0.2, 0.5, 0.1, 0.1], track=42)]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:2": edited})
    (tr,) = cvat.tasks[tid]["ann"]["tracks"]
    assert tr["id"] == 42 and [s["frame"] for s in tr["shapes"]] == [0, 2, 4]  # updated, same id
    assert tr["shapes"][1]["points"] == pytest.approx([20.0, 50.0, 30.0, 60.0])
    assert tr["shapes"][0]["points"] == [0, 0, 10, 10] and tr["shapes"][2]["points"] == [40, 0, 50, 10]
    assert cvat.tasks[tid]["ann"]["shapes"] == []  # no plain shape was written
    labels = read(pid)
    assert labels[2][0].bbox == pytest.approx([0.2, 0.5, 0.1, 0.1])
    assert labels[1][0].bbox[1] == pytest.approx(0.25)  # CVAT interpolates towards the new keyframe


def test_write_removing_tracked_object_adds_outside_and_restore_keyframes(cvat: FakeCvat) -> None:
    pid, tid = tracked(cvat, [kf(0, [0, 0, 10, 10]), kf(4, [40, 0, 50, 10])], nframes=6)
    before = read(pid)
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:2": []})
    (tr,) = cvat.tasks[tid]["ann"]["tracks"]
    assert [(s["frame"], s["outside"]) for s in tr["shapes"]] == [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (4, False),
    ]
    assert tr["shapes"][3]["points"] == pytest.approx([30, 0, 40, 10])
    after = read(pid)
    assert after[2] == []
    for f in (0, 1, 3, 4, 5):
        assert links.same_labels(before[f], after[f])


def test_write_unknown_track_becomes_plain_shape(cvat: FakeCvat) -> None:
    pid, tid = tracked(cvat, [kf(0, [0, 0, 10, 10])], nframes=3)
    ghost = [Label("cat", bbox=[0, 0, 0.5, 0.5], track=999)]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:1": ghost})
    ann = cvat.tasks[tid]["ann"]
    assert [(s["frame"], s["type"]) for s in ann["shapes"]] == [(1, "rectangle")]
    # the real track is hidden on that frame and shows again right after
    assert [(s["frame"], s["outside"]) for s in ann["tracks"][0]["shapes"]] == [
        (0, False),
        (1, True),
        (2, False),
    ]


def test_write_other_class_detaches_label_from_track(cvat: FakeCvat) -> None:
    pid, tid = tracked(cvat, [kf(0, [0, 0, 10, 10])], nframes=2)
    moved = [Label("dog", bbox=[0, 0, 0.5, 0.5], track=42)]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:1": moved})
    ann = cvat.tasks[tid]["ann"]
    assert ann["shapes"][0]["label_id"] == cvat.label_id(pid, "dog")
    assert ann["tracks"][0]["shapes"][-1]["outside"] is True


def test_read_then_write_every_frame_is_idempotent(cvat: FakeCvat) -> None:
    shapes = [kf(0, [0, 0, 10, 10]), kf(2, [20, 0, 30, 10], outside=True), kf(4, [40, 0, 50, 10])]
    pid, tid = tracked(cvat, shapes)
    before = read(pid)
    items = CvatRemote().items(SETTINGS, str(pid), Task.DETECTION)
    CvatRemote().write(SETTINGS, str(pid), project(), {i.key: i.labels for i in items})
    assert [s["frame"] for s in cvat.tasks[tid]["ann"]["tracks"][0]["shapes"]] == [0, 2, 4]
    after = read(pid)
    assert all(links.same_labels(a, b) for a, b in zip(before, after, strict=True))
    assert cvat.tasks[tid]["ann"]["shapes"] == []


def test_labels_hash_and_same_labels_use_track() -> None:
    a, b, c = (Label("cat", bbox=[0, 0, 1, 1], track=t) for t in (1, 1, 2))
    assert links.label_hash([a]) == links.label_hash([b]) != links.label_hash([c])
    assert links.label_hash([Label("cat", bbox=[0, 0, 1, 1])]) != links.label_hash([a])
    assert links.same_labels([a], [b]) and not links.same_labels([a], [c])
    assert not links.same_labels([a], [Label("cat", bbox=[0, 0, 1, 1])])


def test_video_task_meta_lists_every_frame():
    meta = {
        "size": 3,
        "start_frame": 30,
        "frame_filter": "step=15",
        "frames": [{"width": 640, "height": 480, "name": "clip.mp4"}],
    }
    frames = _meta_frames(meta)
    assert [f["name"] for f in frames] == [
        "clip_frame_000030.jpg",
        "clip_frame_000045.jpg",
        "clip_frame_000060.jpg",
    ]
    assert all(f["width"] == 640 for f in frames)
    images = {
        "size": 2,
        "frames": [{"width": 1, "height": 1, "name": "a.jpg"}, {"width": 2, "height": 2, "name": "b.jpg"}],
    }
    assert _meta_frames(images) == images["frames"]


def video_task(cvat: FakeCvat, zip_names: list[str]) -> int:
    pid, _ = seeded(cvat)
    tid = cvat.add_task(pid, [])
    cvat.tasks[tid]["frames"] = [{"name": "clip.mp4", "width": 64, "height": 48}]
    cvat.tasks[tid]["meta"] = {"size": 5, "start_frame": 10, "stop_frame": 50, "frame_filter": "step=10"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in zip_names:
            zf.writestr(n, b"video-bytes" if n.startswith("data/") else b"{}")
    cvat.backup_zip = buf.getvalue()
    return tid


def test_video_source_downloads_the_video_from_the_task_backup(cvat: FakeCvat, tmp_path: Any) -> None:
    tid = video_task(cvat, ["task.json", "annotations.json", "data/clip.mp4"])
    found = CvatRemote().video_source(SETTINGS, str(tid), tmp_path / "dl")
    assert found is not None
    path, sel = found
    assert path == tmp_path / "dl" / "clip.mp4" and path.read_bytes() == b"video-bytes"
    assert sel == {"start": 10, "stop": 50, "step": 10}
    assert not (tmp_path / "dl" / "backup.zip").exists()
    assert ("POST", f"/api/tasks/{tid}/backup/export") in [(m, p) for m, p, _ in cvat.calls]


def test_video_source_is_none_for_image_tasks_and_errors_are_readable(cvat: FakeCvat, tmp_path: Any) -> None:
    pid, tid = seeded(cvat)
    assert CvatRemote().video_source(SETTINGS, str(tid), tmp_path) is None
    assert not any("backup" in p for _, p, _ in cvat.calls)
    vid = video_task(cvat, ["task.json"])
    with pytest.raises(ToolError, match="original video"):
        CvatRemote().video_source(SETTINGS, str(vid), tmp_path / "a")
    cvat.fail[("POST", f"/api/tasks/{vid}/backup")] = 500
    with pytest.raises(ToolError, match="500"):
        CvatRemote().video_source(SETTINGS, str(vid), tmp_path / "b")


# -- safety of write / create / errors -------------------------------------------------------------
def frame_ann(cvat: FakeCvat, pid: int, tid: int) -> dict[str, Any]:
    cat = cvat.label_id(pid, "cat")
    cvat.tasks[tid]["ann"] = {
        "version": 1,
        "tags": [{"frame": 0, "label_id": cat, "group": 0, "source": "manual", "attributes": []}],
        "shapes": [
            {
                **rect(0, cat, [20, 10, 120, 60]),
                "occluded": True,
                "rotation": 0,
                "group": 5,
                "z_order": 3,
                "source": "auto",
                "attributes": [{"spec_id": 1, "value": "red"}],
            },
            {"type": "polyline", "frame": 0, "label_id": cat, "points": [1, 1, 9, 9], "attributes": []},
            {"type": "ellipse", "frame": 0, "label_id": cat, "points": [5, 5, 9, 9], "attributes": []},
        ],
        "tracks": [],
    }
    return cvat.tasks[tid]["ann"]


def test_write_keeps_unconverted_shapes_tags_and_attributes_of_unchanged_shapes(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    frame_ann(cvat, pid, tid)
    labels = [Label("cat", bbox=[0.1, 0.1, 0.5, 0.5]), Label("dog", bbox=[0.5, 0.5, 0.25, 0.25])]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:0": labels})
    ann = cvat.tasks[tid]["ann"]
    shapes = [s for s in ann["shapes"] if s["frame"] == 0]
    assert sorted(s["type"] for s in shapes) == ["ellipse", "polyline", "rectangle", "rectangle"]
    cat = next(s for s in shapes if s["type"] == "rectangle" and s["label_id"] == cvat.label_id(pid, "cat"))
    assert (cat["group"], cat["z_order"], cat["source"], cat["occluded"]) == (5, 3, "auto", True)
    assert cat["attributes"] == [{"spec_id": 1, "value": "red"}]
    dog = next(s for s in shapes if s["label_id"] == cvat.label_id(pid, "dog"))
    assert (
        dog["source"] == "manual" and dog["attributes"] == [] and dog["points"] == [100.0, 50.0, 150.0, 75.0]
    )
    assert len(ann["tags"]) == 1  # the tag is not converted in a detection project


def test_write_replaces_only_what_changed(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    frame_ann(cvat, pid, tid)
    moved = [Label("cat", bbox=[0.5, 0.5, 0.25, 0.25])]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:0": moved})
    shapes = [s for s in cvat.tasks[tid]["ann"]["shapes"] if s["frame"] == 0]
    rects = [s for s in shapes if s["type"] == "rectangle"]
    assert len(rects) == 1 and rects[0]["points"] == [100.0, 50.0, 150.0, 75.0]
    assert rects[0]["group"] == 0 and rects[0]["attributes"] == []  # a moved shape is a new one
    assert sorted(s["type"] for s in shapes if s["type"] != "rectangle") == ["ellipse", "polyline"]
    # one pixel of noise is the same shape
    frame_ann(cvat, pid, tid)
    CvatRemote().write(
        SETTINGS, str(pid), project(), {f"{tid}:0": [Label("cat", bbox=[0.1001, 0.1, 0.5, 0.5])]}
    )
    kept = [s for s in cvat.tasks[tid]["ann"]["shapes"] if s["type"] == "rectangle"]
    assert kept[0]["group"] == 5


def test_write_classification_keeps_shapes_and_matches_tags(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    frame_ann(cvat, pid, tid)
    cat, dog = cvat.label_id(pid, "cat"), cvat.label_id(pid, "dog")
    cvat.tasks[tid]["ann"]["tags"][0]["group"] = 9
    CvatRemote().write(SETTINGS, str(pid), project(Task.CLASSIFICATION), {f"{tid}:0": [Label("cat")]})
    ann = cvat.tasks[tid]["ann"]
    assert [t["group"] for t in ann["tags"] if t["frame"] == 0] == [9]  # same class: the old tag stays
    assert len([s for s in ann["shapes"] if s["frame"] == 0]) == 3  # no shape is converted
    CvatRemote().write(SETTINGS, str(pid), project(Task.CLASSIFICATION), {f"{tid}:0": [Label("dog")]})
    tags = [t for t in cvat.tasks[tid]["ann"]["tags"] if t["frame"] == 0]
    assert [t["label_id"] for t in tags] == [dog] and cat != dog


def test_failed_task_creation_deletes_the_new_task(
    cvat: FakeCvat, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = seeded(cvat)
    monkeypatch.setattr(FakeCvat, "post_data", lambda self, kw, t: Resp(500, {}))
    before = set(cvat.tasks)
    with pytest.raises(ToolError, match="error 500"):
        CvatRemote().add_task(SETTINGS, str(pid), project(), "T", None, make_images(tmp_path, ["a.jpg"]))
    assert set(cvat.tasks) == before  # the half-made task is gone
    assert any(c[0] == "DELETE" for c in cvat.calls)


@pytest.mark.parametrize(
    "body", [{"name": "x"}, {"name": "x", "project_id": None}, {"name": "x", "project_id": 0}]
)
def test_a_task_without_project_is_refused_before_any_request(cvat: FakeCvat, body: dict[str, Any]) -> None:
    api = CvatRemote._api(SETTINGS)
    with pytest.raises(ToolError, match="outside a project"):
        api.request("POST", "/api/tasks", json=body)
    assert not cvat.calls and not cvat.tasks


def test_a_task_with_project_is_created(cvat: FakeCvat) -> None:
    pid, _ = seeded(cvat)
    answer = CvatRemote._api(SETTINGS).request("POST", "/api/tasks", json={"name": "ok", "project_id": pid})
    assert cvat.tasks[answer["id"]]["project_id"] == pid


def test_failed_upload_of_a_later_batch_deletes_only_that_task(
    cvat: FakeCvat, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = seeded(cvat)
    monkeypatch.setattr(cvat_remote, "BATCH", 1)
    real = FakeCvat.post_data
    uploads: list[str] = []

    def second_upload_fails(self: FakeCvat, kw: dict[str, Any], t: str) -> Resp:
        uploads.append(t)
        return real(self, kw, t) if len(uploads) == 1 else Resp(500, {})

    monkeypatch.setattr(FakeCvat, "post_data", second_upload_fails)
    before = set(cvat.tasks)
    with pytest.raises(ToolError, match="error 500"):
        CvatRemote().add(SETTINGS, str(pid), project(), make_images(tmp_path, ["a.jpg", "b.jpg"]))
    (first,) = set(cvat.tasks) - before  # the first batch stays, the half-made second one is gone
    assert str(first) == uploads[0]


def test_failed_cleanup_keeps_the_original_error(
    cvat: FakeCvat, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = seeded(cvat)
    monkeypatch.setattr(FakeCvat, "post_data", lambda self, kw, t: Resp(500, {}))
    cvat.fail[("DELETE", "/api/tasks")] = 500
    with pytest.raises(ToolError, match="error 500 on /api/tasks/\\d+/data"):
        CvatRemote().add_task(SETTINGS, str(pid), project(), "T", None, make_images(tmp_path, ["a.jpg"]))


def test_forbidden_json_body_becomes_a_sentence(cvat: FakeCvat, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"detail": "You have reached the maximum number of tasks for the free plan"}
    monkeypatch.setattr(FakeCvat, "list_projects", lambda self, kw: Resp(403, body))
    with pytest.raises(ToolError) as err:
        CvatRemote().find(SETTINGS, "x")
    assert str(err.value) == (
        "CVAT refused the request. CVAT says: You have reached the maximum number of tasks for the free plan."
    )


@pytest.mark.parametrize(
    ("task", "label", "kept_type"),
    [
        (
            Task.SEGMENTATION,
            Label("cat", bbox=[0.1, 0.1, 0.2, 0.2], polygons=[[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]]),
            "rectangle",
        ),
        (Task.KEYPOINTS, Label("cat", bbox=[0.1, 0.1, 0.2, 0.2], keypoints=[[0.2, 0.2, 2]]), "rectangle"),
    ],
)
def test_push_keeps_shapes_the_project_cannot_see(task, label, kept_type):
    """A shape of a known class but of a type this project does not read stays in CVAT."""
    ids = {"cat": 7}
    unseen = {
        "rectangle": {"type": "rectangle", "label_id": 7, "frame": 0, "points": [10, 10, 30, 30]},
        "points": {"type": "points", "label_id": 7, "frame": 0, "points": [5, 5]},
    }[kept_type]
    shapes, _ = cvat_remote._merge_frame([unseen], [], [label], ids, 0, (100, 100), task)
    assert unseen in shapes and len(shapes) == 2
    removed, _ = cvat_remote._merge_frame([unseen], [], [], ids, 0, (100, 100), task)
    assert removed == [unseen]  # nothing written for this frame: the unseen shape is still not removed


def test_push_still_removes_a_deleted_box():
    old = {"type": "rectangle", "label_id": 7, "frame": 0, "points": [10, 10, 30, 30]}
    shapes, _ = cvat_remote._merge_frame([old], [], [], {"cat": 7}, 0, (100, 100), Task.DETECTION)
    assert shapes == []


# -- differences (PATCH) ------------------------------------------------------------------------------
def test_write_sends_differences_and_never_a_put(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    frame_ann(cvat, pid, tid)
    before = {s["id"]: s for s in CvatRemote()._api(SETTINGS).get(f"/api/tasks/{tid}/annotations")["shapes"]}
    moved = [Label("cat", bbox=[0.11, 0.1, 0.5, 0.5]), Label("dog", bbox=[0.6, 0.6, 0.2, 0.2])]
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:0": moved})
    assert not [c for c in cvat.calls if c[0] == "PUT"]
    actions = [c[2]["params"]["action"] for c in cvat.calls if c[0] == "PATCH" and "annotations" in c[1]]
    assert actions == ["update", "create"]  # nothing to delete; the cat moved, the dog is new
    after = {s["id"]: s for s in cvat.tasks[tid]["ann"]["shapes"]}
    assert set(before) <= set(after)  # every shape keeps its id
    cat = next(
        s for s in after.values() if s["type"] == "rectangle" and s["label_id"] == cvat.label_id(pid, "cat")
    )
    assert cat["points"] == pytest.approx([22.0, 10.0, 122.0, 60.0]) and cat["attributes"] == [
        {"spec_id": 1, "value": "red"}
    ]


def test_write_deletes_only_what_it_read(cvat: FakeCvat) -> None:
    pid, tid = seeded(cvat)
    CvatRemote().write(SETTINGS, str(pid), project(), {f"{tid}:0": []})
    (delete,) = [c[2]["json"] for c in cvat.calls if c[0] == "PATCH" and "annotations" in c[1]]
    assert [s["frame"] for s in delete["shapes"]] == [0] and delete["tracks"] == [] and delete["tags"] == []
    assert [s["frame"] for s in cvat.tasks[tid]["ann"]["shapes"]] == [1]


def test_overlap_needs_the_same_class_and_type() -> None:
    a = {"type": "rectangle", "label_id": 1, "points": [0, 0, 10, 10]}
    assert cvat_remote._overlap(a, {**a, "points": [1, 0, 11, 10]}) > 0.8
    assert cvat_remote._overlap(a, {**a, "label_id": 2}) == 0
    assert cvat_remote._overlap(a, {**a, "type": "polygon"}) == 0
    assert cvat_remote._overlap(a, {**a, "points": [50, 50, 60, 60]}) == 0
    point = {"type": "points", "label_id": 1, "points": [5, 5]}
    assert cvat_remote._overlap(point, point) == 1  # one point still has an area (one pixel around it)


def test_add_delta_sorts_objects_into_actions() -> None:
    kept, moved, gone = ({"id": n, "points": [n]} for n in (1, 2, 3))
    patch = cvat_remote._empty_patch()
    cvat_remote._add_delta(
        patch, "shapes", [kept, moved, gone], [kept, {**moved, "points": [9]}, {"points": [7]}]
    )
    assert patch["update"]["shapes"] == [{"id": 2, "points": [9]}]
    assert patch["create"]["shapes"] == [{"points": [7]}]
    assert patch["delete"]["shapes"] == [gone]


# -- rotated rectangles (#359) ------------------------------------------------------------------------
def test_enclosing_box_of_a_turned_rectangle() -> None:
    assert cvat_remote.enclosing_box([10, 10, 30, 20], 0) == [10, 10, 30, 20]
    assert cvat_remote.enclosing_box([10, 10, 30, 20], 90) == pytest.approx([15, 5, 25, 25])
    half = 50**0.5  # half the diagonal of a 10 x 10 square
    assert cvat_remote.enclosing_box([0, 0, 10, 10], 45) == pytest.approx(
        [5 - half, 5 - half, 5 + half, 5 + half]
    )
    assert cvat_remote.enclosing_box([0, 0, 10, 10], 30) == pytest.approx(
        cvat_remote.enclosing_box([0, 0, 10, 10], 330)
    )


def test_turned_rectangle_is_read_as_its_enclosing_box() -> None:
    shape = {"type": "rectangle", "points": [10, 10, 30, 20], "rotation": 90}
    label = cvat_remote.shape_to_label(shape, "cat", 100, 100)
    assert label is not None and label.bbox == pytest.approx([0.15, 0.05, 0.1, 0.2])


def test_edited_turned_rectangle_moves_with_its_turn_or_becomes_the_new_box() -> None:
    old = {"id": 3, "type": "rectangle", "label_id": 1, "points": [10, 10, 30, 20], "rotation": 90}
    moved = cvat_remote._edited(old, {"points": [20, 5, 30, 25]}, 100, 100)  # the enclosing box, 5 px right
    assert moved["points"] == pytest.approx([15, 10, 35, 20]) and moved["rotation"] == 90 and moved["id"] == 3
    resized = cvat_remote._edited(old, {"points": [15, 5, 45, 25]}, 100, 100)
    assert resized["points"] == [15, 5, 45, 25] and resized["rotation"] == 0 and resized["id"] == 3


def test_track_rotation_is_interpolated_the_short_way() -> None:
    a = {"type": "rectangle", "frame": 0, "points": [0, 0, 10, 10], "rotation": 350}
    b = {"type": "rectangle", "frame": 2, "points": [0, 0, 10, 10], "rotation": 10}
    shown = cvat_remote.track_frames({"shapes": [a, b]}, 3)
    assert shown[1][0]["rotation"] == pytest.approx(0)
