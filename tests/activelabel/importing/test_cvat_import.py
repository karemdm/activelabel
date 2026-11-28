import threading
import time
import uuid
from pathlib import Path

import fiftyone as fo
import pytest
import requests
from api_client import make_client
from fiftyone.utils import cvat as fouc
from import_fakes import FakeResp, FakeSession, make_images
from sync_fakes import FakeRemote, box

from activelabel import fiftyone_bridge as bridge
from activelabel import optional
from activelabel.annotate.base import ToolError, ToolSettings
from activelabel.images.manifest import Manifest
from activelabel.importing import cvat_import as ci
from activelabel.importing.common import RemoteNotFound, read_origin
from activelabel.jobs import Cancelled, Job, Progress
from activelabel.labels import Label, load_annotations
from activelabel.projects import ProjectStore
from activelabel.sync import engine, links, registry
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore, jobs

BASE = "http://cvat.local"
SETTINGS = ToolSettings("cvat", BASE, "user", "pw-secret")


# Stand-ins for FiftyOne labels: they are recognised by class name.
class Detection:
    def __init__(self, label, box):
        self.label, self.bounding_box, self.mask = label, box, None


class Detections:
    def __init__(self, *items):
        self.detections = list(items)


class Polyline:
    def __init__(self, label, ring):
        self.label, self.points, self.filled = label, [ring], True


class Polylines:
    def __init__(self, *items):
        self.polylines = list(items)


class Classification:
    def __init__(self, label):
        self.label = label


RING = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.6)]


@pytest.fixture
def cvat(monkeypatch):
    projects = {
        "count": 2,
        "next": None,
        "results": [
            {"id": 1, "name": "Street", "created_date": "2026-02-01", "labels": [{"id": 1, "name": "car"}]},
            {"id": 2, "name": "Empty", "created_date": "2026-01-01", "labels": {"count": 1, "url": "x"}},
        ],
    }
    tasks = {
        "count": 3,
        "next": None,
        "results": [
            {
                "id": 12,
                "name": "Loose",
                "project_id": None,
                "size": 5,
                "created_date": "2026-04-01",
                "labels": [],
            },
            {"id": 11, "name": "t2", "project_id": 1, "size": 7, "created_date": "2026-03-01", "labels": []},
            {"id": 10, "name": "t1", "project_id": 1, "size": 3, "created_date": "2026-02-01", "labels": []},
        ],
    }
    routes = {
        ("GET", "/api/projects/1"): FakeResp(200, projects["results"][0]),
        ("GET", "/api/projects/99"): FakeResp(404),
        ("GET", "/api/projects"): FakeResp(200, projects),
        ("GET", "/api/tasks/12"): FakeResp(200, tasks["results"][0]),
        ("GET", "/api/tasks/11"): FakeResp(200, tasks["results"][1]),
        ("GET", "/api/tasks"): FakeResp(200, tasks),
        ("GET", "/api/labels"): FakeResp(
            200, {"results": [{"name": "tree"}, {"name": "arm", "parent_id": 3}]}
        ),
        ("GET", "/api/projects/401"): FakeResp(401),
    }
    fake = FakeSession(routes, BASE)
    monkeypatch.setattr(requests, "Session", lambda: fake)
    return fake


def test_list_remote(cvat):
    items = ci.list_remote(SETTINGS)
    assert [(i["kind"], i["name"], i["images"]) for i in items] == [
        ("task", "Loose", 5),
        ("project", "Street", 10),
        ("project", "Empty", 0),
    ]
    assert items[1]["classes"] == ["car"] and items[2]["classes"] == ["tree"]  # inline and fetched labels
    assert items[1]["created"] == "2026-02-01" and items[0]["id"] == 12
    assert [i["name"] for i in ci.list_remote(SETTINGS, "STR")] == ["Street"]
    assert all(c[2].get("auth") == ("user", "pw-secret") for c in cvat.calls)  # basic auth, never in the URL
    assert "pw-secret" not in str([c[1] for c in cvat.calls])


def test_list_remote_pagination(monkeypatch):
    def page(kw):
        n = kw["params"]["page"]
        return FakeResp(
            200,
            {
                "next": "more" if n < 5 else None,
                "results": [{"id": n * 100 + i, "name": "p"} for i in range(100)],
            },
        )

    fake = FakeSession(
        {("GET", "/api/projects"): page, ("GET", "/api/tasks"): FakeResp(200, {"results": []})}, BASE
    )
    monkeypatch.setattr(requests, "Session", lambda: fake)
    items = ci.list_remote(SETTINGS)
    assert len(items) == 200  # capped


def test_list_remote_errors(monkeypatch):
    fake = FakeSession({("GET", "/api/projects"): FakeResp(401)}, BASE)
    monkeypatch.setattr(requests, "Session", lambda: fake)
    with pytest.raises(ToolError, match="login failed"):
        ci.list_remote(SETTINGS)

    def down(*a, **k):
        raise requests.ConnectionError("pw-secret leaked?")

    class Dead:
        get = down

    monkeypatch.setattr(requests, "Session", Dead)
    with pytest.raises(ToolError, match="Cannot reach CVAT") as err:
        ci.list_remote(SETTINGS)
    assert "pw-secret" not in str(err.value)
    with pytest.raises(ToolError, match="needs the server"):
        ci.list_remote(ToolSettings("cvat", BASE, "", ""))


def test_check_remote(cvat):
    assert ci.check_remote(SETTINGS, "project", "1") == {"name": "Street", "classes": ["car"]}
    assert ci.check_remote(SETTINGS, "task", "12") == {"name": "Loose", "classes": [], "project_id": None}
    assert ci.check_remote(SETTINGS, "task", "11")["project_id"] == "1"
    with pytest.raises(RemoteNotFound):
        ci.check_remote(SETTINGS, "project", "99")
    with pytest.raises(RemoteNotFound):
        ci.check_remote(SETTINGS, "project", "x")
    with pytest.raises(ToolError):
        ci.check_remote(SETTINGS, "folder", "1")


def fake_download(files, calls=None):
    """Replacement of ``_download``: writes the images and answers with labels per image."""

    def download(settings, kind, remote_id, data_dir):
        if calls is not None:
            calls.append((kind, remote_id, data_dir))
        sources = make_images(data_dir.parent / "src", len(files))
        out = []
        for name, values, source in zip(files, files.values(), sources, strict=True):
            target = data_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            out.append(ci.SampleLabels(target, values))
        return out

    return download


def run(store, tmp_path, progress=None, **kw):
    progress = progress or Progress(Job("j", 0, "import_project"), threading.Event())
    args = {"kind": "task", "remote_id": "12", "name": "", "description": "", "task": Task.DETECTION}
    return ci.import_project(store, SETTINGS, tmp_path / "ws", progress=progress, **{**args, **kw})


def test_import_merges_fields_and_infers_task(cvat, tmp_path, monkeypatch):
    files = {
        "a.png": [Detections(Detection("car", [0.1, 0.1, 0.3, 0.3])), Polylines(Polyline("tree", RING))],
        "sub/b.png": [
            Detections(Detection("car", [0.2, 0.2, 0.2, 0.2]), Detection("bus", [0.5, 0.5, 0.2, 0.2]))
        ],
        "c.png": [Detections(), None],
        "d.png": [Classification("night")],
    }
    calls = []
    monkeypatch.setattr(ci, "_download", fake_download(files, calls))
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path)
    assert calls[0][:2] == ("task", "12")
    assert result["task"] == "detection" and result["name"] == "Loose"
    assert result["images_added"] == 4 and result["images_rejected"] == 0
    assert result["labels"] == 4  # 3 boxes + the converted polygon; the tag is left out
    assert result["classes"] == ["car", "tree", "bus"]  # declared first, then used
    problems = {p["code"]: p["count"] for p in result["problems"]}
    assert problems == {"converted_to_box": 1, "dropped_other_kind": 1}
    ann = load_annotations(store.dir(result["project_id"]))
    assert ann is not None and len(ann.images) == 2  # images without annotations stay unannotated
    names = {r.name for r in Manifest(store.dir(result["project_id"])).all()}
    assert names == {"a.png", "sub/b.png", "c.png", "d.png"}
    origin = read_origin(store.dir(result["project_id"]))
    assert origin is not None and origin["tool"] == "cvat" and origin["url"] == BASE
    assert origin["remote_id"] == "12" and origin["kind"] == "task"
    assert "pw-secret" not in str(origin) and "pw-secret" not in str(result)
    assert list((tmp_path / "ws" / "tmp").glob("import-*")) == []


def test_import_task_without_annotations(cvat, tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_download", fake_download({"a.png": [Detections()]}))
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="task", remote_id="12", task=Task.KEYPOINTS, name="Mine")
    assert (result["task"], result["labels"], result["name"]) == ("keypoints", 0, "Mine")
    assert store.get(result["project_id"]).description == ""


def test_unreadable_files_are_reported(cvat, tmp_path, monkeypatch):
    def download(settings, kind, remote_id, data_dir):
        (data_dir / "bad.png").write_bytes(b"not an image at all")
        return [ci.SampleLabels(data_dir / "bad.png", [Detections(Detection("car", [0.1, 0.1, 0.3, 0.3]))])]

    monkeypatch.setattr(ci, "_download", download)
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path)
    assert result["images_added"] == 0 and result["images_rejected"] == 1
    assert any(p["code"] == "images_rejected" for p in result["problems"])


def test_download_failure_creates_nothing(cvat, tmp_path, monkeypatch):
    def boom(*a):
        raise ToolError("CVAT login failed")

    monkeypatch.setattr(ci, "_download", boom)
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(ToolError, match="login failed"):
        run(store, tmp_path)
    assert store.list() == [] and list((tmp_path / "ws" / "tmp").glob("import-*")) == []


def test_empty_remote_is_an_error(cvat, tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_download", lambda *a: [])
    with pytest.raises(ToolError, match="no images"):
        run(ProjectStore(tmp_path / "projects"), tmp_path)


def test_rollback_when_conversion_fails(cvat, tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_download", fake_download({"a.png": [Detections()]}))
    monkeypatch.setattr(ci.common, "finish", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(RuntimeError):
        run(store, tmp_path)
    assert store.list() == []
    assert not list(Path(tmp_path / "projects").glob("*/project.json"))
    assert list((tmp_path / "ws" / "tmp").glob("import-*")) == []


def test_cancel_while_reading_annotations(cvat, tmp_path, monkeypatch):
    cancel = threading.Event()
    files = {"a.png": [Detections()], "b.png": [Detections()]}
    inner = fake_download(files)

    def download(*a):
        samples = inner(*a)
        orig = ci.common.file_id
        monkeypatch.setattr(ci.common, "file_id", lambda p: (cancel.set(), orig(p))[1])
        return samples

    monkeypatch.setattr(ci, "_download", download)
    store = ProjectStore(tmp_path / "projects")
    with pytest.raises(Cancelled):
        run(store, tmp_path, Progress(Job("j", 0, "import_project"), cancel))
    assert store.list() == []


@pytest.mark.integration
def test_real_fiftyone_dataset(cvat, tmp_path, monkeypatch):
    """The real temporary dataset, with ``import_annotations`` replaced by a function that fills it."""

    made = []

    def fake_import(
        dataset, data_path=None, download_media=False, url=None, username=None, password=None, **kw
    ):
        assert download_media and kw == {"task_ids": [12]} and password == "pw-secret"
        sources = make_images(Path(data_path).parent / "src", 2)
        made.append(dataset.name)
        for i, source in enumerate(sources):
            target = Path(data_path) / f"f{i}.png"
            target.write_bytes(source.read_bytes())
            sample = fo.Sample(filepath=str(target))
            sample["detections"] = fo.Detections(
                detections=[fo.Detection(label="car", bounding_box=[0.1, 0.1, 0.3, 0.3])]
            )
            if i == 0:
                sample["polylines"] = fo.Polylines(
                    polylines=[fo.Polyline(label="tree", points=[RING], filled=True, closed=True)]
                )
                sample["tags_field"] = fo.Classification(label="day")
            dataset.add_sample(sample)

    monkeypatch.setattr(fouc, "import_annotations", fake_import)
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path)
    assert result["task"] == "detection" and result["images_added"] == 2
    assert result["labels"] == 3 and result["classes"] == ["car", "tree"]
    assert made and made[0] not in fo.list_datasets()  # the temporary dataset is gone


# -- project import through the synchronisation engine ----------------------------------
@pytest.fixture
def tasks_remote(cvat, monkeypatch):
    """A CVAT-like in-memory remote (id 1): task A (4 frames, job size 2, 2 boxes) and task B (1 frame)."""

    fake = FakeRemote("cvat")
    monkeypatch.setitem(registry.OVERRIDES, "cvat", fake)
    fake.seed("Street", classes=("car",))
    fake.new_task("1", "t1", [(f"a{i}.png", [box("car")] if i < 2 else []) for i in range(4)], 2, "10")
    fake.new_task("1", "t2", [("b0.png", [box("bus")])], None, "11")
    return fake


def local_tasks(store, result):
    return TaskStore(store.dir(result["project_id"])).load()


def test_import_project_keeps_tasks_jobs_and_link(tasks_remote, tmp_path):
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="project", remote_id="1")
    assert result["linked"] is True and result["tasks"] == 2
    assert result["task"] == "detection" and result["name"] == "Street"
    assert result["images_added"] == 5 and result["labels"] == 3
    assert result["classes"] == ["car", "bus"]  # declared first, then used
    tasks = local_tasks(store, result)
    assert [(t.name, len(t.images), t.job_size) for t in tasks] == [("t1", 4, 2), ("t2", 1, None)]
    manifest = {r.id: r.name for r in Manifest(store.dir(result["project_id"])).all()}
    assert [manifest[i] for i in tasks[0].images] == ["a0.png", "a1.png", "a2.png", "a3.png"]
    assert [len(j.image_ids) for j in jobs(tasks[0])] == [2, 2]
    origin = read_origin(store.dir(result["project_id"]))
    assert origin is not None and origin["remote_id"] == "1" and origin["kind"] == "project"
    link = links.get_link(store.dir(result["project_id"]), "cvat")
    assert link is not None and link["remote_id"] == "1" and len(link["items"]) == 5
    assert link["tasks"] == {
        str(tasks[0].id): {"remote_task": "10", "name": "t1", "jobs": {"1": "1001", "2": "1002"}},
        str(tasks[1].id): {"remote_task": "11", "name": "t2", "jobs": {"1": "1101"}},
    }
    progress = Progress(Job("j", 0, "sync"), threading.Event())
    plan = engine.make_plan(
        store, result["project_id"], "cvat", SETTINGS, registry.get_remote("cvat"), progress
    )
    assert plan["tasks_new"] == [] and plan["tasks_create"] == [] and plan["tasks_delete"] == []
    assert (plan["pull"], plan["push"], plan["add"], plan["conflicts"]) == (0, 0, 0, [])
    assert plan["remove_local"] == plan["remove_remote"] == 0
    assert plan["restore_local"] == plan["restore_remote"] == 0
    assert plan["unchanged"] == 5 and plan["remote_only"]["count"] == 0


def test_import_one_task_of_a_project(tasks_remote, tmp_path):
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="task", remote_id="11")
    assert result["tasks"] == 1 and result["images_added"] == 1 and result["classes"] == ["bus"]
    assert [(t.name, len(t.images)) for t in local_tasks(store, result)] == [("t2", 1)]
    origin = read_origin(store.dir(result["project_id"]))
    assert origin is not None and (origin["remote_id"], origin["kind"]) == ("11", "task")
    link = links.get_link(store.dir(result["project_id"]), "cvat")
    assert link is not None and link["remote_id"] == "1"  # the CVAT project, not the task
    assert [m["remote_task"] for m in link["tasks"].values()] == ["11"]
    assert all(c == ["11"] for c in tasks_remote.item_calls)  # the other task is never read
    assert not any(c[0] == "frame" and c[1] == "10" for c in tasks_remote.calls)


def test_import_project_infers_task_from_polygons(tasks_remote, tmp_path):
    ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]
    tasks_remote.new_task(
        "1", "t3", [("c0.png", [Label("tree", bbox=[0.1, 0.1, 0.4, 0.5], polygons=[ring])])] * 1, None, "12"
    )
    for key in ("1", "2", "3"):  # make polygons the most frequent kind
        tasks_remote.put("1", f"p{key}.png", [Label("tree", bbox=[0.1, 0.1, 0.4, 0.5], polygons=[ring])])
    for tid in ("10", "11"):
        for it in tasks_remote.projects["1"]["items"].values():
            if it.key.startswith(f"{tid}:"):
                it.labels = [Label("tree", bbox=[0.1, 0.1, 0.4, 0.5], polygons=[ring])]
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="project", remote_id="1")
    assert result["task"] == "segmentation" and "tree" in result["classes"]


def test_import_project_failure_rolls_back(tasks_remote, tmp_path):
    store = ProjectStore(tmp_path / "projects")
    tasks_remote.projects["1"]["items"].clear()
    tasks_remote.task_meta["1"].clear()
    with pytest.raises(ToolError, match="no images"):
        run(store, tmp_path, kind="project", remote_id="1")
    assert store.list() == []


def test_an_address_that_is_not_the_cvat_api(monkeypatch):
    class Page(FakeResp):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1")

    page = Page(200, content=b"<html>", content_type="text/html")
    fake = FakeSession({("GET", "/api/projects"): page}, BASE)
    monkeypatch.setattr(requests, "Session", lambda: fake)
    with pytest.raises(ToolError, match="did not answer as the CVAT API"):
        ci.list_remote(SETTINGS)


@pytest.mark.parametrize(
    "text",
    [
        "403 Forbidden: You have reached the maximum number of tasks",
        "HTTP 403: please upgrade your plan",
    ],
)
def test_plan_limit_is_not_a_login_problem(text):
    err = ci.explain(RuntimeError(text), "https://cvat.example")
    assert "plan limit" in str(err) and "login" not in str(err)


def test_status_codes_are_matched_as_numbers_not_inside_ids():
    assert "login failed" in str(ci.explain(RuntimeError("HTTP 401 Unauthorized")))
    assert "refused" in str(ci.explain(RuntimeError("403 Forbidden")))
    assert "slow down" in str(ci.explain(RuntimeError("Error 429")))
    for text in ("task 14031 failed", "/api/tasks/403/data broke", "id 4010 and 2401"):
        assert str(ci.explain(RuntimeError(text))) == f"CVAT error: {text}"


def test_explain_gives_codes():
    url = "https://cvat.example"
    assert ci.explain(RuntimeError("HTTP 401 Unauthorized"), url).code == "cvat_login_failed"
    assert ci.explain(RuntimeError("403 Forbidden"), url).code == "cvat_refused"
    assert ci.explain(RuntimeError("please upgrade your plan"), url).code == "cvat_plan_limit"
    assert ci.explain(RuntimeError("Error 429"), url).code == "cvat_rate_limited"
    assert ci.explain(requests.ConnectionError("x"), url).params == {"url": url}
    assert ci.explain(RuntimeError("HTTP 404"), url).code == "cvat_not_found"
    other = ci.explain(RuntimeError("something odd"), url)
    assert (other.code, other.params) == ("cvat_error_other", {"reason": "something odd"})


def test_http_failures_of_the_import_client_carry_codes(monkeypatch):
    def ask(status):
        fake = FakeSession({("GET", "/api/projects"): FakeResp(status, {})}, BASE)
        monkeypatch.setattr(requests, "Session", lambda: fake)
        with pytest.raises(ToolError) as err:
            ci.list_remote(SETTINGS)
        return err.value

    assert ask(401).code == "cvat_login_failed"
    assert ask(429).code == "cvat_rate_limited"
    server = ask(500)
    assert (server.code, server.params) == ("cvat_error", {"http": 500})
    assert ask(404).code == "cvat_not_found"
    with pytest.raises(ToolError) as err:
        ci.list_remote(ToolSettings("cvat", BASE, "", ""))
    assert err.value.code == "cvat_config_missing"
    with pytest.raises(ToolError) as err:
        ci.list_remote(ToolSettings("cvat", "ftp://x", "u", "p"))
    assert err.value.code == "cvat_url_invalid"


def test_a_missing_remote_item_has_a_code(cvat):
    with pytest.raises(RemoteNotFound) as err:
        ci.check_remote(SETTINGS, "project", "99")
    assert err.value.code == "cvat_not_found"
    with pytest.raises(RemoteNotFound) as err:
        ci.check_remote(SETTINGS, "project", "x")
    assert err.value.code == "cvat_item_not_found"


class RecordingProgress(Progress):
    """Keeps every (done, total, message) the job reports."""

    def __init__(self) -> None:
        super().__init__(Job("j", 0, "import_project"), threading.Event())
        self.seen: list[tuple[int, int, str]] = []

    def update(self, done=None, total=None, message=None):
        super().update(done, total, message)
        self.seen.append((self._job.done, self._job.total, self._job.message))


def test_import_project_reports_each_downloaded_image(tasks_remote, tmp_path):
    """#564: the bar stayed still while the frames were downloaded; it now counts them (5 frames here)."""
    store = ProjectStore(tmp_path / "projects")
    progress = RecordingProgress()
    run(store, tmp_path, progress=progress, kind="project", remote_id="1")
    downloads = [(done, total) for done, total, message in progress.seen if message == "Downloading images"]
    assert (1, 5) in downloads and (5, 5) in downloads
    assert [d for d, _ in downloads] == sorted(d for d, _ in downloads)  # never goes back


@pytest.mark.integration
def test_a_project_imported_from_cvat_is_linked_and_opens_in_fiftyone_right_away(
    tasks_remote, tmp_path, monkeypatch
):
    """#567: right after the import, Open in CVAT, Compare and Edit in FiftyOne all work."""
    client, state = make_client(tmp_path)
    login = {"url": BASE, "username": SETTINGS.username, "secret": SETTINGS.secret}
    assert client.put("/api/annotation-tools/cvat", json=login).status_code == 200
    result = run(state.store, tmp_path, kind="project", remote_id="1", name="Imported")
    pid = result["project_id"]
    (link,) = client.get(f"/api/projects/{pid}/links").json()["links"]
    assert link["tool"] == "cvat" and link["remote_id"] == "1" and link["supported"] is True
    tasks = client.get(f"/api/projects/{pid}/tasks").json()["tasks"]
    assert [t["remote"]["task"] for t in tasks] == ["10", "11"]  # each task opens in CVAT
    prefix = f"activelabel-import-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(bridge, "DATASET_PREFIX", prefix)
    monkeypatch.setattr(bridge, "launch_app", lambda *args, **kwargs: "http://fake-app/datasets/x")
    try:
        sent = client.post(f"/api/projects/{pid}/annotate", json={"tool": "fiftyone", "scope": "all"})
        assert sent.status_code == 200, sent.text
        job = wait_job(client, sent.json()["id"])
        assert job["status"] == "done", job
        assert job["result"]["session"]["images"] == result["images_added"]
    finally:
        for name in fo.list_datasets():
            if name.startswith(prefix):
                fo.delete_dataset(name)


def wait_job(client, job_id):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_import_project_keeps_the_typed_name_and_description(tasks_remote, tmp_path):
    """#565: the name typed in the form ("meu_dataset_round_1") must not become the CVAT name."""
    store = ProjectStore(tmp_path / "projects")
    result = run(
        store, tmp_path, kind="project", remote_id="1", name="meu_dataset_round_1", description="Minha"
    )
    project = store.get(result["project_id"])
    assert (result["name"], project.name, project.description) == ("meu_dataset_round_1",) * 2 + ("Minha",)


def test_import_project_reports_what_it_could_not_bring(tasks_remote, tmp_path):
    """Shapes that are not converted and the labels of identical images are counted, never lost silently."""
    items = tasks_remote.projects["1"]["items"]
    items["10:2"].not_converted = {"ellipse": 1}
    items["10:3"].not_converted = {"ellipse": 1, "tracks": 2}
    tasks_remote.frame_bytes[("11", 0)] = tasks_remote.frame_bytes[("10", 0)]  # b0 is the same picture as a0
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="project", remote_id="1")
    problems = {p["code"]: p for p in result["problems"]}
    assert problems["shapes_not_converted"]["count"] == 4
    assert problems["shapes_not_converted"]["examples"] == ["ellipse: 2", "tracks: 2"]
    assert result["images_added"] == 4 and result["labels"] == 2
    assert problems["duplicate_labels_dropped"]["count"] == 1  # the bus box of b0


def test_importing_a_project_twice_is_refused_naming_the_first_one(tasks_remote, tmp_path):
    store = ProjectStore(tmp_path / "projects")
    first = run(store, tmp_path, kind="project", remote_id="1")
    folder = store.dir(first["project_id"])
    link = links.get_link(folder, "cvat")
    links.save_link(
        folder, "cvat", {**link, "url": f"{BASE}/projects/1"}
    )  # the fake remote has its own address
    for kind, remote_id in (("project", "1"), ("task", "11")):  # a task goes to the same remote project
        with pytest.raises(ToolError) as err:
            run(store, tmp_path, kind=kind, remote_id=remote_id)
        assert err.value.code == "cvat_already_imported"
        assert err.value.params == {"project_id": first["project_id"], "name": "Street"}
    assert len(store.list()) == 1
    assert ci.linked_projects(store, SETTINGS)["1"].id == first["project_id"]
    assert ci.linked_projects(store, ToolSettings("cvat", "http://other-server", "u", "pw")) == {}


def test_a_video_that_could_not_be_used_is_reported(tasks_remote, tmp_path, monkeypatch):
    """The tasks stay as images and the report names them with a code, not only the server log."""
    tasks_remote.mark_video("10", frames=4)
    tasks_remote.mark_video("11")

    def refuse(*args, **kwargs):
        raise ToolError("backup refused")

    monkeypatch.setattr(tasks_remote, "video_source", refuse)
    store = ProjectStore(tmp_path / "projects")
    result = run(store, tmp_path, kind="project", remote_id="1")
    problems = {p["code"]: p for p in result["problems"]}
    assert problems["video_as_images"]["level"] == "warning"
    assert problems["video_as_images"]["count"] == 2
    assert problems["video_as_images"]["examples"] == ["t1", "t2"]
    assert result["tasks"] == 2


def test_a_video_that_is_used_is_not_reported(tasks_remote, tmp_path):
    tasks_remote.mark_video("10", frames=4)
    result = run(ProjectStore(tmp_path / "projects"), tmp_path, kind="project", remote_id="1")
    assert "video_as_images" not in {p["code"] for p in result["problems"]}


def test_rotated_boxes_are_reported_apart_from_the_shapes_left_out(tasks_remote, tmp_path):
    items = tasks_remote.projects["1"]["items"]
    items["10:2"].not_converted = {"rotated_rectangle": 2}
    items["10:3"].not_converted = {"ellipse": 1}
    result = run(ProjectStore(tmp_path / "projects"), tmp_path, kind="project", remote_id="1")
    problems = {p["code"]: p for p in result["problems"]}
    assert problems["rotated_boxes_flattened"]["count"] == 2
    assert problems["shapes_not_converted"]["examples"] == ["ellipse: 1"]


def test_download_without_fiftyone_stops_before_reading(monkeypatch, tmp_path):
    monkeypatch.setattr(optional, "FIFTYONE", False)
    settings = ToolSettings("cvat", url="http://cvat", username="u", secret="s")
    with pytest.raises(optional.FeatureMissing) as caught:
        ci._download(settings, "project", "7", tmp_path)
    assert caught.value.code == "fiftyone_not_installed"
