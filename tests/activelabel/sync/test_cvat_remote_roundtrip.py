"""End-to-end synchronisation with CVAT against the strict in-memory server of ``cvat_fake``.

The sync engine (``make_plan`` / ``apply_plan``) drives the real ``CvatRemote``; only the HTTP layer is the
fake. Pixels are checked exactly: images are 200 x 100, so relative x is pixels / 200 and y pixels / 100.

Tests named ``test_defect_*`` fail on purpose: each pins down a real defect of the adapter or the engine
(see the QA report); they must pass once it is fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from cvat_fake import FakeCvatServer
from sample_images import make_image
from sync_fakes import progress

from activelabel.annotate.base import ToolSettings
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import ACTIVE, EXCLUDED, Manifest
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.projects import Project, ProjectStore
from activelabel.storage import write_json
from activelabel.sync import engine, links
from activelabel.sync.cvat_remote import CvatRemote
from activelabel.tasks import Task
from activelabel.tasks_store import ImageTask

SETTINGS = ToolSettings("cvat", "https://cvat.test", "ana", "not-a-real-secret")
W, H = 200, 100
TOL = 1e-6


@dataclass
class Env:
    """A project on a temp dir, the fake CVAT and the real adapter between them."""

    store: ProjectStore
    project: Project
    ids: list[str]
    dir: Path
    server: FakeCvatServer
    remote: CvatRemote
    rid: str = ""
    tid: int = 0

    @property
    def pid(self) -> int:
        """The CVAT project id."""
        return int(self.rid)

    def label_id(self, name: str) -> int:
        """Id of a label of the CVAT project.

        Args:
            name: the label name.

        Returns:
            The label id.
        """
        return self.server.label_id(self.pid, name)


# -- helpers --------------------------------------------------------------------------------------------
def make_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: Task = Task.DETECTION,
    count: int = 3,
    labels: dict[int, list[Label]] | None = None,
    job_size: int | None = None,
    names: list[str] | None = None,
) -> Env:
    """A project with ``count`` 200 x 100 images in one local task, its ground truth and a fake CVAT.

    Args:
        tmp_path: pytest temp dir.
        monkeypatch: pytest monkeypatch (the HTTP layer is replaced).
        task: project type.
        count: number of images.
        labels: ground truth ``{image index: [Label]}``.
        job_size: images per job of the local task.
        names: image file names (``img0.png`` ...).

    Returns:
        The environment (not linked to CVAT yet).
    """
    server = FakeCvatServer().install(monkeypatch)
    store = ProjectStore(tmp_path / "projects")
    classes = ["cat", "dog"]
    project = store.create("Pets", "", task.value, classes)
    folder = tmp_path / "src"
    names = names or [f"img{i}.png" for i in range(count)]
    for i, name in enumerate(names):
        make_image(folder / name, size=(W, H), seed=i + 1, fmt="PNG" if name.endswith(".png") else "JPEG")
    manifest = Manifest(store.dir(project.id))
    ingest_folder(manifest, folder, folder)
    by_name = {r.name: r.id for r in manifest.all()}
    ids = [by_name[n] for n in names]
    directory = store.dir(project.id)
    local_task = ImageTask(1, "T", "2026-01-01T00:00:00", job_size, ids)
    write_json(directory / "tasks.json", {"next_id": 2, "tasks": [local_task.as_dict()]})
    env = Env(store, project, ids, directory, server, CvatRemote())
    for index, found in (labels or {}).items():
        set_local(env, index, found)
    return env


def plan(env: Env, **kw: Any) -> dict[str, Any]:
    """Compare the project with CVAT (``engine.make_plan``).

    Args:
        env: the environment.
        **kw: options of ``make_plan`` (``direction``, ``scope``, ``task_id``, ``job``, ``create``).

    Returns:
        The plan summary.
    """
    return engine.make_plan(env.store, env.project.id, "cvat", SETTINGS, env.remote, progress(), **kw)


def apply(
    env: Env, plan_id: str, resolutions: dict[str, str] | None = None, default: str | None = None
) -> dict[str, Any]:
    """Apply a plan (``engine.apply_plan``).

    Args:
        env: the environment.
        plan_id: the plan.
        resolutions: ``{image_id: "local" | "remote"}`` for conflicts.
        default: choice for the conflicts without a resolution.

    Returns:
        The counts.
    """
    return engine.apply_plan(
        env.store,
        env.project.id,
        "cvat",
        SETTINGS,
        env.remote,
        plan_id,
        resolutions or {},
        default,
        progress(),
    )


def sync(env: Env, **kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plan and apply in one go.

    Args:
        env: the environment.
        **kw: options of ``make_plan``.

    Returns:
        ``(plan, result)``.
    """
    p = plan(env, **kw)
    return p, apply(env, p["plan_id"])


def first_push(env: Env) -> dict[str, Any]:
    """Send the whole project to a new CVAT project and remember its ids.

    Args:
        env: the environment.

    Returns:
        The apply result.
    """
    p = plan(env, create=True)
    result = apply(env, p["plan_id"])
    link = links.get_link(env.dir, "cvat") or {}
    env.rid = str(link["remote_id"])
    env.tid = int(link["tasks"]["1"]["remote_task"])
    return result


def set_local(env: Env, index: int, labels: list[Label]) -> None:
    """Replace the ground truth of image ``index``.

    Args:
        env: the environment.
        index: image position.
        labels: its new labels.
    """
    found = load_annotations(env.dir) or AnnotationSet(env.project.task, list(env.project.classes), {}, {})
    found.images[env.ids[index]] = labels
    save_annotations(env.dir, found)


def local(env: Env, index: int) -> list[Label]:
    """Ground truth of image ``index`` (sorted by class and position).

    Args:
        env: the environment.
        index: image position.

    Returns:
        Its labels.
    """
    found = load_annotations(env.dir)
    labels = found.images.get(env.ids[index], []) if found else []
    return sorted(labels, key=lambda lb: (lb.label, flat(lb.bbox), flat(lb.keypoints)))


def frame_of(env: Env, index: int) -> int:
    """CVAT task frame of image ``index`` (from the link).

    Args:
        env: the environment.
        index: image position.

    Returns:
        The frame number.
    """
    entry = (links.get_link(env.dir, "cvat") or {})["items"][env.ids[index]]
    return int(entry["key"].partition(":")[2])


def flat(value: Any) -> list[float]:
    """Every number of a nested list, in order.

    Args:
        value: numbers, nested lists or None.

    Returns:
        The flat list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in flat(item)]
    return [float(value)]


def shapes(env: Env, frame: int | None = None) -> list[dict[str, Any]]:
    """Shapes CVAT holds for the task (of one frame), in id order.

    Args:
        env: the environment.
        frame: only this task frame.

    Returns:
        The stored shapes.
    """
    found = env.server.annotations(env.tid)["shapes"]
    return [s for s in found if frame is None or s["frame"] == frame]


def rows(env: Env, frame: int | None = None) -> list[tuple[int, int, str, tuple[float, ...]]]:
    """``(frame, label id, type, points)`` of the CVAT shapes, points rounded to 6 decimals, sorted.

    Args:
        env: the environment.
        frame: only this task frame.

    Returns:
        One tuple per shape.
    """
    return sorted(
        (s["frame"], s["label_id"], s["type"], tuple(round(v, 6) for v in s["points"]))
        for s in shapes(env, frame)
    )


def px(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Relative ``[x, y, w, h]`` of a pixel rectangle ``x0, y0, x1, y1`` of a 200 x 100 image.

    Args:
        x0: left.
        y0: top.
        x1: right.
        y1: bottom.

    Returns:
        The relative box.
    """
    return [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]


def rel_box(name: str, x0: float, y0: float, x1: float, y1: float) -> Label:
    """A detection label from a pixel rectangle.

    Args:
        name: the class.
        x0: left.
        y0: top.
        x1: right.
        y1: bottom.

    Returns:
        The label.
    """
    return Label(name, bbox=px(x0, y0, x1, y1))


def rel_poly(name: str, pts: list[tuple[float, float]]) -> Label:
    """A segmentation label from pixel vertices (its box is the box of the vertices).

    Args:
        name: the class.
        pts: ``(x, y)`` in pixels.

    Returns:
        The label.
    """
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Label(
        name, bbox=px(min(xs), min(ys), max(xs), max(ys)), polygons=[[[x / W, y / H] for x, y in pts]]
    )


def rect(frame: int, label_id: int, pts: list[float], **extra: Any) -> dict[str, Any]:
    """A CVAT rectangle shape.

    Args:
        frame: task frame.
        label_id: the label.
        pts: ``[xtl, ytl, xbr, ybr]`` in pixels.
        **extra: more fields (``rotation``, ``occluded``, ``z_order`` ...).

    Returns:
        The shape.
    """
    return {"type": "rectangle", "frame": frame, "label_id": label_id, "points": pts, **extra}


def same_without_ids(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two CVAT objects are equal apart from their ids.

    Args:
        a: one object.
        b: the other.

    Returns:
        True when equal.
    """
    return {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}


def rotated_box(pts: list[float], degrees: float) -> list[float]:
    """Pixel box ``[x0, y0, x1, y1]`` that encloses a rectangle rotated clockwise around its centre.

    Args:
        pts: ``[xtl, ytl, xbr, ybr]``.
        degrees: rotation.

    Returns:
        The enclosing axis-aligned rectangle.
    """
    cx, cy = (pts[0] + pts[2]) / 2, (pts[1] + pts[3]) / 2
    hw, hh = (pts[2] - pts[0]) / 2, (pts[3] - pts[1]) / 2
    a = math.radians(degrees)
    ex = hw * abs(math.cos(a)) + hh * abs(math.sin(a))
    ey = hw * abs(math.sin(a)) + hh * abs(math.cos(a))
    return [cx - ex, cy - ey, cx + ex, cy + ey]


class DrawOnce:
    """A hook for ``FakeCvatServer.after``: a person draws a box between the read and the PUT of a push."""

    def __init__(self, env: Env, points: list[float]) -> None:
        """Remember what to draw.

        Args:
            env: the environment.
            points: the rectangle, in pixels.
        """
        self.env, self.points, self.reads = env, points, 0

    def __call__(self) -> None:
        """Draw the box right after the second read of the task (the one of ``write``)."""
        self.reads += 1
        if self.reads == 2:
            self.env.server.add_shape(self.env.tid, rect(1, self.env.label_id("dog"), self.points))


@pytest.fixture
def detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    """Three images; img0 has a cat and a dog, img1 a cat, img2 nothing; already sent to CVAT."""
    env = make_env(
        tmp_path,
        monkeypatch,
        labels={
            0: [rel_box("cat", 20, 20, 80, 60), rel_box("dog", 100, 10, 140, 40)],
            1: [rel_box("cat", 50, 25, 150, 75)],
        },
    )
    first_push(env)
    return env


# -- the fake itself: it must be as strict as CVAT, or the tests below prove nothing --------------------
def test_fake_refuses_what_cvat_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch)
    first_push(env)
    server, cat = env.server, env.label_id("cat")
    good = rect(0, cat, [1, 2, 3, 4])
    bad_bodies = {
        "unknown label": {"shapes": [{**good, "label_id": 99999}]},
        "unknown field": {"shapes": [{**good, "color": "red"}]},
        "3 numbers": {"shapes": [{**good, "points": [1, 2, 3]}]},
        "frame outside": {"shapes": [{**good, "frame": 3}]},
        "negative frame": {"shapes": [{**good, "frame": -1}]},
        "id in a PUT": {"shapes": [{**good, "id": 7}]},
        "polygon with 2 points": {"shapes": [{**good, "type": "polygon", "points": [1, 2, 3, 4]}]},
        "rotated polygon": {
            "shapes": [{**good, "type": "polygon", "points": [1, 2, 3, 4, 5, 6], "rotation": 5}]
        },
        "attribute of another label": {"shapes": [{**good, "attributes": [{"spec_id": 1, "value": "x"}]}]},
        "unknown top key": {"shapes": [], "extra": []},
        "skeleton without parts": {"shapes": [{**good, "type": "skeleton", "points": []}]},
        "track on a bad frame": {
            "tracks": [
                {
                    "frame": 9,
                    "label_id": cat,
                    "shapes": [{"type": "rectangle", "frame": 9, "points": [1, 2, 3, 4]}],
                }
            ]
        },
    }
    for why, body in bad_bodies.items():
        before = server.annotations(env.tid)
        answer = server.request("PUT", f"https://cvat.test/api/tasks/{env.tid}/annotations", json=body)
        assert answer.status_code == 400, why
        assert server.annotations(env.tid) == before, why  # nothing half written
    assert server.request("PUT", "https://cvat.test/api/tasks/99999/annotations", json={}).status_code == 404


def test_fake_put_replaces_and_assigns_new_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch)
    first_push(env)
    server, cat = env.server, env.label_id("cat")
    body = {"shapes": [rect(0, cat, [1, 2, 3, 4])]}
    url = f"https://cvat.test/api/tasks/{env.tid}/annotations"
    assert server.request("PUT", url, json=body).status_code == 200
    first = server.annotations(env.tid)["shapes"][0]
    assert first["rotation"] == 0 and first["occluded"] is False and first["source"] == "manual"
    server.request("PUT", url, json=body)
    second = server.annotations(env.tid)["shapes"][0]
    assert second["id"] != first["id"] and len(server.annotations(env.tid)["shapes"]) == 1
    # PATCH create / update / delete
    made = server.request(
        "PATCH", url, params={"action": "create"}, json={"shapes": [rect(1, cat, [5, 6, 7, 8])]}
    )
    new_id = made.json()["shapes"][0]["id"]
    upd = {"shapes": [{**rect(1, cat, [9, 9, 9, 9]), "id": new_id}]}
    assert server.request("PATCH", url, params={"action": "update"}, json=upd).status_code == 200
    assert server.shape(env.tid, new_id)["points"] == [9, 9, 9, 9]
    ghost = {"shapes": [{**rect(1, cat, [9, 9, 9, 9]), "id": 424242}]}
    assert server.request("PATCH", url, params={"action": "update"}, json=ghost).status_code == 400
    assert server.request("PATCH", url, params={"action": "delete"}, json=upd).status_code == 200
    assert [s["frame"] for s in shapes(env)] == [0]


def test_fake_job_endpoints_use_task_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, count=7, job_size=3)
    first_push(env)
    server, cat = env.server, env.label_id("cat")
    jobs = server.jobs_of(env.tid)
    assert [(j["start_frame"], j["stop_frame"]) for j in jobs] == [(0, 2), (3, 5), (6, 6)]
    server.put(env.tid, {"shapes": [rect(1, cat, [1, 1, 2, 2]), rect(4, cat, [3, 3, 4, 4])]})
    second = f"https://cvat.test/api/jobs/{jobs[1]['id']}/annotations"
    assert [s["frame"] for s in server.request("GET", second).json()["shapes"]] == [4]  # task frame, not 1
    wrong = server.request(
        "PATCH", second, params={"action": "create"}, json={"shapes": [rect(1, cat, [1, 1, 2, 2])]}
    )
    assert wrong.status_code == 400  # frame 1 is not in the second job
    server.request("PUT", second, json={"shapes": [rect(5, cat, [7, 7, 8, 8])]})
    assert sorted(s["frame"] for s in shapes(env)) == [1, 5]  # the other jobs are untouched


# -- push: local labels -> CVAT payloads ----------------------------------------------------------------
def test_push_detection_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(
        tmp_path,
        monkeypatch,
        labels={
            0: [rel_box("cat", 20, 20, 80, 60)],
            1: [rel_box("dog", 100, 10, 140, 40), rel_box("cat", 0, 0, 200, 100)],
        },
    )
    result = first_push(env)
    assert result["tasks_created"] == 1 and env.server.label_names(env.pid) == ["cat", "dog"]
    assert all(x["type"] == "any" for x in env.server.labels.values())
    put = env.server.calls("PUT", f"/api/tasks/{env.tid}/annotations")[-1].body
    assert set(put) == {"version", "tags", "shapes", "tracks"} and put["tags"] == [] and put["tracks"] == []
    got = sorted((s["frame"], s["label_id"], tuple(round(v, 6) for v in s["points"])) for s in put["shapes"])
    cat, dog = env.label_id("cat"), env.label_id("dog")
    assert got == [(0, cat, (20, 20, 80, 60)), (1, cat, (0, 0, 200, 100)), (1, dog, (100, 10, 140, 40))]
    for s in put["shapes"]:  # the defaults CVAT needs, and no ids
        assert s["type"] == "rectangle" and s["rotation"] == 0 and s["occluded"] is False and "id" not in s
        assert s["source"] == "manual" and s["attributes"] == [] and s["z_order"] == 0 and s["group"] == 0
    assert [f["name"] for f in env.server.tasks[env.tid].frames] == ["img0.png", "img1.png", "img2.png"]
    assert [frame_of(env, i) for i in range(3)] == [0, 1, 2]


def test_push_segmentation_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ring_a = [(20, 10), (100, 10), (60, 80)]
    ring_b = [(120, 20), (180, 20), (180, 60), (120, 60)]
    two = Label(
        "dog",
        bbox=px(20, 10, 180, 80),
        polygons=[[[x / W, y / H] for x, y in ring_a], [[x / W, y / H] for x, y in ring_b]],
    )
    env = make_env(tmp_path, monkeypatch, Task.SEGMENTATION, labels={0: [rel_poly("cat", ring_a)], 1: [two]})
    first_push(env)
    cat, dog = env.label_id("cat"), env.label_id("dog")
    assert rows(env) == [
        (0, cat, "polygon", (20, 10, 100, 10, 60, 80)),
        (1, dog, "polygon", (20, 10, 100, 10, 60, 80)),
        (1, dog, "polygon", (120, 20, 180, 20, 180, 60, 120, 60)),
    ]


def test_push_keypoints_payload_with_missing_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kps = [[0.1, 0.2, 2.0], [0.0, 0.0, 0.0], [0.5, 0.5, 2.0]]
    label = Label("cat", bbox=px(20, 20, 100, 50), keypoints=kps)
    env = make_env(tmp_path, monkeypatch, Task.KEYPOINTS, labels={0: [label]})
    first_push(env)
    (shape,) = shapes(env)
    assert shape["type"] == "points" and shape["frame"] == 0 and shape["label_id"] == env.label_id("cat")
    assert shape["points"] == pytest.approx([20, 20, 0, 0, 100, 50])  # missing point: the top-left corner
    assert shape["occluded"] is False


def test_push_classification_payload_is_tags_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, Task.CLASSIFICATION, labels={0: [Label("cat")], 2: [Label("dog")]})
    first_push(env)
    ann = env.server.annotations(env.tid)
    assert ann["shapes"] == [] and ann["tracks"] == []
    assert sorted((t["frame"], t["label_id"]) for t in ann["tags"]) == [
        (0, env.label_id("cat")),
        (2, env.label_id("dog")),
    ]


# -- pull: edits made in CVAT -> local store ------------------------------------------------------------
def test_pull_detection_edits_from_cvat(detection: Env) -> None:
    env, srv = detection, detection.server
    a, b = shapes(env, 0)
    (c,) = shapes(env, 1)
    srv.edit_shape(env.tid, a["id"], points=[30, 25, 90, 70])  # moved and resized
    srv.edit_shape(env.tid, b["id"], label_id=env.label_id("cat"))  # class changed
    srv.delete_shape(env.tid, c["id"])
    srv.add_shape(env.tid, rect(2, env.label_id("dog"), [40, 10, 100, 50]))
    p, result = sync(env)
    assert (p["pull"], p["push"], p["conflicts"]) == (3, 0, []) and result["pulled"] == 3
    assert [(lb.label, flat(lb.bbox)) for lb in local(env, 0)] == [
        ("cat", pytest.approx(px(30, 25, 90, 70), abs=TOL)),
        ("cat", pytest.approx(px(100, 10, 140, 40), abs=TOL)),
    ]
    assert local(env, 1) == []
    assert [(lb.label, flat(lb.bbox)) for lb in local(env, 2)] == [
        ("dog", pytest.approx(px(40, 10, 100, 50), abs=TOL))
    ]
    p2 = plan(env)  # and now both sides agree
    assert (p2["pull"], p2["push"], p2["conflicts"]) == (0, 0, [])


def test_pull_occluded_z_order_and_group_are_not_changes(detection: Env) -> None:
    env, srv = detection, detection.server
    for s in shapes(env, 0):
        srv.edit_shape(env.tid, s["id"], occluded=True, z_order=4, group=2, source="auto")
    assert plan(env)["pull"] == 0
    assert [lb.label for lb in local(env, 0)] == ["cat", "dog"]  # still there, untouched


def test_defect_rotated_rectangle_is_read_as_unrotated(detection: Env) -> None:
    """A rectangle rotated by 30 degrees in CVAT must reach the project as the box that encloses it."""
    env, srv = detection, detection.server
    cat_shape = shapes(env, 1)[0]  # [50, 25, 150, 75]
    srv.edit_shape(env.tid, cat_shape["id"], rotation=30)
    sync(env)
    (label,) = local(env, 1)
    x0, y0, x1, y1 = rotated_box([50, 25, 150, 75], 30)
    assert flat(label.bbox) == pytest.approx(px(x0, y0, x1, y1), abs=1e-4)


def test_rotated_rectangle_survives_a_push_of_its_frame(detection: Env) -> None:
    """The project keeps the enclosing box; a push of another box of the frame keeps the rotated one."""
    env, srv = detection, detection.server
    a, _ = shapes(env, 0)
    srv.edit_shape(env.tid, a["id"], rotation=30, z_order=2)
    sync(env)  # whatever the project reads, a local edit of another box of the frame follows
    set_local(env, 0, [*local(env, 0)[:1], rel_box("dog", 110, 15, 150, 45)])
    sync(env)
    kept = [s for s in shapes(env, 0) if s["rotation"] == 30]
    assert len(kept) == 1 and kept[0]["z_order"] == 2


def test_pull_ignores_shapes_of_unknown_label_ids_and_deleted_frames(detection: Env) -> None:
    env, srv = detection, detection.server
    srv.delete_frames(env.tid, [1])
    p, _ = sync(env)
    assert p["remove_local"] == 1
    manifest = {r.id: r for r in Manifest(env.dir).all()}
    assert [manifest[i].status for i in env.ids] == [ACTIVE, EXCLUDED, ACTIVE]
    assert shapes(env, 1)  # CVAT keeps the annotations of a deleted frame
    srv.tasks[env.tid].deleted = []  # restored in CVAT
    p2, _ = sync(env)
    assert p2["restore_local"] == 1
    assert {r.id: r.status for r in Manifest(env.dir).all()}[env.ids[1]] == ACTIVE


def test_pull_segmentation_edits_from_cvat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tri = [(20, 10), (100, 10), (60, 80)]
    env = make_env(
        tmp_path,
        monkeypatch,
        Task.SEGMENTATION,
        labels={0: [rel_poly("cat", tri)], 1: [rel_poly("dog", tri)]},
    )
    first_push(env)
    srv = env.server
    (p0,) = shapes(env, 0)
    (p1,) = shapes(env, 1)
    srv.edit_shape(env.tid, p0["id"], points=[20, 10, 100, 10, 80, 90])  # one vertex moved
    srv.delete_shape(env.tid, p1["id"])
    srv.add_shape(
        env.tid,
        {
            "type": "polygon",
            "frame": 2,
            "label_id": env.label_id("dog"),
            "points": [10, 10, 50, 10, 50, 50, 10, 50],
        },
    )
    p, _ = sync(env)
    assert p["pull"] == 3
    (moved,) = local(env, 0)
    assert flat(moved.polygons) == pytest.approx([0.1, 0.1, 0.5, 0.1, 0.4, 0.9], abs=TOL)
    assert flat(moved.bbox) == pytest.approx(px(20, 10, 100, 90), abs=TOL)
    assert local(env, 1) == []
    (added,) = local(env, 2)
    assert (
        added.label == "dog"
        and len(added.polygons[0]) == 4
        and flat(added.bbox) == pytest.approx(px(10, 10, 50, 50), abs=TOL)
    )
    assert plan(env)["pull"] == 0


def test_pull_keypoints_edits_from_cvat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    label = Label(
        "cat", bbox=px(20, 20, 100, 50), keypoints=[[0.1, 0.2, 2.0], [0.5, 0.25, 2.0], [0.25, 0.5, 2.0]]
    )
    env = make_env(tmp_path, monkeypatch, Task.KEYPOINTS, labels={0: [label], 1: [label]})
    first_push(env)
    (s0,) = shapes(env, 0)
    (s1,) = shapes(env, 1)
    env.server.edit_shape(env.tid, s0["id"], points=[20, 20, 120, 30, 50, 50])  # the second point moved
    env.server.edit_shape(
        env.tid, s1["id"], points=[20, 20, 0, 0, 50, 50]
    )  # the second point is "not labeled"
    p, _ = sync(env)
    assert p["pull"] == 2
    assert flat(local(env, 0)[0].keypoints) == pytest.approx(
        [0.1, 0.2, 2, 0.6, 0.3, 2, 0.25, 0.5, 2], abs=TOL
    )
    assert flat(local(env, 1)[0].keypoints) == pytest.approx([0.1, 0.2, 2, 0, 0, 0, 0.25, 0.5, 2], abs=TOL)
    env.server.edit_shape(env.tid, s0["id"], occluded=True)  # occluded: labeled but not visible
    sync(env)
    assert [k[2] for k in local(env, 0)[0].keypoints] == [1.0, 1.0, 1.0]


def test_pull_classification_tags_from_cvat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, Task.CLASSIFICATION, labels={0: [Label("cat")], 1: [Label("dog")]})
    first_push(env)
    tags = env.server.annotations(env.tid)["tags"]
    t0 = next(t for t in tags if t["frame"] == 0)
    env.server.anns[env.tid]["tags"] = [t for t in tags if t["frame"] != 1]  # tag of img1 deleted
    env.server.anns[env.tid]["tags"][0] = {**t0, "label_id": env.label_id("dog")}  # img0 cat -> dog
    env.server.add_tag(env.tid, 2, env.label_id("cat"))
    p, _ = sync(env)
    assert p["pull"] == 3
    assert (
        [lb.label for lb in local(env, 0)] == ["dog"]
        and local(env, 1) == []
        and [lb.label for lb in local(env, 2)] == ["cat"]
    )


# -- push: local edits -> CVAT, untouched shapes stay ---------------------------------------------------
def decorate(env: Env) -> dict[int, dict[str, Any]]:
    """Give the shapes of img0 everything a person adds in CVAT; return them by label id.

    Args:
        env: the environment (synchronised).

    Returns:
        ``{label id: stored shape}``.
    """
    srv = env.server
    spec = srv.add_attribute(env.label_id("dog"), "colour")
    spec_cat = srv.add_attribute(env.label_id("cat"), "colour")
    a, b = shapes(env, 0)
    srv.edit_shape(
        env.tid,
        a["id"],
        occluded=True,
        z_order=3,
        group=1,
        attributes=[{"spec_id": spec_cat, "value": "grey"}],
    )
    srv.edit_shape(
        env.tid,
        b["id"],
        rotation=30,
        z_order=5,
        source="auto",
        attributes=[{"spec_id": spec, "value": "brown"}],
    )
    return {s["label_id"]: s for s in shapes(env, 0)}


def test_push_keeps_untouched_shapes_exactly(detection: Env) -> None:
    env = detection
    cat, dog = env.label_id("cat"), env.label_id("dog")
    before = decorate(env)
    # the same image gets a third box locally; the other two boxes are not edited
    sync(env)  # the decorations change nothing the project sees (the rotation gives the enclosing box)
    set_local(env, 0, [*local(env, 0), rel_box("cat", 150, 60, 190, 90)])
    p, _ = sync(env)
    assert p["push"] == 1
    after = {(s["label_id"], tuple(round(v, 6) for v in s["points"])): s for s in shapes(env, 0)}
    assert same_without_ids(after[(cat, (20, 20, 80, 60))], before[cat])
    assert same_without_ids(after[(dog, (100, 10, 140, 40))], before[dog])
    new = after[(cat, (150, 60, 190, 90))]
    assert (
        new["rotation"] == 0 and new["occluded"] is False and new["attributes"] == [] and new["z_order"] == 0
    )


def test_defect_push_renumbers_the_ids_of_untouched_shapes(detection: Env) -> None:
    """A push is a PUT of the whole task: every shape gets a new id, also the ones nobody touched."""
    env = detection
    ids_before = {s["id"] for s in shapes(env)}
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    sync(env)
    assert ids_before <= {s["id"] for s in shapes(env)}


def test_defect_edited_box_loses_its_attributes_in_cvat(detection: Env) -> None:
    """Moving a box locally replaces the CVAT shape by a fresh one: attributes, occluded, z_order are lost."""
    env = detection
    before = decorate(env)
    cat = env.label_id("cat")
    sync(env)  # the dog turned in CVAT comes as its enclosing box (#359)
    set_local(env, 0, [rel_box("cat", 25, 22, 85, 62), local(env, 0)[1]])  # the cat moved a little
    p, _ = sync(env)
    assert p["push"] == 1
    moved = next(s for s in shapes(env, 0) if s["label_id"] == cat)
    assert moved["points"] == pytest.approx([25, 22, 85, 62])
    assert (
        moved["attributes"] == before[cat]["attributes"]
        and moved["occluded"] is True
        and moved["z_order"] == 3
    )


def test_push_local_edits_detection(detection: Env) -> None:
    env = detection
    set_local(
        env, 0, [rel_box("cat", 30, 25, 90, 70), rel_box("cat", 100, 10, 140, 40)]
    )  # moved, class changed
    set_local(env, 1, [])
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    p, result = sync(env)
    assert (p["push"], p["pull"]) == (3, 0) and result["pushed"] == 3
    cat, dog = env.label_id("cat"), env.label_id("dog")
    assert rows(env) == [
        (0, cat, "rectangle", (30, 25, 90, 70)),
        (0, cat, "rectangle", (100, 10, 140, 40)),
        (2, dog, "rectangle", (40, 10, 100, 50)),
    ]
    assert plan(env)["push"] == 0


def test_push_local_edits_segmentation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tri = [(20, 10), (100, 10), (60, 80)]
    env = make_env(tmp_path, monkeypatch, Task.SEGMENTATION, labels={0: [rel_poly("cat", tri)]})
    first_push(env)
    set_local(env, 0, [rel_poly("cat", [(20, 10), (100, 10), (70, 90)])])
    set_local(env, 1, [rel_poly("dog", [(10, 10), (50, 10), (50, 50), (10, 50)])])
    p, _ = sync(env)
    assert p["push"] == 2
    cat, dog = env.label_id("cat"), env.label_id("dog")
    assert rows(env) == [
        (0, cat, "polygon", (20, 10, 100, 10, 70, 90)),
        (1, dog, "polygon", (10, 10, 50, 10, 50, 50, 10, 50)),
    ]
    assert plan(env)["pull"] == 0


def test_push_local_edits_keypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    label = Label(
        "cat", bbox=px(20, 20, 100, 50), keypoints=[[0.1, 0.2, 2.0], [0.5, 0.25, 2.0], [0.25, 0.5, 2.0]]
    )
    env = make_env(tmp_path, monkeypatch, Task.KEYPOINTS, labels={0: [label]})
    first_push(env)
    moved = Label(
        "cat", bbox=px(20, 20, 50, 50), keypoints=[[0.1, 0.2, 2.0], [0.0, 0.0, 0.0], [0.25, 0.5, 2.0]]
    )
    set_local(env, 0, [moved])
    p, _ = sync(env)
    assert p["push"] == 1
    (shape,) = shapes(env)
    assert shape["points"] == pytest.approx([20, 20, 0, 0, 50, 50])
    assert plan(env)["pull"] == 0


def test_defect_keypoint_visibility_one_is_lost_in_the_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A point 'labeled but not visible' (visibility 1) is sent as a plain point and comes back as visible."""
    pts = [[0.1, 0.2, 2.0], [0.5, 0.25, 1.0], [0.25, 0.5, 2.0]]
    label = Label("cat", bbox=px(20, 20, 100, 50), keypoints=pts)
    env = make_env(tmp_path, monkeypatch, Task.KEYPOINTS, labels={0: [label]})
    first_push(env)
    sync(env)
    assert [k[2] for k in local(env, 0)[0].keypoints] == [2.0, 1.0, 2.0]


def test_defect_keypoint_bbox_of_the_project_is_replaced_by_the_points_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pose object has its own box (COCO); CVAT has none: the next sync replaces it by the points' box."""
    label = Label(
        "cat", bbox=px(10, 10, 190, 90), keypoints=[[0.1, 0.2, 2.0], [0.5, 0.25, 2.0], [0.25, 0.5, 2.0]]
    )
    env = make_env(tmp_path, monkeypatch, Task.KEYPOINTS, labels={0: [label]})
    first_push(env)
    p, _ = sync(env)
    assert (p["pull"], p["push"]) == (0, 0)
    assert flat(local(env, 0)[0].bbox) == pytest.approx(px(10, 10, 190, 90), abs=TOL)


def test_push_local_edits_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, Task.CLASSIFICATION, labels={0: [Label("cat")], 1: [Label("dog")]})
    first_push(env)
    set_local(env, 0, [Label("dog")])
    set_local(env, 1, [])
    set_local(env, 2, [Label("cat")])
    p, _ = sync(env)
    assert p["push"] == 3
    tags = sorted((t["frame"], t["label_id"]) for t in env.server.annotations(env.tid)["tags"])
    assert tags == [(0, env.label_id("dog")), (2, env.label_id("cat"))]
    assert plan(env)["push"] == 0


def test_second_sync_is_a_no_op_for_every_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tri = [(20, 10), (100, 10), (60, 80)]
    kps = [[0.1, 0.2, 2.0], [0.5, 0.25, 2.0], [0.25, 0.5, 2.0]]
    cases = {
        Task.DETECTION: [rel_box("cat", 20, 20, 80, 60)],
        Task.SEGMENTATION: [rel_poly("cat", tri)],
        Task.KEYPOINTS: [Label("cat", bbox=px(20, 20, 100, 50), keypoints=kps)],
        Task.CLASSIFICATION: [Label("cat")],
    }
    for task, labels in cases.items():
        env = make_env(tmp_path / task.value, monkeypatch, task, labels={0: labels})
        first_push(env)
        before = [r.body for r in env.server.calls("PUT", "/api/tasks")]
        p, _ = sync(env)
        assert (p["pull"], p["push"], p["add"], p["conflicts"]) == (0, 0, 0, []), task
        assert [r.body for r in env.server.calls("PUT", "/api/tasks")] == before, task  # no second PUT


# -- both sides changed ---------------------------------------------------------------------------------
def test_conflict_both_sides_changed_the_same_image(detection: Env) -> None:
    env, srv = detection, detection.server
    (c,) = shapes(env, 1)
    srv.edit_shape(env.tid, c["id"], points=[10, 10, 60, 60])
    set_local(env, 1, [rel_box("cat", 100, 20, 190, 90)])
    (d,) = [s for s in shapes(env, 0) if s["label_id"] == env.label_id("dog")]
    srv.edit_shape(env.tid, d["id"], points=[110, 10, 150, 40])  # img0: changed only in CVAT
    p = plan(env)
    assert [c["name"] for c in p["conflicts"]] == ["img1.png"] and p["pull"] == 1 and p["push"] == 0
    left = apply(env, p["plan_id"])  # no choice: the conflict is left as it is
    assert left["conflicts_left"] == 1 and left["pulled"] == 1
    assert flat(local(env, 1)[0].bbox) == pytest.approx(px(100, 20, 190, 90), abs=TOL)
    assert shapes(env, 1)[0]["points"] == [10, 10, 60, 60]
    # the conflict is still there, and resolving it "local" writes this project's side to CVAT
    p2 = plan(env)
    assert len(p2["conflicts"]) == 1
    apply(env, p2["plan_id"], default="local")
    assert shapes(env, 1)[0]["points"] == pytest.approx([100, 20, 190, 90])
    assert plan(env)["conflicts"] == []


def test_conflict_resolved_with_remote_overwrites_the_project(detection: Env) -> None:
    env = detection
    (c,) = shapes(env, 1)
    env.server.edit_shape(env.tid, c["id"], points=[10, 10, 60, 60])
    set_local(env, 1, [rel_box("cat", 100, 20, 190, 90)])
    p = plan(env)
    apply(env, p["plan_id"], resolutions={env.ids[1]: "remote"})
    assert flat(local(env, 1)[0].bbox) == pytest.approx(px(10, 10, 60, 60), abs=TOL)
    assert shapes(env, 1)[0]["points"] == [10, 10, 60, 60]


def test_conflict_edited_here_deleted_frame_there(detection: Env) -> None:
    env = detection
    env.server.delete_frames(env.tid, [1])
    set_local(env, 1, [rel_box("cat", 100, 20, 190, 90)])
    p = plan(env)
    assert p["conflicts"][0]["name"] == "img1.png" and p["remove_local"] == 0
    apply(env, p["plan_id"], default="local")  # keep my edit: the frame is restored in CVAT
    assert env.server.tasks[env.tid].deleted == []
    assert shapes(env, 1)[0]["points"] == pytest.approx([100, 20, 190, 90])


def test_direction_pull_and_push_never_conflict(detection: Env) -> None:
    env = detection
    (c,) = shapes(env, 1)
    env.server.edit_shape(env.tid, c["id"], points=[10, 10, 60, 60])
    set_local(env, 1, [rel_box("cat", 100, 20, 190, 90)])
    only_push = plan(env, direction="push")
    assert only_push["conflicts"] == [] and only_push["push"] == 1
    only_pull = plan(env, direction="pull")
    assert only_pull["conflicts"] == [] and only_pull["pull"] == 1
    apply(env, only_pull["plan_id"])
    assert flat(local(env, 1)[0].bbox) == pytest.approx(px(10, 10, 60, 60), abs=TOL)


def test_apply_refuses_when_cvat_changed_after_the_plan(detection: Env) -> None:
    env = detection
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    p = plan(env)
    env.server.add_shape(env.tid, rect(2, env.label_id("cat"), [1, 1, 30, 30]))  # drawn after the comparison
    puts = len(env.server.calls("PUT", "/api/tasks"))
    with pytest.raises(engine.RemoteChanged):
        apply(env, p["plan_id"])
    assert len(env.server.calls("PUT", "/api/tasks")) == puts


def test_defect_a_shape_drawn_in_cvat_during_a_push_is_overwritten(detection: Env) -> None:
    """The push reads the task, then PUTs all of it: whatever a person draws in between is deleted."""
    env, srv = detection, detection.server
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    p = plan(env)
    srv.after[("GET", f"/api/tasks/{env.tid}/annotations")] = DrawOnce(env, [1, 1, 30, 30])
    apply(env, p["plan_id"])
    assert any(s["points"] == [1, 1, 30, 30] for s in shapes(env, 1))


# -- labels ---------------------------------------------------------------------------------------------
def test_label_renamed_in_cvat_comes_back_with_the_new_name(detection: Env) -> None:
    env = detection
    env.server.rename_label(env.label_id("cat"), "kitten")
    p, _ = sync(env)
    assert p["pull"] == 2  # img0 and img1 show the new name
    assert [lb.label for lb in local(env, 1)] == ["kitten"]
    assert "kitten" in (load_annotations(env.dir).classes)


@pytest.mark.xfail(strict=True, reason="#364")
def test_defect_label_renamed_in_cvat_is_created_again_by_the_next_push(detection: Env) -> None:
    """The old class stays in the project, so a later push adds a 'cat' label again to the CVAT project."""
    env = detection
    env.server.rename_label(env.label_id("cat"), "kitten")
    sync(env)
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    sync(env)
    assert env.server.label_names(env.pid) == ["kitten", "dog"]


@pytest.mark.xfail(strict=True, reason="#364")
def test_defect_labels_that_differ_only_in_case_are_merged_and_the_cvat_shape_changes_class(
    detection: Env,
) -> None:
    """CVAT has 'cat' and 'Cat'; the project folds them into 'cat' and a push relabels the 'Cat' shape."""
    env, srv = detection, detection.server
    big = srv._add_label(env.pid, {"name": "Cat"})["id"]  # noqa: SLF001 - what a person does in CVAT
    srv.add_shape(env.tid, rect(0, big, [150, 60, 190, 90]))
    sync(env)  # the 'Cat' shape arrives as class 'cat'
    set_local(env, 0, [*local(env, 0), rel_box("dog", 5, 5, 20, 20)])  # unrelated edit on the same frame
    sync(env)
    assert [s["label_id"] for s in shapes(env, 0) if s["points"] == [150, 60, 190, 90]] == [big]


# -- shape types the project does not convert -----------------------------------------------------------
def add_unsupported(env: Env) -> None:
    """Draw an ellipse, a polyline and a mask on img0 and an oblong cuboid-free set of extras.

    Args:
        env: the environment (synchronised, detection).
    """
    any_label = env.label_id("cat")
    env.server.add_shape(
        env.tid, {"type": "ellipse", "frame": 0, "label_id": any_label, "points": [100, 60, 140, 80]}
    )
    env.server.add_shape(
        env.tid, {"type": "polyline", "frame": 0, "label_id": any_label, "points": [10, 10, 40, 40, 60, 10]}
    )
    env.server.add_shape(
        env.tid,
        {"type": "mask", "frame": 0, "label_id": any_label, "points": [0, 0, 5, 3, 7, 10, 10, 20, 12]},
    )


def test_unsupported_shapes_are_kept_when_the_frame_is_pushed(detection: Env) -> None:
    env = detection
    add_unsupported(env)
    sync(env)
    set_local(env, 0, [*local(env, 0), rel_box("dog", 5, 5, 20, 20)])
    sync(env)
    kinds = sorted(s["type"] for s in shapes(env, 0))
    assert kinds == ["ellipse", "mask", "polyline", "rectangle", "rectangle", "rectangle"]


@pytest.mark.xfail(strict=True, reason="#365")
def test_defect_unsupported_shapes_are_silently_invisible(detection: Env) -> None:
    """Ellipses, polylines, masks, cuboids and skeletons never reach the project and nothing says so."""
    env = detection
    add_unsupported(env)
    p, result = sync(env)
    text = repr((p, result)).lower()
    assert any(
        word in text
        for word in ("ellipse", "mask", "polyline", "not converted", "unsupported", "skipped shapes")
    )


@pytest.mark.xfail(strict=True, reason="#365")
def test_defect_cvat_skeletons_are_not_read_as_keypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native way to annotate a pose in CVAT is a skeleton label; its shapes never arrive."""
    env = make_env(
        tmp_path,
        monkeypatch,
        Task.KEYPOINTS,
        labels={0: [Label("cat", bbox=px(20, 20, 100, 50), keypoints=[[0.1, 0.2, 2.0]])]},
    )
    first_push(env)
    srv = env.server
    sk = srv._add_label(  # noqa: SLF001 - what a person does in CVAT
        env.pid,
        {
            "name": "body",
            "type": "skeleton",
            "sublabels": [{"name": "head", "type": "points"}, {"name": "tail", "type": "points"}],
        },
    )
    head, tail = (s["id"] for s in sk["sublabels"])
    srv.add_shape(
        env.tid,
        {
            "type": "skeleton",
            "frame": 1,
            "label_id": sk["id"],
            "points": [],
            "elements": [
                {"type": "points", "label_id": head, "points": [40, 30]},
                {"type": "points", "label_id": tail, "points": [120, 60], "occluded": True},
            ],
        },
    )
    sync(env)
    (found,) = local(env, 1)
    assert flat(found.keypoints) == pytest.approx([0.2, 0.3, 2, 0.6, 0.6, 1], abs=TOL)


# -- tasks, jobs, frames, pagination --------------------------------------------------------------------
def test_remote_tasks_and_jobs_use_task_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, count=7, job_size=3)
    first_push(env)
    gt = env.server.add_job(env.tid, 0, 1, kind="ground_truth")  # a validation job is not a work job
    (task,) = env.remote.tasks(SETTINGS, env.rid)
    assert (task.size, task.job_size) == (7, 3)
    assert [(j.n, j.start, j.stop) for j in task.jobs] == [(1, 0, 2), (2, 3, 5), (3, 6, 6)]
    assert str(gt) not in [j.id for j in task.jobs]
    assert [(j.state, j.stage) for j in task.jobs] == [("new", "annotation")] * 3


def test_job_scope_pulls_only_the_images_of_that_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cats = {i: [rel_box("cat", 20, 20, 80, 60)] for i in range(7)}
    env = make_env(tmp_path, monkeypatch, count=7, job_size=3, labels=cats)
    first_push(env)
    srv = env.server
    for frame in (1, 4, 6):
        shape = next(s for s in shapes(env, frame))
        srv.edit_shape(env.tid, shape["id"], points=[30, 30, 90, 70])
    p = plan(env, scope="job", task_id=1, job=2)
    assert p["pull"] == 1 and p["scope"]["job"] == 2
    apply(env, p["plan_id"])
    moved = [i for i in range(7) if flat(local(env, i)[0].bbox) == pytest.approx(px(30, 30, 90, 70), abs=TOL)]
    assert moved == [4]  # image 4 = task frame 4 = the middle of job 2; frames 1 and 6 are outside the scope
    whole = plan(env)
    assert whole["pull"] == 2


def test_push_of_one_job_writes_only_its_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cats = {i: [rel_box("cat", 20, 20, 80, 60)] for i in range(7)}
    env = make_env(tmp_path, monkeypatch, count=7, job_size=3, labels=cats)
    first_push(env)
    for i in (1, 4):
        set_local(env, i, [rel_box("dog", 30, 30, 90, 70)])
    p, _ = sync(env, scope="job", task_id=1, job=1)
    assert p["push"] == 1
    dog = env.label_id("dog")
    assert [s["frame"] for s in shapes(env) if s["label_id"] == dog] == [1]  # image 4 (job 2) is not sent


def test_frames_of_a_video_task_follow_start_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(tmp_path, monkeypatch, count=1)
    srv = env.server
    pid = srv.add_project("P", ["cat"])["id"]
    tid = srv.add_task(
        pid, "clip", [("clip.mp4", 200, 100)], segment_size=3, video={"size": 6, "start": 10, "step": 5}
    )
    srv.delete_frames(tid, [2])
    names = env.remote.frames(SETTINGS, str(tid))
    assert [n for n, _ in names] == [f"clip_frame_{10 + 5 * i:06d}.jpg" for i in range(6)]
    assert [d for _, d in names] == [False, False, True, False, False, False]
    (task,) = env.remote.tasks(SETTINGS, str(pid))
    assert (task.size, task.job_size, [(j.start, j.stop) for j in task.jobs]) == (6, 3, [(0, 2), (3, 5)])
    meta = srv.tasks[tid]
    assert (meta.start_frame, meta.stop_frame, meta.frame_filter) == (10, 35, "step=5")


def test_pagination_of_tasks_labels_and_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(tmp_path, monkeypatch, count=1)
    srv = env.server
    for n in range(120):  # more than one page of projects
        srv.add_project(f"Filler {n}", [])
    pid = srv.add_project("ActiveLabel 9 - Many", [f"c{n}" for n in range(230)])["id"]
    for n in range(205):
        srv.add_task(pid, f"t{n}", [("a.png", 20, 10)])
    found = env.remote.find(SETTINGS, "ActiveLabel 9 - Many")
    assert found is not None and found.images == 205
    assert len(env.remote.tasks(SETTINGS, str(pid))) == 205
    before = len(srv.calls("PATCH", "/api/projects"))
    env.remote.ensure_classes(SETTINGS, str(pid), ["c0", "c229"])  # both exist, on the 1st and the 3rd page
    assert len(srv.calls("PATCH", "/api/projects")) == before
    assert len(srv.label_names(pid)) == 230


def test_set_frames_and_restore_go_through_the_meta(detection: Env) -> None:
    env = detection
    env.server.delete_frames(env.tid, [0])  # someone else deleted frame 0
    set_local(env, 2, [])
    Manifest(env.dir).set_status([env.ids[2]], EXCLUDED, "test")
    p, _ = sync(env)
    assert p["remove_remote"] == 1
    assert env.server.tasks[env.tid].deleted == [0, 2]  # merged with what was there
    assert shapes(env, 1)  # annotations are kept by CVAT


# -- tracks (video) -------------------------------------------------------------------------------------
@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    """A video task (start 10, step 5, 6 frames, 2 jobs) with one tracked cat, linked by name."""
    names = [f"clip_frame_{10 + 5 * i:06d}.jpg" for i in range(6)]
    env = make_env(tmp_path, monkeypatch, count=6, names=names, job_size=3)
    srv = env.server
    info = srv.add_project(engine.remote_name(env.project), ["cat", "dog"])
    env.rid = str(info["id"])
    env.tid = srv.add_task(
        info["id"], "clip", [("clip.mp4", W, H)], segment_size=3, video={"size": 6, "start": 10, "step": 5}
    )
    cat = env.label_id("cat")
    track = {
        "frame": 0,
        "label_id": cat,
        "group": 0,
        "source": "manual",
        "shapes": [
            {"type": "rectangle", "frame": 0, "points": [20, 20, 60, 60]},
            {"type": "rectangle", "frame": 3, "points": [80, 20, 120, 60]},
            {"type": "rectangle", "frame": 4, "points": [80, 20, 120, 60], "outside": True},
        ],
    }
    srv.put(env.tid, {"tracks": [track]})
    return env


def track_boxes(env: Env) -> dict[int, list[float]]:
    """Pixel boxes of the tracked labels per image index (from the project ground truth).

    Args:
        env: the video environment.

    Returns:
        ``{image index: relative bbox}`` for the images that have a label.
    """
    out: dict[int, list[float]] = {}
    for i in range(len(env.ids)):
        labels = local(env, i)
        if labels:
            out[i] = flat(labels[0].bbox)
    return out


def test_video_pull_track_interpolates_between_keyframes(video: Env) -> None:
    env = video
    p, _ = sync(env)
    assert p["pull"] == 6 - 2 and p["push"] == 0  # visible on frames 0..3
    got = track_boxes(env)
    assert sorted(got) == [0, 1, 2, 3]
    expected = {0: (20, 20, 60, 60), 1: (40, 20, 80, 60), 2: (60, 20, 100, 60), 3: (80, 20, 120, 60)}
    for i, box in expected.items():
        assert got[i] == pytest.approx(px(*box), abs=TOL), i
    track_id = env.server.annotations(env.tid)["tracks"][0]["id"]
    assert {local(env, i)[0].track for i in range(4)} == {track_id}
    assert (
        plan(env)["pull"] == 0
        and (links.get_link(env.dir, "cvat") or {})["items"][env.ids[3]]["key"] == f"{env.tid}:3"
    )


def test_video_keyframe_edit_in_cvat_moves_the_neighbours(video: Env) -> None:
    env, srv = video, video.server
    sync(env)
    track = srv.annotations(env.tid)["tracks"][0]
    srv.edit_track(env.tid, track["id"], 3, points=[100, 40, 140, 80])  # the second keyframe moved
    srv.edit_track(
        env.tid, track["id"], 2, points=[50, 20, 90, 60], occluded=True
    )  # a keyframe added between
    p, _ = sync(env)
    assert p["pull"] == 3 and p["push"] == 0
    got = track_boxes(env)
    expected = {0: (20, 20, 60, 60), 1: (35, 20, 75, 60), 2: (50, 20, 90, 60), 3: (100, 40, 140, 80)}
    assert sorted(got) == [0, 1, 2, 3]
    for i, box in expected.items():
        assert got[i] == pytest.approx(px(*box), abs=TOL), i


def test_video_outside_keyframe_hides_the_object(video: Env) -> None:
    env, srv = video, video.server
    track = srv.annotations(env.tid)["tracks"][0]
    srv.edit_track(env.tid, track["id"], 2, outside=True)  # gone on frame 2, back on frame 3
    sync(env)
    got = track_boxes(env)
    assert sorted(got) == [0, 1, 3]
    assert (
        got[0] == pytest.approx(px(20, 20, 60, 60), abs=TOL) and got[1] == got[0]
    )  # held until the outside frame
    assert got[3] == pytest.approx(px(80, 20, 120, 60), abs=TOL)


def test_video_local_edit_becomes_a_keyframe_and_neighbours_follow(video: Env) -> None:
    env, srv = video, video.server
    sync(env)
    (cat,) = [lb for lb in local(env, 1)]
    set_local(env, 1, [Label("cat", bbox=px(50, 25, 90, 65), track=cat.track)])
    p, _ = sync(env)
    assert p["push"] == 1
    (track,) = srv.annotations(env.tid)["tracks"]
    frames = {s["frame"]: s for s in track["shapes"]}
    assert sorted(frames) == [0, 1, 3, 4]
    assert frames[1]["points"] == pytest.approx([50, 25, 90, 65]) and frames[4]["outside"] is True
    # CVAT interpolates frame 2 again; the project takes that over, so nothing is pending
    assert track_boxes(env)[2] == pytest.approx(px(65, 22.5, 105, 62.5), abs=TOL)
    p2 = plan(env)
    assert (p2["pull"], p2["push"], p2["conflicts"]) == (0, 0, [])
    assert {local(env, i)[0].track for i in range(4)} == {track["id"]}


def test_video_deleting_a_tracked_label_adds_an_outside_keyframe(video: Env) -> None:
    env, srv = video, video.server
    sync(env)
    set_local(env, 2, [])
    p, _ = sync(env)
    assert p["push"] == 1
    got = track_boxes(env)
    assert sorted(got) == [0, 1, 3]  # hidden on frame 2 only, as before everywhere else
    assert got[3] == pytest.approx(px(80, 20, 120, 60), abs=TOL) and got[1] == pytest.approx(
        px(40, 20, 80, 60), abs=TOL
    )
    assert len(srv.annotations(env.tid)["tracks"]) == 1


def test_defect_rotated_keyframes_of_a_track_are_read_unrotated(video: Env) -> None:
    env, srv = video, video.server
    track = srv.annotations(env.tid)["tracks"][0]
    srv.edit_track(env.tid, track["id"], 0, rotation=45)
    sync(env)
    x0, y0, x1, y1 = rotated_box([20, 20, 60, 60], 45)
    assert track_boxes(env)[0] == pytest.approx(px(x0, y0, x1, y1), abs=1e-4)
