"""Smoke tests of the CVAT synchronisation against a REAL CVAT server (nightly workflow ``cvat-live.yml``).

The other sync tests run the adapter against the strict in-memory fake of ``cvat_fake``. Only a real CVAT
proves what the fake can only imitate: the serializer (unknown fields, ids in a PUT, ``version``), the ids it
assigns, real interpolation and rotation. These tests mirror the assertions of ``test_cvat_remote_roundtrip``.

They are skipped unless the three variables below are set, so the normal test run never needs a server:

* ``ACTIVELABEL_CVAT_LIVE_URL``: the throw-away CVAT of the workflow (``http://localhost:8080``);
* ``ACTIVELABEL_CVAT_LIVE_USER`` and ``ACTIVELABEL_CVAT_LIVE_PASSWORD``: its superuser.

Never point them at a server somebody uses: every test creates a CVAT project with a unique name and deletes
it at the end, but a test that fails half way is not a test for a shared server.
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests
from sample_images import make_image
from sync_fakes import progress

from activelabel.annotate.base import ToolSettings
from activelabel.images.ingest import ingest_folder
from activelabel.images.manifest import Manifest
from activelabel.labels import AnnotationSet, Label, load_annotations, save_annotations
from activelabel.projects import Project, ProjectStore
from activelabel.storage import write_json
from activelabel.sync import engine, links
from activelabel.sync.cvat_remote import CvatRemote
from activelabel.tasks import Task
from activelabel.tasks_store import ImageTask

URL = os.environ.get("ACTIVELABEL_CVAT_LIVE_URL", "").rstrip("/")
USER = os.environ.get("ACTIVELABEL_CVAT_LIVE_USER", "")
PASSWORD = os.environ.get("ACTIVELABEL_CVAT_LIVE_PASSWORD", "")

pytestmark = [
    pytest.mark.cvat_live,
    pytest.mark.skipif(
        not (URL and USER and PASSWORD),
        reason="needs ACTIVELABEL_CVAT_LIVE_URL, _USER and _PASSWORD (a throw-away CVAT)",
    ),
]

W, H = 200, 100
TOL = 1e-6
TIMEOUT = 120


@dataclass
class LiveEnv:
    """A local project, a real CVAT and the real adapter between them."""

    store: ProjectStore
    project: Project
    ids: list[str]
    dir: Path
    settings: ToolSettings
    remote: CvatRemote
    rid: str = ""
    tid: int = 0


# -- CVAT REST (only what the tests need to act like a person in the CVAT UI, and to clean up) ----------
def api(method: str, path: str, **kw: Any) -> requests.Response:
    """One call to the CVAT REST API with the superuser of the workflow.

    Args:
        method: HTTP method.
        path: path below the server, for example ``/api/projects``.
        **kw: more arguments of ``requests.request`` (``json``, ``params``).

    Returns:
        The response (the status is not checked).
    """
    return requests.request(method, URL + path, auth=(USER, PASSWORD), timeout=TIMEOUT, **kw)


def cvat_annotations(env: LiveEnv) -> dict[str, Any]:
    """Annotations CVAT holds for the task.

    Args:
        env: the environment (already sent to CVAT).

    Returns:
        ``{"version", "tags", "shapes", "tracks"}``.
    """
    answer = api("GET", f"/api/tasks/{env.tid}/annotations")
    answer.raise_for_status()
    return dict(answer.json())


def cvat_shapes(env: LiveEnv, frame: int | None = None) -> list[dict[str, Any]]:
    """Shapes of the task (of one frame), in id order.

    Args:
        env: the environment.
        frame: only this task frame.

    Returns:
        The stored shapes.
    """
    found = cvat_annotations(env)["shapes"]
    return sorted((s for s in found if frame is None or s["frame"] == frame), key=lambda s: s["id"])


def cvat_edit_shape(env: LiveEnv, shape: dict[str, Any], **changes: Any) -> None:
    """Change one shape like a person does in CVAT (``PATCH ...?action=update``).

    Args:
        env: the environment.
        shape: the stored shape (with its id), as read from CVAT.
        **changes: the fields to change (``points``, ``rotation`` ...).
    """
    body = {"shapes": [{**shape, **changes}]}
    answer = api("PATCH", f"/api/tasks/{env.tid}/annotations", params={"action": "update"}, json=body)
    assert answer.status_code == 200, answer.text


def label_id(env: LiveEnv, name: str) -> int:
    """Id of a label of the CVAT project.

    Args:
        env: the environment.
        name: the label name.

    Returns:
        The label id.
    """
    answer = api("GET", "/api/labels", params={"project_id": env.rid, "page_size": 100})
    answer.raise_for_status()
    return int(next(x["id"] for x in answer.json()["results"] if x["name"] == name))


def delete_projects(token: str) -> None:
    """Delete every CVAT project whose name holds ``token`` (the unique part of a test project name).

    Args:
        token: the unique text of one test.
    """
    answer = api("GET", "/api/projects", params={"search": token, "page_size": 100})
    if answer.status_code != 200:
        return
    for found in answer.json()["results"]:
        if token in found["name"]:
            api("DELETE", f"/api/projects/{found['id']}")


@pytest.fixture
def tokens() -> Iterator[list[str]]:
    """The unique names of the projects a test created; their CVAT projects are deleted afterwards."""
    made: list[str] = []
    yield made
    for token in made:
        delete_projects(token)


# -- the local project ----------------------------------------------------------------------------------
def make_live(
    tmp_path: Path,
    tokens: list[str],
    task: Task = Task.DETECTION,
    labels: dict[int, list[Label]] | None = None,
) -> LiveEnv:
    """A project with three 200 x 100 images in one local task and its ground truth (not sent yet).

    Args:
        tmp_path: pytest temp dir.
        tokens: the fixture list; the unique name of this project is added to it.
        task: project type.
        labels: ground truth ``{image index: [Label]}``.

    Returns:
        The environment.
    """
    token = uuid.uuid4().hex[:12]
    tokens.append(token)
    store = ProjectStore(tmp_path / "projects")
    project = store.create(f"Live {token}", "", task.value, ["cat", "dog"])
    folder = tmp_path / "src"
    names = [f"img{i}.png" for i in range(3)]
    for i, name in enumerate(names):
        make_image(folder / name, size=(W, H), seed=i + 1, fmt="PNG")
    manifest = Manifest(store.dir(project.id))
    ingest_folder(manifest, folder, folder)
    by_name = {r.name: r.id for r in manifest.all()}
    ids = [by_name[n] for n in names]
    directory = store.dir(project.id)
    write_json(
        directory / "tasks.json",
        {"next_id": 2, "tasks": [ImageTask(1, "T", "2026-01-01T00:00:00", None, ids).as_dict()]},
    )
    settings = ToolSettings("cvat", URL, USER, PASSWORD)
    env = LiveEnv(store, project, ids, directory, settings, CvatRemote())
    for index, found in (labels or {}).items():
        set_local(env, index, found)
    return env


def plan(env: LiveEnv, **kw: Any) -> dict[str, Any]:
    """Compare the project with CVAT (``engine.make_plan``).

    Args:
        env: the environment.
        **kw: options of ``make_plan``.

    Returns:
        The plan summary.
    """
    return engine.make_plan(env.store, env.project.id, "cvat", env.settings, env.remote, progress(), **kw)


def sync(env: LiveEnv, **kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plan and apply in one go.

    Args:
        env: the environment.
        **kw: options of ``make_plan``.

    Returns:
        ``(plan, result)``.
    """
    p = plan(env, **kw)
    result = engine.apply_plan(
        env.store, env.project.id, "cvat", env.settings, env.remote, p["plan_id"], {}, None, progress()
    )
    return p, result


def first_push(env: LiveEnv) -> dict[str, Any]:
    """Send the whole project to a new CVAT project and remember its ids.

    Args:
        env: the environment.

    Returns:
        The apply result.
    """
    p = plan(env, create=True)
    result = engine.apply_plan(
        env.store, env.project.id, "cvat", env.settings, env.remote, p["plan_id"], {}, None, progress()
    )
    link = links.get_link(env.dir, "cvat") or {}
    env.rid = str(link["remote_id"])
    env.tid = int(link["tasks"]["1"]["remote_task"])
    return result


def start(
    tmp_path: Path,
    tokens: list[str],
    task: Task = Task.DETECTION,
    labels: dict[int, list[Label]] | None = None,
) -> LiveEnv:
    """``make_live`` followed by the first push.

    Args:
        tmp_path: pytest temp dir.
        tokens: the fixture list of unique project names.
        task: project type.
        labels: ground truth ``{image index: [Label]}``.

    Returns:
        The environment, already sent to CVAT.
    """
    env = make_live(tmp_path, tokens, task, labels)
    first_push(env)
    return env


def set_local(env: LiveEnv, index: int, labels: list[Label]) -> None:
    """Replace the ground truth of image ``index``.

    Args:
        env: the environment.
        index: image position.
        labels: its new labels.
    """
    found = load_annotations(env.dir) or AnnotationSet(env.project.task, list(env.project.classes), {}, {})
    found.images[env.ids[index]] = labels
    save_annotations(env.dir, found)


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


def local(env: LiveEnv, index: int) -> list[Label]:
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


def px(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Relative ``[x, y, w, h]`` of a pixel rectangle of a 200 x 100 image.

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


def rows(env: LiveEnv) -> list[tuple[int, int, str, tuple[float, ...]]]:
    """``(frame, label id, type, points)`` of the CVAT shapes, points rounded to 4 decimals, sorted.

    Args:
        env: the environment.

    Returns:
        One tuple per shape.
    """
    return sorted(
        (s["frame"], s["label_id"], s["type"], tuple(round(v, 4) for v in s["points"]))
        for s in cvat_shapes(env)
    )


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


def detection(tmp_path: Path, tokens: list[str]) -> LiveEnv:
    """The detection project of most tests, already sent: img0 a cat and a dog, img1 a cat, img2 nothing.

    Args:
        tmp_path: pytest temp dir.
        tokens: the fixture list of unique project names.

    Returns:
        The environment.
    """
    labels = {
        0: [rel_box("cat", 20, 20, 80, 60), rel_box("dog", 100, 10, 140, 40)],
        1: [rel_box("cat", 50, 25, 150, 75)],
    }
    return start(tmp_path, tokens, labels=labels)


# -- tests ----------------------------------------------------------------------------------------------
def test_first_push_detection_arrives_in_pixels(tmp_path: Path, tokens: list[str]) -> None:
    env = detection(tmp_path, tokens)
    cat, dog = label_id(env, "cat"), label_id(env, "dog")
    assert rows(env) == [
        (0, cat, "rectangle", (20, 20, 80, 60)),
        (0, dog, "rectangle", (100, 10, 140, 40)),
        (1, cat, "rectangle", (50, 25, 150, 75)),
    ]
    names = api("GET", f"/api/tasks/{env.tid}/data/meta").json()["frames"]
    assert [f["name"] for f in names] == ["img0.png", "img1.png", "img2.png"]


def test_second_sync_is_a_no_op(tmp_path: Path, tokens: list[str]) -> None:
    env = detection(tmp_path, tokens)
    before = {s["id"] for s in cvat_shapes(env)}
    p, _ = sync(env)
    assert (p["pull"], p["push"], p["conflicts"]) == (0, 0, [])
    assert {s["id"] for s in cvat_shapes(env)} == before


def test_edit_made_in_cvat_is_pulled(tmp_path: Path, tokens: list[str]) -> None:
    env = detection(tmp_path, tokens)
    (cat_shape,) = cvat_shapes(env, 1)
    cvat_edit_shape(env, cat_shape, points=[60, 30, 160, 80])
    p, _ = sync(env)
    assert (p["pull"], p["push"], p["conflicts"]) == (1, 0, [])
    assert flat(local(env, 1)[0].bbox) == pytest.approx(px(60, 30, 160, 80), abs=TOL)
    p2 = plan(env)
    assert (p2["pull"], p2["push"], p2["conflicts"]) == (0, 0, [])


def test_local_edit_is_pushed_and_untouched_shapes_keep_their_ids(tmp_path: Path, tokens: list[str]) -> None:
    env = detection(tmp_path, tokens)
    ids_before = {s["id"] for s in cvat_shapes(env)}
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    p, _ = sync(env)
    assert (p["push"], p["pull"]) == (1, 0)
    assert (2, label_id(env, "dog"), "rectangle", (40, 10, 100, 50)) in rows(env)
    assert ids_before <= {s["id"] for s in cvat_shapes(env)}
    assert plan(env)["push"] == 0


def test_rotated_rectangle_is_pulled_as_its_enclosing_box(tmp_path: Path, tokens: list[str]) -> None:
    env = detection(tmp_path, tokens)
    (cat_shape,) = cvat_shapes(env, 1)
    cvat_edit_shape(env, cat_shape, rotation=30)
    sync(env)
    (label,) = local(env, 1)
    x0, y0, x1, y1 = rotated_box([50, 25, 150, 75], 30)
    assert flat(label.bbox) == pytest.approx(px(x0, y0, x1, y1), abs=1e-4)


def test_segmentation_round_trip(tmp_path: Path, tokens: list[str]) -> None:
    tri = [(20, 10), (100, 10), (60, 80)]
    env = start(tmp_path, tokens, Task.SEGMENTATION, {0: [rel_poly("cat", tri)]})
    assert rows(env) == [(0, label_id(env, "cat"), "polygon", (20, 10, 100, 10, 60, 80))]
    (shape,) = cvat_shapes(env, 0)
    cvat_edit_shape(env, shape, points=[20, 10, 100, 10, 80, 90])
    p, _ = sync(env)
    assert p["pull"] == 1
    (moved,) = local(env, 0)
    assert flat(moved.polygons) == pytest.approx([0.1, 0.1, 0.5, 0.1, 0.4, 0.9], abs=TOL)
    assert plan(env)["pull"] == 0


def test_keypoints_round_trip(tmp_path: Path, tokens: list[str]) -> None:
    kps = [[0.1, 0.2, 2.0], [0.5, 0.25, 2.0], [0.25, 0.5, 2.0]]
    label = Label("cat", bbox=px(20, 20, 100, 50), keypoints=kps)
    env = start(tmp_path, tokens, Task.KEYPOINTS, {0: [label]})
    (shape,) = cvat_shapes(env)
    assert shape["type"] == "points" and shape["frame"] == 0
    assert shape["points"] == pytest.approx([20, 20, 100, 25, 50, 50])
    p, _ = sync(env)
    assert (p["pull"], p["push"]) == (0, 0)
    cvat_edit_shape(env, shape, points=[20, 20, 120, 30, 50, 50])
    sync(env)
    assert flat(local(env, 0)[0].keypoints) == pytest.approx(
        [0.1, 0.2, 2, 0.6, 0.3, 2, 0.25, 0.5, 2], abs=TOL
    )


def test_classification_tags_round_trip(tmp_path: Path, tokens: list[str]) -> None:
    env = start(tmp_path, tokens, Task.CLASSIFICATION, {0: [Label("cat")], 2: [Label("dog")]})
    ann = cvat_annotations(env)
    assert ann["shapes"] == [] and ann["tracks"] == []
    cat, dog = label_id(env, "cat"), label_id(env, "dog")
    assert sorted((t["frame"], t["label_id"]) for t in ann["tags"]) == [(0, cat), (2, dog)]
    set_local(env, 0, [Label("dog")])
    p, _ = sync(env)
    assert p["push"] == 1
    tags = sorted((t["frame"], t["label_id"]) for t in cvat_annotations(env)["tags"])
    assert tags == [(0, dog), (2, dog)]
    assert plan(env)["push"] == 0


def test_pose_visibility_and_box_survive_the_round_trip(tmp_path: Path, tokens: list[str]) -> None:
    """A pose mixing visibilities 1 and 2 adds the 'visibility' attribute to its CVAT label (#363)."""
    kps = [[0.1, 0.2, 2.0], [0.5, 0.25, 1.0], [0.25, 0.5, 2.0]]
    env = start(
        tmp_path, tokens, Task.KEYPOINTS, {0: [Label("cat", bbox=px(10, 10, 190, 90), keypoints=kps)]}
    )
    labels = api("GET", "/api/labels", params={"project_id": env.rid, "page_size": 100}).json()["results"]
    cat = next(x for x in labels if x["name"] == "cat")
    assert [a["name"] for a in cat["attributes"]] == ["visibility"]
    p, _ = sync(env)
    assert (p["pull"], p["push"]) == (0, 0)
    (pose,) = local(env, 0)
    assert [k[2] for k in pose.keypoints or []] == [2.0, 1.0, 2.0]
    assert flat(pose.bbox) == pytest.approx(px(10, 10, 190, 90), abs=TOL)


def test_label_renamed_in_cvat_is_renamed_here(tmp_path: Path, tokens: list[str]) -> None:
    """Renaming a label in CVAT keeps its id: the class is renamed here, not created again (#364)."""
    env = detection(tmp_path, tokens)
    cat = label_id(env, "cat")
    answer = api("PATCH", f"/api/projects/{env.rid}", json={"labels": [{"id": cat, "name": "kitten"}]})
    assert answer.status_code == 200, answer.text
    p, _ = sync(env)
    assert p["renamed_classes"] == [{"from": "cat", "to": "kitten"}]
    set_local(env, 2, [rel_box("dog", 40, 10, 100, 50)])
    sync(env)
    names = api("GET", "/api/labels", params={"project_id": env.rid, "page_size": 100}).json()["results"]
    assert sorted(x["name"] for x in names if not x.get("parent_id")) == ["dog", "kitten"]
