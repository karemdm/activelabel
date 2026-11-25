"""Do edits made to the ground truth reach the exact input of a training run?

Each test builds a project through the API (non-square images, so x/y mix-ups show), imports labels,
freezes version 1, edits the labels by importing a changed file (replace mode) and by removing a class
(PATCH), freezes version 2 and then follows the labels through ``build_spec`` and the worker functions
that turn the spec into pixels. No training is started and nothing outside the temporary workspace is
touched.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from api_client import make_client, upload
from PIL import Image, ImageDraw
from sample_images import make_image

from activelabel.datasets import DatasetStore
from activelabel.images.manifest import Manifest
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig
from activelabel.training.runner import build_spec

WORKERS = Path(__file__).parents[3] / "src" / "activelabel" / "training" / "workers"
CLASSES = ["cat", "dog", "bird"]
SIZES = [(64, 48), (80, 40)]  # width, height: never square, alternating per image
IMAGES = 10
SPLIT = {"train": 60, "val": 20, "test": 20}
KEYPOINT_NAMES = ["head", "hand", "foot"]

Ring = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Obj:
    """One expected object, in absolute pixels (the way a COCO file writes it).

    Attributes:
        cls: class name.
        bbox: ``(x, y, w, h)`` or None.
        rings: polygon rings.
        points: keypoints ``(x, y, visibility)``.
    """

    cls: str
    bbox: tuple[float, float, float, float] | None = None
    rings: tuple[Ring, ...] = ()
    points: tuple[tuple[float, float, int], ...] = ()


Truth = dict[int, list[Obj]]


@dataclass
class Scene:
    """A project with its images.

    Attributes:
        client: API test client.
        state: application state of the temporary workspace.
        pid: project id.
        ids: image id by image index.
    """

    client: Any
    state: Any
    pid: int
    ids: dict[int, str]

    @property
    def project_dir(self) -> Path:
        """Folder of the project."""
        return self.state.store.dir(self.pid)


def load_worker(name: str) -> Any:
    """Import a standalone worker script as a module.

    Args:
        name: file name without ``.py`` (``detection`` or ``segmentation``).

    Returns:
        The loaded module (the test is skipped where torch is missing: the workers import it at the top).
    """
    pytest.importorskip("torch", reason="the training workers need torch (the 'training workers' CI job)")
    spec = importlib.util.spec_from_file_location(f"{name}_worker_rt", WORKERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_worker_rt"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def det() -> Any:
    """The detection worker (also trains instance segmentation and keypoints)."""
    return load_worker("detection")


@pytest.fixture(scope="module")
def seg() -> Any:
    """The semantic segmentation worker."""
    return load_worker("segmentation")


def image_name(index: int) -> str:
    """File name of image ``index``."""
    return f"img{index}.png"


def size_of(index: int) -> tuple[int, int]:
    """``(width, height)`` of image ``index``."""
    return SIZES[index % 2]


def make_scene(tmp_path: Path, task: str, classes: list[str] | None = None) -> Scene:
    """Create a project of ``task`` and upload ``IMAGES`` images of alternating non-square sizes.

    Args:
        tmp_path: temporary folder (the workspace goes inside).
        task: project type.
        classes: class names of the project (default ``CLASSES``).

    Returns:
        The scene.
    """
    client, state = make_client(tmp_path)
    body = {"name": "P", "task": task, "classes": classes or CLASSES}
    assert client.post("/api/projects", json=body).status_code == 200
    paths = [
        make_image(tmp_path / "src" / image_name(k), size=size_of(k), seed=k, fmt="PNG")
        for k in range(IMAGES)
    ]
    assert upload(client, 1, paths).status_code == 200
    by_name = {r.name: r.id for r in state.manifest(1).all()}
    return Scene(client, state, 1, {k: by_name[image_name(k)] for k in range(IMAGES)})


def post_labels(scene: Scene, filename: str, content: bytes, mode: str = "replace") -> dict[str, Any]:
    """Upload one annotation file and apply it as the person would (check, then ``import``).

    Args:
        scene: the project.
        filename: name of the file (decides the format by content).
        content: file bytes.
        mode: ``replace`` or ``merge``.

    Returns:
        The summary of the apply call.
    """
    url = f"/api/projects/{scene.pid}/annotations"
    reply = scene.client.post(url, files=[("files", (filename, content))])
    assert reply.status_code == 200, reply.text
    report = reply.json()
    assert report["can_import"] and "import" in report["options"], report
    applied = scene.client.post(f"{url}/{report['token']}/apply", json={"action": "import", "mode": mode})
    assert applied.status_code == 200, applied.text
    return applied.json()


def coco_bytes(truth: Truth, categories: list[str], keypoint_names: list[str] | None = None) -> bytes:
    """A COCO file for the truth (absolute pixels, the real image sizes).

    Args:
        truth: objects per image index.
        categories: category names (ids are positions + 1).
        keypoint_names: names of the points of each category, for pose.

    Returns:
        JSON bytes.
    """
    images = [
        {"id": k + 1, "file_name": image_name(k), "width": size_of(k)[0], "height": size_of(k)[1]}
        for k in range(IMAGES)
    ]
    cats = [
        {"id": i + 1, "name": name, **({"keypoints": keypoint_names} if keypoint_names else {})}
        for i, name in enumerate(categories)
    ]
    annotations: list[dict[str, Any]] = []
    for k, objects in sorted(truth.items()):
        for obj in objects:
            ann: dict[str, Any] = {
                "id": len(annotations) + 1,
                "image_id": k + 1,
                "category_id": categories.index(obj.cls) + 1,
                "iscrowd": 0,
            }
            if obj.bbox:
                ann["bbox"] = list(obj.bbox)
            if obj.rings:
                ann["segmentation"] = [[v for point in ring for v in point] for ring in obj.rings]
            if obj.points:
                ann["keypoints"] = [v for point in obj.points for v in point]
                ann["num_keypoints"] = sum(1 for p in obj.points if p[2] > 0)
            annotations.append(ann)
    return json.dumps({"images": images, "categories": cats, "annotations": annotations}).encode()


def tags_bytes(tags: dict[int, str]) -> bytes:
    """A CVAT for images XML file with one class tag per image (image classification).

    Args:
        tags: class name by image index; images that are not in it are listed without a tag.

    Returns:
        XML bytes.
    """
    rows = ""
    for k in range(IMAGES):
        width, height = size_of(k)
        tag = f'<tag label="{tags[k]}" source="manual"/>' if k in tags else ""
        rows += f'<image id="{k}" name="{image_name(k)}" width="{width}" height="{height}">{tag}</image>'
    return f'<?xml version="1.0"?><annotations><version>1.1</version>{rows}</annotations>'.encode()


def new_version(scene: Scene) -> int:
    """Create a dataset version with a fixed seed and a train/val/test split.

    Args:
        scene: the project.

    Returns:
        The version number.
    """
    reply = scene.client.post(f"/api/projects/{scene.pid}/datasets", json={"split": SPLIT, "seed": 7})
    assert reply.status_code == 200, reply.text
    return reply.json()["version"]


def remove_class(scene: Scene, remaining: list[str]) -> None:
    """Change the classes of the project to ``remaining``, removing the annotations of the others.

    Args:
        scene: the project.
        remaining: the class list after the change.
    """
    body = {"classes": remaining, "classes_action": "remove_annotations"}
    reply = scene.client.patch(f"/api/projects/{scene.pid}", json=body)
    assert reply.status_code == 200, reply.text


def spec_of(scene: Scene, number: int, model_id: str) -> dict[str, Any]:
    """The ``spec.json`` content of a run on a version.

    Args:
        scene: the project.
        number: dataset version.
        model_id: catalogue model.

    Returns:
        The spec.
    """
    model = get_model(model_id)
    assert model is not None
    version = DatasetStore(scene.project_dir).read(number)
    run_dir = scene.project_dir / "runs" / "r1"
    return build_spec(scene.project_dir, RunConfig(dataset=number, model=model_id), model, run_dir, version)


def items_by_index(scene: Scene, spec: dict[str, Any]) -> dict[int, tuple[str, dict[str, Any]]]:
    """Items of a spec by image index, with the part each one is in.

    Args:
        scene: the project.
        spec: a spec.

    Returns:
        ``{index: (part, item)}``; an image that is in two places fails the test.
    """
    manifest = Manifest(scene.project_dir)
    record_of = {str(manifest.file_path(r).resolve()): r.name for r in manifest.all()}
    out: dict[int, tuple[str, dict[str, Any]]] = {}
    for part, items in spec["parts"].items():
        for item in items:
            index = int(record_of[item["path"]][3:].split(".")[0])
            assert index not in out, f"image {index} is in two parts"
            out[index] = (part, item)
    return out


def kept(truth: Truth, removed: str | None) -> Truth:
    """The truth without the objects of a removed class (images left empty are dropped).

    Args:
        truth: objects per image index.
        removed: the class name that was removed, or None.

    Returns:
        The expected truth.
    """
    out = {k: [o for o in v if o.cls != removed] for k, v in truth.items()}
    return {k: v for k, v in out.items() if v}


def edited(truth: Truth, index: int, objects: list[Obj]) -> Truth:
    """A copy of the truth with the objects of one image replaced (empty list: no objects).

    Args:
        truth: objects per image index.
        index: the image.
        objects: its new objects.

    Returns:
        The new truth.
    """
    out = {k: list(v) for k, v in truth.items()}
    out[index] = objects
    return out


def assert_same_images(scene: Scene, spec: dict[str, Any], truth: Truth) -> dict[int, tuple[str, dict]]:
    """The spec has exactly the images that have labels, in the three parts.

    Args:
        scene: the project.
        spec: a spec.
        truth: the expected objects per image index.

    Returns:
        The items by image index.
    """
    items = items_by_index(scene, spec)
    assert set(items) == {k for k, v in truth.items() if v}
    assert all(spec["parts"][p] for p in ("train", "val", "test"))
    return items


def extent(rings: tuple[Ring, ...]) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` around rings."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def poly(*rings: Ring, bbox: tuple[float, float, float, float] | None = None, cls: str = "cat") -> Obj:
    """A polygon object whose box is the extent of its rings (unless ``bbox`` is given).

    Args:
        rings: the rings, absolute pixels.
        bbox: a box to store instead of the extent.
        cls: class name.

    Returns:
        The object.
    """
    return Obj(cls, bbox or extent(rings), rings)


def paint(rings: tuple[Ring, ...], width: int, height: int, value: int = 1) -> np.ndarray:
    """An independent rendering of rings (absolute pixels) with PIL.

    Args:
        rings: the rings.
        width: image width.
        height: image height.
        value: pixel value inside.

    Returns:
        ``(height, width)`` uint8 array.
    """
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for ring in rings:
        draw.polygon([tuple(p) for p in ring], fill=value)
    return np.array(image, dtype=np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks (1.0 when both are empty)."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def rel_box(obj: Obj, width: int, height: int) -> list[float]:
    """Relative ``[x, y, w, h]`` of an object's box."""
    assert obj.bbox is not None
    x, y, w, h = obj.bbox
    return [x / width, y / height, w / width, h / height]


# ---------------------------------------------------------------------------------------------------
# truth of each project type: version 1, the edit, and what is expected afterwards
# ---------------------------------------------------------------------------------------------------
def box_truth() -> Truth:
    """Two boxes per image; x, y, w, h all different so a mix-up shows."""
    out: Truth = {}
    for k in range(IMAGES):
        out[k] = [
            Obj(CLASSES[k % 3], (4 + k, 3 + k, 20 + k, 10 + k)),
            Obj(CLASSES[(k + 1) % 3], (30, 20, 12 + k % 3, 10)),
        ]
    return out


def edit_boxes(truth: Truth) -> Truth:
    """Class change, move and resize, add, delete (the class removal comes after)."""
    out = edited(truth, 0, [replace(truth[0][0], cls="bird"), truth[0][1]])
    out = edited(out, 1, [replace(out[1][0], bbox=(10, 12, 30, 20)), out[1][1]])
    out = edited(out, 2, [*out[2], Obj("dog", (2, 2, 15, 9))])
    return edited(out, 3, [out[3][0]])


def ring_truth() -> Truth:
    """A triangle and (on even images) a rectangle per image, boxes equal to the extent."""
    out: Truth = {}
    for k in range(IMAGES):
        triangle: Ring = ((6 + k, 6), (30 + k, 8), (10 + k, 30))
        rectangle: Ring = ((36, 18), (56, 18), (56, 34), (36, 34))
        objects = [poly(triangle, cls=CLASSES[k % 3])]
        if k % 2 == 0:
            objects.append(poly(rectangle, cls=CLASSES[(k + 1) % 3]))
        out[k] = objects
    return out


MOVED: Ring = ((7, 6), (31, 28), (11, 30))  # image 1: the second vertex of the triangle moved down


def edit_rings(truth: Truth) -> Truth:
    """Class change, one vertex moved, objects added (counter-clockwise, closed ring), one deleted."""
    out = edited(truth, 0, [replace(truth[0][0], cls="bird"), truth[0][1]])
    out = edited(out, 1, [poly(MOVED, cls=truth[1][0].cls)])
    out = edited(out, 2, [*out[2], poly(((20, 10), (40, 10), (30, 30)), cls="dog")])  # counter-clockwise
    out = edited(out, 3, [poly(((6, 6), (20, 6), (20, 30), (6, 30), (6, 6)), cls="dog")])  # closed ring
    return edited(out, 4, [out[4][0]])


def pose_truth() -> Truth:
    """One object per image with three points: head, hand, foot."""
    out: Truth = {}
    for k in range(IMAGES):
        points = ((8 + k, 6, 2), (28 + k, 12, 2), (14 + k, 30, 2))
        out[k] = [Obj(CLASSES[k % 3], (6 + k, 4, 26, 30), points=points)]
    return out


def edit_pose(truth: Truth) -> Truth:
    """Move a point, hide one (v=0) and occlude one (v=1), add an object without a box, delete one."""
    moved = replace(truth[0][0], points=((10, 9, 2), *truth[0][0].points[1:]))
    out = edited(truth, 0, [moved])
    out = edited(out, 1, [replace(out[1][0], points=(out[1][0].points[0], (33, 21, 0), (20, 28, 1)))])
    out = edited(out, 2, [*out[2], Obj("dog", None, points=((20, 8, 2), (40, 14, 2), (25, 30, 1)))])
    return edited(out, 3, [replace(truth[3][0], cls="dog")])


# ---------------------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------------------
def tag_truth() -> dict[int, str]:
    """Image 0..8 labelled in turn cat, dog, bird; image 9 is not labelled."""
    return {k: CLASSES[k % 3] for k in range(9)}


def test_classification_edits_reach_the_spec(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, "classification")
    tags = tag_truth()
    post_labels(scene, "a.xml", tags_bytes(tags))
    v1 = new_version(scene)
    assert v1 == 1

    tags = {**tags, 0: "dog", 1: "bird", 9: "dog"}  # class changes and a label added
    del tags[3]  # a label deleted
    post_labels(scene, "b.xml", tags_bytes(tags))
    remove_class(scene, ["dog", "bird"])  # "cat" (index 0) goes: the others shift down
    expected = {k: c for k, c in tags.items() if c != "cat"}
    v2 = new_version(scene)

    spec = spec_of(scene, v2, "resnet18")
    assert spec["classes"] == ["dog", "bird"]
    items = items_by_index(scene, spec)
    assert set(items) == set(expected)  # deleted, removed-class and never-labelled images are not there
    for k, (_, item) in items.items():
        assert item["label"] == spec["classes"].index(expected[k]), f"image {k}"

    old = spec_of(scene, v1, "resnet18")  # version 1 is frozen: old labels, old class order
    assert old["classes"] == CLASSES
    old_items = items_by_index(scene, old)
    assert set(old_items) == set(tag_truth())
    assert all(old_items[k][1]["label"] == CLASSES.index(c) for k, c in tag_truth().items())


def test_classification_case_only_rename_keeps_labels_and_classes(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, "classification")
    post_labels(scene, "a.xml", tags_bytes(tag_truth()))
    reply = scene.client.patch(f"/api/projects/{scene.pid}", json={"classes": ["Cat", "dog", "bird"]})
    assert reply.status_code == 200, reply.text
    version = DatasetStore(scene.project_dir).read(new_version(scene))
    labels = {lb["label"] for items in version["labels"].values() for lb in items}
    assert version["classes"] == ["Cat", "dog", "bird"], version["classes"]  # no second "cat"
    assert labels <= set(version["classes"]), f"labels {labels} are not in the classes"
    spec = spec_of(scene, 1, "resnet18")
    assert len(items_by_index(scene, spec)) == 9  # no image silently dropped


@pytest.mark.xfail(strict=True, reason="#354")
def test_version_summary_signals_that_labels_changed_since(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, "classification")
    post_labels(scene, "a.xml", tags_bytes(tag_truth()))
    new_version(scene)
    before = scene.client.get(f"/api/projects/{scene.pid}/datasets").json()["versions"][0]
    post_labels(scene, "b.xml", tags_bytes({**tag_truth(), 0: "bird"}))
    after = scene.client.get(f"/api/projects/{scene.pid}/datasets").json()["versions"][0]
    detail = scene.client.get(f"/api/projects/{scene.pid}/datasets/1").json()
    words = ("stale", "changed", "outdated", "drift", "modified")
    flags = {k for k in {*after, *detail} if any(w in k for w in words)}
    assert flags or after != before, "nothing in the version summary says the labels changed since"


def test_old_version_cannot_start_a_run_after_a_class_was_removed(tmp_path: Path) -> None:
    scene = make_scene(tmp_path, "classification")
    post_labels(scene, "a.xml", tags_bytes(tag_truth()))
    v1 = new_version(scene)
    remove_class(scene, ["dog", "bird"])
    reply = scene.client.post(f"/api/projects/{scene.pid}/runs", json={"dataset": v1, "model": "resnet18"})
    assert reply.status_code == 400 and reply.json()["code"] == "dataset_classes_removed", reply.text


# ---------------------------------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------------------------------
def check_boxes(det: Any, scene: Scene, spec: dict[str, Any], truth: Truth) -> None:
    """Class indexes, relative boxes (spec) and pixel boxes (worker target) of every image.

    Args:
        det: the detection worker.
        scene: the project.
        spec: the spec of a ``fasterrcnn`` run.
        truth: expected objects per image index.
    """
    classes = spec["classes"]
    for k, (_, item) in assert_same_images(scene, spec, truth).items():
        width, height = size_of(k)
        want = truth[k]
        assert [o["label"] for o in item["objects"]] == [classes.index(o.cls) for o in want], f"image {k}"
        for got, obj in zip(item["objects"], want, strict=True):
            assert got["bbox"] == pytest.approx(rel_box(obj, width, height), abs=1e-6), f"image {k}"
        target = det.build_target(item, width, height, "detection", 0)
        x, y, w, h = zip(*[o.bbox for o in want if o.bbox], strict=True)
        expected = np.array([x, y, np.add(x, w), np.add(y, h)]).T
        assert target["boxes"] == pytest.approx(expected, abs=1e-3), f"image {k}"
        assert target["labels"].tolist() == [classes.index(o.cls) + 1 for o in want]


def test_detection_edits_reach_the_boxes_the_worker_trains_on(tmp_path: Path, det: Any) -> None:
    scene = make_scene(tmp_path, "detection")
    first = box_truth()
    post_labels(scene, "a.json", coco_bytes(first, CLASSES))
    v1 = new_version(scene)
    check_boxes(det, scene, spec_of(scene, v1, "fasterrcnn_resnet50_fpn"), first)

    second = edit_boxes(first)
    post_labels(scene, "b.json", coco_bytes(second, CLASSES))
    remove_class(scene, ["dog", "bird"])
    v2 = new_version(scene)
    spec = spec_of(scene, v2, "fasterrcnn_resnet50_fpn")
    assert spec["classes"] == ["dog", "bird"]
    check_boxes(det, scene, spec, kept(second, "cat"))

    old = DatasetStore(scene.project_dir).read(v1)  # frozen: still the first labels, all three classes
    assert old["classes"] == CLASSES
    old_first = old["labels"][scene.ids[0]][0]
    assert old_first["label"] == "cat" and old_first["bbox"] == pytest.approx(rel_box(first[0][0], 64, 48))


def test_yolo_normalised_boxes_reach_the_worker_in_pixels(tmp_path: Path, det: Any) -> None:
    scene = make_scene(tmp_path, "detection")
    truth = box_truth()
    post_labels(scene, "a.json", coco_bytes(truth, CLASSES))
    moved = edited(truth, 1, [Obj("bird", (10, 12, 30, 20))])
    lines = {}
    for k, objects in moved.items():
        width, height = size_of(k)
        lines[k] = "\n".join(
            f"{CLASSES.index(o.cls)} {(o.bbox[0] + o.bbox[2] / 2) / width:.6f} "
            f"{(o.bbox[1] + o.bbox[3] / 2) / height:.6f} {o.bbox[2] / width:.6f} {o.bbox[3] / height:.6f}"
            for o in objects
            if o.bbox
        )
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    reply = scene.client.post(
        f"/api/projects/{scene.pid}/annotations",
        files=[("files", ("data.yaml", f"names:\n{names}\n".encode()))]
        + [("files", (f"labels/{image_name(k)[:-4]}.txt", text.encode())) for k, text in lines.items()],
    )
    assert reply.status_code == 200, reply.text
    token = reply.json()["token"]
    applied = scene.client.post(
        f"/api/projects/{scene.pid}/annotations/{token}/apply", json={"action": "import", "mode": "replace"}
    )
    assert applied.status_code == 200, applied.text
    spec = spec_of(scene, new_version(scene), "fasterrcnn_resnet50_fpn")
    check_boxes(det, scene, spec, moved)


# ---------------------------------------------------------------------------------------------------
# segmentation: instance (detection worker) and semantic (segmentation worker)
# ---------------------------------------------------------------------------------------------------
def check_instance(det: Any, scene: Scene, spec: dict[str, Any], truth: Truth) -> None:
    """Boxes, class indexes and masks of every image of an instance segmentation spec.

    Args:
        det: the detection worker.
        scene: the project.
        spec: the spec of a ``maskrcnn`` run.
        truth: expected objects per image index.
    """
    classes = spec["classes"]
    for k, (_, item) in assert_same_images(scene, spec, truth).items():
        width, height = size_of(k)
        want = truth[k]
        assert [o["label"] for o in item["objects"]] == [classes.index(o.cls) for o in want], f"image {k}"
        target = det.build_target(item, width, height, "instance", 0)
        assert len(target["masks"]) == len(want)
        for n, obj in enumerate(want):
            assert iou(target["masks"][n], paint(obj.rings, width, height)) > 0.97, f"image {k} object {n}"
            x, y, w, h = extent(obj.rings)
            assert target["boxes"][n].tolist() == pytest.approx([x, y, x + w, y + h], abs=1e-3)


def check_semantic(seg: Any, scene: Scene, spec: dict[str, Any], truth: Truth) -> None:
    """The class mask of every image of a semantic spec (later objects paint over earlier ones).

    Args:
        seg: the segmentation worker.
        scene: the project.
        spec: the spec of a ``deeplabv3`` run.
        truth: expected objects per image index.
    """
    classes = spec["classes"]
    for k, (_, item) in assert_same_images(scene, spec, truth).items():
        width, height = size_of(k)
        mask = np.array(seg.rasterize_mask(item["objects"], width, height, classes))
        expected = np.zeros((height, width), dtype=np.uint8)
        for obj in truth[k]:
            painted = paint(obj.rings, width, height, classes.index(obj.cls) + 1)
            expected[painted > 0] = painted[painted > 0]
        assert (mask != expected).mean() < 0.01, f"image {k}"
        assert set(np.unique(mask)) <= {0, *range(1, len(classes) + 1)}


def test_segmentation_edits_reach_masks_and_boxes(tmp_path: Path, det: Any, seg: Any) -> None:
    scene = make_scene(tmp_path, "segmentation")
    first = ring_truth()
    post_labels(scene, "a.json", coco_bytes(first, CLASSES))
    v1 = new_version(scene)
    check_instance(det, scene, spec_of(scene, v1, "maskrcnn_resnet50_fpn"), first)
    check_semantic(seg, scene, spec_of(scene, v1, "deeplabv3_resnet50"), first)

    second = edit_rings(first)
    post_labels(scene, "b.json", coco_bytes(second, CLASSES))
    remove_class(scene, ["dog", "bird"])
    v2 = new_version(scene)
    want = kept(second, "cat")
    instance = spec_of(scene, v2, "maskrcnn_resnet50_fpn")
    semantic = spec_of(scene, v2, "deeplabv3_resnet50")
    assert instance["classes"] == semantic["classes"] == ["dog", "bird"]
    check_instance(det, scene, instance, want)
    check_semantic(seg, scene, semantic, want)


def test_moved_polygon_vertex_changes_the_mask_between_versions(tmp_path: Path, det: Any, seg: Any) -> None:
    scene = make_scene(tmp_path, "segmentation")
    first = ring_truth()
    post_labels(scene, "a.json", coco_bytes(first, CLASSES))
    v1 = new_version(scene)
    post_labels(scene, "b.json", coco_bytes(edit_rings(first), CLASSES))
    v2 = new_version(scene)
    width, height = size_of(1)
    inside_new, inside_old = (28, 27), (28, 9)  # (x, y): only in the moved / only in the original triangle
    for model, kind in (("maskrcnn_resnet50_fpn", "instance"), ("deeplabv3_resnet50", "semantic")):
        masks = []
        for number in (v1, v2):
            spec = spec_of(scene, number, model)
            item = items_by_index(scene, spec)[1][1]
            if kind == "instance":
                masks.append(det.build_target(item, width, height, "instance", 0)["masks"][0] > 0)
            else:
                masks.append(
                    np.array(seg.rasterize_mask(item["objects"], width, height, spec["classes"])) > 0
                )
        old, new = masks
        assert old[inside_old[1], inside_old[0]] and not new[inside_old[1], inside_old[0]], kind
        assert new[inside_new[1], inside_new[0]] and not old[inside_new[1], inside_new[0]], kind


@pytest.mark.xfail(strict=True, reason="#355")
def test_instance_box_agrees_with_polygons_when_the_file_box_disagrees(tmp_path: Path, det: Any) -> None:
    scene = make_scene(tmp_path, "segmentation")
    truth = ring_truth()
    ring: Ring = ((10, 8), (40, 8), (40, 30), (10, 30))
    truth[0] = [poly(ring, bbox=(30, 20, 10, 10))]  # the stored box does not match the polygon
    post_labels(scene, "a.json", coco_bytes(truth, CLASSES))
    spec = spec_of(scene, new_version(scene), "maskrcnn_resnet50_fpn")
    width, height = size_of(0)
    item = items_by_index(scene, spec)[0][1]
    target = det.build_target(item, width, height, "instance", 0)
    ys, xs = np.nonzero(target["masks"][0])
    mask_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert target["boxes"][0].tolist() == pytest.approx(mask_box, abs=1.5), (
        "box and mask of one object disagree"
    )


def test_ring_orientation_and_closure_do_not_change_the_mask(det: Any, seg: Any) -> None:
    clockwise = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.7], [0.1, 0.7]]
    counter = clockwise[::-1]
    closed = [*clockwise, clockwise[0]]
    base = det.polygon_mask([clockwise], 80, 40)
    assert base.sum() > 500
    for ring in (counter, closed, [*counter, counter[0]]):
        assert (det.polygon_mask([ring], 80, 40) == base).all()
        want = np.array(seg.rasterize_mask([{"label": 0, "polygons": [clockwise]}], 80, 40, ["a"]))
        got = np.array(seg.rasterize_mask([{"label": 0, "polygons": [ring]}], 80, 40, ["a"]))
        assert (got == want).all()


def test_polygon_with_two_rings_fills_both_and_a_nested_ring_is_not_a_hole(det: Any, seg: Any) -> None:
    outer = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    inner = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    far = [[0.92, 0.92], [0.99, 0.92], [0.99, 0.99], [0.92, 0.99]]
    mask = det.polygon_mask([outer, far], 100, 100)
    assert mask[95, 95] == 1 and mask[50, 50] == 1
    assert det.polygon_mask([outer, inner], 100, 100)[50, 50] == 1  # documented: rings are not holes
    classes = seg.rasterize_mask([{"label": 0, "polygons": [outer, inner]}], 100, 100, ["a"])
    assert classes.getpixel((50, 50)) == 1


# ---------------------------------------------------------------------------------------------------
# keypoints
# ---------------------------------------------------------------------------------------------------
def check_pose(det: Any, scene: Scene, spec: dict[str, Any], truth: Truth) -> None:
    """Class indexes, relative points (spec) and pixel points and boxes (worker target).

    Args:
        det: the detection worker.
        scene: the project.
        spec: the spec of a ``keypointrcnn`` run.
        truth: expected objects per image index.
    """
    classes = spec["classes"]
    assert spec["num_keypoints"] == len(KEYPOINT_NAMES)
    for k, (_, item) in assert_same_images(scene, spec, truth).items():
        width, height = size_of(k)
        want = truth[k]
        assert [o["label"] for o in item["objects"]] == [classes.index(o.cls) for o in want], f"image {k}"
        target = det.build_target(item, width, height, "keypoints", spec["num_keypoints"])
        for n, obj in enumerate(want):
            got = item["objects"][n]["keypoints"]
            relative = [[x / width, y / height, v] for x, y, v in obj.points]
            assert np.array(got) == pytest.approx(np.array(relative), abs=1e-6), f"image {k}"
            pixels = [[x, y, v] if v > 0 else [0, 0, 0] for x, y, v in obj.points]
            assert target["keypoints"][n] == pytest.approx(np.array(pixels), abs=1e-3), f"image {k}"
            if obj.bbox:
                x, y, w, h = obj.bbox
                box = [x, y, x + w, y + h]
            else:  # no stored box: the extent of the visible points
                shown = [p for p in obj.points if p[2] > 0]
                box = [min(p[0] for p in shown), min(p[1] for p in shown)]
                box += [max(p[0] for p in shown), max(p[1] for p in shown)]
            assert target["boxes"][n].tolist() == pytest.approx(box, abs=1e-3), f"image {k}"
        assert target["labels"].tolist() == [classes.index(o.cls) + 1 for o in want]


def test_keypoint_edits_reach_the_points_the_worker_trains_on(tmp_path: Path, det: Any) -> None:
    scene = make_scene(tmp_path, "keypoints")
    first = pose_truth()
    post_labels(scene, "a.json", coco_bytes(first, CLASSES, KEYPOINT_NAMES))
    v1 = new_version(scene)
    check_pose(det, scene, spec_of(scene, v1, "keypointrcnn_resnet50_fpn"), first)

    second = edit_pose(first)
    post_labels(scene, "b.json", coco_bytes(second, CLASSES, KEYPOINT_NAMES))
    remove_class(scene, ["dog", "bird"])
    v2 = new_version(scene)
    spec = spec_of(scene, v2, "keypointrcnn_resnet50_fpn")
    assert spec["classes"] == ["dog", "bird"]
    check_pose(det, scene, spec, kept(second, "cat"))

    old = spec_of(scene, v1, "keypointrcnn_resnet50_fpn")  # frozen: the old point, the old class order
    moved = items_by_index(scene, old)[0][1]["objects"][0]
    assert moved["label"] == 0 and moved["keypoints"][0][:2] == pytest.approx([8 / 64, 6 / 48])


def test_hidden_keypoint_with_coordinates_is_zeroed_and_occluded_is_kept(det: Any) -> None:
    item = {
        "path": "x",
        "objects": [
            {
                "label": 0,
                "bbox": [0.1, 0.1, 0.5, 0.5],
                "keypoints": [[0.5, 0.5, 0.0], [0.25, 0.5, 1.0], [0.75, 0.5, 2.0]],
            }
        ],
    }
    points = det.build_target(item, 80, 40, "keypoints", 3)["keypoints"][0]
    assert points.tolist() == [[0.0, 0.0, 0.0], [20.0, 20.0, 1.0], [60.0, 20.0, 2.0]]


@pytest.mark.xfail(strict=True, reason="#355")
def test_keypoint_box_covers_the_points_when_a_point_was_moved_outside_the_stored_box(
    tmp_path: Path, det: Any
) -> None:
    scene = make_scene(tmp_path, "keypoints")
    truth = pose_truth()
    stale = replace(truth[0][0], bbox=(8, 6, 10, 10), points=((50, 40, 2), (12, 10, 2), (14, 12, 2)))
    post_labels(scene, "a.json", coco_bytes(edited(truth, 0, [stale]), CLASSES, KEYPOINT_NAMES))
    spec = spec_of(scene, new_version(scene), "keypointrcnn_resnet50_fpn")
    width, height = size_of(0)
    target = det.build_target(items_by_index(scene, spec)[0][1], width, height, "keypoints", 3)
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    inside = [
        (x1 - 1 <= x <= x2 + 1 and y1 - 1 <= y <= y2 + 1) for x, y, _ in target["keypoints"][0].tolist()
    ]
    assert all(inside), f"a visible point lies outside the training box {target['boxes'][0].tolist()}"
