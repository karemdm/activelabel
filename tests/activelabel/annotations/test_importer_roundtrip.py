"""Evidence tests: annotation imports and re-imports through the API are exact.

Projects have no export of their labels (only a dataset version is exported, see
``activelabel.exporting``), so a "round trip" here is: import a file, edit the file, import it again
(replace or merge) and compare the stored labels with what the file says. Images are 64x48 pixels (not
square), so an x/y mix-up shows. Every number below is in pixels of the 64x48 images, converted to the
relative form the importer stores.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from api_client import make_client, upload
from fastapi.testclient import TestClient
from make_video import make_video
from sample_images import make_image

from activelabel.labels import Label, load_annotations
from activelabel.tasks_store import TaskStore
from activelabel.web.state import AppState

W, H = 64, 48
NAMES = ("a.png", "b.png", "c.png", "d.png", "e.png")
CLASSES = ["cat", "dog", "bird"]
COCO_IDS = {"cat": 4, "dog": 1, "bird": 9, "fish": 0}  # not sorted, not 1..n, one id 0
Pixels = tuple[float, ...]
Boxes = dict[str, list[tuple[str, int, int, int, int]]]
FORMATS = ["coco", "yolo", "voc", "cvat"]


# -- expected labels, written in pixels ---------------------------------------------------------------
def rel_box(x: float, y: float, w: float, h: float) -> list[float]:
    """Relative ``[x, y, w, h]`` of a box given in pixels of a 64x48 image.

    Args:
        x: left edge in pixels.
        y: top edge in pixels.
        w: width in pixels.
        h: height in pixels.

    Returns:
        The box as the importer stores it (0..1).
    """
    return [x / W, y / H, w / W, h / H]


def box(name: str, x: float, y: float, w: float, h: float, track: int | None = None) -> Label:
    """Expected detection label from a box in pixels.

    Args:
        name: class name.
        x: left edge in pixels.
        y: top edge in pixels.
        w: width in pixels.
        h: height in pixels.
        track: expected track id, or None.

    Returns:
        The label the importer should store.
    """
    return Label(name, bbox=rel_box(x, y, w, h), track=track)


def poly(name: str, rings: list[list[tuple[float, float]]], bbox: Pixels | None = None) -> Label:
    """Expected segmentation label from rings in pixels.

    Args:
        name: class name.
        rings: polygon rings as ``(x, y)`` pixel points.
        bbox: ``(x, y, w, h)`` in pixels when the file gives one; else the box around all points.

    Returns:
        The label the importer should store.
    """
    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    x, y, w, h = bbox if bbox else (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    polygons = [[[px / W, py / H] for px, py in ring] for ring in rings]
    return Label(name, bbox=rel_box(x, y, w, h), polygons=polygons)


def pose(name: str, bbox: Pixels, points: list[tuple[float, float, int]]) -> Label:
    """Expected pose label from a box and ``(x, y, visibility)`` points in pixels.

    Args:
        name: class name.
        bbox: ``(x, y, w, h)`` in pixels.
        points: keypoints as ``(x, y, v)`` with v 0, 1 or 2.

    Returns:
        The label the importer should store.
    """
    return Label(name, bbox=rel_box(*bbox), keypoints=[[px / W, py / H, float(v)] for px, py, v in points])


def flat(value: Any) -> list[float]:
    """All numbers of a nested list, in order.

    Args:
        value: a number, a list, or None.

    Returns:
        The numbers (empty for None).
    """
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [n for item in value for n in flat(item)]


def same_labels(got: list[Label] | None, want: list[Label], where: str) -> None:
    """Assert two label lists are equal (same order), numbers within 1e-6 relative.

    Args:
        got: the stored labels (None means the image has no entry).
        want: the expected labels.
        where: text for the failure message.
    """
    assert got is not None, f"{where}: no labels stored, expected {len(want)}"
    assert [lb.label for lb in got] == [lb.label for lb in want], f"{where}: classes {got} vs {want}"
    assert [lb.track for lb in got] == [lb.track for lb in want], f"{where}: tracks"
    for g, w in zip(got, want, strict=True):
        for part in ("bbox", "polygons", "keypoints"):
            gv, wv = flat(getattr(g, part)), flat(getattr(w, part))
            assert gv == pytest.approx(wv, rel=1e-6, abs=1e-9), f"{where}: {part} {gv} vs {wv}"
            assert (getattr(g, part) is None) == (getattr(w, part) is None), f"{where}: {part} presence"


def same_images(got: dict[Any, list[Label] | None], want: dict[Any, list[Label] | None], where: str) -> None:
    """Assert the stored labels of every image equal the expected ones (None = no entry or empty).

    Args:
        got: stored labels by image name (or by frame).
        want: expected labels by the same keys.
        where: text for the failure message.
    """
    for key, labels in want.items():
        if labels is None:
            assert not got.get(key), f"{where}: {key} should have no labels, has {got.get(key)}"
        else:
            same_labels(got.get(key), labels, f"{where} {key}")
    for key in got:
        assert key in want, f"{where}: unexpected image {key}"


# -- a workspace with one project -----------------------------------------------------------------------
@dataclass
class Rig:
    """A temporary workspace with project 1 and its images."""

    client: TestClient
    state: AppState
    task: str

    def check(self, files: list[tuple[str, str]], **form: Any) -> dict[str, Any]:
        """Upload annotation files and return the validation report (status 200).

        Args:
            files: ``(file name, text)`` pairs.
            **form: extra form fields (``task`` for a video task, ``class_name``).

        Returns:
            The report with its ``token``.
        """
        response = self.client.post(
            "/api/projects/1/annotations",
            files=[("files", (name, text.encode())) for name, text in files],
            data={k: str(v) for k, v in form.items()},
        )
        assert response.status_code == 200, response.text
        return response.json()

    def load(
        self, files: list[tuple[str, str]], mode: str = "replace", action: str = "import", **form: Any
    ) -> dict[str, Any]:
        """Check and apply annotation files; returns the apply summary (status 200).

        Args:
            files: ``(file name, text)`` pairs.
            mode: ``replace`` or ``merge``.
            action: the apply action.
            **form: extra form fields of the check.

        Returns:
            The summary of the import.
        """
        report = self.check(files, **form)
        assert report["can_import"], report
        done = self.client.post(
            f"/api/projects/1/annotations/{report['token']}/apply", json={"action": action, "mode": mode}
        )
        assert done.status_code == 200, done.text
        return done.json()

    def stored(self) -> dict[str, list[Label] | None]:
        """Stored ground truth by image name (None when the image has no entry)."""
        records = self.state.manifest(1).load()
        found = load_annotations(self.state.store.dir(1))
        images = found.images if found else {}
        return {rec.name: images.get(rec.id) for rec in records.values()}

    def classes(self) -> list[str]:
        """Classes of the project."""
        return list(self.state.store.get(1).classes)

    def frames(self, task_id: int = 1) -> dict[int, list[Label] | None]:
        """Stored labels of a video task by video frame (None when the frame has no entry)."""
        stored = TaskStore(self.state.store.dir(1)).get(task_id)
        records = self.state.manifest(1).load()
        found = load_annotations(self.state.store.dir(1))
        images = found.images if found else {}
        return {records[i].frame: images.get(i) for i in stored.images}


def make_rig(tmp_path: Path, task: str = "detection", classes: list[str] | None = None) -> Rig:
    """Workspace with project 1 of this task and five 64x48 images ``a.png`` .. ``e.png``.

    Args:
        tmp_path: pytest temporary folder.
        task: project task.
        classes: classes of the project.

    Returns:
        The rig.
    """
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": task, "classes": classes or []})
    paths = [make_image(tmp_path / "src" / n, size=(W, H), seed=i) for i, n in enumerate(NAMES)]
    assert upload(client, 1, paths, list(NAMES)).status_code == 200
    return Rig(client, state, task)


def wait(client: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    """Wait for a job to end.

    Args:
        client: API client.
        job: the job as returned when it was created.

    Returns:
        The finished job.
    """
    for _ in range(400):
        current = next(j for j in client.get("/api/projects/1/jobs").json() if j["id"] == job["id"])
        if current["status"] in ("done", "failed", "cancelled"):
            return current
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def make_video_rig(tmp_path: Path, tasks: int = 1) -> Rig:
    """Workspace with a detection project and a 60-frame 64x48 video; each task takes frames 10..50 step 10.

    Args:
        tmp_path: pytest temporary folder.
        tasks: number of video tasks.

    Returns:
        The rig.
    """
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection"})
    path = make_video(tmp_path / "clip.mp4")
    video = client.post("/api/projects/1/videos", files={"file": (path.name, path.read_bytes())}).json()["id"]
    for i in range(tasks):
        job = client.post(
            "/api/projects/1/tasks/from-video",
            json={"video": video, "name": f"T{i}", "start": 10, "stop": 50, "step": 10},
        )
        assert wait(client, job.json())["status"] == "done"
    return Rig(client, state, "detection")


# -- files of each format, written from boxes -----------------------------------------------------------
def expected_boxes(boxes: Boxes) -> dict[str, list[Label] | None]:
    """Labels the importer should store for these boxes (an image listed with no box has an empty list).

    Args:
        boxes: ``{image name: [(class, x, y, w, h) in pixels]}``.

    Returns:
        Expected labels by image name.
    """
    return {n: [box(c, *b) for c, *b in items] for n, items in boxes.items()}


def coco_files(boxes: Boxes, classes: list[str] | None = None) -> list[tuple[str, str]]:
    """One COCO file for these boxes (image ids and category ids are not 0..n-1).

    Args:
        boxes: ``{image name: [(class, x, y, w, h)]}``.
        classes: category names in file order.

    Returns:
        ``[(file name, text)]``.
    """
    names = list(boxes)
    images = [{"id": 10 + 5 * i, "file_name": n, "width": W, "height": H} for i, n in enumerate(names)]
    anns = []
    for i, name in enumerate(names):
        for cls, x, y, w, h in boxes[name]:
            anns.append(
                {
                    "id": len(anns) + 100,
                    "image_id": 10 + 5 * i,
                    "category_id": COCO_IDS[cls],
                    "bbox": [x, y, w, h],
                }
            )
    cats = [{"id": COCO_IDS[c], "name": c} for c in (classes or CLASSES)]
    return [("instances.json", json.dumps({"images": images, "annotations": anns, "categories": cats}))]


def yolo_files(boxes: Boxes, classes: list[str] | None = None) -> list[tuple[str, str]]:
    """YOLO txt files (and data.yaml) for these boxes.

    Args:
        boxes: ``{image name: [(class, x, y, w, h)]}``.
        classes: names in index order for ``data.yaml``.

    Returns:
        ``[(file name, text)]``.
    """
    names = classes or CLASSES
    files = [("data.yaml", "names: " + json.dumps(names))]
    for image, items in boxes.items():
        lines = [
            f"{names.index(c)} {(x + w / 2) / W!r} {(y + h / 2) / H!r} {w / W!r} {h / H!r}"
            for c, x, y, w, h in items
        ]
        files.append((f"{Path(image).stem}.txt", "\n".join(lines) + ("\n" if lines else "")))
    return files


def voc_files(boxes: Boxes, classes: list[str] | None = None) -> list[tuple[str, str]]:
    """One Pascal VOC XML per image for these boxes (coordinates as written).

    Args:
        boxes: ``{image name: [(class, x, y, w, h)]}``.
        classes: unused (VOC has no class list); kept so all builders have one signature.

    Returns:
        ``[(file name, text)]``.
    """
    files = []
    for image, items in boxes.items():
        objects = "".join(
            f"<object><name>{c}</name><bndbox><xmin>{x}</xmin><ymin>{y}</ymin><xmax>{x + w}</xmax>"
            f"<ymax>{y + h}</ymax></bndbox></object>"
            for c, x, y, w, h in items
        )
        xml = (
            f"<annotation><filename>{image}</filename>"
            f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>{objects}</annotation>"
        )
        files.append((f"{Path(image).stem}.xml", xml))
    return files


def cvat_files(boxes: Boxes, classes: list[str] | None = None) -> list[tuple[str, str]]:
    """One CVAT for images 1.1 XML for these boxes.

    Args:
        boxes: ``{image name: [(class, x, y, w, h)]}``.
        classes: labels declared in the meta.

    Returns:
        ``[(file name, text)]``.
    """
    labels = "".join(f"<label><name>{c}</name></label>" for c in (classes or CLASSES))
    images = []
    for i, (image, items) in enumerate(boxes.items()):
        shapes = "".join(
            f'<box label="{c}" occluded="0" source="manual" xtl="{x}" ytl="{y}" xbr="{x + w}" ybr="{y + h}" '
            'z_order="0"/>'
            for c, x, y, w, h in items
        )
        images.append(f'<image id="{i}" name="{image}" width="{W}" height="{H}">{shapes}</image>')
    xml = (
        f"<annotations><version>1.1</version><meta><task><labels>{labels}</labels></task></meta>"
        + "".join(images)
        + "</annotations>"
    )
    return [("annotations.xml", xml)]


BUILDERS = {"coco": coco_files, "yolo": yolo_files, "voc": voc_files, "cvat": cvat_files}

V1: Boxes = {
    "a.png": [("cat", 5, 5, 20, 10), ("dog", 30, 10, 20, 20)],
    "b.png": [("cat", 0, 0, 10, 10)],
    "c.png": [],
    "d.png": [("bird", 8, 8, 40, 30)],
}
# v1 edited: a box moved, a box deleted, another added; a class changed; an empty image gets a box; d dropped
V2: Boxes = {
    "a.png": [("cat", 6, 7, 20, 10), ("bird", 40, 30, 10, 10)],
    "b.png": [("dog", 0, 0, 10, 10)],
    "c.png": [("cat", 1, 1, 5, 5)],
}


# -- boxes in every image format ------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FORMATS)
def test_boxes_import_is_exact(tmp_path, fmt):
    rig = make_rig(tmp_path)
    summary = rig.load(BUILDERS[fmt](V1))
    assert summary["format"] == fmt and summary["labels_imported"] == 4
    same_images(rig.stored(), {**expected_boxes(V1), "e.png": None}, fmt)
    assert rig.classes() == CLASSES


@pytest.mark.parametrize("fmt", FORMATS)
def test_boxes_edit_then_replace_equals_the_edited_file(tmp_path, fmt):
    rig = make_rig(tmp_path)
    rig.load(BUILDERS[fmt](V1))
    summary = rig.load(BUILDERS[fmt](V2), mode="replace")
    assert summary["labels_imported"] == 4  # nothing from v1 is counted again
    got = rig.stored()
    same_images(got, {**expected_boxes(V2), "d.png": None, "e.png": None}, f"{fmt} replace")
    assert sum(len(v or []) for v in got.values()) == 4, "labels left over or duplicated"


@pytest.mark.parametrize("fmt", FORMATS)
def test_boxes_replace_twice_with_the_same_file_changes_nothing(tmp_path, fmt):
    rig = make_rig(tmp_path)
    rig.load(BUILDERS[fmt](V1))
    first = {k: [lb.as_dict() for lb in v or []] for k, v in rig.stored().items()}
    rig.load(BUILDERS[fmt](V1))
    assert {k: [lb.as_dict() for lb in v or []] for k, v in rig.stored().items()} == first


@pytest.mark.parametrize("fmt", FORMATS)
def test_boxes_merge_adds_only_to_images_without_labels(tmp_path, fmt):
    rig = make_rig(tmp_path)
    rig.load(BUILDERS[fmt](V1))
    merged: Boxes = {
        "a.png": [("cat", 1, 1, 3, 3)],  # a has labels: kept as it was
        "b.png": [],
        "c.png": [("dog", 2, 2, 8, 8)],  # c had none: gets it
        "e.png": [("bird", 3, 3, 9, 9)],  # e had none: gets it
    }
    summary = rig.load(BUILDERS[fmt](merged), mode="merge")
    assert summary["images_kept"] == 2 and summary["images_added"] == 2 and summary["labels_imported"] == 2
    want = expected_boxes(V1)
    want["c.png"] = [box("dog", 2, 2, 8, 8)]
    want["e.png"] = [box("bird", 3, 3, 9, 9)]
    same_images(rig.stored(), want, f"{fmt} merge")


@pytest.mark.parametrize("fmt", FORMATS)
def test_boxes_same_file_in_merge_twice_adds_no_duplicates(tmp_path, fmt):
    rig = make_rig(tmp_path)
    rig.load(BUILDERS[fmt](V1), mode="merge")  # merge into a project without labels
    same_images(rig.stored(), {**expected_boxes(V1), "e.png": None}, f"{fmt} first merge")
    summary = rig.load(BUILDERS[fmt](V1), mode="merge")
    assert summary["images_added"] == 0 and summary["labels_imported"] == 0
    same_images(rig.stored(), {**expected_boxes(V1), "e.png": None}, f"{fmt} second merge")


@pytest.mark.parametrize("fmt", ["coco", "yolo", "cvat"])
def test_merge_adds_only_the_classes_of_the_labels_it_added(tmp_path, fmt):
    rig = make_rig(tmp_path, classes=["cat"])
    rig.load(BUILDERS[fmt]({"a.png": [("cat", 1, 1, 5, 5)]}, ["cat"]))
    only = {"a.png": [("fish", 2, 2, 5, 5)], "b.png": [("dog", 2, 2, 5, 5)]}
    rig.load(BUILDERS[fmt](only, ["cat", "dog", "fish"]), mode="merge")
    assert rig.classes() == ["cat", "dog"]  # fish was only on a labelled image that was skipped


# -- classes ---------------------------------------------------------------------------------------------
def test_coco_category_ids_are_mapped_by_id_not_by_position(tmp_path):
    rig = make_rig(tmp_path)
    boxes: Boxes = {"a.png": [("fish", 1, 1, 9, 9), ("dog", 2, 2, 9, 9), ("cat", 3, 3, 9, 9)]}
    rig.load(coco_files(boxes, ["cat", "dog", "bird", "fish"]))  # bird is declared but unused
    same_images(
        rig.stored(),
        {"a.png": expected_boxes(boxes)["a.png"], "b.png": None, "c.png": None, "d.png": None, "e.png": None},
        "coco ids",
    )
    assert sorted(rig.classes()) == ["bird", "cat", "dog", "fish"]  # an unused class is kept


def test_class_names_differing_only_in_case_use_the_projects_spelling(tmp_path):
    rig = make_rig(tmp_path, classes=["Cat", "Dog"])
    rig.load(coco_files({"a.png": [("cat", 1, 1, 9, 9)], "b.png": [("dog", 2, 2, 9, 9)]}, ["cat", "dog"]))
    assert [lb.label for lb in rig.stored()["a.png"]] == ["Cat"]
    assert [lb.label for lb in rig.stored()["b.png"]] == ["Dog"]
    assert rig.classes() == ["Cat", "Dog"]  # no "cat" next to "Cat"


def test_yolo_class_index_follows_data_yaml_not_the_projects_order(tmp_path):
    rig = make_rig(tmp_path, classes=["ant", "zebra"])
    lines = "0 0.5 0.5 0.25 0.25\n1 0.25 0.25 0.125 0.125\n"
    rig.load([("data.yaml", "names:\n  0: zebra\n  3: ant\n"), ("a.txt", lines.replace("\n1 ", "\n3 "))])
    assert [lb.label for lb in rig.stored()["a.png"]] == ["zebra", "ant"]
    assert rig.classes() == ["ant", "zebra"]


def test_yolo_classes_txt_gives_the_names_by_line(tmp_path):
    rig = make_rig(tmp_path)
    rig.load([("classes.txt", "zebra\nant\n"), ("a.txt", "1 0.5 0.5 0.25 0.25\n")])
    assert [lb.label for lb in rig.stored()["a.png"]] == ["ant"]
    assert sorted(rig.classes()) == ["ant", "zebra"]  # declared and unused "zebra" is kept


def test_yolo_without_names_uses_class_n_and_warns(tmp_path):
    rig = make_rig(tmp_path)
    report = rig.check([("a.txt", "2 0.5 0.5 0.25 0.25\n")])
    assert any(p["code"] == "missing_class_names" for p in report["problems"])
    rig.load([("a.txt", "2 0.5 0.5 0.25 0.25\n")])
    assert [lb.label for lb in rig.stored()["a.png"]] == ["class_2"]


def test_deleted_class_comes_back_with_the_same_labels_on_reimport(tmp_path):
    rig = make_rig(tmp_path)
    rig.load(coco_files(V1))
    patched = rig.client.patch(
        "/api/projects/1", json={"classes": ["cat", "bird"], "classes_action": "remove_annotations"}
    )
    assert patched.status_code == 200, patched.text
    assert all(lb.label != "dog" for v in rig.stored().values() for lb in v or [])
    rig.load(coco_files(V1))
    same_images(rig.stored(), {**expected_boxes(V1), "e.png": None}, "after class deletion")
    assert "dog" in rig.classes()


# -- COCO: polygons, keypoints, size ------------------------------------------------------------------
def coco_doc(anns: list[dict[str, Any]], names: tuple[str, ...] = NAMES[:3], **extra: Any) -> str:
    """A COCO file with these annotations (``image`` is an image name) and one category per class.

    Args:
        anns: annotations; ``image`` (a file name) is replaced by ``image_id``.
        names: images listed in the file.
        **extra: extra keys of the category (``keypoints``).

    Returns:
        The JSON text.
    """
    images = [{"id": 10 + 5 * i, "file_name": n, "width": W, "height": H} for i, n in enumerate(names)]
    out = []
    for i, ann in enumerate(anns):
        ann = dict(ann)
        ann["image_id"] = 10 + 5 * names.index(ann.pop("image"))
        ann.setdefault("id", 100 + i)
        out.append(ann)
    cats = [{"id": COCO_IDS[c], "name": c, **extra} for c in CLASSES]
    return json.dumps({"images": images, "annotations": out, "categories": cats})


def seg(image: str, cls: str, rings: list[list[tuple[float, float]]], bbox: list[float] | None = None):
    """A COCO segmentation annotation.

    Args:
        image: image file name.
        cls: class name.
        rings: polygon rings in pixels.
        bbox: ``[x, y, w, h]`` or None to leave it out.

    Returns:
        The annotation dict.
    """
    ann: dict[str, Any] = {
        "image": image,
        "category_id": COCO_IDS[cls],
        "segmentation": [[v for p in ring for v in p] for ring in rings],
        "iscrowd": 0,
    }
    if bbox:
        ann["bbox"] = bbox
    return ann


RING_A = [(10, 5), (40, 8), (50, 30), (20, 40)]
RING_HOLE = [(15, 15), (25, 15), (25, 25), (15, 25)]
RING_B = [(0, 0), (30, 0), (0, 20)]


def test_coco_segmentation_import_and_edit(tmp_path):
    rig = make_rig(tmp_path, task="segmentation")
    v1 = coco_doc(
        [
            seg("a.png", "cat", [RING_A, RING_HOLE], [10, 5, 40, 35]),  # two rings and the file's box
            seg("a.png", "dog", [RING_B]),  # no box in the file: taken around the polygon
            seg("b.png", "bird", [RING_B]),
        ]
    )
    rig.load([("seg.json", v1)])
    same_images(
        rig.stored(),
        {
            "a.png": [poly("cat", [RING_A, RING_HOLE], (10, 5, 40, 35)), poly("dog", [RING_B])],
            "b.png": [poly("bird", [RING_B])],
            "c.png": [],
            "d.png": None,
            "e.png": None,
        },
        "seg v1",
    )
    moved = [(10, 5), (40, 8), (51, 31), (20, 40)]  # one vertex changed
    v2 = coco_doc([seg("a.png", "cat", [moved, RING_HOLE], [10, 5, 41, 35]), seg("c.png", "dog", [RING_B])])
    rig.load([("seg.json", v2)])
    same_images(
        rig.stored(),
        {
            "a.png": [poly("cat", [moved, RING_HOLE], (10, 5, 41, 35))],
            "b.png": None,
            "c.png": [poly("dog", [RING_B])],
            "d.png": None,
            "e.png": None,
        },
        "seg v2",
    )


KP_NAMES = ["nose", "left", "right"]


def kp_ann(image: str, cls: str, bbox: list[float], pts: list[tuple[float, float, int]]) -> dict[str, Any]:
    """A COCO keypoints annotation.

    Args:
        image: image file name.
        cls: class name.
        bbox: ``[x, y, w, h]`` in pixels.
        pts: ``(x, y, v)`` per keypoint in pixels.

    Returns:
        The annotation dict.
    """
    return {
        "image": image,
        "category_id": COCO_IDS[cls],
        "bbox": bbox,
        "keypoints": [n for p in pts for n in p],
        "num_keypoints": sum(1 for p in pts if p[2] > 0),
    }


def test_coco_keypoints_visibility_and_edit(tmp_path):
    rig = make_rig(tmp_path, task="keypoints")
    p1 = [(10, 10, 2), (20, 30, 1), (0, 0, 0)]  # visible, hidden, not labelled
    p2 = [(40, 5, 2), (50, 6, 2), (45, 40, 2)]
    v1 = coco_doc(
        [kp_ann("a.png", "cat", [5, 5, 30, 30], p1), kp_ann("b.png", "dog", [30, 2, 25, 40], p2)],
        keypoints=KP_NAMES,
    )
    rig.load([("kp.json", v1)])
    same_images(
        rig.stored(),
        {
            "a.png": [pose("cat", (5, 5, 30, 30), p1)],
            "b.png": [pose("dog", (30, 2, 25, 40), p2)],
            "c.png": [],
            "d.png": None,
            "e.png": None,
        },
        "kp v1",
    )
    p1_moved = [(11, 10, 2), (20, 31, 2), (0, 0, 0)]  # one point moved, one made visible
    v2 = coco_doc(
        [kp_ann("a.png", "cat", [5, 5, 30, 30], p1_moved), kp_ann("c.png", "bird", [1, 1, 10, 10], p2)],
        keypoints=KP_NAMES,
    )
    rig.load([("kp.json", v2)])
    same_images(
        rig.stored(),
        {
            "a.png": [pose("cat", (5, 5, 30, 30), p1_moved)],
            "b.png": None,
            "c.png": [pose("bird", (1, 1, 10, 10), p2)],
            "d.png": None,
            "e.png": None,
        },
        "kp v2",
    )


def test_coco_declared_image_size_scales_the_coordinates_and_is_reported(tmp_path):
    rig = make_rig(tmp_path)
    text = json.dumps(
        {
            "images": [{"id": 1, "file_name": "a.png", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [64, 96, 320, 120]}],
            "categories": [{"id": 1, "name": "cat"}],
        }
    )
    report = rig.check([("c.json", text)])
    assert any(p["code"] == "size_mismatch" for p in report["problems"])
    rig.load([("c.json", text)])
    same_images(rig.stored(), {"a.png": [box("cat", 6.4, 9.6, 32, 12)], **dict.fromkeys(NAMES[1:])}, "scaled")


@pytest.mark.parametrize("kind", ["polygon", "keypoints"])
def test_coco_geometry_outside_the_image_is_reported_not_changed_silently(tmp_path, kind):
    rig = make_rig(tmp_path, task="segmentation" if kind == "polygon" else "keypoints")
    if kind == "polygon":
        ann = seg("a.png", "cat", [[(10, 5), (80, 8), (50, 30)]])  # one vertex 16 px outside
        text = coco_doc([ann])
    else:
        text = coco_doc([kp_ann("a.png", "cat", [5, 5, 30, 30], [(10, 10, 2), (90, 30, 2), (20, 20, 2)])])
    report = rig.check([("x.json", text)])
    codes = {p["code"] for p in report["problems"]}
    assert codes & {"box_clipped", "geometry_clipped", "polygon_clipped", "point_clipped"}, (
        f"the {kind} was moved to the image border without any warning: {sorted(codes)}"
    )


# -- YOLO: segmentation and pose ------------------------------------------------------------------------
def yolo_poly(cls: int, ring: list[tuple[float, float]]) -> str:
    """A YOLO segmentation line from pixel points.

    Args:
        cls: class index.
        ring: ``(x, y)`` in pixels.

    Returns:
        The line.
    """
    return f"{cls} " + " ".join(f"{x / W!r} {y / H!r}" for x, y in ring) + "\n"


def test_yolo_segmentation_import_and_edit(tmp_path):
    rig = make_rig(tmp_path, task="segmentation")
    yaml = ("data.yaml", "names: [cat, dog, bird]")
    rig.load([yaml, ("a.txt", yolo_poly(0, RING_A) + yolo_poly(1, RING_B)), ("b.txt", yolo_poly(2, RING_B))])
    same_images(
        rig.stored(),
        {
            "a.png": [poly("cat", [RING_A]), poly("dog", [RING_B])],
            "b.png": [poly("bird", [RING_B])],
            "c.png": None,
            "d.png": None,
            "e.png": None,
        },
        "yolo seg v1",
    )
    moved = [(10, 5), (40, 8), (51, 31), (20, 40)]
    rig.load([yaml, ("a.txt", yolo_poly(0, moved)), ("c.txt", yolo_poly(1, RING_B))])
    same_images(
        rig.stored(),
        {
            "a.png": [poly("cat", [moved])],
            "b.png": None,
            "c.png": [poly("dog", [RING_B])],
            "d.png": None,
            "e.png": None,
        },
        "yolo seg v2",
    )


def yolo_pose(cls: int, bbox: tuple[float, float, float, float], pts: list[tuple[float, float, int]]) -> str:
    """A YOLO pose line (box then ``x y v`` per point) from pixels.

    Args:
        cls: class index.
        bbox: ``(x, y, w, h)`` in pixels.
        pts: ``(x, y, v)`` in pixels.

    Returns:
        The line.
    """
    x, y, w, h = bbox
    head = f"{cls} {(x + w / 2) / W!r} {(y + h / 2) / H!r} {w / W!r} {h / H!r} "
    return head + " ".join(f"{px / W!r} {py / H!r} {v}" for px, py, v in pts) + "\n"


def test_yolo_pose_with_kpt_shape_import_and_edit(tmp_path):
    rig = make_rig(tmp_path, task="keypoints")
    yaml = ("data.yaml", "names: [cat, dog]\nkpt_shape: [3, 3]\n")
    p1 = [(10, 10, 2), (20, 30, 1), (0, 0, 0)]
    p2 = [(40, 5, 2), (50, 6, 2), (45, 40, 2)]
    box1, box2 = (5, 5, 30, 30), (30, 2, 25, 40)
    rig.load([yaml, ("a.txt", yolo_pose(0, box1, p1)), ("b.txt", yolo_pose(1, box2, p2))])
    same_images(
        rig.stored(),
        {
            "a.png": [pose("cat", box1, p1)],
            "b.png": [pose("dog", box2, p2)],
            "c.png": None,
            "d.png": None,
            "e.png": None,
        },
        "yolo pose v1",
    )
    p1_moved = [(10, 11, 2), (20, 30, 2), (0, 0, 0)]
    rig.load([yaml, ("a.txt", yolo_pose(0, box1, p1_moved)), ("c.txt", yolo_pose(1, box2, p2))])
    same_images(
        rig.stored(),
        {
            "a.png": [pose("cat", box1, p1_moved)],
            "b.png": None,
            "c.png": [pose("dog", box2, p2)],
            "d.png": None,
            "e.png": None,
        },
        "yolo pose v2",
    )


def test_yolo_pose_with_two_values_per_point_marks_unset_points_not_labelled(tmp_path):
    rig = make_rig(tmp_path, task="keypoints")
    line = "0 0.5 0.5 0.5 0.5 0.25 0.25 0 0 0.75 0.5\n"
    rig.load([("data.yaml", "names: [cat]\nkpt_shape: [3, 2]\n"), ("a.txt", line)])
    got = rig.stored()["a.png"][0].keypoints
    assert flat(got) == pytest.approx([0.25, 0.25, 2, 0, 0, 0, 0.75, 0.5, 2])


@pytest.mark.parametrize(
    "points",
    [
        1,
        2,
        3,
        4,
        5,
        6,
    ],
)
def test_yolo_pose_without_kpt_shape_is_read_as_pose_for_any_point_count(tmp_path, points):
    rig = make_rig(tmp_path, task="keypoints")
    pts = [(10 + i, 10 + 2 * i, 2) for i in range(points)]
    report = rig.check([("data.yaml", "names: [cat]"), ("a.txt", yolo_pose(0, (5, 5, 30, 30), pts))])
    assert report["inferred_task"] == "keypoints", (
        f"{points} points ({5 + 3 * points} values) were read as {report['inferred_task']}"
    )


# -- Pascal VOC ----------------------------------------------------------------------------------------
def test_voc_pixels_are_used_as_written_one_pixel_convention_is_not_applied(tmp_path):
    """The docstring promises "no 1-pixel shift": xmin=1 means 1/64, a 1-based tool would mean pixel 0."""
    rig = make_rig(tmp_path)
    xml = (
        f"<annotation><filename>a.png</filename><size><width>{W}</width><height>{H}</height></size>"
        f"<object><name>cat</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{W}</xmax><ymax>{H}</ymax>"
        "</bndbox></object></annotation>"
    )
    rig.load([("a.xml", xml)])
    assert flat(rig.stored()["a.png"][0].bbox) == pytest.approx(rel_box(1, 1, W - 1, H - 1))


VOC_OBJECT = (
    "<object><name>dog</name><pose>Left</pose><truncated>1</truncated><difficult>0</difficult>"
    "<bndbox><xmin>4</xmin><ymin>6</ymin><xmax>24</xmax><ymax>26</ymax></bndbox></object>"
)


def test_voc_filename_decides_the_image_and_extra_fields_are_ignored(tmp_path):
    rig = make_rig(tmp_path)
    rig.load([("one.xml", f"<annotation><filename>b.png</filename>{VOC_OBJECT}</annotation>")])
    same_images(
        rig.stored(),
        {"b.png": [box("dog", 4, 6, 20, 20)], **dict.fromkeys(("a.png", "c.png", "d.png", "e.png"))},
        "voc",
    )


def test_voc_without_filename_falls_back_to_the_xml_name(tmp_path):
    """The xml name is a fallback (``c.xml`` -> ``c``), but without extension ``c.png`` is not found."""
    rig = make_rig(tmp_path)
    report = rig.check([("c.xml", f"<annotation>{VOC_OBJECT}</annotation>")])
    assert report["counts"]["images_matched"] == 1, report["problems"]


# -- CVAT for images -----------------------------------------------------------------------------------
def cvat_doc(images: str, labels: str = "cat dog bird") -> str:
    """A CVAT for images 1.1 file.

    Args:
        images: the ``<image>`` elements.
        labels: label names declared in the meta, separated by spaces.

    Returns:
        The XML text.
    """
    declared = "".join(f"<label><name>{n}</name></label>" for n in labels.split())
    return (
        f"<annotations><version>1.1</version><meta><task><labels>{declared}</labels></task></meta>"
        f"{images}</annotations>"
    )


def cvat_image(name: str, shapes: str) -> str:
    """An ``<image>`` element of 64x48.

    Args:
        name: image file name.
        shapes: the shape elements.

    Returns:
        The XML text.
    """
    return f'<image id="0" name="{name}" width="{W}" height="{H}">{shapes}</image>'


def cvat_points(ring: list[tuple[float, float]]) -> str:
    """CVAT ``points`` attribute text.

    Args:
        ring: ``(x, y)`` in pixels.

    Returns:
        ``"x,y;x,y"``.
    """
    return ";".join(f"{x},{y}" for x, y in ring)


def test_cvat_polygon_import_and_edit(tmp_path):
    rig = make_rig(tmp_path, task="segmentation")
    shape = '<polygon label="{c}" source="manual" occluded="0" points="{p}" z_order="0"/>'
    v1 = cvat_doc(
        cvat_image(
            "a.png",
            shape.format(c="cat", p=cvat_points(RING_A)) + shape.format(c="dog", p=cvat_points(RING_B)),
        )
        + cvat_image("b.png", shape.format(c="bird", p="1.5,2.25;30.75,3;20,40.5"))
    )
    rig.load([("c.xml", v1)])
    same_images(
        rig.stored(),
        {
            "a.png": [poly("cat", [RING_A]), poly("dog", [RING_B])],
            "b.png": [poly("bird", [[(1.5, 2.25), (30.75, 3), (20, 40.5)]])],
            "c.png": None,
            "d.png": None,
            "e.png": None,
        },
        "cvat polygon v1",
    )
    moved = [(10, 5), (40, 8), (51, 31), (20, 40)]
    v2 = cvat_doc(cvat_image("a.png", shape.format(c="cat", p=cvat_points(moved))))
    rig.load([("c.xml", v2)])
    same_images(
        rig.stored(),
        {"a.png": [poly("cat", [moved])], "b.png": None, "c.png": None, "d.png": None, "e.png": None},
        "cvat polygon v2",
    )


def test_cvat_points_occluded_and_skeleton(tmp_path):
    rig = make_rig(tmp_path, task="keypoints")
    skeleton = (
        '<skeleton label="cat" occluded="0"><points label="nose" outside="0" occluded="0" points="10,10"/>'
        '<points label="left" outside="0" occluded="1" points="20,30"/>'
        '<points label="right" outside="1" occluded="0" points="0,0"/></skeleton>'
    )
    points = '<points label="dog" occluded="1" points="40,5;50,6;45,40"/>'
    rig.load([("c.xml", cvat_doc(cvat_image("a.png", skeleton) + cvat_image("b.png", points)))])
    got = rig.stored()
    assert flat(got["a.png"][0].keypoints) == pytest.approx(
        flat([[10 / W, 10 / H, 2], [20 / W, 30 / H, 1], [0, 0, 0]])
    )
    assert flat(got["a.png"][0].bbox) == pytest.approx(rel_box(10, 10, 10, 20))  # around the visible points
    assert flat(got["b.png"][0].keypoints) == pytest.approx(
        flat([[40 / W, 5 / H, 1], [50 / W, 6 / H, 1], [45 / W, 40 / H, 1]])
    )


def test_cvat_tags_are_image_classes(tmp_path):
    rig = make_rig(tmp_path, task="classification")
    xml = cvat_doc(
        cvat_image("a.png", '<tag label="cat" source="manual"/>')
        + cvat_image("b.png", '<tag label="dog" source="manual"/>')
        + cvat_image("c.png", "")
    )
    rig.load([("c.xml", xml)])
    got = rig.stored()
    assert [lb.label for lb in got["a.png"]] == ["cat"] and [lb.label for lb in got["b.png"]] == ["dog"]
    assert got["a.png"][0].bbox is None


def test_cvat_rotated_box_is_imported_as_its_enclosing_box(tmp_path):
    """A CVAT box with ``rotation`` has xtl/ytl/xbr/ybr of the UNROTATED rectangle (#359, #520)."""
    rig = make_rig(tmp_path)
    shape = (
        '<box label="cat" occluded="0" source="manual" xtl="10" ytl="10" xbr="30" ybr="20" '
        'rotation="90.0" z_order="0"/>'
    )
    rig.load([("c.xml", cvat_doc(cvat_image("a.png", shape)))])
    stored = flat(rig.stored()["a.png"][0].bbox)
    assert stored == pytest.approx(rel_box(15, 5, 10, 20))


def test_cvat_unsupported_shapes_and_unlabelled_shapes_are_reported_not_stored(tmp_path):
    rig = make_rig(tmp_path)
    shapes = (
        '<box label="cat" occluded="0" xtl="1" ytl="1" xbr="11" ybr="11"/>'
        '<polyline label="dog" points="1,1;5,5;9,1"/>'
        '<box occluded="0" xtl="2" ytl="2" xbr="12" ybr="12"/>'
    )
    report = rig.check([("c.xml", cvat_doc(cvat_image("a.png", shapes)))])
    assert {"unsupported_shape", "invalid_annotation"} <= {p["code"] for p in report["problems"]}
    rig.load([("c.xml", cvat_doc(cvat_image("a.png", shapes)))])
    same_images(rig.stored(), {"a.png": [box("cat", 1, 1, 10, 10)], **dict.fromkeys(NAMES[1:])}, "cvat")


# -- video formats: frame numbers ---------------------------------------------------------------------
TRACKS = (
    '<track id="1" label="car">'
    '<box frame="{f0}" outside="0" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="16" ybr="12"/>'
    '<box frame="{f1}" outside="0" occluded="0" keyframe="1" xtl="16" ytl="12" xbr="32" ybr="24"/>'
    '<box frame="{f2}" outside="1" occluded="0" keyframe="1" xtl="16" ytl="12" xbr="32" ybr="24"/></track>'
    '<track id="2" label="person">'
    '<box frame="{f1}" outside="0" occluded="0" keyframe="1" xtl="32" ytl="24" xbr="48" ybr="36"/>'
    '<box frame="{f2}" outside="0" occluded="0" keyframe="1" xtl="33" ytl="25" xbr="49" ybr="37"/></track>'
)
META = (
    "<meta><task><start_frame>10</start_frame><stop_frame>50</stop_frame><frame_filter>step=10"
    "</frame_filter><labels><label><name>car</name></label><label><name>person</name></label></labels>"
    "</task></meta>"
)
CAR1, CAR2 = box("car", 0, 0, 16, 12, 1), box("car", 16, 12, 16, 12, 1)
PERSON1, PERSON2 = box("person", 32, 24, 16, 12, 2), box("person", 33, 25, 16, 12, 2)


def cvat_video(meta: str, frames: tuple[int, int, int]) -> list[tuple[str, str]]:
    """A CVAT for video file with the tracks above at three frame numbers.

    Args:
        meta: the ``<meta>`` element.
        frames: the three frame numbers written in the file.

    Returns:
        ``[(file name, text)]``.
    """
    body = TRACKS.format(f0=frames[0], f1=frames[1], f2=frames[2])
    return [("video.xml", f"<annotations><version>1.1</version>{meta}{body}</annotations>")]


def test_cvat_video_task_numbered_frames_land_on_the_right_images(tmp_path):
    rig = make_video_rig(tmp_path)
    rig.load(cvat_video(META, (0, 1, 2)))  # task frames 0, 1, 2 = video frames 10, 20, 30
    same_images(
        rig.frames(),
        {10: [CAR1], 20: [CAR2, PERSON1], 30: [PERSON2], 40: [], 50: []},
        "cvat video task numbers",
    )


def test_cvat_video_video_numbered_frames_give_the_same_result_as_task_numbered(tmp_path):
    """CVAT's own export numbers by video frame (10, 20, 30) with the same meta."""
    rig = make_video_rig(tmp_path)
    rig.load(cvat_video(META, (10, 20, 30)))
    same_images(
        rig.frames(),
        {10: [CAR1], 20: [CAR2, PERSON1], 30: [PERSON2], 40: [], 50: []},
        "cvat video video numbers",
    )


def test_cvat_video_without_meta_frame_numbers_are_video_frames(tmp_path):
    rig = make_video_rig(tmp_path)
    report = rig.check(cvat_video("", (10, 20, 25)))  # 25 is not a frame of the task
    assert any(p["code"] == "frames_not_in_task" for p in report["problems"])
    rig.load(cvat_video("", (10, 20, 25)))
    same_images(rig.frames(), {10: [CAR1], 20: [CAR2, PERSON1], 30: None, 40: None, 50: None}, "cvat no meta")


def test_cvat_video_original_size_in_the_meta_scales_the_coordinates(tmp_path):
    rig = make_video_rig(tmp_path)
    meta = "<meta><task><original_size><width>128</width><height>96</height></original_size></task></meta>"
    xml = (
        f'<annotations>{meta}<track id="5" label="car"><box frame="10" outside="0" occluded="0" '
        'keyframe="1" xtl="32" ytl="24" xbr="96" ybr="72"/></track></annotations>'
    )
    rig.load([("v.xml", xml)])
    same_images(
        rig.frames(), {10: [box("car", 16, 12, 32, 24, 5)], 20: None, 30: None, 40: None, 50: None}, "size"
    )


MOT_V1 = (
    "11,1,6,5,20,10,1,1,1\n"  # frame 11 (1-based) = video frame 10
    "21,1,8,6,20,10,1,1,1\n"
    "21,2,30,20,16,12,1,2,1\n"
    "31,2,31,21,16,12,1,2,1\n"
)
MOT_LABELS = ("labels.txt", "car\nperson\n")


def test_mot_frames_are_one_based_and_land_on_the_right_images(tmp_path):
    rig = make_video_rig(tmp_path)
    rig.load([MOT_LABELS, ("gt.txt", MOT_V1)])
    same_images(
        rig.frames(),
        {
            10: [box("car", 6, 5, 20, 10, 1)],
            20: [box("car", 8, 6, 20, 10, 1), box("person", 30, 20, 16, 12, 2)],
            30: [box("person", 31, 21, 16, 12, 2)],
            40: None,
            50: None,
        },
        "mot v1",
    )


def test_mot_zero_based_file_does_not_land_on_neighbour_frames(tmp_path):
    """A file whose first frame is 0 (not MOT) must not be shifted silently onto other frames."""
    rig = make_video_rig(tmp_path)
    report = rig.check([MOT_LABELS, ("gt.txt", "0,1,6,5,20,10,1,1,1\n10,1,8,6,20,10,1,1,1\n")])
    codes = {p["code"] for p in report["problems"]}
    assert "invalid_annotation" in codes  # frame 0 cannot be read
    assert "frames_not_in_task" in codes  # frame 10 is video frame 9, not a task frame


def test_mot_edit_then_replace_scopes_to_the_task_and_merge_keeps_labelled_frames(tmp_path):
    rig = make_video_rig(tmp_path, tasks=2)
    other = [("frame_20.txt", "0 0.5 0.5 0.25 0.25\n"), ("classes.txt", "car\n")]
    rig.load(other, task=2)
    rig.load([MOT_LABELS, ("gt.txt", MOT_V1)], task=1)
    mot_v2 = (
        "11,1,7,5,20,10,1,1,1\n"  # moved
        "21,2,30,20,16,12,1,2,1\n"  # the car of frame 20 deleted
        "31,2,33,21,16,12,1,2,1\n"  # moved
        "41,3,0,0,10,10,1,1,1\n"  # added at video frame 40
    )
    rig.load([MOT_LABELS, ("gt.txt", mot_v2)], task=1, mode="replace")
    want = {
        10: [box("car", 7, 5, 20, 10, 1)],
        20: [box("person", 30, 20, 16, 12, 2)],
        30: [box("person", 33, 21, 16, 12, 2)],
        40: [box("car", 0, 0, 10, 10, 3)],
        50: None,
    }
    same_images(rig.frames(1), want, "mot v2 replace")
    assert len(rig.frames(2)[20]) == 1 and rig.frames(2)[20][0].label == "car"  # the other task is untouched
    # merge of v1 again: every frame with labels is kept, nothing duplicated
    summary = rig.load([MOT_LABELS, ("gt.txt", MOT_V1)], task=1, mode="merge")
    assert summary["images_added"] == 0
    same_images(rig.frames(1), want, "mot merge")


def test_yolo_per_frame_files_use_the_trailing_number_as_video_frame(tmp_path):
    rig = make_video_rig(tmp_path)
    rig.load(
        [
            ("classes.txt", "car\nperson\n"),
            ("frame_000010.txt", "0 0.25 0.25 0.5 0.5\n"),
            ("video_20.txt", "1 0.5 0.5 0.25 0.25\n"),
            ("clip_000035.txt", "0 0.5 0.5 0.25 0.25\n"),  # not a task frame
        ],
        task=1,
    )
    same_images(
        rig.frames(),
        {
            10: [Label("car", bbox=[0.0, 0.0, 0.5, 0.5])],
            20: [Label("person", bbox=[0.375, 0.375, 0.25, 0.25])],
            30: None,
            40: None,
            50: None,
        },
        "yolo frames",
    )


def test_coco_frames_use_frame_id_or_the_number_in_the_file_name(tmp_path):
    rig = make_video_rig(tmp_path)
    data = {
        "images": [
            {"id": 1, "file_name": "x.jpg", "frame_id": 10, "width": W, "height": H},
            {"id": 2, "file_name": "clip_0030.jpg", "width": W, "height": H},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [6.4, 4.8, 32, 24], "track_id": 7},
            {"id": 2, "image_id": 2, "category_id": 3, "bbox": [0, 0, 16, 12]},
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    rig.load([("v.json", json.dumps(data))], task=1)
    same_images(
        rig.frames(),
        {10: [box("car", 6.4, 4.8, 32, 24, 7)], 20: None, 30: [box("car", 0, 0, 16, 12)], 40: None, 50: None},
        "coco frames",
    )


def test_only_the_version_export_writes_annotation_files():
    """Round trips are import, edit, re-import: only the dataset version export writes COCO, YOLO or VOC."""
    root = Path(__file__).resolve().parents[3] / "src" / "activelabel"
    markers = ("write_coco", "to_coco", "export_coco", "write_yolo", "export_yolo", "write_voc", "export_voc")
    hits = [
        str(p)
        for p in root.rglob("*.py")
        if "exporting" not in p.parts and any(m in p.read_text(encoding="utf-8") for m in markers)
    ]
    assert hits == []
