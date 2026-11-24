import importlib.util
import json
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from training_fakes import tmp_dirs, wait_for

from activelabel import preannotate
from activelabel.annotate.sessions import select_images
from activelabel.datasets import DatasetOptions, plan_split
from activelabel.labels import Label, load_annotations, save_annotations
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore
from activelabel.training.config import RunConfig
from activelabel.training.runs import RunStore

FAKE_PREDICT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    (Path(spec["checkpoint"]).parent / "predict_spec.json").write_text(json.dumps(spec))
    if spec["threshold"] > 0.95:
        print(json.dumps({"event": "error", "message": "too strict"}), flush=True)
        sys.exit(1)
    for step, item in enumerate(spec["items"], start=1):
        labels = [] if step == 1 else [{"label": "a", "score": 0.9, "model": spec["model"]}]
        print(json.dumps({"event": "prediction", "id": item["id"], "labels": labels}), flush=True)
        print(json.dumps({"event": "progress", "step": step, "steps": len(spec["items"])}), flush=True)
    print(json.dumps({"event": "done", "results": {"images": len(spec["items"]) - 1}}), flush=True)
    """
)


@pytest.fixture
def pre(setup, tmp_path):
    """Project with 20 images: 6 without labels; a finished run saved as model m1."""
    client, state, runner = setup
    (tmp_path / "predict.py").write_text(FAKE_PREDICT)
    project_dir = state.store.dir(1)
    annotations = load_annotations(project_dir)
    ids = sorted(annotations.images)
    for image_id in ids[:6]:
        del annotations.images[image_id]
    save_annotations(project_dir, annotations)
    store = RunStore(project_dir)
    n = store.create(RunConfig(dataset=1, model="resnet18", params={"epochs": 2}, metric="accuracy"))[
        "number"
    ]
    runner.enqueue(1, n)
    wait_for(lambda: store.get(n)["status"] == "done")
    assert client.post("/api/projects/1/models", json={"run": n}).status_code == 200
    return client, state, runner, ids[:6]


def worker_module():
    path = Path(__file__).parents[3] / "src/activelabel/training/workers/predict.py"
    spec = importlib.util.spec_from_file_location("predict_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_label_conversion():
    w = worker_module()
    assert w.classification_labels([0.1, 0.7, 0.2], ["a", "b", "c"], 0.5, "m1") == [
        {"label": "b", "score": 0.7, "model": "m1"}
    ]
    assert w.classification_labels([0.4, 0.35, 0.25], ["a", "b", "c"], 0.5, "m1") == []
    out = w.detection_labels(
        [[10, 20, 60, 120], [0, 0, 5, 5], [90, 90, 300, 300]],
        [1, 2, 0],
        [0.9, 0.2, 0.99],
        200,
        100,
        ["car", "person"],
        0.5,
        "m2",
    )
    assert out == [
        {"label": "car", "bbox": [0.05, 0.2, 0.25, 0.8], "score": 0.9, "model": "m2"}
    ]  # clamped to the image
    kp = w.detection_labels(
        [[0, 0, 100, 100]], [1], [0.8], 100, 100, ["p"], 0.5, "m1", keypoints=[[[50, 50, 1], [10, 10, 0]]]
    )
    assert kp[0]["keypoints"] == [[0.5, 0.5, 2], [0.1, 0.1, 0]]


def test_rules_never_overwrite_people(pre):
    _, state, _, empty = pre
    project_dir, project = state.store.dir(1), state.project(1)
    task_id = TaskStore(project_dir).load()[0].id
    assert sorted(preannotate.targets(project_dir, task_id, "unlabeled")) == sorted(empty)
    assert preannotate.targets(project_dir, task_id, "unreviewed") == []
    labelled = sorted(load_annotations(project_dir).images)[0]
    predicted = {i: [Label("a", score=0.8, model="m1")] for i in empty[:4]}
    predicted[labelled] = [Label("b", score=0.99, model="m1")]  # a person's image: left alone
    predicted[empty[4]] = []  # the model found nothing
    predicted[empty[5]] = [Label("zebra", score=0.9, model="m1")]  # a class the project does not have
    counts = preannotate.apply_predictions(project_dir, project, predicted, "unlabeled")
    assert counts == {
        "images": 4,
        "objects": 4,
        "skipped": 1,
        "empty": 2,
        "dropped_class": 1,
        "dropped_geometry": 0,
    }
    images = load_annotations(project_dir).images
    assert images[labelled][0].score is None and empty[4] not in images and empty[5] not in images
    assert images[empty[0]][0].as_dict() == {"label": "a", "score": 0.8, "model": "m1"}
    assert sorted(preannotate.targets(project_dir, task_id, "unreviewed")) == sorted(empty[:4])
    counts_by_task = preannotate.task_counts(project_dir)[0]
    assert (counts_by_task["unlabeled"], counts_by_task["unreviewed"], counts_by_task["models"]) == (
        2,
        4,
        ["m1"],
    )
    # a dataset version ignores pre-annotations: those images count as unlabelled
    plan = plan_split(project_dir, project, DatasetOptions())
    used = {i for part in plan.parts.values() for i in part}
    assert not used & set(empty[:4])
    # review scope for FiftyOne / CVAT, then accept
    assert sorted(select_images(state.manifest(1), "preannotated")) == sorted(empty[:4])
    assert preannotate.accept(project_dir, empty[:2]) == 2
    assert load_annotations(project_dir).images[empty[0]][0].as_dict() == {"label": "a"}
    assert sorted(preannotate.unreviewed_images(project_dir)) == sorted(empty[2:4])
    # redoing unreviewed images replaces only pre-annotations
    redo = {empty[2]: [Label("b", score=0.7, model="m2")], empty[0]: [Label("b", score=0.7, model="m2")]}
    assert preannotate.apply_predictions(project_dir, project, redo, "unreviewed")["skipped"] == 1
    assert load_annotations(project_dir).images[empty[2]][0].model == "m2"
    with pytest.raises(ValueError, match="images must be"):
        preannotate.targets(project_dir, task_id, "everything")
    with pytest.raises(ValueError, match="task does not exist"):
        preannotate.targets(project_dir, 99, "unlabeled")


def test_preannotate_end_to_end(pre):
    client, state, runner, empty = pre
    overview = client.get("/api/projects/1/preannotations").json()
    assert overview["models"][0]["number"] == 1 and overview["models"][0]["supported"]
    assert overview["tasks"][0]["unlabeled"] == 6 and overview["runs"] == []
    task_id = overview["tasks"][0]["id"]
    resp = client.post("/api/projects/1/preannotations", json={"model": 1, "task": task_id, "threshold": 0.4})
    assert resp.status_code == 200, resp.text
    n = resp.json()["number"]
    run = wait_for(
        lambda: (
            (r := client.get("/api/projects/1/preannotations").json()["runs"][0])["status"] == "done" and r
        )
    )
    assert (
        run["counts"]
        == {
            "images": 5,
            "objects": 5,
            "skipped": 0,
            "empty": 1,
            "dropped_class": 0,
            "dropped_geometry": 0,
            "unreadable": 0,
            "removed_classes": [],
        }
        and run["done"] == 6
    )
    assert run["model_label"] == "ResNet-18" and run["scope"] == "unlabeled" and run["threshold"] == 0.4
    spec = json.loads((state.store.dir(1) / "models" / "m1" / "predict_spec.json").read_text())
    assert spec["model"] == "m1" and spec["classes"] == ["a", "b"] and spec["worker"] == "fake_worker"
    assert sorted(i["id"] for i in spec["items"]) == sorted(empty) and spec["threshold"] == 0.4
    assert "torch" in runner.environment.ensured[-1]
    assert (state.store.dir(1) / "preannotations" / f"p{n}.log").exists()
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    tasks = client.get("/api/projects/1/preannotations").json()["tasks"][0]
    assert (tasks["unlabeled"], tasks["unreviewed"]) == (1, 5)
    # nothing left to do for unlabeled images but one: the model found nothing there
    again = client.post(
        "/api/projects/1/preannotations",
        json={"model": 1, "task": task_id, "scope": "unreviewed", "threshold": 0.99},
    )
    failed = wait_for(
        lambda: (
            (r := client.get("/api/projects/1/preannotations").json()["runs"][0])["number"]
            == again.json()["number"]
            and r["status"] == "failed"
            and r
        )
    )
    assert failed["error"] == "too strict"
    assert (
        client.get("/api/projects/1/preannotations").json()["tasks"][0]["unreviewed"] == 5
    )  # nothing written
    assert client.post("/api/projects/1/preannotations/accept", json={"task": task_id}).json() == {
        "accepted": 5
    }
    assert client.get("/api/projects/1/preannotations").json()["tasks"][0]["unreviewed"] == 0


def test_api_validation_and_cancel(pre):
    client, state, runner, _ = pre
    task_id = client.get("/api/projects/1/preannotations").json()["tasks"][0]["id"]
    post = lambda **b: client.post("/api/projects/1/preannotations", json={"model": 1, "task": task_id, **b})  # noqa: E731
    assert post(model=9).status_code == 404
    assert post(threshold=0).status_code == 400 and post(threshold=1).status_code == 400
    assert post(scope="all").status_code == 400
    assert post(scope="unreviewed").json()["detail"] == "No image of this task matches this choice."
    assert client.post("/api/projects/1/preannotations", json={"model": 1, "task": 99}).status_code == 400
    assert client.post("/api/projects/1/preannotations/5/cancel").status_code == 404
    store = preannotate.PreannotationStore(state.store.dir(1))
    queued = store.create(model=1, task=task_id, scope="unlabeled", threshold=0.5)
    assert (
        client.post(f"/api/projects/1/preannotations/{queued['number']}/cancel").json()["status"]
        == "cancelled"
    )
    store.update(queued["number"], status="running")
    runner.recover()
    assert (
        store.get(queued["number"])["status"] == "failed"
        and store.get(queued["number"])["error"] == "interrupted"
    )


def test_review_fixes(pre):
    """Regressions found in review: empty reviews, deleted images, project type, duplicates, counts."""
    client, state, runner, empty = pre
    project_dir, project = state.store.dir(1), state.project(1)
    task_id = TaskStore(project_dir).load()[0].id
    # a person reviewed an image and found nothing: it is not "without annotations" any more
    annotations = load_annotations(project_dir)
    annotations.images[empty[0]] = []
    save_annotations(project_dir, annotations)
    assert empty[0] not in preannotate.targets(project_dir, task_id, "unlabeled")
    counts = preannotate.apply_predictions(
        project_dir, project, {empty[0]: [Label("a", score=0.9, model="m1")]}, "unlabeled"
    )
    assert counts["skipped"] == 1 and load_annotations(project_dir).images[empty[0]] == []
    # an image deleted while the run went is skipped
    state.manifest(1).delete([empty[1]])
    counts = preannotate.apply_predictions(
        project_dir, project, {empty[1]: [Label("a", score=0.9, model="m1")]}, "unlabeled"
    )
    assert counts["skipped"] == 1 and empty[1] not in load_annotations(project_dir).images
    # counts: reviewed images apart from images waiting for review
    preannotate.apply_predictions(
        project_dir, project, {empty[2]: [Label("b", score=0.8, model="m1")]}, "unlabeled"
    )
    overview = client.get("/api/projects/1").json()["counts"]
    assert overview["preannotated"] == 1 and overview["annotated"] == 15  # 14 labelled + the empty review
    summary = client.get("/api/projects/1/annotations").json()
    assert summary["images_preannotated"] == 1 and summary["images_annotated"] == 15
    # one active run per task and scope
    store = preannotate.PreannotationStore(project_dir)
    store.create(model=1, task=task_id, scope="unlabeled", threshold=0.5, status="running")
    again = client.post("/api/projects/1/preannotations", json={"model": 1, "task": task_id})
    assert again.status_code == 409
    # a model of another project type is not offered
    state.store.update(1, task="detection")
    model = client.get("/api/projects/1/preannotations").json()["models"][0]
    assert model["supported"] is False
    resp = client.post(
        "/api/projects/1/preannotations", json={"model": 1, "task": task_id, "scope": "unreviewed"}
    )
    assert resp.status_code == 400 and "does not use" in resp.json()["detail"]


def test_no_readable_image_fails_readably(pre):
    client, state, runner, empty = pre
    manifest = state.manifest(1)
    records = manifest.load()
    for image_id in empty:  # the files of every image waiting for pre-annotation are gone
        manifest.file_path(records[image_id]).unlink()
    task_id = client.get("/api/projects/1/preannotations").json()["tasks"][0]["id"]
    resp = client.post("/api/projects/1/preannotations", json={"model": 1, "task": task_id})
    assert resp.status_code == 200, resp.text
    failed = wait_for(
        lambda: (
            (r := client.get("/api/projects/1/preannotations").json()["runs"][0])["status"] == "failed" and r
        )
    )
    assert failed["error"] == "None of the images of this task could be read."
    wait_for(lambda: not runner.has_active(1))
    assert runner._pre_jobs == {} and tmp_dirs(runner) == []


def test_masks_become_polygon_labels():
    import numpy as np

    w = worker_module()
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 10:30] = True  # region 1
    mask[5:35, 50:75] = True  # region 2
    mask[0, 79] = True  # a speck: noise
    encoded = w.encode_mask(mask)
    assert (preannotate.decode_mask(encoded) == mask).all()
    assert (preannotate.decode_mask(w.encode_mask(~mask)) == ~mask).all()
    rings = preannotate.mask_polygons(mask)
    assert len(rings) == 2 and all(len(r) >= 4 for r in rings)
    xs = sorted(p[0] for p in rings[0] + rings[1])
    assert xs[0] == pytest.approx(10.5 / 80) and xs[-1] == pytest.approx(74.5 / 80)

    instance = w.detection_labels([[10, 5, 75, 35]], [1], [0.9], 80, 40, ["a"], 0.5, "m1", None, [mask])
    (one,) = preannotate.mask_labels(instance)
    assert one.label == "a" and one.score == 0.9 and len(one.polygons) == 2 and len(one.bbox) == 4
    assert one.bbox[0] == pytest.approx(10.5 / 80) and one.bbox[2] == pytest.approx(64 / 80)

    classes_map = np.where(mask, 1, 0)
    semantic = w.semantic_labels(classes_map, np.full(mask.shape, 0.8), ["a", "b"], 0.5, "m2")
    assert [x["label"] for x in semantic] == ["a"] and semantic[0]["score"] == 0.8
    assert w.semantic_labels(classes_map, np.full(mask.shape, 0.3), ["a"], 0.5, "m2") == []
    objects = preannotate.mask_labels(semantic)
    assert len(objects) == 2 and all(len(o.polygons) == 1 and o.bbox for o in objects)  # one per region
    assert preannotate.mask_labels([{"label": "a", "mask": w.encode_mask(np.zeros((9, 9), bool))}]) == []
    with pytest.raises(ValueError):
        preannotate.decode_mask({"size": [3, 3], "counts": [0, 4]})
    assert set(preannotate.KINDS) >= {"instance", "semantic"}
    assert preannotate.fits("semantic", Task.SEGMENTATION) and preannotate.fits("instance", Task.SEGMENTATION)


def test_only_current_classes_and_the_task_shape_are_written(pre):
    client, state, runner, empty = pre
    project_dir, project = state.store.dir(1), state.project(1)
    annotations = load_annotations(project_dir)
    annotations.classes = [*annotations.classes, "old"]  # a class the project no longer has
    save_annotations(project_dir, annotations)
    box = Label("a", bbox=[0.1, 0.1, 0.2, 0.2], score=0.9, model="m1")  # a box is not a classification
    counts = preannotate.apply_predictions(
        project_dir,
        project,
        {empty[0]: [Label("old", score=0.9, model="m1"), Label("a", score=0.9, model="m1")], empty[1]: [box]},
        "unlabeled",
    )
    assert counts["objects"] == 2 and counts["dropped_class"] == 1 and counts["dropped_geometry"] == 0
    detection = replace(project, task=Task.DETECTION)
    bare = Label("a", score=0.9, model="m1")  # no box
    counts = preannotate.apply_predictions(project_dir, detection, {empty[2]: [bare, box]}, "unlabeled")
    assert counts["objects"] == 1 and counts["dropped_geometry"] == 1
    assert [lb.label for lb in load_annotations(project_dir).images[empty[0]]] == ["a"]


def test_nothing_is_written_when_the_project_changed_during_the_run(pre, monkeypatch):
    client, state, runner, empty = pre
    task_id = TaskStore(state.store.dir(1)).load()[0].id
    original = runner._run_export

    def changing(*args, **kwargs):
        original(*args, **kwargs)
        state.store.update(1, classes=["a"])  # a class was removed while the model ran

    monkeypatch.setattr(runner, "_run_export", changing)
    resp = client.post("/api/projects/1/preannotations", json={"model": 1, "task": task_id, "threshold": 0.4})
    run = wait_for(
        lambda: (
            (r := client.get("/api/projects/1/preannotations").json()["runs"][0])["number"]
            == resp.json()["number"]
            and r["status"] in ("done", "failed")
            and r
        )
    )
    assert run["status"] == "failed" and "nothing was written" in run["error"]
    assert not [i for i in load_annotations(state.store.dir(1)).images if i in empty]


def _finished(client, number):
    return wait_for(
        lambda: (
            (r := client.get("/api/projects/1/preannotations").json()["runs"][0])["number"] == number
            and r["status"] in ("done", "failed")
            and r
        )
    )


def test_classes_the_project_lost_are_reported_or_refuse_the_run(pre):
    client, state, runner, empty = pre
    task_id = TaskStore(state.store.dir(1)).load()[0].id
    body = {"model": 1, "task": task_id, "threshold": 0.4}
    state.store.update(1, classes=["a"])  # the model also knows "b"
    run = _finished(client, client.post("/api/projects/1/preannotations", json=body).json()["number"])
    assert run["status"] == "done" and run["counts"]["removed_classes"] == ["b"]
    state.store.update(1, classes=["c"])  # nothing of the model is left
    body["scope"] = "unreviewed"
    run = _finished(client, client.post("/api/projects/1/preannotations", json=body).json()["number"])
    assert run["status"] == "failed" and "None of the classes of this model" in run["error"]
