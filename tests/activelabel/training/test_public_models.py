import importlib.util
import json
import textwrap
from pathlib import Path

import pytest
from api_client import upload
from sample_images import make_image
from training_fakes import tmp_dirs, wait_for

from activelabel.labels import load_annotations
from activelabel.training.public_models import (
    COCO_CATEGORIES,
    PUBLIC_MODELS,
    get_public,
    public_for,
    suggest_mapping,
)

FAKE_PREDICT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    Path(spec["items"][0]["path"]).with_name("public_spec.json").write_text(json.dumps(spec))
    mapping = spec["public"]["mapping"]
    for step, item in enumerate(spec["items"], start=1):
        box = [0.1, 0.1, 0.2, 0.3]
        labels = [{"label": mapping["person"], "bbox": box, "score": 0.8, "model": spec["model"]}]
        print(json.dumps({"event": "prediction", "id": item["id"], "labels": labels}), flush=True)
        print(json.dumps({"event": "progress", "step": step, "steps": len(spec["items"])}), flush=True)
    print(json.dumps({"event": "done", "results": {"images": len(spec["items"])}}), flush=True)
    """
)


def test_catalogue_and_suggestions():
    assert {m.task_kind for m in PUBLIC_MODELS} == {"detection", "keypoints"}
    detector = get_public("coco_fasterrcnn_mobilenet_v3")
    assert detector is not None and detector.categories == COCO_CATEGORIES and "N/A" not in detector.classes()
    assert len(detector.classes()) == 80 and detector.classes()[0] == "person"
    assert [m.id for m in public_for("detection")] == [
        m.id for m in PUBLIC_MODELS if m.task_kind == "detection"
    ]
    assert [m.task_kind for m in public_for("keypoints")] == ["keypoints"] and public_for(
        "classification"
    ) == []
    assert suggest_mapping(detector, ["Person", "traffic_light", "weld"]) == {
        "person": "Person",
        "traffic light": "traffic_light",
    }
    data = detector.as_dict()
    assert data["classes"][0] == "person" and "torchvision" in data["source"] and get_public("nope") is None


@pytest.fixture
def det(setup, tmp_path):
    """A detection project (person, car) with 3 images in one task, and the fake predict script."""
    client, state, runner = setup
    (tmp_path / "predict.py").write_text(FAKE_PREDICT)
    client.post("/api/projects", json={"name": "D", "task": "detection", "classes": ["person", "car"]})
    paths = [make_image(tmp_path / "det" / f"{k}.png", seed=50 + k, fmt="PNG") for k in range(3)]
    assert upload(client, 2, paths).status_code == 200
    return client, state, runner


def test_add_public_model_and_preannotate(det):
    client, state, runner = det
    catalog = client.get("/api/projects/2/models/public-catalog").json()
    assert catalog["project_classes"] == ["person", "car"]
    first = catalog["models"][0]
    assert first["id"] == "coco_fasterrcnn_mobilenet_v3" and first["suggested"] == {
        "person": "person",
        "car": "car",
    }
    assert (
        client.get("/api/projects/1/models/public-catalog").json()["models"] == []
    )  # classification project

    def post(**body):
        payload = {"public": first["id"], "mapping": {"person": "person"}, **body}
        return client.post("/api/projects/2/models/public", json=payload)

    assert post(public="nope").status_code == 404
    assert post(mapping={}).status_code == 400
    assert post(mapping={"unicorn": "person"}).json()["detail"].startswith("The model has no category")
    assert post(mapping={"person": "pedestrian"}).json()["detail"].startswith("The project has no class")
    keypoints = client.post(
        "/api/projects/2/models/public",
        json={"public": "coco_keypointrcnn_resnet50", "mapping": {"person": "person"}},
    )
    assert keypoints.status_code == 400  # keypoint labels: not for a detection project
    card = post(mapping={"person": "person", "truck": "car", "car": "car"}, note=" COCO ").json()
    assert card["number"] == 1 and card["current"] and card["source"] == "public" and card["note"] == "COCO"
    assert card["classes"] == ["person", "car"] and card["mapping"]["truck"] == "car" and card["run"] is None
    listed = client.get("/api/projects/2/models").json()["models"][0]
    assert listed["label"] == "Faster R-CNN MobileNetV3 (COCO)" and listed["results"] == {}
    assert client.get("/api/projects/2/models/1/download/pytorch").status_code == 404  # no local weights

    overview = client.get("/api/projects/2/preannotations").json()
    model = overview["models"][0]
    assert model["supported"] and model["public"] and model["label"].endswith("(COCO)")
    task = overview["tasks"][0]["id"]
    resp = client.post("/api/projects/2/preannotations", json={"model": 1, "task": task, "threshold": 0.3})
    assert resp.status_code == 200, resp.text
    run = wait_for(
        lambda: (
            (r := client.get("/api/projects/2/preannotations").json()["runs"][0])["status"]
            in ("done", "failed")
            and r
        )
    )
    assert run["status"] == "done", run
    assert run["counts"]["images"] == 3 and run["counts"]["objects"] == 3
    spec_file = next(Path(state.store.dir(2)).rglob("public_spec.json"))
    spec = json.loads(spec_file.read_text())
    assert "checkpoint" not in spec and spec["public"]["builder"] == "fasterrcnn_mobilenet_v3_large_fpn"
    assert spec["public"]["categories"][1] == "person" and spec["public"]["mapping"]["truck"] == "car"
    assert spec["model"] == "m1" and spec["task"] == "detection"
    assert runner.environment.ensured[-1][:2] == ["torch", "torchvision"]
    labels = next(iter(load_annotations(state.store.dir(2)).images.values()))
    assert labels[0].label == "person" and labels[0].model == "m1" and labels[0].score == 0.8
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written


def test_worker_skips_unmapped_categories():
    pytest.importorskip("torch", reason="the training workers import torch, absent from the app environment")
    path = Path(__file__).parents[3] / "src/activelabel/training/workers/predict.py"
    spec = importlib.util.spec_from_file_location("predict_public_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [None, "car", None]  # category 1 unmapped, 2 → car, 3 unmapped
    out = module.detection_labels(
        [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
        [1, 2, 3],
        [0.9, 0.9, 0.9],
        100,
        100,
        classes,
        0.5,
        "m1",
    )
    assert [o["label"] for o in out] == ["car"]


def test_public_model_has_no_export_or_download_and_can_leave_the_catalogue(det, monkeypatch):
    client, state, runner = det
    body = {"public": "coco_fasterrcnn_mobilenet_v3", "mapping": {"person": "person"}}
    assert client.post("/api/projects/2/models/public", json=body).status_code == 200
    export = client.post("/api/projects/2/models/1/export", json={"format": "onnx"})
    assert export.status_code == 400 and "public model" in export.json()["detail"]
    for kind in ("pytorch", "onnx"):
        download = client.get(f"/api/projects/2/models/1/download/{kind}")
        assert download.status_code == 404 and "public model" in download.json()["detail"]

    from activelabel.training import public_models

    monkeypatch.setattr(public_models, "PUBLIC_MODELS", ())  # a later version without this model
    model = client.get("/api/projects/2/preannotations").json()["models"][0]
    assert model["missing"] and not model["supported"]
    task = client.get("/api/projects/2/preannotations").json()["tasks"][0]["id"]
    start = client.post("/api/projects/2/preannotations", json={"model": 1, "task": task, "threshold": 0.3})
    assert start.status_code == 400 and "catalogue" in start.json()["detail"]


def test_threshold_is_checked_as_stored_and_model_delete_waits(det):
    client, state, runner = det
    body = {"public": "coco_fasterrcnn_mobilenet_v3", "mapping": {"person": "person"}}
    client.post("/api/projects/2/models/public", json=body)
    task = client.get("/api/projects/2/preannotations").json()["tasks"][0]["id"]
    for bad in (0.0004, 0.9996, 0, 1):
        r = client.post("/api/projects/2/preannotations", json={"model": 1, "task": task, "threshold": bad})
        assert r.status_code == 400, bad

    from activelabel.preannotate import PreannotationStore

    PreannotationStore(state.store.dir(2)).create(model=1, task=task, scope="unlabeled", threshold=0.3)
    delete = client.request("DELETE", "/api/projects/2/models/1", json={"confirm": True})
    assert delete.status_code == 409 and "pre-annotation" in delete.json()["detail"]
    assert client.get("/api/projects/2/models").json()["models"][0]["number"] == 1


def test_a_keypoint_model_with_another_skeleton_is_refused(det):
    from activelabel.labels import AnnotationSet, Label, save_annotations
    from activelabel.tasks import Task

    client, state, runner = det
    client.post("/api/projects", json={"name": "K", "task": "keypoints", "classes": ["person"]})
    paths = [make_image(tmp_path_of(state) / f"k{k}.png", seed=90 + k, fmt="PNG") for k in range(3)]
    assert upload(client, 3, paths).status_code == 200
    ids = [r.id for r in state.manifest(3).all()]
    points = [[0.2, 0.2, 2], [0.4, 0.4, 2], [0.6, 0.6, 2]]  # a 3-point skeleton, not COCO's 17
    labeled = {ids[0]: [Label("person", bbox=[0.1, 0.1, 0.8, 0.8], keypoints=points)]}
    save_annotations(state.store.dir(3), AnnotationSet(Task.KEYPOINTS, ["person"], labeled, {}))
    card = client.post(
        "/api/projects/3/models/public",
        json={"public": "coco_keypointrcnn_resnet50", "mapping": {"person": "person"}},
    ).json()
    assert card["num_keypoints"] == 17
    task = client.get("/api/projects/3/preannotations").json()["tasks"][0]["id"]
    client.post(
        "/api/projects/3/preannotations", json={"model": card["number"], "task": task, "threshold": 0.3}
    )
    run = wait_for(
        lambda: (
            (r := client.get("/api/projects/3/preannotations").json()["runs"][0])["status"]
            in ("done", "failed")
            and r
        )
    )
    assert run["status"] == "failed" and "17 keypoints" in run["error"] and "have 3" in run["error"]


def tmp_path_of(state):
    return state.store.root.parent / "kimg"


def test_suggestions_translate_portuguese_names_and_keep_exact_ones_first():
    detector = get_public("coco_fasterrcnn_mobilenet_v3")
    assert detector is not None
    suggested = suggest_mapping(detector, ["gato", "Cão", "pessoa", "carro", "solda"])
    assert suggested == {"person": "pessoa", "car": "carro", "cat": "gato", "dog": "Cão"}
    assert suggest_mapping(detector, ["gato", "cat"])["cat"] == "cat"
