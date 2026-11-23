from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training.runs import RunStore

RUNS = "/api/projects/1/runs"


def _setup(tmp_path, kfold=None):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a", "b"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(20)]
    assert upload(client, 1, paths).status_code == 200
    ids = [r.id for r in state.manifest(1).all()]
    images = {i: [Label("a" if k % 3 else "b", bbox=[0, 0, 0.5, 0.5])] for k, i in enumerate(ids)}
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["a", "b"], images, {}))
    assert client.post("/api/projects/1/datasets", json={"kfold": kfold}).status_code == 200
    return client, state


def test_catalog(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/api/training/catalog", params={"type": "segmentation"}).json()
    assert {m["task_kind"] for m in body["models"]} == {"semantic", "instance"}
    assert all(m["params"] and m["packages"] for m in body["models"])
    assert body["metrics"]["semantic"]["default"] == "miou"
    assert len(client.get("/api/training/catalog").json()["models"]) > len(body["models"])
    assert client.get("/api/training/catalog", params={"type": "x"}).status_code == 400


def test_create_list_get_reproduce_delete(tmp_path):
    client, _ = _setup(tmp_path)
    created = client.post(
        RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn", "params": {"epochs": 2}}
    )
    assert created.status_code == 200, created.text
    run = created.json()
    assert run["number"] == 1 and run["status"] == "queued" and run["model"] == "fasterrcnn_resnet50_fpn"
    assert [r["number"] for r in client.get(RUNS).json()["runs"]] == [1]

    one = client.get(f"{RUNS}/1").json()
    assert one["config"]["params"] == {"epochs": 2} and one["config"]["metric"] == "map50_95"
    assert one["config"]["dataset_fingerprint"] and one["metrics"] == [] and one["results"] is None

    again = client.post(f"{RUNS}/1/reproduce")
    assert again.status_code == 200 and again.json()["number"] == 2
    assert client.get(f"{RUNS}/2").json()["config"] == one["config"]

    assert client.request("DELETE", f"{RUNS}/1").status_code == 400
    assert client.request("DELETE", f"{RUNS}/1", json={"confirm": True}).json() == {"deleted": True}
    assert client.get(f"{RUNS}/1").status_code == 404
    assert client.post(f"{RUNS}/9/reproduce").status_code == 404


def test_running_run_is_not_deleted(tmp_path):
    client, state = _setup(tmp_path)
    client.post(RUNS, json={"dataset": 1, "model": "yolov8n"})
    RunStore(state.store.dir(1)).update(1, status="running")
    assert client.request("DELETE", f"{RUNS}/1", json={"confirm": True}).status_code == 409


def test_readable_errors(tmp_path):
    client, _ = _setup(tmp_path)
    cases = [
        ({"dataset": 1, "model": "resnet18"}, 400, "not for detection"),
        ({"dataset": 1, "model": "yolov8n", "params": {"epochs": 0}}, 400, "at least"),
        ({"dataset": 1, "model": "yolov8n", "device": "cuda"}, 400, "Unknown option"),
        ({"dataset": 1, "model": "yolov8n", "fold": 0}, 400, "no folds"),
        ({"dataset": 5, "model": "yolov8n"}, 404, "does not exist"),
        ({"model": "yolov8n"}, 400, "dataset"),
    ]
    for body, status, text in cases:
        response = client.post(RUNS, json=body)
        assert response.status_code == status and text in response.json()["detail"], body
    assert client.get(RUNS).json() == {"runs": []}
    assert client.get("/api/projects/9/runs").status_code == 404


def test_fold_only_for_kfold_versions(tmp_path):
    client, _ = _setup(tmp_path, kfold=3)
    assert client.post(RUNS, json={"dataset": 1, "model": "yolov8n"}).status_code == 200
    assert client.get(f"{RUNS}/1").json()["config"]["fold"] == "all"
    assert client.post(RUNS, json={"dataset": 1, "model": "yolov8n", "fold": 1}).status_code == 200
    assert client.post(RUNS, json={"dataset": 1, "model": "yolov8n", "fold": 3}).status_code == 400


def test_cancel_queued_run_and_unknown(tmp_path):
    client, _ = _setup(tmp_path)
    client.post(RUNS, json={"dataset": 1, "model": "yolov8n"})
    assert client.post(f"{RUNS}/1/cancel").json()["status"] == "cancelled"
    assert client.post(f"{RUNS}/7/cancel").status_code == 404
