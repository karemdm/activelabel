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
    client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"})
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
    assert client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}).status_code == 200
    assert client.get(f"{RUNS}/1").json()["config"]["fold"] == "all"
    assert (
        client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn", "fold": 1}).status_code
        == 200
    )
    assert (
        client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn", "fold": 3}).status_code
        == 400
    )


def test_cancel_queued_run_and_unknown(tmp_path):
    client, _ = _setup(tmp_path)
    client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"})
    assert client.post(f"{RUNS}/1/cancel").json()["status"] == "cancelled"
    assert client.post(f"{RUNS}/7/cancel").status_code == 404


def test_log_tail_metric_and_environment(tmp_path, monkeypatch):
    from activelabel.training import environment

    client, state = _setup(tmp_path)
    run = client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}).json()
    assert run["metric"] == "map50_95"  # filled with the default of the model's kind
    folder = RunStore(state.store.dir(1)).path(1)
    (folder / "log.txt").write_text("\n".join(f"line {k}" for k in range(300)), encoding="utf-8")
    assert client.get(f"{RUNS}/1/log", params={"lines": 3}).json()["lines"] == [
        "line 297",
        "line 298",
        "line 299",
    ]
    assert client.get(f"{RUNS}/9/log").status_code == 404
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu", lambda self: None)
    monkeypatch.setattr(environment.TrainingEnvironment, "exists", lambda self: False)
    body = client.get("/api/training/environment").json()
    assert body == {"exists": False, "packages": {}, "gpu": None}


def test_a_damaged_run_does_not_hide_the_others(tmp_path):
    client, state = _setup(tmp_path)
    for _ in range(2):
        client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn", "params": {"epochs": 1}})
    (RunStore(state.store.dir(1)).path(1) / "config.yaml").unlink()
    runs = client.get(RUNS).json()["runs"]
    assert [r["number"] for r in runs] == [2, 1]
    assert runs[1]["config_error"] == "config_missing" and runs[0]["model"] == "fasterrcnn_resnet50_fpn"
    assert "config.yaml" not in runs[1]["config_message"] and "\\" not in runs[1]["config_message"]
    assert str(state.store.dir(1)) not in str(runs[1])
    (RunStore(state.store.dir(1)).path(1) / "config.yaml").write_text("{not: [valid", encoding="utf-8")
    damaged = client.get(RUNS).json()["runs"][1]
    assert damaged["config_error"] == "config_damaged" and str(state.store.dir(1)) not in str(damaged)
    one = client.get(f"{RUNS}/1")
    assert one.status_code == 200 and one.json()["config"] is None
    assert client.post(f"{RUNS}/1/reproduce").status_code == 409


def test_log_tail_reads_only_the_end(tmp_path):
    import random

    from activelabel.web.training_api import _tail

    rng = random.Random(7)
    for case in range(60):
        lines = ["x" * rng.randint(0, 40) + f" {k} ção" for k in range(rng.randint(0, 120))]
        path = tmp_path / f"log{case}.txt"
        path.write_text("\n".join(lines) + ("\n" if rng.random() < 0.5 else ""), encoding="utf-8")
        for count in (1, 3, 50, 2000):
            assert _tail(path, count, chunk=rng.randint(1, 64)) == lines[-count:], (case, count)
    assert _tail(tmp_path / "missing.txt", 5) == []


def test_a_version_without_training_images_is_refused(tmp_path):
    import json

    from activelabel.datasets import DatasetStore

    client, state = _setup(tmp_path)
    path = DatasetStore(state.store.dir(1)).dir / "v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["parts"]["train"] = []  # e.g. every training image was excluded by the split options
    path.write_text(json.dumps(data), encoding="utf-8")
    response = client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"})
    assert response.status_code == 400 and "training part" in response.json()["detail"]
