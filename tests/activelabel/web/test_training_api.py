import json
import random

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.datasets import DatasetStore
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training import environment
from activelabel.training.config import RunConfig
from activelabel.training.runs import RunStore
from activelabel.web.training_api import _tail

PINS = environment.PINNED_VERSIONS  # torch and torchvision count as installed only at these versions

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


def test_augmentation_catalog(tmp_path):
    client, _ = make_client(tmp_path)
    body = client.get("/api/training/augmentations").json()
    assert body["levels"] == ["none", "light", "standard", "strong", "custom"]
    assert {"classification", "keypoints"} <= set(body["kinds"])
    assert {"techniques", "presets"} == set(body["kinds"]["detection"])
    only = client.get("/api/training/augmentations", params={"type": "keypoints"}).json()
    assert list(only["kinds"]) == ["keypoints"]
    bad = client.get("/api/training/augmentations", params={"type": "bad"})
    assert bad.status_code == 400 and bad.json()["code"] == "type_invalid"


def test_create_list_get_reproduce_delete(tmp_path):
    client, _ = _setup(tmp_path)
    created = client.post(
        RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn", "params": {"epochs": 2}}
    )
    assert created.status_code == 200, created.text
    run = created.json()
    assert run["number"] == 1 and run["status"] == "queued" and run["model"] == "fasterrcnn_resnet50_fpn"
    assert run["augmentations"] is None
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
        ({"dataset": 1, "model": "yolov8n", "gpu": "cuda"}, 400, "Unknown option"),
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
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu_count", lambda self: 0)
    monkeypatch.setattr(environment.TrainingEnvironment, "mps_hint", staticmethod(lambda: False))
    body = client.get("/api/training/environment").json()
    assert body == {"exists": False, "packages": {}, "gpu": None, "devices": ["auto", "cpu"]}
    first = client.get("/api/training/environment", params={"model": "resnet18"}).json()
    assert first["install_needed"] is True and {"torch", "torchvision", "pillow"} <= set(first["missing"])
    assert client.get("/api/training/environment", params={"model": "nope"}).status_code == 404
    monkeypatch.setattr(environment.TrainingEnvironment, "exists", lambda self: True)
    monkeypatch.setattr(
        environment.TrainingEnvironment, "installed", lambda self: {"torch": PINS["torch"], "pillow": "1"}
    )
    some = client.get("/api/training/environment", params={"model": "resnet18"}).json()
    assert some["missing"] == ["torchvision", "numpy"] and some["install_needed"] is True
    monkeypatch.setattr(
        environment.TrainingEnvironment,
        "installed",
        lambda self: {
            "torch": PINS["torch"],
            "torchvision": PINS["torchvision"],
            "pillow": "1",
            "numpy": "1",
        },
    )
    assert (
        client.get("/api/training/environment", params={"model": "resnet18"}).json()["install_needed"]
        is False
    )


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
    rng = random.Random(7)
    for case in range(60):
        lines = ["x" * rng.randint(0, 40) + f" {k} ção" for k in range(rng.randint(0, 120))]
        path = tmp_path / f"log{case}.txt"
        path.write_text("\n".join(lines) + ("\n" if rng.random() < 0.5 else ""), encoding="utf-8")
        for count in (1, 3, 50, 2000):
            assert _tail(path, count, chunk=rng.randint(1, 64)) == lines[-count:], (case, count)
    assert _tail(tmp_path / "missing.txt", 5) == []


def test_a_version_without_training_images_is_refused(tmp_path):
    client, state = _setup(tmp_path)
    path = DatasetStore(state.store.dir(1)).dir / "v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["parts"]["train"] = []  # e.g. every training image was excluded by the split options
    path.write_text(json.dumps(data), encoding="utf-8")
    response = client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"})
    assert response.status_code == 400 and "training part" in response.json()["detail"]


def test_deleting_a_version_checks_the_runs_that_use_it(tmp_path):
    client, state = _setup(tmp_path)
    store = RunStore(state.store.dir(1))
    queued = client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}).json()["number"]
    path = "/api/projects/1/datasets/1"
    blocked = client.request("DELETE", path, json={"confirm": True, "confirm_runs": True})
    assert blocked.status_code == 409 and f"r{queued}" in blocked.json()["detail"]
    store.update(queued, status="done")  # the run finished
    asked = client.request("DELETE", path, json={"confirm": True})
    assert asked.status_code == 409 and f"r{queued}" in asked.json()["detail"]
    assert client.get(path).status_code == 200
    assert client.request("DELETE", path, json={"confirm": True, "confirm_runs": True}).status_code == 200


def test_a_version_with_classes_the_project_lost_cannot_train(tmp_path):
    client, state = _setup(tmp_path)
    state.store.update(1, classes=["a"])  # the version has "a" and "b"
    refused = client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"})
    assert refused.status_code == 400 and refused.json()["code"] == "dataset_classes_removed"
    assert refused.json()["params"]["classes"] == ["b"]
    state.store.update(1, classes=["a", "b", "c"])  # a class that was only added is fine
    assert client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}).status_code == 200


def test_reading_the_environment_writes_nothing(tmp_path, monkeypatch):
    """GET /training/environment only reads: no env.json appears, even for a folder outside the workspace."""
    monkeypatch.setattr(environment, "IS_WINDOWS", True)
    monkeypatch.setattr(environment, "MAX_PATH_LENGTH", 0)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu", lambda self: None)
    client, state = make_client(tmp_path)
    assert client.get("/api/training/environment").status_code == 200
    assert not (state.settings.workspace / "training").exists()


BODY = {"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}


def _gpus(monkeypatch, count=0, mps=False):
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu_count", lambda self: count)
    monkeypatch.setattr(environment.TrainingEnvironment, "mps_hint", staticmethod(lambda: mps))


def _boom(self):
    raise AssertionError("the computer must not be looked at")


def test_a_run_stores_the_device_it_asked_for(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu_count", _boom)
    client, state = _setup(tmp_path)
    assert client.post(RUNS, json={**BODY, "device": "cpu"}).status_code == 200
    assert client.post(RUNS, json=BODY).status_code == 200
    assert client.post(RUNS, json={**BODY, "device": None}).status_code == 200
    store = RunStore(state.store.dir(1))
    assert [store.get(n)["device_choice"] for n in (1, 2, 3)] == ["cpu", "auto", "auto"]
    assert "device" not in client.get(f"{RUNS}/1").json()["config"]  # not part of the configuration


def test_a_gpu_that_exists_is_accepted(tmp_path, monkeypatch):
    _gpus(monkeypatch, count=2)
    client, state = _setup(tmp_path)
    assert client.post(RUNS, json={**BODY, "device": "cuda:1"}).status_code == 200
    assert RunStore(state.store.dir(1)).get(1)["device_choice"] == "cuda:1"


def test_a_device_this_computer_lacks_is_refused_and_nothing_is_queued(tmp_path, monkeypatch):
    _gpus(monkeypatch, count=1)
    client, state = _setup(tmp_path)
    for device in ("cuda:7", "mps"):
        refused = client.post(RUNS, json={**BODY, "device": device})
        assert refused.status_code == 400 and refused.json()["code"] == "device_unavailable", refused.text
    assert client.get(RUNS).json()["runs"] == []
    assert RunStore(state.store.dir(1)).list() == []


@pytest.mark.parametrize("device", ["bogus", "gpu1", 5])
def test_a_device_that_is_not_a_device_is_refused(tmp_path, monkeypatch, device):
    _gpus(monkeypatch)
    client, _ = _setup(tmp_path)
    refused = client.post(RUNS, json={**BODY, "device": device})
    assert refused.status_code == 400 and refused.json()["code"] == "device_invalid", refused.text
    assert client.get(RUNS).json()["runs"] == []


def test_reproducing_keeps_the_device_choice(tmp_path, monkeypatch):
    _gpus(monkeypatch, count=1)
    client, state = _setup(tmp_path)
    assert client.post(RUNS, json={**BODY, "device": "cuda:0"}).status_code == 200
    assert client.post(RUNS, json=BODY).status_code == 200
    assert client.post(f"{RUNS}/1/reproduce").json()["number"] == 3
    assert client.post(f"{RUNS}/2/reproduce").json()["number"] == 4
    store = RunStore(state.store.dir(1))
    assert [store.get(n)["device_choice"] for n in (3, 4)] == ["cuda:0", "auto"]


def test_reproducing_a_run_without_a_stored_device_uses_auto(tmp_path, monkeypatch):
    _gpus(monkeypatch)
    client, state = _setup(tmp_path)
    client.post(RUNS, json=BODY)
    store = RunStore(state.store.dir(1))
    store.update(1, device_choice=None)
    assert client.post(f"{RUNS}/1/reproduce").status_code == 200
    assert store.get(2)["device_choice"] == "auto"


def test_the_environment_lists_the_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.TrainingEnvironment, "gpu", lambda self: None)
    monkeypatch.setattr(environment.TrainingEnvironment, "exists", lambda self: False)
    client, _ = make_client(tmp_path)
    _gpus(monkeypatch, count=2, mps=False)
    assert client.get("/api/training/environment").json()["devices"] == ["auto", "cpu", "cuda:0", "cuda:1"]
    _gpus(monkeypatch, count=0, mps=True)
    assert client.get("/api/training/environment").json()["devices"] == ["auto", "cpu", "mps"]
    _gpus(monkeypatch)
    assert client.get("/api/training/environment").json()["devices"] == ["auto", "cpu"]


def test_warnings_and_errors_come_with_their_codes(tmp_path):
    client, state = _setup(tmp_path)
    store = RunStore(state.store.dir(1))
    n = store.create(RunConfig(dataset=1, model="fasterrcnn_resnet50_fpn"))["number"]
    coded = {"code": "batch_halved", "params": {"from": 8, "to": 4}, "message": "halved"}
    store.update(
        n,
        status="failed",
        warnings=["an old plain warning", coded],
        error="no memory",
        error_code="out_of_memory",
        error_params={},
    )
    run = client.get(f"{RUNS}/{n}").json()
    assert run["warnings"] == [{"code": "", "params": {}, "message": "an old plain warning"}, coded]
    assert run["error_code"] == "out_of_memory" and run["error_params"] == {}
    listed = client.get(RUNS).json()["runs"][0]
    assert listed["warnings"][1]["code"] == "batch_halved" and listed["error_code"] == "out_of_memory"


def test_licences_are_listed_and_accepted_with_a_date(tmp_path):
    client, state = make_client(tmp_path)
    [entry] = client.get("/api/training/licences").json()["licences"]
    assert entry["id"] == "ultralytics" and entry["name"] == "AGPL-3.0" and entry["accepted"] is False
    assert (
        client.post("/api/training/licences/ultralytics/accept", json={}).json()["code"]
        == "licence_not_confirmed"
    )
    assert client.post("/api/training/licences/ultralytics/accept", json={"accept": False}).status_code == 400
    missing = client.post("/api/training/licences/nothing/accept", json={"accept": True})
    assert missing.status_code == 404 and missing.json()["code"] == "licence_not_found"
    accepted = client.post("/api/training/licences/ultralytics/accept", json={"accept": True}).json()
    assert accepted["accepted"] is True and accepted["accepted_at"]
    assert client.get("/api/training/licences").json()["licences"] == [accepted]
    again = client.post("/api/training/licences/ultralytics/accept", json={"accept": True}).json()
    assert again["accepted_at"] == accepted["accepted_at"]  # the first date stays
    assert (state.settings.workspace / "settings.json").exists()


def test_yolo_run_is_refused_until_the_licence_is_accepted(tmp_path):
    client, _ = _setup(tmp_path)
    body = {"dataset": 1, "model": "yolov8n", "params": {"epochs": 1}}
    refused = client.post(RUNS, json=body)
    assert refused.status_code == 400 and refused.json()["code"] == "licence_not_accepted"
    assert refused.json()["params"] == {"licence": "ultralytics", "name": "AGPL-3.0"}
    assert client.get(RUNS).json() == {"runs": []}
    assert client.post(RUNS, json={"dataset": 1, "model": "fasterrcnn_resnet50_fpn"}).status_code == 200
    client.post("/api/training/licences/ultralytics/accept", json={"accept": True})
    created = client.post(RUNS, json=body)
    assert created.status_code == 200 and created.json()["model"] == "yolov8n"


def test_environment_tells_whether_the_licence_of_a_model_is_accepted(tmp_path):
    client, _ = make_client(tmp_path)
    yolo = client.get("/api/training/environment", params={"model": "yolov8n"}).json()
    assert yolo["licence"] == "ultralytics" and yolo["licence_accepted"] is False
    client.post("/api/training/licences/ultralytics/accept", json={"accept": True})
    assert (
        client.get("/api/training/environment", params={"model": "yolov8n"}).json()["licence_accepted"]
        is True
    )
    plain = client.get("/api/training/environment", params={"model": "resnet18"}).json()
    assert plain["licence"] is None and plain["licence_accepted"] is True
