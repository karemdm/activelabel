import importlib.util
import json
import textwrap
from pathlib import Path

import pytest
from training_fakes import FAKE_WORKER, FakeEnv, tmp_dirs, wait_for

from activelabel.training.config import RunConfig
from activelabel.training.models import ModelStore
from activelabel.training.runner import TrainingRunner
from activelabel.training.runs import RunStore

FAKE_EXPORT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    (Path(spec["checkpoint"]).parent / "export_spec.json").write_text(json.dumps(spec))
    if spec["build"]["params"].get("fake_export") == "fail":
        print(json.dumps({"event": "error", "message": "no onnx"}), flush=True)
        sys.exit(1)
    Path(spec["output"]).write_bytes(b"onnx" * 10)
    print(json.dumps({"event": "done", "results": {"size": 40, "opset": 17}}), flush=True)
    """
)


@pytest.fixture
def models(setup, tmp_path):
    client, state, runner = setup
    (tmp_path / "export_onnx.py").write_text(FAKE_EXPORT)
    return client, state, runner


def finished_run(state, runner, **params):
    store = RunStore(state.store.dir(1))
    n = store.create(
        RunConfig(dataset=1, model="resnet18", params={"epochs": 2, **params}, metric="accuracy")
    )
    runner.enqueue(1, n["number"])
    wait_for(lambda: store.get(n["number"])["status"] == "done")
    return n["number"]


def export_state(client, n):
    return client.get(f"/api/projects/1/models/{n}").json()["exports"].get("onnx", {})


def test_save_list_current_and_delete(models):
    client, state, runner = models
    first = finished_run(state, runner)
    runs = RunStore(state.store.dir(1))
    assert (runs.path(first) / "last.pt").exists()
    resp = client.post("/api/projects/1/models", json={"run": first, "note": " baseline "})
    assert resp.status_code == 200, resp.text
    card = resp.json()
    assert card["number"] == 1 and card["current"] and card["note"] == "baseline"
    assert card["run"] == first and card["model"] == "resnet18" and card["metric"] == "accuracy"
    assert card["classes"] == ["a", "b"] and card["dataset"] == 1 and card["size"] > 0
    assert card["value"] == 1.0 and card["task_kind"] == "classification"
    assert not (runs.path(first) / "last.pt").exists() and not (runs.path(first) / "best.pt").exists()
    assert (ModelStore(state.store.dir(1)).path(1) / "best.pt").exists()  # the weights exist once
    again = client.post("/api/projects/1/models", json={"run": first})
    assert again.status_code == 400 and "m1" in again.json()["detail"]
    assert client.post("/api/projects/1/models", json={"run": 99}).status_code == 404
    queued = runs.create(RunConfig(dataset=1, model="resnet18"))["number"]
    assert client.post("/api/projects/1/models", json={"run": queued}).status_code == 400

    second = finished_run(state, runner, score=0.7)
    assert client.post("/api/projects/1/models", json={"run": second}).json()["current"] is False
    listed = client.get("/api/projects/1/models").json()
    assert [m["number"] for m in listed["models"]] == [2, 1] and listed["current"] == 1
    assert listed["models"][0]["results"]["val"]["accuracy"] == 0.7
    run_list = client.get("/api/projects/1/runs").json()["runs"]
    assert {r["number"]: r["model_number"] for r in run_list} == {second: 2, queued: None, first: 1}

    assert client.patch("/api/projects/1/models/2", json={"current": True, "note": "better"}).json()[
        "current"
    ]
    assert client.get("/api/projects/1/models/1").json()["current"] is False
    assert client.get("/api/projects/1/models/2").json()["note"] == "better"
    client.patch("/api/projects/1/models/1", json={"current": False})  # not current: nothing changes
    assert client.get("/api/projects/1/models").json()["current"] == 2
    assert client.patch("/api/projects/1/models/9", json={"note": "x"}).status_code == 404

    pt = client.get("/api/projects/1/models/1/download/pytorch")
    assert (
        pt.status_code == 200
        and pt.content == (ModelStore(state.store.dir(1)).path(1) / "best.pt").read_bytes()
    )
    assert 'filename="P-m1-resnet18.pt"' in pt.headers["content-disposition"]
    assert client.get("/api/projects/1/models/1/download/onnx").status_code == 404
    assert client.get("/api/projects/1/models/1/download/other").status_code == 404

    assert client.request("DELETE", "/api/projects/1/models/2", json={}).status_code == 400
    assert client.request("DELETE", "/api/projects/1/models/2", json={"confirm": True}).json() == {
        "deleted": True
    }
    assert client.get("/api/projects/1/models").json()["current"] is None
    assert client.get("/api/projects/1/models/2").status_code == 404
    third = finished_run(state, runner)
    assert client.post("/api/projects/1/models", json={"run": third}).json()["number"] == 3  # never reused


def test_export_onnx(models):
    client, state, runner = models
    run = finished_run(state, runner)
    client.post("/api/projects/1/models", json={"run": run})
    resp = client.post("/api/projects/1/models/1/export", json={"format": "onnx"})
    assert resp.status_code == 200 and resp.json()["exports"]["onnx"]["status"] in (
        "queued",
        "running",
        "done",
    )
    done = wait_for(lambda: (e := export_state(client, 1)).get("status") == "done" and e)
    assert done["size"] == 40 and done["opset"] == 17 and done["error"] == ""
    assert "onnx" in runner.environment.ensured[-1] and "torch" in runner.environment.ensured[-1]
    folder = ModelStore(state.store.dir(1)).path(1)
    spec = json.loads((folder / "export_spec.json").read_text())
    assert spec["worker"] == "fake_worker" and spec["task"] == "classification"
    assert spec["build"]["classes"] == ["a", "b"] and spec["build"]["params"]["epochs"] == 2
    assert spec["output"].endswith("model.onnx") and (folder / "export.log").exists()
    onnx = client.get("/api/projects/1/models/1/download/onnx")
    assert onnx.status_code == 200 and onnx.content == b"onnx" * 10
    assert 'filename="P-m1-resnet18.onnx"' in onnx.headers["content-disposition"]
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    assert client.post("/api/projects/1/models/1/export", json={"format": "tflite"}).status_code == 400
    assert client.post("/api/projects/1/models/7/export", json={"format": "onnx"}).status_code == 404


def test_export_failure_and_recover(models):
    client, state, runner = models
    run = finished_run(state, runner, fake_export="fail")
    client.post("/api/projects/1/models", json={"run": run})
    client.post("/api/projects/1/models/1/export", json={"format": "onnx"})
    failed = wait_for(lambda: (e := export_state(client, 1)).get("status") == "failed" and e)
    assert failed["error"] == "no onnx"
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    store = ModelStore(state.store.dir(1))
    store.set_export(1, "onnx", status="running")
    assert client.request("DELETE", "/api/projects/1/models/1", json={"confirm": True}).status_code == 409
    fresh = TrainingRunner(
        runner.workspace, state.store.dir, lambda: [1], environment=FakeEnv(runner.workspace)
    )
    fresh.recover()
    fresh.shutdown()
    assert export_state(client, 1)["status"] == "failed" and export_state(client, 1)["error"] == "interrupted"


def test_detection_export_comes_later(models, tmp_path):
    client, state, runner = models
    run = finished_run(state, runner)
    client.post("/api/projects/1/models", json={"run": run})
    store = ModelStore(state.store.dir(1))
    store.update(1, model="fasterrcnn_resnet50_fpn")
    before = len(runner.environment.ensured)
    client.post("/api/projects/1/models/1/export", json={"format": "onnx"})
    failed = wait_for(lambda: (e := export_state(client, 1)).get("status") == "failed" and e)
    assert "comes later" in failed["error"] and "PyTorch" in failed["error"]
    assert len(runner.environment.ensured) == before  # nothing installed


def test_export_worker_key_check():
    path = Path(__file__).parents[3] / "src/activelabel/training/workers/export_onnx.py"
    spec = importlib.util.spec_from_file_location("export_onnx_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_keys(["backbone.bn1.num_batches_tracked"], ["aux_classifier.4.weight"])
    with pytest.raises(ValueError, match="do not match"):
        module.check_keys(["head.weight"], [])
    with pytest.raises(ValueError, match="do not match"):
        module.check_keys([], ["extra.weight"])
    assert module.KINDS == ("classification", "semantic")


def test_fake_worker_is_shared():
    assert "best.pt" in FAKE_WORKER
