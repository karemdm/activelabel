import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sample_images import make_image
from training_fakes import finished, tmp_dirs, wait_for

from activelabel.datasets import DatasetStore
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig
from activelabel.training.runner import RunnerError, build_spec, download_env
from activelabel.training.runs import RunStore


def new_run(state, **params):
    config = RunConfig(dataset=1, model="resnet18", params=params, metric="accuracy")
    return RunStore(state.store.dir(1)).create(config)["number"]


def test_success(setup):
    _, state, runner = setup
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    run = wait_for(finished(store, n))
    assert run["status"] == "done", run
    assert run["device"] == {"cuda": False, "torch": "2.0.0"}
    assert run["packages"] == {"torch": "2.0.0", "pillow": "10.0"}
    assert run["activelabel_version"] and run["best"]["epoch"] == 2 and run["epoch"] == 2
    assert [m["epoch"] for m in store.metrics(n)] == [1, 2]
    assert store.results(n)["best_epoch"] == 2
    log = (store.path(n) / "log.txt").read_text()
    assert "hello" in log and "not json noise" in log
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    assert "torch" in runner.environment.ensured[0] and "pillow" in runner.environment.ensured[0]
    seen = json.loads((store.path(n) / "env.json").read_text())
    assert seen["cwd"].startswith(str(runner.tmp_root))
    assert seen["env"]["TORCH_HOME"].startswith(seen["cwd"])
    assert {seen["env"][k] for k in ("TMPDIR", "TEMP", "TMP")} == {seen["cwd"]}
    assert seen["env"]["HF_HOME"].startswith(seen["cwd"]) and seen["env"]["PIP_CACHE_DIR"].startswith(
        seen["cwd"]
    )


def test_failure_cleans_up(setup):
    _, state, runner = setup
    n = new_run(state, fake="fail")
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and run["error"] == "boom"
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written


def test_not_yet_supported_fails_before_install(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = store.create(RunConfig(dataset=1, model="yolov8n"))["number"]
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and "comes later" in run["error"]
    assert not hasattr(runner.environment, "ensured")


def composite(state, **kwargs):
    config = RunConfig(model="resnet18", metric="accuracy", **{"dataset": 1, **kwargs})
    return RunStore(state.store.dir(1)).create(config)["number"]


def test_grid_search_ranks_and_keeps_best_weights(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, mode="grid", params={"epochs": 2}, grid={"score": [0.3, 0.9, 0.5]})
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "done", run
    assert [c["id"] for c in run["children"]] == ["c1", "c2", "c3"] and run["trainings"] == 3
    assert [c["values"] for c in run["children"]] == [{"score": 0.3}, {"score": 0.9}, {"score": 0.5}]
    assert all(c["status"] == "done" and c["best"]["epoch"] == 2 for c in run["children"])
    assert run["best_child"] == "c2" and run["best"] == {"epoch": 2, "value": 0.9} and run["current"] is None
    results = store.results(n)
    assert [r["combo"] for r in results["ranking"]] == [2, 3, 1] and results["best_combo"] == 2
    assert results["ranking"][0]["rank"] == 1 and results["ranking"][0]["std"] == 0.0
    assert results["val"] == {"accuracy": 0.9} and results["metric"] == "accuracy"
    assert (store.path(n) / "best.pt").exists() and (store.path(n) / "last.pt").exists()
    assert not list(store.path(n).glob("c*/*.pt"))
    assert [m["epoch"] for m in store.metrics(n)] == [1, 2] == [m["epoch"] for m in store.metrics(n, "c1")]
    assert store.child_config(n, "c3").params == {"epochs": 2, "score": 0.5}
    assert store.child_config(n, "c3").mode == "single" and store.results(n, "c1")["val"]["accuracy"] == 0.3
    log = (store.path(n) / "log.txt").read_text()
    assert "== c1: score=0.3 ==" in log and "== c3: score=0.5 ==" in log
    assert len(runner.environment.ensured) == 1
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    child = client.get(f"/api/projects/1/runs/{n}/children/c3").json()
    assert child["values"] == {"score": 0.5} and child["config"]["params"]["score"] == 0.5
    assert [m["epoch"] for m in child["metrics"]] == [1, 2] and child["results"]["val"]["accuracy"] == 0.5
    assert client.get(f"/api/projects/1/runs/{n}/children/c9").status_code == 404
    assert client.get(f"/api/projects/1/runs/{n}/children/cx").status_code == 404
    detail = client.get(f"/api/projects/1/runs/{n}").json()
    assert detail["results"]["best_child"] == "c2" and len(detail["metrics"]) == 2


def test_all_folds_with_grid_mean_and_std(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, dataset=2, fold="all", mode="grid", grid={"score": [0.8, 0.4]})
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "done", run
    assert [(c["combo"], c["fold"]) for c in run["children"]] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    results = store.results(n)
    top = results["ranking"][0]
    assert top["combo"] == 1 and top["children"] == ["c1", "c2"] and results["folds"] == 2
    assert top["mean"] == pytest.approx(0.805) and top["std"] == pytest.approx(0.005)
    assert run["best_child"] == "c2" and run["best"]["value"] == pytest.approx(0.805)
    folded = composite(state, dataset=2, fold="all")
    runner.enqueue(1, folded)
    run = wait_for(finished(store, folded))
    assert run["status"] == "done" and [c["fold"] for c in run["children"]] == [0, 1]
    assert run["children"][1]["values"] == {} and "fold 2" in (store.path(folded) / "log.txt").read_text()


def test_grid_child_failure(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, mode="grid", grid={"fake": ["fail", "ok"]})
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "done" and run["best_child"] == "c2"
    assert run["children"][0]["status"] == "failed" and run["children"][0]["error"] == "boom"
    assert store.results(n)["ranking"][-1]["mean"] is None
    n = composite(state, mode="grid", grid={"fake": ["fail"], "epochs": [1, 2]})
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and "Every training" in run["error"] and "boom" in run["error"]


def test_cancel_grid(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, mode="grid", grid={"fake": ["sleep"], "epochs": [1, 2]})
    runner.enqueue(1, n)
    wait_for(lambda: (store.path(n) / "c1" / "metrics.jsonl").exists() and store.metrics(n, "c1"))
    assert store.get(n)["current"] == "c1" and store.get(n)["epoch"] == 1
    assert client.get(f"/api/projects/1/runs/{n}").json()["metrics"][0]["epoch"] == 1  # live child
    assert client.post(f"/api/projects/1/runs/{n}/cancel").json()["status"] == "cancelled"
    assert [c["status"] for c in store.get(n)["children"]] == ["cancelled", "cancelled"]
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written


def test_cancel_running(setup):
    client, state, runner = setup
    n = new_run(state, fake="sleep")
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.metrics(n))
    resp = client.post(f"/api/projects/1/runs/{n}/cancel")
    assert resp.status_code == 200 and resp.json()["status"] == "cancelled"
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written
    assert client.post("/api/projects/1/runs/99/cancel").status_code == 404
    assert client.post(f"/api/projects/1/runs/{n}/cancel").json()["status"] == "cancelled"


def test_one_at_a_time_and_queue_position(setup, tmp_path):
    _, state, runner = setup
    gate = tmp_path / "gate"
    first, second, third = (new_run(state, gate=str(gate)) for _ in range(3))
    for n in (first, second, third):
        runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.get(first)["status"] == "running" and store.metrics(first))
    assert store.get(second)["status"] == "queued" and store.get(second)["queue_position"] == 1
    assert store.get(third)["queue_position"] == 2
    runner.cancel(1, third)  # a queued run is cancelled without starting
    assert store.get(third)["status"] == "cancelled"
    gate.write_text("go")
    wait_for(lambda: store.get(second)["status"] == "done")
    assert store.get(first)["status"] == "done" and store.get(second)["queue_position"] is None
    t1 = json.loads((store.path(first) / "times.json").read_text())
    t2 = json.loads((store.path(second) / "times.json").read_text())
    assert t1[1] <= t2[1] and not (store.path(third) / "env.json").exists()
    assert store.get(second)["started"] >= store.get(first)["finished"]


def test_recover_marks_interrupted(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state)
    store.update(n, status="running")
    runner.recover()
    assert store.get(n)["status"] == "failed" and store.get(n)["error"] == "interrupted"


def test_build_spec_parts_and_folds(setup):
    _, state, _ = setup
    project_dir = state.store.dir(1)
    store = DatasetStore(project_dir)
    model = get_model("resnet18")
    run_dir = project_dir / "runs" / "r1"
    spec = build_spec(project_dir, RunConfig(dataset=1, model="resnet18"), model, run_dir, store.read(1))
    assert spec["classes"] == ["a", "b"] and spec["start"] == "transfer" and spec["params"]["epochs"] == 10
    assert all(Path(i["path"]).is_absolute() and i["label"] in (0, 1) for i in spec["parts"]["train"])
    assert spec["fold"] is None and spec["metric"] == "accuracy"
    folded = store.read(2)
    config = RunConfig(dataset=2, model="resnet18", fold=1)
    spec = build_spec(project_dir, config, model, run_dir, folded)
    assert len(spec["parts"]["val"]) == len(folded["folds"][1]) and spec["fold"] == 1
    assert len(spec["parts"]["train"]) == len(folded["folds"][0])
    config = RunConfig(dataset=2, model="resnet18", fold="all")
    with pytest.raises(RunnerError, match="one child at a time"):
        build_spec(project_dir, config, model, run_dir, folded)


def test_download_env(tmp_path):
    env = download_env(tmp_path)
    assert all(env[k].startswith(str(tmp_path)) for k in ("TORCH_HOME", "HF_HOME", "TMPDIR", "TEMP", "TMP"))


# -- worker helpers (no torch) ------------------------------------------------------------------

WORKER = Path(__file__).parents[3] / "src" / "activelabel" / "training" / "workers" / "classification.py"


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("classification_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["classification_worker"] = module
    spec.loader.exec_module(module)
    return module


def test_worker_metrics(worker):
    truth, pred = [0, 0, 1, 1, 2], [0, 1, 1, 1, 0]
    ev = worker.evaluation(truth, pred, ["a", "b", "c"])
    assert ev["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert ev["accuracy"] == pytest.approx(0.6)
    assert ev["per_class"]["a"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert ev["per_class"]["b"]["precision"] == pytest.approx(2 / 3) and ev["per_class"]["b"]["recall"] == 1
    assert ev["per_class"]["c"]["f1"] == 0
    assert ev["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert worker.macro_f1([[2, 0], [0, 0]]) == 1.0  # a class absent everywhere does not count
    assert worker.accuracy([[0, 0], [0, 0]]) == 0.0


def test_worker_spec_and_plans(worker, tmp_path):
    spec = {
        "task": "classification",
        "model": "resnet18",
        "family": "torchvision",
        "start": "transfer",
        "params": {},
        "classes": ["a"],
        "parts": {"train": [1], "val": [1]},
        "output_dir": "x",
        "metric": "accuracy",
    }
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    assert worker.load_spec(file)["model"] == "resnet18"
    for change, message in (
        ({"fold": "all"}, "comes later"),
        ({"task": "detection"}, "classification"),
        ({"metric": "map"}, "metric"),
        ({"parts": {"train": [1], "val": []}}, "validation"),
    ):
        file.write_text(json.dumps({**spec, **change}))
        with pytest.raises(worker.SpecError, match=message):
            worker.load_spec(file)
    assert worker.augmentation_plan("none", 64) == [("Resize", {"size": (64, 64)})]
    assert len(worker.augmentation_plan("strong", 64)) > len(worker.augmentation_plan("light", 64))
    with pytest.raises(worker.SpecError):
        worker.augmentation_plan("x", 64)
    assert worker.head_path("resnet18") == "fc" and worker.head_path("mobilenet_v3_large") == "classifier.3"


def test_worker_dataset_needs_torch(worker, tmp_path):
    pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms")
    image = make_image(tmp_path / "i.png", fmt="PNG")
    data = worker.ImageDataset([{"path": str(image), "label": 1}], transforms.ToTensor())
    tensor, label = data[0]
    assert label == 1 and tensor.shape[0] == 3
