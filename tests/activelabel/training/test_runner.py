import json
from pathlib import Path

import pytest
from training_fakes import finished, tmp_dirs, wait_for

from activelabel.datasets import DatasetStore
from activelabel.labels import Label
from activelabel.training import runner as runner_module
from activelabel.training.augmentations import preset
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig, validate
from activelabel.training.runner import (
    RunnerError,
    _objects,
    build_spec,
    download_env,
    failure_fields,
    pose_box,
    warning_entry,
)
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
    assert run["setup"] == {"device": "cpu", "amp": False, "batch_size": 4} and "setup: " in log
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
    assert (
        run["best_child"] == "c2"
        and run["best"] == {"epoch": 2, "value": 0.9, "best_fold_value": 0.9}
        and run["current"] is None
    )
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


def test_cancel_between_two_children(setup, monkeypatch):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, mode="grid", grid={"fake": ["ok"], "epochs": [1, 2, 3]})
    real_train = runner._train

    def train_then_cancel(*args, **kwargs):
        real_train(*args, **kwargs)  # the first child ends normally...
        client.post(f"/api/projects/1/runs/{n}/cancel")  # ...and the run is cancelled before the next

    monkeypatch.setattr(runner, "_train", train_then_cancel)
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "cancelled"
    assert [c["status"] for c in run["children"]] == ["done", "cancelled", "cancelled"]
    wait_for(lambda: tmp_dirs(runner) == [])


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


def test_build_spec_augmentations(setup):
    _, state, _ = setup
    project_dir = state.store.dir(1)
    version = DatasetStore(project_dir).read(1)
    model = get_model("resnet18")
    run_dir = project_dir / "runs" / "r1"
    default = build_spec(project_dir, RunConfig(dataset=1, model="resnet18"), model, run_dir, version)
    assert default["augmentations"] == preset("light", "classification")
    strong = RunConfig(dataset=1, model="resnet18", params={"augmentation": "strong"})
    spec = build_spec(project_dir, strong, model, run_dir, version)
    assert spec["augmentations"] == preset("strong", "classification")
    custom = validate(
        RunConfig(dataset=1, model="resnet18", augmentations=[{"name": "blur"}]), "classification"
    )
    spec = build_spec(project_dir, custom, model, run_dir, version)
    assert spec["augmentations"] == [{"name": "blur", "params": {"p": 0.2, "sigma_max": 1.5}}]


def test_download_env(tmp_path):
    env = download_env(tmp_path)
    assert all(env[k].startswith(str(tmp_path)) for k in ("TORCH_HOME", "HF_HOME", "TMPDIR", "TEMP", "TMP"))


def test_jobs_carry_the_number_of_their_run(setup):
    client, state, runner = setup
    n = new_run(state)
    runner.enqueue(1, n)
    wait_for(lambda: store_status(state, n) == "done")
    (job,) = [j for j in runner._jobs.list(1) if j.kind == "training"]
    assert job.as_dict()["params"] == {"number": n}
    assert runner._jobs.submit(1, "analyze", lambda progress: {}).as_dict()["params"] == {}


def store_status(state, n):
    return RunStore(state.store.dir(1)).get(n)["status"]


def test_pose_box_holds_the_stored_box_and_every_visible_point():
    inside = Label("p", bbox=[0.1, 0.1, 0.5, 0.5], keypoints=[[0.2, 0.2, 2], [0.3, 0.3, 1]])
    assert pose_box(inside) == [0.1, 0.1, 0.5, 0.5]
    moved = Label("p", bbox=[0.1, 0.1, 0.2, 0.2], keypoints=[[0.8, 0.7, 2], [0.9, 0.9, 0]])  # hidden: ignored
    assert pose_box(moved) == pytest.approx([0.1, 0.1, 0.7, 0.6])
    assert pose_box(Label("p", keypoints=[[0.2, 0.4, 2], [0.6, 0.5, 2]])) == pytest.approx(
        [0.2, 0.4, 0.4, 0.1]
    )
    assert pose_box(Label("p", keypoints=[[0.0, 0.0, 0]])) is None


def test_instance_objects_carry_no_stored_box_and_detection_keeps_it():
    ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    own = [{"label": "cat", "bbox": [0.3, 0.3, 0.1, 0.1], "polygons": [ring]}]
    assert _objects(own, ["cat"], "instance", None) == [{"label": 0, "polygons": [ring]}]
    assert _objects(own, ["cat"], "detection", None) == [{"label": 0, "bbox": [0.3, 0.3, 0.1, 0.1]}]


def record_devices(monkeypatch):
    """Make build_spec note the device it is given (the real one still runs)."""
    seen = []
    real = runner_module.build_spec

    def spy(*args, **kwargs):
        spec = real(*args, **kwargs)
        seen.append(spec["device"])
        return spec

    monkeypatch.setattr(runner_module, "build_spec", spy)
    return seen


def test_build_spec_carries_the_device(setup):
    _, state, _ = setup
    project_dir = state.store.dir(1)
    version = DatasetStore(project_dir).read(1)
    model, run_dir, config = (
        get_model("resnet18"),
        project_dir / "runs" / "r1",
        RunConfig(dataset=1, model="resnet18"),
    )
    assert build_spec(project_dir, config, model, run_dir, version)["device"] == "auto"
    assert build_spec(project_dir, config, model, run_dir, version, "cpu")["device"] == "cpu"
    assert build_spec(project_dir, config, model, run_dir, version, device="cuda:1")["device"] == "cuda:1"


def test_run_store_keeps_the_device_choice(setup):
    _, state, _ = setup
    store = RunStore(state.store.dir(1))
    config = RunConfig(dataset=1, model="resnet18")
    assert store.create(config)["device_choice"] == "auto"
    assert store.create(config, "cpu")["device_choice"] == "cpu"
    assert store.get(2)["device_choice"] == "cpu"


def test_a_run_trains_on_the_device_it_was_given(setup, monkeypatch):
    _, state, runner = setup
    seen = record_devices(monkeypatch)
    store = RunStore(state.store.dir(1))
    config = RunConfig(dataset=1, model="resnet18", params={"epochs": 2}, metric="accuracy")
    n = store.create(config, "cpu")["number"]
    runner.enqueue(1, n)
    assert wait_for(finished(store, n))["status"] == "done"
    assert seen == ["cpu"]


def test_a_run_without_a_device_choice_uses_auto(setup, monkeypatch):
    _, state, runner = setup
    seen = record_devices(monkeypatch)
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    store.update(n, device_choice=None)  # a run made before the device existed
    runner.enqueue(1, n)
    assert wait_for(finished(store, n))["status"] == "done"
    assert seen == ["auto"]


def test_the_children_of_a_grid_use_the_device_of_the_run(setup, monkeypatch):
    _, state, runner = setup
    seen = record_devices(monkeypatch)
    store = RunStore(state.store.dir(1))
    config = RunConfig(
        dataset=1,
        model="resnet18",
        metric="accuracy",
        mode="grid",
        params={"epochs": 2},
        grid={"score": [0.3, 0.9]},
    )
    n = store.create(config, "cpu")["number"]
    runner.enqueue(1, n)
    assert wait_for(finished(store, n))["status"] == "done"
    assert seen == ["cpu", "cpu"]


def test_failures_and_warnings_keep_their_code_and_params():
    coded = RunnerError("no memory", "out_of_memory", {"smallest": 1})
    assert failure_fields(coded) == {
        "error": "no memory",
        "error_code": "out_of_memory",
        "error_params": {"smallest": 1},
    }
    assert failure_fields(ValueError("x")) == {"error": "x", "error_code": "", "error_params": {}}
    assert warning_entry({"message": "m", "code": "batch_halved", "params": {"from": 4, "to": 2}}) == {
        "code": "batch_halved",
        "params": {"from": 4, "to": 2},
        "message": "m",
    }
    assert warning_entry({"message": "m", "params": "bad"}) == {"code": "", "params": {}, "message": "m"}
