import json
import threading
import time
from pathlib import Path

import pytest
from training_fakes import FakeEnv, finished, tmp_dirs, wait_for

from activelabel.datasets import DatasetStore
from activelabel.images.manifest import Manifest
from activelabel.jobs import CANCELLED, QUEUED
from activelabel.labels import Label
from activelabel.preannotate import PreannotationStore
from activelabel.scheduler import Scheduler
from activelabel.training import runner as runner_module
from activelabel.training.augmentations import preset
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig, validate
from activelabel.training.licences import LICENCES, LicenceNotAccepted, LicenceStore, licences_of_packages
from activelabel.training.models import ModelStore
from activelabel.training.runner import (
    RunnerError,
    TrainingRunner,
    _objects,
    build_spec,
    download_env,
    failure_fields,
    pose_box,
    val_image_count,
    warning_entry,
    worker_key,
    worker_script,
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
    val = len(DatasetStore(state.store.dir(1)).read(1)["parts"]["val"])
    assert run["alerts"] == ([{"code": "few_val_images", "params": {"n": val}}] if val < 10 else [])
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


class LicensedEnv(FakeEnv):
    """The fake environment with the licence check of the real one."""

    def ensure(self, packages, check, step):
        LicenceStore(self.workspace).require(licences_of_packages(packages))
        super().ensure(packages, check, step)


def test_yolo_run_needs_the_accepted_licence_before_anything_is_installed(setup):
    _, state, runner = setup
    runner.environment = LicensedEnv(runner.workspace)
    store = RunStore(state.store.dir(1))
    n = store.create(RunConfig(dataset=1, model="yolov8n"))["number"]
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and run["error_code"] == "licence_not_accepted"
    assert run["error_params"] == {"licence": "ultralytics", "name": "AGPL-3.0"}
    assert not hasattr(runner.environment, "ensured")
    LicenceStore(runner.workspace).accept("ultralytics")
    again = store.create(RunConfig(dataset=1, model="yolov8n"))["number"]
    runner.enqueue(1, again)
    run = wait_for(finished(store, again))
    assert (
        run["error_code"] != "licence_not_accepted"
    )  # past the licence: this classification version has no boxes
    assert runner.environment.ensured[0][0] == "ultralytics"


def test_worker_key_and_script_pick_the_yolo_worker_for_its_family():
    assert worker_key(get_model("yolov8n")) == "ultralytics" == worker_key(get_model("yolo11s-pose"))
    assert worker_key(get_model("resnet18")) == "classification"
    assert worker_key(get_model("maskrcnn_resnet50_fpn")) == "instance"
    assert (
        worker_script("ultralytics").name == "yolo.py" and worker_script("semantic").name == "segmentation.py"
    )
    with pytest.raises(RunnerError, match="comes later"):
        worker_script("nothing")


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
    assert isinstance(run["alerts"], list)  # the alerts are those of the best child (see quality.py)
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


def test_val_image_count_reads_the_part_or_the_fold_of_the_best_child(setup):
    _, state, _ = setup
    project = state.store.dir(1)
    store = RunStore(project)
    data = DatasetStore(project)
    single = new_run(state)
    assert val_image_count(project, store, single) == len(data.read(1)["parts"]["val"])
    folded = composite(state, dataset=2, mode="single", fold=1)
    assert val_image_count(project, store, folded) == len(data.read(2)["folds"][1])
    store.update(single, best_child="c1")  # a grid run asks its best child (missing here: no answer)
    assert val_image_count(project, store, single) is None
    gone = composite(state, dataset=99)
    assert val_image_count(project, store, gone) is None


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
    licence = failure_fields(LicenceNotAccepted(LICENCES["ultralytics"]))
    assert (
        licence["error_code"] == "licence_not_accepted"
        and licence["error_params"]["licence"] == "ultralytics"
    )
    assert warning_entry({"message": "m", "code": "batch_halved", "params": {"from": 4, "to": 2}}) == {
        "code": "batch_halved",
        "params": {"from": 4, "to": 2},
        "message": "m",
    }
    assert warning_entry({"message": "m", "params": "bad"}) == {"code": "", "params": {}, "message": "m"}


# -- resuming an interrupted run (#321) ------------------------------------------------

FAKE_RESUMABLE = """
import json, sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
out = Path(spec["output_dir"])
start = 1
if spec.get("resume"):
    start = json.loads((out / "last.pt").read_text())["epoch"] + 1
    print(json.dumps({"event": "resumed", "epoch": start - 1}), flush=True)
for epoch in range(start, 4):
    (out / "last.pt").write_text(json.dumps({"epoch": epoch}))
    row = {"event": "epoch", "epoch": epoch, "val_accuracy": 0.1 * epoch, "best": True, "checkpoint": True}
    print(json.dumps(row), flush=True)
    if epoch == 2 and not spec.get("resume") and spec["params"].get("fake") == "crash":
        print(json.dumps({"event": "error", "message": "crashed"}), flush=True)
        sys.exit(1)
(out / "best.pt").write_text("w")
results = {"best_epoch": 3, "val": {"accuracy": 0.3}, "test": None}
print(json.dumps({"event": "done", "results": results}), flush=True)
"""


@pytest.fixture
def resumable(setup, tmp_path):
    client, state, runner = setup
    script = tmp_path / "resumable_worker.py"
    script.write_text(FAKE_RESUMABLE)
    runner._scripts = lambda kind: script
    return client, state, runner


def crashed_run(state, runner):
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=3, fake="crash")
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and run["checkpoint_epoch"] == 2 and run["digest"], run
    return store, n


def test_a_failed_run_resumes_from_its_checkpoint(resumable):
    client, state, runner = resumable
    store, n = crashed_run(state, runner)
    with (store.path(n) / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 3, "val_accuracy": 9.0}) + "\n")  # written after the checkpoint
    answer = client.post(f"/api/projects/1/runs/{n}/resume")
    assert answer.status_code == 200, answer.text
    assert answer.json()["resumes"][0]["from_epoch"] == 2 and answer.json()["error"] == ""
    run = wait_for(finished(store, n))
    assert run["status"] == "done" and run["resume"] is False and run["checkpoint_epoch"] == 3
    assert [m["epoch"] for m in store.metrics(n)] == [1, 2, 3] and store.metrics(n)[-1]["val_accuracy"] < 1
    assert "checkpoint" not in store.metrics(n)[0]
    assert "resumed after epoch 2" in (store.path(n) / "log.txt").read_text()
    again = client.post(f"/api/projects/1/runs/{n}/resume")
    assert again.status_code == 409 and again.json()["code"] == "run_not_resumable"


def test_a_run_saved_before_the_schedule_rule_still_resumes(resumable):
    """#546: warmup/patience longer than the epochs is refused for new runs only, not on resume."""
    client, state, runner = resumable
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=3, warmup_epochs=5, patience=10, fake="crash")
    runner.enqueue(1, n)
    assert wait_for(finished(store, n))["checkpoint_epoch"] == 2
    answer = client.post(f"/api/projects/1/runs/{n}/resume")
    assert answer.status_code == 200, answer.text
    assert wait_for(finished(store, n))["status"] == "done"


def test_resume_is_refused_with_a_reason(resumable):
    client, state, runner = resumable
    store, n = crashed_run(state, runner)
    store.update(n, digest="another training")
    changed = client.post(f"/api/projects/1/runs/{n}/resume")
    assert changed.status_code == 409 and changed.json()["code"] == "resume_config_changed"
    old = new_run(state)  # a run that stopped before checkpoints existed
    store.update(old, status="failed")
    assert client.post(f"/api/projects/1/runs/{old}/resume").json()["code"] == "resume_no_checkpoint"
    folds = RunStore(state.store.dir(1)).create(RunConfig(dataset=2, model="resnet18", fold="all"))["number"]
    store.update(folds, status="cancelled")
    composite = client.post(f"/api/projects/1/runs/{folds}/resume")
    assert composite.status_code == 400 and composite.json()["code"] == "resume_composite"
    assert client.post("/api/projects/1/runs/99/resume").status_code == 404


def test_a_queued_resume_of_a_changed_training_fails_with_its_code(resumable):
    _, state, runner = resumable
    store, n = crashed_run(state, runner)
    store.update(n, status="queued", resume=True, digest="another training")
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and run["error_code"] == "resume_config_changed"


def test_digest_ignores_folder_device_and_resume():
    spec = {"model": "m", "params": {"lr": 0.1}, "output_dir": "a", "device": "cpu"}
    same = runner_module.spec_digest(
        {**spec, "output_dir": "b", "device": "cuda:0", "resume": True, "digest": "x"}
    )
    assert runner_module.spec_digest(spec) == same
    assert runner_module.spec_digest({**spec, "params": {"lr": 0.2}}) != same


def test_tool_jobs_share_the_queue_and_unknown_ones_are_ignored(setup):
    _, _, runner = setup
    job_id = runner.enqueue_tool(1, "tool", lambda progress: {"ok": True}, {"what": "x"})
    job = wait_for(lambda: (j := runner.tool_job(job_id)) and j.status == "done" and j)
    assert job.kind == "tool" and job.result == {"ok": True} and job.params == {"what": "x"}
    assert runner.tool_job("unknown") is None
    runner.cancel_tool(1, "unknown", "embed")  # nothing queued, no process: nothing happens


class Flag:
    """Records that it was called (the owner's cancel of a tool)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def test_a_tool_waits_in_its_device_lane_and_the_queue_cancels_it_through_its_owner(setup):
    _, _, runner = setup
    hold, owner = Hold(), Flag()
    first = runner.enqueue_tool(1, "tool", hold, device="cuda:0")
    assert hold.started.wait(5) and runner.tool_job(first).device == "cuda:0"
    queued = runner.enqueue_tool(1, "tool", lambda p: None, device="cuda:0", on_cancel=owner)
    plain = runner.enqueue_tool(1, "tool", lambda p: None, device="cuda:0")
    assert runner.cancel_job(queued) and owner.calls == 1  # the owner cancels it (and records it)
    assert runner.cancel_job(plain) and runner.tool_job(plain).status == "cancelled"
    hold.release.set()
    wait_for(lambda: runner.tool_job(first).status == "done")
    done = runner.enqueue_tool(1, "tool", lambda p: None, device="cpu", on_cancel=owner)
    wait_for(lambda: runner.tool_job(done).status == "done")
    runner.enqueue_tool(1, "tool", lambda p: None, device="cpu")  # finished tools leave the cancel table
    assert done not in runner._tool_cancels


# -- the global queue: lanes, order and restart (#604) --------------------------------------


def new_run_on(state, device, project=1, **params):
    config = RunConfig(dataset=1, model="resnet18", params=params, metric="accuracy")
    return RunStore(state.store.dir(project)).create(config, device=device)["number"]


class Hold:
    """A job function that holds its lane until released."""

    def __init__(self) -> None:
        self.started, self.release = threading.Event(), threading.Event()

    def __call__(self, progress):
        self.started.set()
        self.release.wait(10)


class Recorder:
    """Stands in for a runner method and records what it was asked to queue (and may fail after that)."""

    def __init__(self, calls: list, kind: str, fail: bool = False) -> None:
        self.calls, self.kind, self.fail = calls, kind, fail

    def __call__(self, *args):
        self.calls.append((self.kind, *args))
        if self.fail:
            raise RuntimeError("cannot queue")


class RunnerFactory:
    """Builds fresh runners on the workspace of a test and shuts them all down at the end."""

    def __init__(self, workspace: Path, project_dir, ids: list[int]) -> None:
        self.workspace, self.project_dir, self.ids, self.runners = workspace, project_dir, ids, []

    def __call__(self, scheduler: Scheduler | None = None) -> TrainingRunner:
        runner = TrainingRunner(
            self.workspace,
            self.project_dir,
            lambda: self.ids,
            environment=FakeEnv(self.workspace),
            scheduler=scheduler,
        )
        self.runners.append(runner)
        return runner


@pytest.fixture
def fresh(setup):
    _, state, runner = setup
    factory = RunnerFactory(runner.workspace, state.store.dir, [1, 2])
    yield factory
    for each in factory.runners:
        each.shutdown()


def times_of(store, n):
    return json.loads((store.path(n) / "times.json").read_text())


def test_two_runs_on_the_same_lane_never_overlap(setup, tmp_path):
    _, state, runner = setup
    runner._lane = lambda device: device
    gate = tmp_path / "gate"
    first, second = (new_run_on(state, "cuda:0", gate=str(gate)) for _ in range(2))
    for n in (first, second):
        runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.get(first)["status"] == "running" and store.metrics(first))
    time.sleep(0.2)
    assert store.get(second)["status"] == "queued"
    gate.write_text("go")
    wait_for(finished(store, second))
    assert times_of(store, first)[1] <= times_of(store, second)[0]


def test_runs_on_different_lanes_overlap(setup, tmp_path):
    _, state, runner = setup
    runner._lane = lambda device: device
    gate = tmp_path / "gate"
    gpu, cpu = new_run_on(state, "cuda:0", gate=str(gate)), new_run_on(state, "cpu", gate=str(gate))
    for n in (gpu, cpu):
        runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: all(store.get(n)["status"] == "running" and store.metrics(n) for n in (gpu, cpu)))
    gate.write_text("go")
    wait_for(finished(store, gpu))
    wait_for(finished(store, cpu))
    a, b = times_of(store, gpu), times_of(store, cpu)
    assert a[0] < b[1] and b[0] < a[1]  # each started before the other ended


def test_the_queue_position_is_counted_inside_the_lane_of_the_run(setup, tmp_path):
    _, state, runner = setup
    runner._lane = lambda device: device
    gate = tmp_path / "gate"
    store = RunStore(state.store.dir(1))
    g1, c1 = new_run_on(state, "cuda:0", gate=str(gate)), new_run_on(state, "cpu", gate=str(gate))
    for n in (g1, c1):
        runner.enqueue(1, n)
    wait_for(lambda: all(store.get(n)["status"] == "running" for n in (g1, c1)))
    g2, c2, g3 = (new_run_on(state, d, gate=str(gate)) for d in ("cuda:0", "cpu", "cuda:0"))
    for n in (g2, c2, g3):
        runner.enqueue(1, n)
    assert [store.get(n)["queue_position"] for n in (g2, c2, g3)] == [1, 1, 2]
    runner.cancel(1, g2)  # the one behind it moves up
    wait_for(lambda: store.get(g3)["queue_position"] == 1)
    gate.write_text("go")
    for n in (g1, c1, c2, g3):
        wait_for(finished(store, n))
    assert store.get(g3)["queue_position"] is None


def test_auto_is_the_lane_of_the_first_gpu_else_the_cpu(fresh):
    runner = fresh()
    runner.environment.gpu_count = lambda: 2
    assert runner._auto_lane("auto") == "cuda:0" and runner._auto_lane("cuda:1") == "cuda:1"
    runner.environment.gpu_count = lambda: 0  # looked up only once
    assert runner._auto_lane("") == "cuda:0"
    other = fresh()
    other.environment.gpu_count = lambda: 0
    other.environment.mps_hint = lambda: False
    assert other._auto_lane("auto") == "cpu" and other._auto_lane("cpu") == "cpu"


def test_the_job_of_a_run_waits_on_the_lane_of_its_device(setup, tmp_path):
    _, state, runner = setup
    runner._lane = lambda device: device
    gate = tmp_path / "gate"
    n = new_run_on(state, "cuda:1", gate=str(gate))
    runner.enqueue(1, n)
    (job,) = [j for j in runner._jobs.list(1) if j.kind == "training"]
    assert job.device == "cuda:1"
    gate.write_text("go")
    wait_for(finished(RunStore(state.store.dir(1)), n))


def test_recover_queues_again_in_the_saved_order_across_projects(setup, fresh, tmp_path):
    client, state, _ = setup
    created = client.post("/api/projects", json={"name": "Q", "task": "classification", "classes": ["a"]})
    assert created.status_code < 300
    a1, a2, a3 = (new_run_on(state, "auto") for _ in range(3))
    b1 = new_run_on(state, "auto", project=2)
    done = new_run_on(state, "auto")
    RunStore(state.store.dir(1)).update(done, status="done")
    saved = [
        {"id": "x1", "project_id": 2, "kind": "training", "status": "queued", "params": {"number": b1}},
        {"id": "x2", "project_id": 1, "kind": "analyze", "status": "queued", "params": {"k": 1}},
        {"id": "x3", "project_id": 1, "kind": "training", "status": "running", "params": {"number": a1}},
        {"id": "x4", "project_id": 1, "kind": "training", "status": "queued", "params": {"number": a2}},
        {"id": "x5", "project_id": 1, "kind": "training", "status": "queued", "params": {"number": done}},
        {"id": "x6", "project_id": 1, "kind": "training", "status": "queued", "params": {"number": 999}},
        {"id": "x7", "project_id": 1, "kind": "training", "status": "queued", "params": {}},
    ]
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"jobs": saved}), encoding="utf-8")
    runner = fresh(Scheduler(queue))
    calls: list = []
    restored: list = []
    runner.enqueue = Recorder(calls, "training")
    runner.recover({"analyze": restored.append})
    # file order first (b1, then a2), then what the file does not list, oldest first (a1, a3)
    assert calls == [("training", 2, b1), ("training", 1, a2), ("training", 1, a1), ("training", 1, a3)]
    assert [e["id"] for e in restored] == ["x2"]  # the rest of the app queues its own jobs again
    assert [e["id"] for e in runner.scheduler.snapshot()["interrupted"]] == ["x3"]


def test_recover_without_a_queue_file_queues_every_queued_run_oldest_first(setup, fresh):
    _, state, _ = setup
    runs = [new_run_on(state, "auto") for _ in range(3)]
    runner = fresh()
    calls: list = []
    runner.enqueue = Recorder(calls, "training")
    runner.recover()
    assert calls == [("training", 1, n) for n in runs]


def test_a_failing_requeue_does_not_stop_the_others(setup, fresh):
    _, state, _ = setup
    first, second = new_run_on(state, "auto"), new_run_on(state, "auto")
    runner = fresh()
    calls: list = []
    runner.enqueue = Recorder(calls, "training", fail=True)
    runner.recover()
    assert calls == [("training", 1, first), ("training", 1, second)]


def finished_model(client, state, runner):
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    wait_for(finished(store, n))
    assert client.post("/api/projects/1/models", json={"run": n}).status_code == 200
    return n


def test_queued_preannotations_and_exports_are_queued_again_and_running_ones_are_interrupted(setup, fresh):
    client, state, runner = setup
    model = finished_model(client, state, runner)
    pre = PreannotationStore(state.store.dir(1))
    queued_pre = pre.create(model=model, task=1, device="cpu")["number"]
    running_pre = pre.create(model=model, task=1, device="cpu", status="running")["number"]
    models = ModelStore(state.store.dir(1))
    models.set_export(model, "onnx", status="queued")
    models.set_export(model, "other", status="running")
    again = fresh()
    calls: list = []
    again.enqueue_preannotation = Recorder(calls, "preannotate")
    again._submit_export = Recorder(calls, "export")
    again.recover()
    assert calls == [("preannotate", 1, queued_pre), ("export", 1, model, "onnx")]
    assert pre.get(queued_pre)["status"] == "queued"  # not failed: it goes back in line
    assert pre.get(running_pre)["status"] == "failed" and pre.get(running_pre)["error"] == "interrupted"
    exports = models.get(model)["exports"]
    assert exports["onnx"]["status"] == "queued" and exports["other"]["status"] == "failed"
    assert exports["other"]["error"] == "interrupted"


def test_a_queued_preannotation_and_export_submit_their_jobs_on_the_right_lane(setup, fresh):
    client, state, runner = setup
    model = finished_model(client, state, runner)
    pre = PreannotationStore(state.store.dir(1))
    number = pre.create(model=model, task=1, device="cpu")["number"]
    ModelStore(state.store.dir(1)).set_export(model, "onnx", status="queued")
    again = fresh()
    hold = Hold()
    again._jobs.submit(1, "hold", hold, device="cpu")  # keeps the cpu lane busy: nothing below starts
    assert hold.started.wait(2)
    again.recover()
    kinds = {j.kind: j for j in again._jobs.list(1)}
    assert kinds["preannotate"].device == "cpu" and kinds["preannotate"].params == {"number": number}
    assert kinds["export"].device == "cpu" and kinds["export"].params == {"number": model, "format": "onnx"}
    assert kinds["preannotate"].status == QUEUED and kinds["export"].status == QUEUED
    hold.release.set()


def test_a_run_interrupted_with_a_checkpoint_can_be_resumed(resumable):
    client, state, runner = resumable
    store, n = crashed_run(state, runner)
    store.update(n, status="running", error="", checkpoint_epoch=2)
    runner.recover()
    run = store.get(n)
    assert run["status"] == "failed" and run["error"] == "interrupted"
    answer = client.post(f"/api/projects/1/runs/{n}/resume")
    assert answer.status_code == 200, answer.text
    assert wait_for(finished(store, n))["status"] == "done"


def test_shutdown_during_a_training_records_it_as_interrupted(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state, fake="sleep")
    runner.enqueue(1, n)
    wait_for(lambda: store.metrics(n))
    runner.shutdown()
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and run["error"] == "interrupted"
    wait_for(lambda: tmp_dirs(runner) == [])


def test_cancel_job_of_a_running_and_of_a_queued_training(setup, tmp_path):
    _, state, runner = setup
    gate = tmp_path / "gate"
    store = RunStore(state.store.dir(1))
    first, second = (new_run(state, gate=str(gate)) for _ in range(2))
    for n in (first, second):
        runner.enqueue(1, n)
    wait_for(lambda: store.get(first)["status"] == "running" and store.metrics(first))
    assert runner.cancel_job(runner._job_ids[(1, second)]) is True
    assert store.get(second)["status"] == "cancelled"
    assert runner.cancel_job(runner._job_ids[(1, first)]) is True
    assert wait_for(finished(store, first))["status"] == "cancelled"


def test_cancel_job_of_a_queued_preannotation_and_export(setup):
    _, state, runner = setup
    hold = Hold()
    runner._jobs.submit(1, "hold", hold, device="cpu")
    assert hold.started.wait(2)
    pre = PreannotationStore(state.store.dir(1))
    number = pre.create(model=1, task=1, device="cpu")["number"]
    runner.enqueue_preannotation(1, number)
    (pre_job,) = [j for j in runner._jobs.list(1) if j.kind == "preannotate"]
    assert runner.cancel_job(pre_job.id) is True
    assert pre.get(number)["status"] == "cancelled"
    runner._submit_export(1, 5, "onnx")
    (export_job,) = [j for j in runner._jobs.list(1) if j.kind == "export"]
    assert runner.cancel_job(export_job.id) is True
    assert export_job.status == CANCELLED
    hold.release.set()


def test_cancel_job_of_an_unknown_job_is_false(setup):
    _, _, runner = setup
    assert runner.cancel_job("nothing") is False


# -- downloaded pretrained weights are deleted with the run (#643) ------------------------------


@pytest.mark.parametrize("fake", ["ok", "fail"])
def test_downloaded_weights_are_deleted_and_recorded(setup, fake):
    _, state, runner = setup
    n = new_run(state, epochs=2, download=64, fake=fake)
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(finished(store, n))
    run = wait_for(lambda: store.get(n) if store.get(n).get("downloads") else None)
    assert run["downloads"] == {
        "files": [
            {"name": "torch/hub/checkpoints/fake-backbone.pth", "bytes": 64},
            {"name": "yolo-fake.pt", "bytes": 64},
        ],
        "bytes": 128,
        "deleted": True,
    }
    assert tmp_dirs(runner) == []
    if fake == "ok":  # the trained weights stay in the run's own folder
        assert (store.path(n) / "best.pt").exists() and (store.path(n) / "last.pt").exists()


def test_cancelled_run_deletes_its_downloads(setup):
    _, state, runner = setup
    n = new_run(state, download=8, fake="sleep")
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.get(n)["status"] == "running" and store.get(n).get("epoch"))
    runner.cancel(1, n)
    run = wait_for(lambda: store.get(n) if store.get(n).get("downloads") else None)
    assert run["status"] == "cancelled" and run["downloads"]["deleted"] is True
    assert run["downloads"]["bytes"] == 16 and tmp_dirs(runner) == []


def test_download_env_sends_ultralytics_settings_inside(tmp_path):
    assert download_env(tmp_path)["YOLO_CONFIG_DIR"] == str(tmp_path / "yolo-config")


def _deleted_run(**_fields):
    raise KeyError("gone")


def _broken_store(**_fields):
    raise OSError("disk")


def test_record_downloads_survives_a_deleted_run_or_a_failing_write(tmp_path, caplog):
    for update in (_deleted_run, _broken_store):
        tmp = tmp_path / update.__name__
        tmp.mkdir()
        runner_module.TrainingRunner._record_downloads(update, tmp)
        assert not tmp.exists()
    assert "Could not record the downloads" in caplog.text


def test_build_spec_refuses_images_missing_or_changed_on_disk_with_a_code(setup):
    """#620: a file deleted or replaced after the import stops the run with a translatable code."""
    _, state, _ = setup
    project_dir = state.store.dir(1)
    version = DatasetStore(project_dir).read(1)
    model = get_model("resnet18")
    run_dir = project_dir / "runs" / "r1"
    manifest = Manifest(project_dir)
    first = manifest.get(version["parts"]["train"][0])
    path = manifest.file_path(first)
    original = path.read_bytes()
    path.write_bytes(original + b"edited")
    with pytest.raises(RunnerError) as info:
        build_spec(project_dir, RunConfig(dataset=1, model="resnet18"), model, run_dir, version)
    assert (info.value.code, info.value.params) == ("dataset_images_changed", {"count": 1})
    path.unlink()
    with pytest.raises(RunnerError) as info:
        build_spec(project_dir, RunConfig(dataset=1, model="resnet18"), model, run_dir, version)
    assert (info.value.code, info.value.params) == ("dataset_images_missing", {"count": 1})
    path.write_bytes(original)
    assert build_spec(project_dir, RunConfig(dataset=1, model="resnet18"), model, run_dir, version)
