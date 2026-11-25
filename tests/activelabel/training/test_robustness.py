"""Robustness of the training runner, the model store and the training API (fixes after a review)."""

import json
import os
import stat
import textwrap

import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import finished, tmp_dirs, wait_for

from activelabel.datasets import DatasetStore
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training import runner as runner_module
from activelabel.training.catalog import ParamSpec
from activelabel.training.config import ConfigError, RunConfig, validate
from activelabel.training.models import ModelNotFound, ModelStore
from activelabel.training.runs import RunStore

FAKE_EXPORT_PARTIAL = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    Path(spec["output"]).with_suffix(".part").write_bytes(b"half")
    print(json.dumps({"event": "error", "message": "disk full"}), flush=True)
    sys.exit(1)
    """
)


def new_run(state, **params):
    config = RunConfig(dataset=1, model="resnet18", params=params, metric="accuracy")
    return RunStore(state.store.dir(1)).create(config)["number"]


def composite(state, **kwargs):
    config = RunConfig(model="resnet18", metric="accuracy", **{"dataset": 1, **kwargs})
    return RunStore(state.store.dir(1)).create(config)["number"]


def save_model(client, state, runner, **params):
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2, **params)
    runner.enqueue(1, n)
    wait_for(lambda: store.get(n)["status"] == "done")
    assert client.post("/api/projects/1/models", json={"run": n}).status_code == 200
    return n


# -- has_active ------------------------------------------------------------------------------


def test_has_active_covers_training_until_it_ends(setup, tmp_path):
    _, state, runner = setup
    gate = tmp_path / "gate"
    assert runner.has_active(1) is False
    n = new_run(state, gate=str(gate))
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    assert runner.has_active(1) is True and runner.has_active(2) is False
    wait_for(lambda: store.get(n)["status"] == "running")
    assert runner.has_active(1) is True
    gate.write_text("go")
    wait_for(lambda: store.get(n)["status"] == "done")
    wait_for(lambda: runner.has_active(1) is False)
    assert runner._job_ids == {}


def test_has_active_covers_export(setup, tmp_path):
    client, state, runner = setup
    (tmp_path / "export_onnx.py").write_text("import json, sys, time\ntime.sleep(60)\n")
    save_model(client, state, runner)
    runner.enqueue_export(1, 1, "onnx")
    assert runner.has_active(1) is True
    wait_for(lambda: ModelStore(state.store.dir(1)).get(1)["exports"]["onnx"]["status"] == "running")
    assert runner.has_active(1) is True


def test_busy_blocks_project_and_image_deletion(setup, tmp_path):
    client, state, runner = setup
    gate = tmp_path / "gate"
    n = new_run(state, gate=str(gate))
    runner.enqueue(1, n)
    wait_for(lambda: RunStore(state.store.dir(1)).get(n)["status"] == "running")
    ids = [r.id for r in state.manifest(1).all()]
    assert (
        client.post("/api/projects/1/images/delete", json={"ids": ids[:1], "confirm": True}).status_code
        == 409
    )
    resp = client.request("DELETE", "/api/projects/1", json={"confirm_name": "P"})
    assert resp.status_code == 409 and "training" in resp.json()["detail"]
    assert state.project(1).name == "P"
    gate.write_text("go")
    wait_for(lambda: RunStore(state.store.dir(1)).get(n)["status"] == "done")
    wait_for(lambda: not runner.has_active(1))
    assert client.request("DELETE", "/api/projects/1", json={"confirm_name": "P"}).status_code == 200


# -- recover ---------------------------------------------------------------------------------


def test_recover_composite_children_and_current(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, mode="grid", grid={"score": [0.1, 0.2, 0.3]})
    store.update(
        n,
        status="running",
        current="c2",
        children=[
            {"id": "c1", "status": "done"},
            {"id": "c2", "status": "running"},
            {"id": "c3", "status": "queued"},
        ],
    )
    runner.recover()
    run = store.get(n)
    assert run["status"] == "failed" and run["error"] == "interrupted" and run["current"] is None
    assert [c["status"] for c in run["children"]] == ["done", "failed", "failed"]
    assert run["children"][1]["error"] == "interrupted" and run["children"][2]["error"] == "interrupted"


def test_recover_never_raises(setup, caplog):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state)
    store.update(n, status="running")
    broken = new_run(state)
    (store.path(broken) / "run.json").write_text("{not json")
    runner._project_ids = lambda: [1, 99]  # project 99 has no folder at all
    runner.recover()
    assert store.get(n)["status"] == "failed"

    def boom():
        raise RuntimeError("no list")

    runner._project_ids = boom
    runner.recover()  # nothing to recover, nothing raised
    assert "Could not list the projects" in caplog.text


def test_recover_removes_leftovers_even_read_only(setup):
    _, _, runner = setup
    leftover = runner.tmp_root / "r1-xyz" / "sub"
    leftover.mkdir(parents=True)
    locked = leftover / "weights.bin"
    locked.write_text("x")
    os.chmod(locked, stat.S_IREAD)
    stray = runner.tmp_root / "stray.txt"
    stray.write_text("y")
    runner.recover()
    assert tmp_dirs(runner) == []


def test_recover_survives_failing_cleanup(setup, monkeypatch):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state)
    store.update(n, status="running")
    runner.tmp_root.mkdir(parents=True, exist_ok=True)

    def bad_iterdir(self):
        raise PermissionError("denied")

    monkeypatch.setattr(type(runner.tmp_root), "iterdir", bad_iterdir)
    runner.recover()
    assert store.get(n)["status"] == "failed"


def test_remove_tree_never_raises(tmp_path, monkeypatch):
    stuck = tmp_path / "stuck"
    stuck.mkdir()
    monkeypatch.setattr(runner_module.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(runner_module.time, "sleep", lambda s: None)
    runner_module._remove_tree(stuck)  # still there, silently
    assert stuck.exists()


# -- dataset version deleted, classes recorded ------------------------------------------------


def test_run_records_classes_and_survives_deleted_version(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "done" and run["classes"] == ["a", "b"]
    DatasetStore(state.store.dir(1)).delete(1)
    resp = client.post("/api/projects/1/models", json={"run": n})
    assert resp.status_code == 200 and resp.json()["classes"] == ["a", "b"]


def test_old_run_without_classes_and_deleted_version_is_a_readable_error(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    wait_for(finished(store, n))
    run = json.loads((store.path(n) / "run.json").read_text())
    del run["classes"]
    (store.path(n) / "run.json").write_text(json.dumps(run))
    DatasetStore(state.store.dir(1)).delete(1)
    resp = client.post("/api/projects/1/models", json={"run": n})
    assert resp.status_code == 400 and "deleted" in resp.json()["detail"]


def test_old_run_without_classes_reads_the_version(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    wait_for(finished(store, n))
    run = json.loads((store.path(n) / "run.json").read_text())
    del run["classes"]
    (store.path(n) / "run.json").write_text(json.dumps(run))
    resp = client.post("/api/projects/1/models", json={"run": n})
    assert resp.status_code == 200 and resp.json()["classes"] == ["a", "b"]


def test_run_fails_readably_when_version_is_deleted(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state)
    DatasetStore(state.store.dir(1)).delete(1)
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed"
    assert run["error"] == "The data split version of this run was deleted."
    wait_for(lambda: tmp_dirs(runner) == [])  # removed right after the status is written


# -- composite runs: best fold value ------------------------------------------------------------


def test_composite_keeps_mean_and_best_fold_value(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = composite(state, dataset=2, fold="all", mode="grid", params={"epochs": 2}, grid={"score": [0.8, 0.4]})
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "done", run
    assert run["best"]["value"] == pytest.approx(0.805)
    assert run["best"]["best_fold_value"] == pytest.approx(0.81)  # fold 2 of the best combination
    results = store.results(n)
    assert results["best_value"] == pytest.approx(0.805) and results["best_fold_value"] == pytest.approx(0.81)
    card = client.post("/api/projects/1/models", json={"run": n}).json()
    assert card["value"] == pytest.approx(0.805) and card["best_fold_value"] == pytest.approx(0.81)


# -- export -----------------------------------------------------------------------------------------


def test_export_missing_files_and_stale_files(setup, tmp_path):
    client, state, runner = setup
    save_model(client, state, runner)
    store = ModelStore(state.store.dir(1))
    folder = store.path(1)
    (tmp_path / "export_onnx.py").write_text(FAKE_EXPORT_PARTIAL)
    (folder / "model.onnx").write_bytes(b"old")
    runner.enqueue_export(1, 1, "onnx")
    failed = wait_for(lambda: (e := store.get(1)["exports"]["onnx"])["status"] == "failed" and e)
    assert failed["error"] == "disk full"
    assert not (folder / "model.onnx").exists() and not (folder / "model.part").exists()
    assert wait_for(lambda: tmp_dirs(runner) == [])

    (folder / "config.yaml").unlink()
    runner.enqueue_export(1, 1, "onnx")
    failed = wait_for(
        lambda: (e := store.get(1)["exports"]["onnx"])["status"] == "failed" and "missing" in e["error"] and e
    )
    assert "best.pt" in failed["error"] and "FileNotFound" not in failed["error"]


def test_export_of_a_model_deleted_midway_is_readable(setup, monkeypatch):
    client, state, runner = setup
    save_model(client, state, runner)
    store = ModelStore(state.store.dir(1))
    original = ModelStore.get
    calls = {"n": 0}

    def vanishing(self, number):
        calls["n"] += 1
        if calls["n"] == 2:  # the first read is inside set_export(running); the model is then "deleted"
            raise ModelNotFound(number)
        return original(self, number)

    monkeypatch.setattr(ModelStore, "get", vanishing)
    runner._export(1, 1, "onnx", _NoProgress())
    monkeypatch.undo()
    failed = store.get(1)["exports"]["onnx"]
    assert failed["status"] == "failed" and failed["error"] == "This model was deleted."


class _NoProgress:
    def check(self):
        pass

    def update(self, **kwargs):
        pass


# -- queue bookkeeping ------------------------------------------------------------------------------


def test_cancelled_waiting_run_leaves_no_job_entry(setup, tmp_path):
    _, state, runner = setup
    gate = tmp_path / "gate"
    first, second = new_run(state, gate=str(gate)), new_run(state)
    runner.enqueue(1, first)
    runner.enqueue(1, second)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.get(first)["status"] == "running")
    runner.cancel(1, second)
    gate.write_text("go")
    wait_for(lambda: store.get(first)["status"] == "done")
    wait_for(lambda: not runner.has_active(1))
    assert runner._job_ids == {} and runner._pre_jobs == {}


# -- API ---------------------------------------------------------------------------------------------


def test_save_note_is_capped(setup):
    client, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = new_run(state, epochs=2)
    runner.enqueue(1, n)
    wait_for(finished(store, n))
    card = client.post("/api/projects/1/models", json={"run": n, "note": "  " + "x" * 900}).json()
    assert card["note"] == "x" * 500


def test_delete_queued_run_leaves_the_queue(setup, tmp_path):
    client, state, runner = setup
    gate = tmp_path / "gate"
    first, second = new_run(state, gate=str(gate)), new_run(state)
    runner.enqueue(1, first)
    runner.enqueue(1, second)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.get(first)["status"] == "running")
    resp = client.request("DELETE", f"/api/projects/1/runs/{second}", json={"confirm": True})
    assert resp.status_code == 200
    assert (second not in [r["number"] for r in store.list()]) and (1, second) not in runner._waiting
    gate.write_text("go")
    wait_for(lambda: store.get(first)["status"] == "done")
    wait_for(lambda: not runner.has_active(1))
    assert runner._job_ids == {}


def test_create_run_rejections(setup):
    client, state, runner = setup
    runs = "/api/projects/1/runs"
    assert client.post(runs, json={"dataset": 1, "model": "resnet18"}).status_code == 200
    # a version with no validation images
    no_val = client.post("/api/projects/1/datasets", json={"split": {"train": 100, "val": 0, "test": 0}})
    assert no_val.status_code == 200, no_val.text
    resp = client.post(runs, json={"dataset": no_val.json()["version"], "model": "resnet18"})
    assert resp.status_code == 400 and "no validation" in resp.json()["detail"]
    # a version that lost images
    state.manifest(1).delete([r.id for r in state.manifest(1).all()][:1])
    resp = client.post(runs, json={"dataset": 1, "model": "resnet18"})
    assert resp.status_code == 400 and "deleted" in resp.json()["detail"]


def test_yolo_is_rejected_before_the_environment_until_the_licence_is_accepted(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(10)]
    upload(client, 1, paths)
    ids = [r.id for r in state.manifest(1).all()]
    images = {i: [Label("a", bbox=[0, 0, 0.5, 0.5])] for i in ids}
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["a"], images, {}))
    assert client.post("/api/projects/1/datasets", json={}).status_code == 200
    resp = client.post("/api/projects/1/runs", json={"dataset": 1, "model": "yolov8n"})
    assert resp.status_code == 400 and resp.json()["code"] == "licence_not_accepted"
    assert client.get("/api/projects/1/runs").json() == {"runs": []}


# -- catalogue and configuration -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_param_rejects_non_finite_and_huge(bad):
    for kind in ("int", "float"):
        spec = ParamSpec("x", kind, 1, 0, 1000)
        with pytest.raises(ValueError):
            spec.check(bad)


def test_semantic_needs_batch_of_two():
    base = {"dataset": 1, "model": "deeplabv3_resnet50"}
    with pytest.raises(ConfigError, match="at least 2"):
        validate(RunConfig(**base, params={"batch_size": 1}), "segmentation")
    with pytest.raises(ConfigError, match="at least 2"):
        validate(RunConfig(**base, mode="grid", grid={"batch_size": [4, 1]}), "segmentation")
    assert validate(RunConfig(**base, params={"batch_size": 2}), "segmentation").params["batch_size"] == 2
    # other kinds keep accepting 1
    validate(RunConfig(dataset=1, model="resnet18", params={"batch_size": 1}), "classification")


def test_partial_export_is_gone_before_the_failure_is_recorded(setup, tmp_path, monkeypatch):
    client, state, runner = setup
    save_model(client, state, runner)
    store = ModelStore(state.store.dir(1))
    folder = store.path(1)
    (tmp_path / "export_onnx.py").write_text(FAKE_EXPORT_PARTIAL)
    seen = []
    original = ModelStore.set_export

    def watching(self, number, fmt, **fields):
        if fields.get("status") == "failed":
            seen.append((folder / "model.part").exists())
        return original(self, number, fmt, **fields)

    monkeypatch.setattr(ModelStore, "set_export", watching)
    runner.enqueue_export(1, 1, "onnx")
    wait_for(lambda: store.get(1)["exports"]["onnx"]["status"] == "failed")
    assert seen == [False]
