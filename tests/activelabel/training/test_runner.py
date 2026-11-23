import importlib.util
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.datasets import DatasetStore
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training.catalog import get_model
from activelabel.training.config import RunConfig
from activelabel.training.environment import TrainingEnvironment
from activelabel.training.runner import RunnerError, TrainingRunner, build_spec, download_env
from activelabel.training.runs import RunStore

FAKE_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path

    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(spec["output_dir"])
    fake = spec["params"].get("fake", "ok")
    gate = spec["params"].get("gate", "")
    started = time.time()
    names = ("TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "TMPDIR", "TEMP", "TMP")
    (out / "env.json").write_text(json.dumps({
        "env": {k: os.environ.get(k) for k in names}, "cwd": os.getcwd(), "spec": sys.argv[1]}))
    print("not json noise", flush=True)
    print(json.dumps({"event": "log", "message": "hello"}), flush=True)
    for epoch in (1, 2):
        print(json.dumps({"event": "progress", "phase": "training", "epoch": epoch, "step": 1, "steps": 2}),
              flush=True)
        print(json.dumps({"event": "epoch", "epoch": epoch, "train_loss": 1.0 / epoch, "val_loss": 0.5,
                          "val_accuracy": 0.5 * epoch, "val_macro_f1": 0.4, "best": True}), flush=True)
        if fake == "sleep":
            time.sleep(60)
        if gate:
            while not Path(gate).exists():
                time.sleep(0.05)
    if fake == "fail":
        print(json.dumps({"event": "error", "message": "boom"}), flush=True)
        sys.exit(1)
    (out / "times.json").write_text(json.dumps([started, time.time()]))
    results = {"best_epoch": 2, "val": {"accuracy": 1.0}, "test": None}
    print(json.dumps({"event": "done", "results": results}), flush=True)
    """
)


class FakeEnv(TrainingEnvironment):
    ensured: list[list[str]]

    def ensure(self, packages, check, step):
        self.ensured = [*getattr(self, "ensured", []), list(packages)]
        step("train_env_install:torch")

    def python(self):
        return Path(sys.executable)

    def installed(self):
        return {"torch": "2.0.0", "Pillow": "10.0", "other": "1"}

    def device_info(self):
        return {"cuda": False, "torch": "2.0.0"}


def wait_for(predicate, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timeout")


@pytest.fixture
def setup(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a", "b"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(20)]
    assert upload(client, 1, paths).status_code == 200
    ids = [r.id for r in state.manifest(1).all()]
    images = {i: [Label("a" if k % 2 else "b")] for k, i in enumerate(ids)}
    save_annotations(state.store.dir(1), AnnotationSet(Task.CLASSIFICATION, ["a", "b"], images, {}))
    assert client.post("/api/projects/1/datasets", json={}).status_code == 200
    assert client.post("/api/projects/1/datasets", json={"kfold": 2}).status_code == 200
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: script,
        poll=0.05,
    )
    state.runner = runner
    yield client, state, runner
    runner.shutdown()


def new_run(state, **params):
    config = RunConfig(dataset=1, model="resnet18", params=params, metric="accuracy")
    return RunStore(state.store.dir(1)).create(config)["number"]


def tmp_dirs(runner):
    return list(runner.tmp_root.iterdir()) if runner.tmp_root.exists() else []


def finished(store, n):
    return lambda: (r := store.get(n))["status"] in ("done", "failed", "cancelled") and r


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
    assert tmp_dirs(runner) == []
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
    assert tmp_dirs(runner) == []


def test_not_yet_supported_fails_before_install(setup):
    _, state, runner = setup
    store = RunStore(state.store.dir(1))
    n = store.create(RunConfig(dataset=2, model="resnet18", fold="all"))["number"]
    runner.enqueue(1, n)
    run = wait_for(finished(store, n))
    assert run["status"] == "failed" and "comes later" in run["error"]
    assert not hasattr(runner.environment, "ensured")


def test_cancel_running(setup):
    client, state, runner = setup
    n = new_run(state, fake="sleep")
    runner.enqueue(1, n)
    store = RunStore(state.store.dir(1))
    wait_for(lambda: store.metrics(n))
    resp = client.post(f"/api/projects/1/runs/{n}/cancel")
    assert resp.status_code == 200 and resp.json()["status"] == "cancelled"
    assert tmp_dirs(runner) == []
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
    with pytest.raises(RunnerError, match="comes later"):
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
