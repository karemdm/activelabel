"""Fakes shared by the training tests: a worker script, a training environment and wait helpers."""

import sys
import textwrap
import time
from pathlib import Path

from activelabel.training.environment import TrainingEnvironment

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
    size = int(spec["params"].get("download", 0))
    if size:  # fake pretrained weights, downloaded where the runner sends the caches
        hub = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
        hub.mkdir(parents=True, exist_ok=True)
        (hub / "fake-backbone.pth").write_bytes(b"w" * size)
        (Path.cwd() / "yolo-fake.pt").write_bytes(b"y" * size)
    print("not json noise", flush=True)
    print(json.dumps({"event": "log", "message": "hello"}), flush=True)
    print(json.dumps({"event": "setup", "device": "cpu", "amp": False}), flush=True)
    print(json.dumps({"event": "setup", "batch_size": 4}), flush=True)
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
    (out / "best.pt").write_text("weights")
    (out / "last.pt").write_text("weights")
    score = spec["params"].get("score", 1.0) + 0.01 * (spec["fold"] or 0)
    results = {"best_epoch": 2, "val": {"accuracy": score}, "test": None}
    print(json.dumps({"event": "done", "results": results}), flush=True)
    """
)

# the embeddings worker (workers/embed.py) without torch: three clear groups by position. The device of the
# spec drives it: "fail" reports an error, "skip" skips the first image (unreadable), "slow" waits a minute
FAKE_EMBED = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np

    spec = json.loads(Path(sys.argv[1]).read_text())
    items, device = spec["items"], spec["device"]
    if device == "fail":
        print(json.dumps({"event": "error", "message": "boom"}), flush=True)
        sys.exit(1)
    ids, rows = [], []
    for step, item in enumerate(items, start=1):
        if device == "skip" and step == 1:
            print(json.dumps({"event": "skipped", "id": item["id"], "message": "unreadable"}), flush=True)
        else:
            centre = np.zeros(8)
            centre[step % 3] = 10.0
            rows.append(centre + np.random.default_rng(step).normal(0, 0.1, 8))
            ids.append(item["id"])
        print(json.dumps({"event": "progress", "step": step, "steps": len(items)}), flush=True)
        if device == "slow":
            time.sleep(60)
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", np.array(rows, dtype=np.float32))
    (out / "ids.json").write_text(json.dumps(ids))
    results = {"images": len(ids), "dims": 8, "backbone": spec["backbone"]}
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


def tmp_dirs(runner):
    return list(runner.tmp_root.iterdir()) if runner.tmp_root.exists() else []


def finished(store, n):
    return lambda: (r := store.get(n))["status"] in ("done", "failed", "cancelled") and r


def settled(store, n):
    """Finished, and the runner's last write done too: after the status, it deletes the temporary folder and
    records ``downloads`` on the run (#643). A test that edits ``run.json`` must wait for it, or that late
    write puts back what the test removed."""
    return lambda: (r := store.get(n))["status"] in ("done", "failed", "cancelled") and "downloads" in r and r
