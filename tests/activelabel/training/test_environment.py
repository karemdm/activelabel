import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from activelabel.training import environment as envmod
from activelabel.training.environment import (
    GpuInfo,
    TrainingEnvError,
    TrainingEnvironment,
    cuda_tag,
    parse_nvidia_smi,
    pip_commands,
)

PLAIN = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.61                 Driver Version: 551.61         CUDA Version: 12.4     |
+-----------------------------------------------------------------------------------------+
"""
QUERY = "NVIDIA GeForce RTX 3060, 551.61\n"


def nothing():
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "IS_WINDOWS", False)
    return TrainingEnvironment(tmp_path)


def test_default_location(env, tmp_path):
    assert env.path == tmp_path / "training" / "env"
    assert not env.record.exists()
    assert env.python() == env.path / "bin" / "python"
    assert not env.exists()


def test_long_windows_path_uses_localappdata_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    workspace = tmp_path / ("x" * 80) / "ws"
    workspace.mkdir(parents=True)
    env = TrainingEnvironment(workspace)
    digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:8]
    expected = tmp_path / "local" / "activelabel" / f"train-{digest}"
    assert env.path == expected
    assert json.loads(env.record.read_text())["path"] == str(expected)
    assert env.python() == expected / "Scripts" / "python.exe"
    # stable: the record wins even if the default rule would change
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "other"))
    assert TrainingEnvironment(workspace).path == expected


def test_short_windows_path_stays_in_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "IS_WINDOWS", True)
    monkeypatch.setattr(envmod, "MAX_PATH_LENGTH", 10_000)
    env = TrainingEnvironment(tmp_path)
    assert env.path == tmp_path / "training" / "env"
    assert not env.record.exists()


def test_parse_nvidia_smi():
    info = parse_nvidia_smi(QUERY, PLAIN)
    assert info == GpuInfo("NVIDIA GeForce RTX 3060", "551.61", "12.4")
    assert parse_nvidia_smi(QUERY).cuda is None  # type: ignore[union-attr]
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("No devices were found\n") is None


def test_gpu_with_without_and_failing(env, monkeypatch):
    def fake(cmd, timeout=30):
        return (0, QUERY) if len(cmd) > 1 else (0, PLAIN)

    monkeypatch.setattr(envmod, "capture", fake)
    gpu = env.gpu()
    assert gpu is not None and gpu.name.endswith("3060") and gpu.cuda == "12.4"
    assert gpu.as_dict()["driver"] == "551.61"

    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (-1, ""))
    assert env.gpu() is None
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (9, "boom"))
    assert env.gpu() is None
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (0, ""))
    assert env.gpu() is None


@pytest.mark.parametrize(
    ("cuda", "tag"),
    [
        ("12.8", "cu126"),
        ("12.6", "cu126"),
        ("12.5", "cu124"),
        ("12.4", "cu124"),
        ("12.2", "cu121"),
        ("11.8", "cu118"),
        ("11.7", None),
        ("13.0", "cu126"),
        (None, None),
        ("abc", None),
    ],
)
def test_cuda_tag(cuda, tag):
    assert cuda_tag(cuda) == tag


PY = Path("py")


def test_pip_commands_gpu_cpu_mac_other():
    gpu = GpuInfo("RTX", "551", "12.4")
    cmds = pip_commands(PY, ["torch", "torchvision", "timm"], gpu, "win32")
    assert cmds[0][-4:] == ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
    assert "--index-url" in cmds[0] and cmds[0][-1].endswith("/cu124")
    assert cmds[1][-1] == "timm" and "--index-url" not in cmds[1]
    assert cmds[0][1:4] == ["-m", "pip", "install"]

    cpu = pip_commands(PY, ["torch", "torchvision"], None, "linux")
    assert len(cpu) == 1 and cpu[0][-1].endswith("/whl/cpu")
    no_cuda = pip_commands(PY, ["Torch"], GpuInfo("RTX", "1", None), "linux")
    assert no_cuda[0][-1].endswith("/whl/cpu")

    mac = pip_commands(PY, ["torch", "torchvision"], None, "darwin")
    assert len(mac) == 1 and "--index-url" not in mac[0]

    other = pip_commands(PY, ["ultralytics", "pycocotools"], gpu, "linux")
    assert len(other) == 1 and other[0][-2:] == ["ultralytics", "pycocotools"]


def make_installed(env, monkeypatch, dists):
    py = env.python()
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")

    def fake(cmd, timeout=30):
        if cmd[0] == str(py):
            return 0, "warning line\n" + json.dumps(dists) + "\n"
        return 1, ""

    monkeypatch.setattr(envmod, "capture", fake)


def test_installed_and_missing(env, monkeypatch):
    assert env.installed() == {}
    assert env.missing(["timm"]) == ["timm"]
    make_installed(env, monkeypatch, {"PyYAML": "6.0", "torch": "2.5.0", "typing_extensions": "4"})
    assert env.installed()["torch"] == "2.5.0"
    assert env.missing(["torch", "timm>=1.0", "pyyaml", "typing-extensions", "onnx"]) == ["timm>=1.0", "onnx"]


def test_device_info(env, monkeypatch):
    assert env.device_info() == {}
    make_installed(env, monkeypatch, {"timm": "1"})
    assert env.device_info() == {}  # no torch

    info = {"torch": "2.5.0", "cuda": False, "device_name": None, "mps": False}

    def fake(cmd, timeout=30):
        out = {"torch": "2.5.0"} if "metadata" in cmd[2] else info
        return 0, json.dumps(out) + "\n"

    monkeypatch.setattr(envmod, "capture", fake)
    assert env.device_info() == info


class Recorder:
    def __init__(self, env, code=0):
        self.env, self.code, self.cmds = env, code, []

    def __call__(self, cmd, log, check, poll=0.5):
        self.cmds.append(cmd)
        if "venv" in cmd:
            py = self.env.python()
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        return self.code


def test_create_runs_venv_once(env, monkeypatch):
    rec = Recorder(env)
    monkeypatch.setattr(envmod, "run_logged", rec)
    steps = []
    env.create(nothing, steps.append)
    env.create(nothing, steps.append)
    assert steps == ["train_env_create"]
    assert rec.cmds == [[envmod.sys.executable, "-m", "venv", str(env.path)]]


def test_create_failure_mentions_log(env, monkeypatch):
    monkeypatch.setattr(envmod, "run_logged", lambda *a, **k: 1)
    with pytest.raises(TrainingEnvError, match="install.log"):
        env.create(nothing, lambda s: None)


def test_ensure_installs_only_missing(env, monkeypatch):
    rec = Recorder(env)
    monkeypatch.setattr(envmod, "run_logged", rec)
    monkeypatch.setattr(envmod.TrainingEnvironment, "gpu", lambda self: GpuInfo("RTX", "551", "12.4"))
    state = {"dists": {}}

    def fake(cmd, timeout=30):
        return 0, json.dumps(state["dists"])

    monkeypatch.setattr(envmod, "capture", fake)
    steps = []
    env.ensure(["torch", "torchvision", "timm"], nothing, steps.append)
    assert steps == ["train_env_create", "train_env_install:torch,torchvision,timm"]
    pips = [c for c in rec.cmds if "pip" in c]
    assert len(pips) == 2 and pips[0][-1].endswith("cu124") and pips[1][-1] == "timm"

    state["dists"] = {"torch": "2", "torchvision": "1", "timm": "1"}
    rec.cmds.clear()
    steps.clear()
    env.ensure(["torch", "timm"], nothing, steps.append)
    assert rec.cmds == [] and steps == []


def test_ensure_install_failure(env, monkeypatch):
    def run(cmd, log, check, poll=0.5):
        if "venv" in cmd:
            return Recorder(env)(cmd, log, check)
        return 1

    monkeypatch.setattr(envmod, "run_logged", run)
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (0, "{}"))
    with pytest.raises(TrainingEnvError) as err:
        env.ensure(["timm"], nothing, lambda s: None)
    assert str(env.log) in str(err.value) and "timm" in str(err.value)


class FakeProc:
    pid = 1234

    def __init__(self):
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        raise subprocess.TimeoutExpired("pip", timeout)


class Cancelled(Exception):
    pass


def test_cancel_kills_process_tree(tmp_path, monkeypatch):
    proc = FakeProc()
    killed = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(envmod, "kill_tree", killed.append)
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 2:
            raise Cancelled

    log = tmp_path / "t" / "install.log"
    with pytest.raises(Cancelled):
        envmod.run_logged(["pip", "install", "x"], log, check, poll=0.01)
    assert killed == [proc] and proc.waits == 2
    assert "pip install x" in log.read_text()


def test_run_logged_returns_exit_code(tmp_path, monkeypatch):
    class Done:
        def wait(self, timeout=None):
            return 3

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Done())
    assert envmod.run_logged(["x"], tmp_path / "l.log", nothing) == 3
