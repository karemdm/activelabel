import hashlib
import json
import re
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
    pin,
    pinned_version,
    pip_commands,
)
from activelabel.training.licences import LicenceNotAccepted, LicenceStore

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
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (0, "{}"))  # pip works
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
    assert not env.record.exists()  # reading the location does not write to the disk
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (0, "{}"))
    monkeypatch.setattr(envmod, "run_logged", Recorder(env))
    env.create(nothing, lambda step: None)
    assert json.loads(env.record.read_text())["path"] == str(
        expected
    )  # it is recorded when the environment is made
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
    ("out", "count"),
    [("NVIDIA A\n", 1), ("NVIDIA A\nNVIDIA B\n", 2), ("NVIDIA A\n\n  \nNVIDIA B", 2), ("", 0), ("\n", 0)],
)
def test_gpu_count_counts_the_lines(env, monkeypatch, out, count):
    seen = []

    def fake(cmd, timeout=30):
        seen.append(cmd)
        return 0, out

    monkeypatch.setattr(envmod, "capture", fake)
    assert env.gpu_count() == count
    assert seen == [["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]]


@pytest.mark.parametrize("code", [-1, 9])
def test_gpu_count_is_zero_when_nvidia_smi_fails(env, monkeypatch, code):
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (code, "NVIDIA A\n"))
    assert env.gpu_count() == 0


@pytest.mark.parametrize(
    ("cuda", "tag"),
    [
        ("13.2", "cu130"),
        ("13.0", "cu130"),
        ("12.8", "cu126"),
        ("12.6", "cu126"),
        ("12.5", None),  # no build of the pinned torch for this driver: CPU
        ("11.8", None),
        (None, None),
        ("abc", None),
    ],
)
def test_cuda_tag(cuda, tag):
    assert cuda_tag(cuda) == tag


PY = Path("py")


def test_pip_commands_gpu_cpu_mac_other():
    gpu = GpuInfo("RTX", "561", "12.6")
    cmds = pip_commands(PY, ["torch", "torchvision", "timm"], gpu, "win32")
    assert cmds[0][-4:] == ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"]
    assert "--index-url" in cmds[0] and cmds[0][-1].endswith("/cu126")
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
    torch = envmod.PINNED_VERSIONS["torch"]
    make_installed(env, monkeypatch, {"PyYAML": "6.0", "torch": f"{torch}+cu126", "typing_extensions": "4"})
    assert env.installed()["torch"] == f"{torch}+cu126"
    assert env.missing(["torch", "timm>=1.0", "pyyaml", "typing-extensions", "onnx"]) == ["timm>=1.0", "onnx"]


def test_a_pinned_package_at_another_version_is_missing(env, monkeypatch):
    make_installed(env, monkeypatch, {"torch": "1.0.0", "timm": "0.1"})
    # a bare name must be at its pinned version; a requirement with its own specifier only needs the package
    assert env.missing(["torch", "timm>=0.1", "Torch"]) == ["torch", "Torch"]


def test_pin_adds_the_tested_version_to_a_bare_pinned_name():
    assert pin("torch") == f"torch=={envmod.PINNED_VERSIONS['torch']}"
    assert pin("Timm") == f"Timm=={envmod.PINNED_VERSIONS['timm']}"
    assert pin("timm>=1.0") == "timm>=1.0"
    assert pin("pillow") == "pillow"
    assert pinned_version("torch[extra]") is None


def test_every_cuda_tag_is_a_build_and_the_torch_packages_are_pinned():
    assert all(re.fullmatch(r"cu\d{3}", tag) for tag in envmod.CUDA_TAGS)
    assert list(envmod.CUDA_TAGS) == sorted(envmod.CUDA_TAGS, reverse=True)
    for name in (*envmod.TORCH_PACKAGES, "timm"):
        assert re.fullmatch(r"\d+(\.\d+)+", envmod.PINNED_VERSIONS[name])


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


def test_create_failure_removes_half_made_folder(env, monkeypatch):
    def run(cmd, log, check, poll=0.5):
        Recorder(env)(cmd, log, check)  # leaves a python behind, then fails
        (env.path / "lib").mkdir()
        return 1

    monkeypatch.setattr(envmod, "run_logged", run)
    with pytest.raises(TrainingEnvError):
        env.create(nothing, lambda s: None)
    assert not env.path.exists()


def test_create_cancel_removes_half_made_folder(env, monkeypatch):
    def run(cmd, log, check, poll=0.5):
        Recorder(env)(cmd, log, check)
        raise KeyboardInterrupt

    monkeypatch.setattr(envmod, "run_logged", run)
    with pytest.raises(KeyboardInterrupt):
        env.create(nothing, lambda s: None)
    assert not env.path.exists()


def test_venv_without_pip_is_recreated(env, monkeypatch):
    rec = Recorder(env)
    monkeypatch.setattr(envmod, "run_logged", rec)
    pip_ok = {"value": False}
    monkeypatch.setattr(envmod, "capture", lambda cmd, timeout=30: (0 if pip_ok["value"] else 1, "{}"))
    env.python().parent.mkdir(parents=True)
    env.python().write_text("")
    (env.path / "stale.txt").write_text("old")
    assert not env.exists()
    pip_ok["value"] = False
    steps = []

    def run_and_fix(cmd, log, check, poll=0.5):
        code = rec(cmd, log, check)
        pip_ok["value"] = True  # the new venv has pip
        return code

    monkeypatch.setattr(envmod, "run_logged", run_and_fix)
    env.create(nothing, steps.append)
    assert steps == ["train_env_create"] and not (env.path / "stale.txt").exists()
    assert env.exists() and len(rec.cmds) == 1


def test_remove_tree_clears_read_only_and_never_raises(tmp_path):
    import os
    import stat

    folder = tmp_path / "x"
    (folder / "sub").mkdir(parents=True)
    locked = folder / "sub" / "f.txt"
    locked.write_text("1")
    os.chmod(locked, stat.S_IREAD)
    envmod.remove_tree(folder)
    assert not folder.exists()
    envmod.remove_tree(tmp_path / "missing")  # nothing to remove: no error


def test_remove_tree_gives_up_quietly(tmp_path, monkeypatch):
    folder = tmp_path / "stuck"
    folder.mkdir()
    monkeypatch.setattr(envmod.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(envmod.time, "sleep", lambda s: None)
    envmod.remove_tree(folder)  # still there, no exception
    assert folder.exists()


def test_ensure_installs_only_missing(env, monkeypatch):
    rec = Recorder(env)
    monkeypatch.setattr(envmod, "run_logged", rec)
    monkeypatch.setattr(envmod.TrainingEnvironment, "gpu", lambda self: GpuInfo("RTX", "561", "12.6"))
    # the CUDA build is chosen on Linux and Windows, never on macOS
    monkeypatch.setattr(envmod.sys, "platform", "linux")
    state = {"dists": {}}

    def fake(cmd, timeout=30):
        return 0, json.dumps(state["dists"])

    monkeypatch.setattr(envmod, "capture", fake)
    steps = []
    env.ensure(["torch", "torchvision", "timm"], nothing, steps.append)
    assert steps == ["train_env_create", "train_env_install:torch,torchvision,timm"]
    pips = [c for c in rec.cmds if "pip" in c]
    pins = envmod.PINNED_VERSIONS
    assert len(pips) == 2 and pips[0][-1].endswith("cu126")
    assert f"torch=={pins['torch']}" in pips[0] and f"torchvision=={pins['torchvision']}" in pips[0]
    assert pips[1][-1] == f"timm=={pins['timm']}"

    state["dists"] = {name: pins[name] for name in ("torch", "torchvision", "timm")}
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


def test_ensure_refuses_a_licensed_package_until_accepted(env, monkeypatch):
    ran = []
    monkeypatch.setattr(envmod, "run_logged", lambda cmd, log, check, poll=0.5: ran.append(cmd) or 0)
    with pytest.raises(LicenceNotAccepted) as caught:
        env.ensure(["ultralytics"], nothing, lambda s: None)
    assert caught.value.code == "licence_not_accepted"
    assert ran == [] and not env.path.exists()  # not even the environment is created
    LicenceStore(env.workspace).accept("ultralytics")
    monkeypatch.setattr(TrainingEnvironment, "create", lambda self, check, step: None)
    monkeypatch.setattr(TrainingEnvironment, "installed", lambda self: {})
    env.ensure(["ultralytics"], nothing, lambda s: None)
    assert any("ultralytics==" + envmod.PINNED_VERSIONS["ultralytics"] in cmd for cmd in ran)
