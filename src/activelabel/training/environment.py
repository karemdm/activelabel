"""The Python environment that holds the training packages (see docs/TRAINING.md, "Principles").

Training packages (PyTorch, timm, ultralytics...) live in their own virtual environment, so
ActiveLabel itself stays light. The environment is created and filled on demand: only the missing
packages are installed, PyTorch with GPU support when an NVIDIA GPU is found. Nothing but the
location of the environment is recorded (``<workspace>/training/env.json``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform == "win32"
MAX_PATH_LENGTH = 90  # longer environment paths break the 260-character limit on Windows
CUDA_TAGS = ("cu126", "cu124", "cu121", "cu118")  # newest first
TORCH_PACKAGES = ("torch", "torchvision")
TORCH_INDEX = "https://download.pytorch.org/whl/"

_DUMP_SCRIPT = (
    "import importlib.metadata as m, json;"
    "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))"
)
_DEVICE_SCRIPT = """
import json, torch
mps = getattr(getattr(torch, "backends", None), "mps", None)
cuda = torch.cuda.is_available()
print(json.dumps({
    "torch": torch.__version__,
    "cuda": cuda,
    "cuda_version": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0) if cuda else None,
    "mps": bool(mps and mps.is_available()),
}))
"""

CheckFn = Callable[[], None]
StepFn = Callable[[str], None]


class TrainingEnvError(RuntimeError):
    """A readable problem with the training environment (creation, install, detection)."""


@dataclass
class GpuInfo:
    """An NVIDIA GPU found with ``nvidia-smi``."""

    name: str
    driver: str
    cuda: str | None  # "12.4"; None when the output does not say

    def as_dict(self) -> dict[str, Any]:
        """Plain data for the API and run.json."""
        return {"name": self.name, "driver": self.driver, "cuda": self.cuda}


def normalize(name: str) -> str:
    """Distribution name as pip compares it (case and ``-_.`` are the same)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """Name part of a requirement such as ``timm>=1.0`` or ``torch[extra]==2``."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1) if match else requirement.strip()


def _is_torch(requirement: str) -> bool:
    return normalize(requirement_name(requirement)) in TORCH_PACKAGES


# ---------------------------------------------------------------------------
# Subprocess layer (replaced in tests)


def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0


def capture(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    """Run a short command and return (exit code, stdout); (-1, "") when it cannot run."""
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_no_window_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return done.returncode, done.stdout or ""


LOG = logging.getLogger(__name__)


def _clear_readonly(function: Callable[[str], Any], path: str, *_: Any) -> None:
    """``rmtree`` error handler: clear the read-only flag and retry once (never raises)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        pass


def remove_tree(path: Path) -> None:
    """Remove a folder or file as well as possible, retrying; never raises (a leftover is cleaned later)."""
    try:
        for _ in range(5):
            if path.is_dir() and not path.is_symlink():
                if sys.version_info >= (3, 12):
                    shutil.rmtree(path, onexc=_clear_readonly)
                else:
                    shutil.rmtree(path, onerror=_clear_readonly)
            else:
                path.unlink(missing_ok=True)
            if not path.exists():
                return
            time.sleep(0.2)
    except OSError:
        pass
    LOG.warning("Could not remove %s", path)


def kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill a process and everything it started (pip starts build subprocesses)."""
    import psutil

    victims: list[Any] = []
    try:
        parent = psutil.Process(proc.pid)
        victims = [*parent.children(recursive=True), parent]
    except psutil.Error:
        pass
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


def run_logged(cmd: list[str], log: Path, check: CheckFn, poll: float = 0.5) -> int:
    """Run ``cmd`` appending its output to ``log``; poll ``check`` and kill the process tree if it raises."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", errors="replace") as out:
        out.write(f"\n$ {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=_no_window_flags(),
        )
        try:
            while True:
                try:
                    return proc.wait(timeout=poll)
                except subprocess.TimeoutExpired:
                    check()
        except BaseException:
            kill_tree(proc)
            raise


# ---------------------------------------------------------------------------
# Pure helpers


def parse_nvidia_smi(query_output: str, plain_output: str = "") -> GpuInfo | None:
    """Parse the CSV query (``name, driver``) and, when given, the CUDA version of the plain output."""
    for line in query_output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", plain_output)
            return GpuInfo(name=parts[0], driver=parts[1], cuda=match.group(1) if match else None)
    return None


def cuda_tag(cuda_version: str | None) -> str | None:
    """Newest PyTorch CUDA build (``cu124``...) not above the driver's CUDA version, or None."""
    match = re.match(r"(\d+)\.(\d+)", cuda_version or "")
    if not match:
        return None
    limit = (int(match.group(1)), int(match.group(2)))
    for tag in CUDA_TAGS:
        digits = tag[2:]  # "126" -> 12.6
        if (int(digits[:-1]), int(digits[-1])) <= limit:
            return tag
    return None


def pip_commands(
    python: Path, packages: Iterable[str], gpu: GpuInfo | None, platform_name: str | None = None
) -> list[list[str]]:
    """The ``pip install`` commands for ``packages``: PyTorch from its own index, the rest from PyPI."""
    platform_name = platform_name or sys.platform
    packages = list(packages)
    base = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    torch_like = [p for p in packages if _is_torch(p)]
    others = [p for p in packages if not _is_torch(p)]
    commands: list[list[str]] = []
    if torch_like:
        if platform_name == "darwin":
            commands.append([*base, *torch_like])
        else:
            tag = cuda_tag(gpu.cuda) if gpu else None
            commands.append([*base, *torch_like, "--index-url", f"{TORCH_INDEX}{tag or 'cpu'}"])
    if others:
        commands.append([*base, *others])
    return commands


# ---------------------------------------------------------------------------


class TrainingEnvironment:
    """The training environment of a workspace."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace)
        self.training_dir = self.workspace / "training"
        self.record = self.training_dir / "env.json"
        self.log = self.training_dir / "install.log"

    # -- location --------------------------------------------------------

    def _default_dir(self) -> Path:
        preferred = self.training_dir / "env"
        if not IS_WINDOWS or len(str(preferred)) <= MAX_PATH_LENGTH:
            return preferred
        digest = hashlib.sha256(str(self.workspace.resolve()).encode("utf-8")).hexdigest()[:8]
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "activelabel" / f"train-{digest}"

    @property
    def path(self) -> Path:
        """Folder of the environment; stable once chosen (recorded in ``env.json`` when not the default)."""
        try:
            recorded = json.loads(self.record.read_text(encoding="utf-8")).get("path")
            if recorded:
                return Path(recorded)
        except (OSError, ValueError, AttributeError):
            pass
        chosen = self._default_dir()
        if chosen != self.training_dir / "env":
            self.training_dir.mkdir(parents=True, exist_ok=True)
            self.record.write_text(json.dumps({"path": str(chosen)}), encoding="utf-8")
        return chosen

    def python(self) -> Path:
        """Python executable of the environment (may not exist yet)."""
        if IS_WINDOWS:
            return self.path / "Scripts" / "python.exe"
        return self.path / "bin" / "python"

    def exists(self) -> bool:
        """Whether the environment has been created and works (a venv without pip does not count)."""
        python = self.python()
        if not python.exists():
            return False
        code, _ = capture([str(python), "-m", "pip", "--version"], timeout=60)
        return code == 0

    # -- detection -------------------------------------------------------

    def gpu(self) -> GpuInfo | None:
        """The NVIDIA GPU, found with ``nvidia-smi`` (no torch needed); None when absent or failing."""
        code, out = capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if code != 0 or parse_nvidia_smi(out) is None:
            return None
        code, plain = capture(["nvidia-smi"])
        return parse_nvidia_smi(out, plain if code == 0 else "")

    @staticmethod
    def mps_hint() -> bool:
        """Apple silicon: PyTorch can use the GPU (MPS) with the plain PyPI build, nothing special."""
        return sys.platform == "darwin" and platform.machine() == "arm64"

    def installed(self) -> dict[str, str]:
        """``{distribution name: version}`` in the environment; empty when it does not exist."""
        if not self.exists():
            return {}
        code, out = capture([str(self.python()), "-c", _DUMP_SCRIPT], timeout=120)
        if code != 0:
            return {}
        try:
            data = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def missing(self, packages: Iterable[str]) -> list[str]:
        """The requirements whose distribution is not installed."""
        have = {normalize(n) for n in self.installed()}
        return [p for p in packages if normalize(requirement_name(p)) not in have]

    # -- changes ---------------------------------------------------------

    def _fail(self, what: str) -> TrainingEnvError:
        return TrainingEnvError(f"{what}. Details in {self.log}")

    def create(self, check: CheckFn, step: StepFn) -> None:
        """Create the virtual environment when it is missing (or broken: no Python or no pip).

        A creation that is cancelled or fails does not leave a half-made folder behind.
        """
        if self.exists():
            return
        step("train_env_create")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():  # left by an earlier failure, or a venv without pip
            remove_tree(self.path)
        try:
            code = run_logged([sys.executable, "-m", "venv", str(self.path)], self.log, check)
            created = code == 0 and self.exists()
        except BaseException:
            remove_tree(self.path)
            raise
        if not created:
            remove_tree(self.path)
            raise self._fail("Could not create the training environment")

    def ensure(self, packages: Iterable[str], check: CheckFn, step: StepFn) -> None:
        """Create the environment if needed and install the missing packages."""
        self.create(check, step)
        todo = self.missing(list(packages))
        if not todo:
            return
        step("train_env_install:" + ",".join(todo))
        gpu = self.gpu() if any(_is_torch(p) for p in todo) else None
        for cmd in pip_commands(self.python(), todo, gpu):
            check()
            if run_logged(cmd, self.log, check) != 0:
                raise self._fail(f"Could not install {', '.join(todo)}")

    def device_info(self) -> dict[str, Any]:
        """Torch version, CUDA and MPS availability and device name (for run.json); empty without torch."""
        if "torch" not in {normalize(n) for n in self.installed()}:
            return {}
        code, out = capture([str(self.python()), "-c", _DEVICE_SCRIPT], timeout=120)
        if code != 0:
            return {}
        try:
            data = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}
        return data if isinstance(data, dict) else {}
