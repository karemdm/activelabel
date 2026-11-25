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

import psutil

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
        """Plain data for the API and run.json.

        Returns:
            ``name``, ``driver`` and ``cuda`` as a dict.
        """
        return {"name": self.name, "driver": self.driver, "cuda": self.cuda}


def normalize(name: str) -> str:
    """Distribution name as pip compares it (case and ``-_.`` are the same).

    Args:
        name: distribution name.

    Returns:
        The lower-case name with each run of ``-``, ``_`` or ``.`` replaced by ``-``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """Name part of a requirement such as ``timm>=1.0`` or ``torch[extra]==2``.

    Args:
        requirement: a pip requirement string.

    Returns:
        The distribution name (the stripped text when no name is found).
    """
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1) if match else requirement.strip()


def _is_torch(requirement: str) -> bool:
    """Whether a requirement is one of the PyTorch packages (``TORCH_PACKAGES``).

    Args:
        requirement: a pip requirement string.

    Returns:
        ``True`` for torch or torchvision.
    """
    return normalize(requirement_name(requirement)) in TORCH_PACKAGES


# ---------------------------------------------------------------------------
# Subprocess layer (replaced in tests)


def _no_window_flags() -> int:
    """Process creation flags that hide the console window on Windows (0 elsewhere).

    Returns:
        The flags for ``subprocess``.
    """
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0


def capture(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    """Run a short command and return (exit code, stdout); (-1, "") when it cannot run.

    Args:
        cmd: command line.
        timeout: seconds before the command is given up.

    Returns:
        The exit code and the standard output (``(-1, "")`` on an OS error or a timeout).
    """
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
    """``rmtree`` error handler: clear the read-only flag and retry once (never raises).

    Args:
        function: the function that failed.
        path: path it failed on.
        *_: the exception information given by ``rmtree`` (ignored).
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        pass


def remove_tree(path: Path) -> None:
    """Remove a folder or file as well as possible, retrying; never raises (a leftover is cleaned later).

    Args:
        path: folder or file to remove (a symbolic link is unlinked, not followed).
    """
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
    """Kill a process and everything it started (pip starts build subprocesses).

    Args:
        proc: the process to kill; it is waited for.
    """
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
    """Run ``cmd`` appending its output to ``log``; poll ``check`` and kill the process tree if it raises.

    Args:
        cmd: command line.
        log: file that receives the command and its output.
        check: function called every ``poll`` seconds; it raises to cancel.
        poll: seconds between calls to ``check``.

    Returns:
        The exit code of the process.
    """
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
    """Parse the CSV query (``name, driver``) and, when given, the CUDA version of the plain output.

    Args:
        query_output: output of ``nvidia-smi --query-gpu=name,driver_version --format=csv,noheader``.
        plain_output: output of a plain ``nvidia-smi`` (where ``CUDA Version`` appears).

    Returns:
        The first GPU found, or ``None`` when no line has a name and a driver.
    """
    for line in query_output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", plain_output)
            return GpuInfo(name=parts[0], driver=parts[1], cuda=match.group(1) if match else None)
    return None


def cuda_tag(cuda_version: str | None) -> str | None:
    """Newest PyTorch CUDA build (``cu124``...) not above the driver's CUDA version, or None.

    Args:
        cuda_version: CUDA version of the driver (``"12.4"``), or ``None``.

    Returns:
        A tag of ``CUDA_TAGS``, or ``None`` when the version is missing or older than all of them.
    """
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
    """The ``pip install`` commands for ``packages``: PyTorch from its own index, the rest from PyPI.

    Args:
        python: Python executable of the environment.
        packages: requirements to install.
        gpu: the NVIDIA GPU, or ``None`` (PyTorch then comes from the CPU index).
        platform_name: ``sys.platform`` value; on ``darwin`` PyTorch comes from PyPI. Defaults to the
            current platform.

    Returns:
        One command for the PyTorch packages (if any) and one for the others (if any).
    """
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
        """Locate the environment of a workspace; nothing is created yet.

        Args:
            workspace: folder of the app's data (the record and log live in ``training/`` inside it).
        """
        self.workspace = Path(workspace)
        self.training_dir = self.workspace / "training"
        self.record = self.training_dir / "env.json"
        self.log = self.training_dir / "install.log"

    # -- location --------------------------------------------------------

    def _default_dir(self) -> Path:
        """Folder to use when none is recorded: inside the workspace, or a short path on Windows.

        Returns:
            ``training/env`` in the workspace, or (on Windows, when that path is too long) a folder in
            ``%LOCALAPPDATA%`` named after a hash of the workspace.
        """
        preferred = self.training_dir / "env"
        if not IS_WINDOWS or len(str(preferred)) <= MAX_PATH_LENGTH:
            return preferred
        digest = hashlib.sha256(str(self.workspace.resolve()).encode("utf-8")).hexdigest()[:8]
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "activelabel" / f"train-{digest}"

    @property
    def path(self) -> Path:
        """Folder of the environment; stable once chosen (recorded in ``env.json`` when it is made).

        The record is written only for a folder that is not the default one; reading the path never writes.
        """
        try:
            recorded = json.loads(self.record.read_text(encoding="utf-8")).get("path")
            if recorded:
                return Path(recorded)
        except (OSError, ValueError, AttributeError):
            pass
        return self._default_dir()

    def _remember_location(self) -> None:
        """Record a folder that is not the default one, so it stays the same when the rule changes."""
        if self.path != self.training_dir / "env":
            self.training_dir.mkdir(parents=True, exist_ok=True)
            self.record.write_text(json.dumps({"path": str(self.path)}), encoding="utf-8")

    def python(self) -> Path:
        """Python executable of the environment (may not exist yet).

        Returns:
            The path of ``python`` inside the environment folder.
        """
        if IS_WINDOWS:
            return self.path / "Scripts" / "python.exe"
        return self.path / "bin" / "python"

    def exists(self) -> bool:
        """Whether the environment has been created and works (a venv without pip does not count).

        Returns:
            ``True`` when its Python runs ``pip --version`` successfully.
        """
        python = self.python()
        if not python.exists():
            return False
        code, _ = capture([str(python), "-m", "pip", "--version"], timeout=60)
        return code == 0

    # -- detection -------------------------------------------------------

    def gpu(self) -> GpuInfo | None:
        """The NVIDIA GPU, found with ``nvidia-smi`` (no torch needed); None when absent or failing.

        Returns:
            The GPU, or ``None``.
        """
        code, out = capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if code != 0 or parse_nvidia_smi(out) is None:
            return None
        code, plain = capture(["nvidia-smi"])
        return parse_nvidia_smi(out, plain if code == 0 else "")

    def gpu_count(self) -> int:
        """How many NVIDIA GPUs ``nvidia-smi`` lists (0 when it is absent or fails).

        Returns:
            The number of GPUs.
        """
        code, out = capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        return len([line for line in out.splitlines() if line.strip()]) if code == 0 else 0

    @staticmethod
    def mps_hint() -> bool:
        """Apple silicon: PyTorch can use the GPU (MPS) with the plain PyPI build, nothing special.

        Returns:
            ``True`` on macOS with an ARM processor.
        """
        return sys.platform == "darwin" and platform.machine() == "arm64"

    def installed(self) -> dict[str, str]:
        """``{distribution name: version}`` in the environment; empty when it does not exist.

        Returns:
            Distribution name to version (empty when the environment is missing or cannot be read).
        """
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
        """The requirements whose distribution is not installed.

        Args:
            packages: requirements to check.

        Returns:
            The requirements (as given) that are not installed.
        """
        have = {normalize(n) for n in self.installed()}
        return [p for p in packages if normalize(requirement_name(p)) not in have]

    # -- changes ---------------------------------------------------------

    def _fail(self, what: str) -> TrainingEnvError:
        """Build the error for a failed step, pointing to the install log.

        Args:
            what: what failed.

        Returns:
            The error to raise.
        """
        return TrainingEnvError(f"{what}. Details in {self.log}")

    def create(self, check: CheckFn, step: StepFn) -> None:
        """Create the virtual environment when it is missing (or broken: no Python or no pip).

        A creation that is cancelled or fails does not leave a half-made folder behind.

        Args:
            check: function called while the command runs; it raises to cancel.
            step: function that receives the step code (``train_env_create``).

        Raises:
            TrainingEnvError: the environment could not be created.
        """
        if self.exists():
            return
        step("train_env_create")
        self._remember_location()
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
        """Create the environment if needed and install the missing packages.

        Args:
            packages: requirements the environment must have.
            check: function called while commands run; it raises to cancel.
            step: function that receives the step code (``train_env_install:<packages>``).

        Raises:
            TrainingEnvError: the environment could not be created or a package could not be installed.
        """
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
        """Torch version, CUDA and MPS availability and device name (for run.json); empty without torch.

        Returns:
            The device data as a dict (empty without torch or when it cannot be read).
        """
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
