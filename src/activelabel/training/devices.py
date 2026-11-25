"""The device a training or a pre-annotation runs on (docs/TRAINING.md).

The device is not part of a run's configuration (a configuration repeats on any computer): it is chosen when
the run is asked for, stored in ``run.json`` as ``device_choice`` (``device`` in a pre-annotation run) and
checked here against what the computer has, so a GPU that does not exist is refused before anything starts.
The worker checks it again with PyTorch (``workers/options.py``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.training.environment import TrainingEnvironment

AUTO = "auto"
#: Choices every computer has (no need to look for GPUs to accept them).
ALWAYS = (AUTO, "cpu")
_PATTERN = re.compile(r"auto|cpu|mps|cuda:(\d+)")


class DeviceError(CodedError):
    """A device choice that is not valid or not available (the message is readable)."""


def choices(gpus: int, mps: bool) -> list[str]:
    """The devices a person can choose on this computer.

    Args:
        gpus: number of NVIDIA GPUs found (``nvidia-smi``).
        mps: whether the computer has an Apple GPU (MPS).

    Returns:
        ``auto`` and ``cpu``, then ``cuda:0``... one per GPU, then ``mps`` when there is one.
    """
    return [AUTO, "cpu", *(f"cuda:{i}" for i in range(gpus)), *(["mps"] if mps else [])]


def available(env: Any) -> list[str]:
    """The devices of :func:`choices` for the computer of a training environment.

    Args:
        env: the training environment (``gpu_count`` and ``mps_hint``).

    Returns:
        The device choices.
    """
    return choices(env.gpu_count(), env.mps_hint())


def check(choice: Any, offered: list[str]) -> str:
    """Validate a device choice against the devices of this computer.

    Args:
        choice: the requested device (``None`` or empty means ``auto``).
        offered: the devices of this computer (:func:`available`).

    Returns:
        The choice, as stored.

    Raises:
        DeviceError: not a device name (``device_invalid``) or not on this computer (``device_unavailable``).
    """
    value = AUTO if choice in (None, "") else choice
    if not isinstance(value, str) or not _PATTERN.fullmatch(value):
        raise DeviceError(
            "device_invalid", "The device must be auto, cpu, cuda:N (a GPU) or mps.", device=str(value)
        )
    if value not in offered:
        raise DeviceError(
            "device_unavailable",
            f"The device {value} is not available on this computer (it has: {', '.join(offered)}).",
            device=value,
            devices=list(offered),
        )
    return value


def check_here(choice: Any, workspace: Path) -> str:
    """Validate a device choice against the computer of the app's training environment.

    ``auto`` and ``cpu`` are accepted without looking for GPUs.

    Args:
        choice: the requested device (``None`` or empty means ``auto``).
        workspace: folder of the app's data (where the training environment lives).

    Returns:
        The choice, as stored.

    Raises:
        DeviceError: not a device name, or not on this computer.
    """
    if choice in (None, "", *ALWAYS) or not isinstance(choice, str) or not _PATTERN.fullmatch(choice):
        return check(choice, list(ALWAYS))  # accepted, or refused as not a device, without looking for GPUs
    return check(choice, available(TrainingEnvironment(workspace)))
