"""A FiftyOne database that belongs to the tests (or to a test server) and to nothing else.

FiftyOne's own database service is shared: a process that has no ``database_uri`` looks for ANY running
``fiftyone/service/main.py --51-service db --multi`` process of the same user and attaches to it, whatever
``FIFTYONE_DATABASE_DIR`` says. When the person's real ActiveLabel is running, that service holds the
datasets of their real projects (``~/.fiftyone``). So the tests never rely on that service: they start a
``mongod`` of their own (the binary that ships with FiftyOne) on a free port, with a temporary data folder,
and point FiftyOne at it with ``FIFTYONE_DATABASE_URI``; with a URI FiftyOne connects directly and never
looks for a service. :func:`problem` is the guard that refuses anything else.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

USER_FIFTYONE = Path.home() / ".fiftyone"  # the person's own FiftyOne folder: never used by tests
TESTS_HOME = Path(tempfile.gettempdir()) / "activelabel-tests" / "fiftyone"
START_TIMEOUT = 60.0  # seconds to wait for the mongod to accept connections
DROPPED = ("FIFTYONE_PRIVATE_DATABASE_PORT",)  # set by a FiftyOne parent process: would override the URI


def mongod_path() -> Path:
    """The ``mongod`` that ships with FiftyOne (found without importing FiftyOne).

    Returns:
        The path of the executable.

    Raises:
        FileNotFoundError: FiftyOne or its database binary is not installed.
    """
    spec = importlib.util.find_spec("fiftyone")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("FiftyOne is not installed.")
    name = "mongod.exe" if sys.platform.startswith("win") else "mongod"
    path = Path(spec.origin).parent / "db" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"FiftyOne's mongod was not found at {path}.")
    return path


def free_port() -> int:
    """A TCP port nobody listens on right now (on 127.0.0.1).

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def database_uri(port: int) -> str:
    """The URI of a mongod of ours on ``port``.

    Args:
        port: The port the mongod listens on.

    Returns:
        A ``mongodb://`` URI on 127.0.0.1.
    """
    return f"mongodb://127.0.0.1:{port}"


def isolated_env(home: Path, port: int) -> dict[str, str]:
    """The FiftyOne variables that point a process at a database and folders of its own.

    Args:
        home: Folder for the datasets and the (absent) config files.
        port: Port of the mongod started for this run (see :class:`OwnDatabase`).

    Returns:
        The variables to set; the ones in ``DROPPED`` must be removed as well.
    """
    return {
        "FIFTYONE_DATABASE_URI": database_uri(port),
        "FIFTYONE_DATABASE_DIR": str(home / "db"),
        "FIFTYONE_DEFAULT_DATASET_DIR": str(home / "datasets"),
        "FIFTYONE_CONFIG_PATH": str(home / "config.json"),
        "FIFTYONE_APP_CONFIG_PATH": str(home / "app_config.json"),
        "FIFTYONE_DO_NOT_TRACK": "true",
    }


def _inside(path: str, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or lies inside it.

    Args:
        path: A path (may be relative or use ``~``).
        folder: The folder.

    Returns:
        True when inside.
    """
    resolved, root = Path(path).expanduser().resolve(), folder.resolve()
    return resolved == root or root in resolved.parents


def problem(environ: dict[str, str], own_uri: str | None, ci: bool) -> str | None:
    """Why the FiftyOne settings of ``environ`` are not safe for tests, or None when they are.

    Safe means: FiftyOne connects by URI (so it never attaches to a database service it did not start),
    the URI is the one of the mongod this run started (``own_uri``) — or, on CI only, the one the job
    provides — and no FiftyOne folder points into the person's ``~/.fiftyone``.

    Args:
        environ: The environment the tests (or the test server) run with.
        own_uri: URI of the mongod this run started, or None when it started none.
        ci: Whether this is a CI job (``CI=true``), whose database service is its own.

    Returns:
        A sentence for the person, or None.
    """
    uri = environ.get("FIFTYONE_DATABASE_URI")
    for name in ("FIFTYONE_DATABASE_DIR", "FIFTYONE_DEFAULT_DATASET_DIR", "FIFTYONE_CONFIG_PATH"):
        value = environ.get(name)
        if value and _inside(value, USER_FIFTYONE):
            return f"{name} points into {USER_FIFTYONE}: the tests never use the person's FiftyOne data."
    if not uri:
        return (
            "FIFTYONE_DATABASE_URI is not set: FiftyOne would attach to any running FiftyOne database "
            "service, such as the one of the person's real ActiveLabel."
        )
    if uri != own_uri and not ci:
        return (
            f"FIFTYONE_DATABASE_URI ({uri}) is not the database this test run started; "
            "outside CI the tests only use their own database."
        )
    if environ.get("FIFTYONE_PRIVATE_DATABASE_PORT"):
        return "FIFTYONE_PRIVATE_DATABASE_PORT is set: FiftyOne would connect to that port, not the URI."
    return None


def _accepts(port: int) -> bool:
    """Whether something accepts TCP connections on ``port`` (127.0.0.1).

    Args:
        port: The port.

    Returns:
        True when a connection succeeds.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class OwnDatabase:
    """A mongod started for one test run (or one test server), stopped by :meth:`stop`."""

    def __init__(self, home: Path, port: int) -> None:
        """Remember where the database lives; nothing starts yet.

        Args:
            home: Folder under which a fresh data folder is made.
            port: The port to listen on (from :func:`free_port`).
        """
        self.home = home
        self.port = port
        self.uri = database_uri(port)
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Start the mongod with a fresh data folder and wait until it accepts connections.

        Raises:
            RuntimeError: it stopped or did not answer in time.
        """
        if self.process is not None:
            return
        self.home.mkdir(parents=True, exist_ok=True)
        data = Path(tempfile.mkdtemp(prefix="db-", dir=self.home))
        args = [str(mongod_path()), "--dbpath", str(data), "--port", str(self.port), "--bind_ip", "127.0.0.1"]
        args += ["--logpath", str(data / "mongod.log")]
        if not sys.platform.startswith("win"):
            args.append("--nounixsocket")
        self.process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"The test mongod stopped at once (see {data / 'mongod.log'}).")
            if _accepts(self.port):
                return
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"The test mongod did not answer on port {self.port} in {START_TIMEOUT:.0f} s.")

    def stop(self) -> None:
        """Stop the mongod (nothing happens when it was not started)."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


def apply(environ: dict[str, str] | os._Environ[str], home: Path, port: int) -> None:
    """Point ``environ`` at the database of this run (see :func:`isolated_env`).

    Args:
        environ: The environment to change in place (``os.environ`` or a copy for a child process).
        home: Folder for the datasets and config files.
        port: Port of the run's mongod.
    """
    environ.update(isolated_env(home, port))
    for name in DROPPED:
        environ.pop(name, None)
