"""The tests' FiftyOne database is their own, even while another FiftyOne database service is running."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fiftyone_isolation as isolation
import isolated_server
import pytest
from fiftyone.service import util as service_util

from activelabel.training.environment import TrainingEnvironment

SERVICE_ARGS = ["--51-service", "db", "--multi"]  # how FiftyOne finds a database service to attach to

# Runs in a child process with the tests' environment: where does FiftyOne really connect?
PROBE = """
import json
from fiftyone.core.odm import database
client = database.get_db_client()
print(json.dumps({"address": list(client.address), "service": database._db_service is not None}))
"""


def foreign_service() -> subprocess.Popen[bytes]:
    """A process that looks like a FiftyOne database service of another program (it only sleeps).

    Returns:
        The running process; the caller stops it.
    """
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)", *SERVICE_ARGS]
    return subprocess.Popen(sleeper, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_until_listed(process: subprocess.Popen[bytes]) -> bool:
    """Whether FiftyOne's own service lookup finds ``process`` (tried for a few seconds).

    Args:
        process: The process started by :func:`foreign_service`.

    Returns:
        True once FiftyOne would consider attaching to it.
    """
    for _ in range(10):  # each lookup scans every process of the user (a few seconds on Windows)
        found = service_util.find_processes_by_args(SERVICE_ARGS)
        # the venv's python.exe may be a launcher: the listed process is then its child
        if any(process.pid in (p.pid, p.ppid()) for p in found):
            return True
        time.sleep(0.5)
    return False


def test_isolated_env_uses_a_uri_and_folders_of_the_run(tmp_path: Path) -> None:
    env = isolation.isolated_env(tmp_path, 40001)
    assert env["FIFTYONE_DATABASE_URI"] == "mongodb://127.0.0.1:40001"
    assert all(str(tmp_path) in env[k] for k in ("FIFTYONE_DATABASE_DIR", "FIFTYONE_DEFAULT_DATASET_DIR"))


def test_apply_drops_the_private_port_a_parent_fiftyone_leaves(tmp_path: Path) -> None:
    environ = {"FIFTYONE_PRIVATE_DATABASE_PORT": "27017"}
    isolation.apply(environ, tmp_path, 40002)
    assert "FIFTYONE_PRIVATE_DATABASE_PORT" not in environ


OWN_1, OTHER = "mongodb://127.0.0.1:1", "mongodb://127.0.0.1:2"
USER_DIR = "~/.fiftyone/var/lib/mongo"


@pytest.mark.parametrize(
    ("environ", "own", "ci", "bad"),
    [
        ({}, OWN_1, False, "not set"),  # would attach to any running service
        ({"FIFTYONE_DATABASE_URI": OTHER}, OWN_1, False, "not the database"),
        ({"FIFTYONE_DATABASE_URI": OTHER}, None, True, None),  # the CI job's own service
        ({"FIFTYONE_DATABASE_URI": OWN_1}, OWN_1, False, None),
        (
            {"FIFTYONE_DATABASE_URI": OWN_1, "FIFTYONE_DATABASE_DIR": USER_DIR},
            OWN_1,
            False,
            ".fiftyone",
        ),
        (
            {"FIFTYONE_DATABASE_URI": OWN_1, "FIFTYONE_PRIVATE_DATABASE_PORT": "27017"},
            OWN_1,
            False,
            "PRIVATE",
        ),
    ],
)
def test_the_guard_refuses_anything_but_the_runs_own_database(
    environ: dict[str, str], own: str | None, ci: bool, bad: str | None
) -> None:
    found = isolation.problem(environ, own, ci)
    assert (found is None) if bad is None else (found is not None and bad in found), found


def test_this_session_passed_the_guard() -> None:
    own = os.environ.get("FIFTYONE_DATABASE_URI")
    ci = os.environ.get("CI", "").lower() == "true"
    assert own and isolation.problem(dict(os.environ), own, ci) is None


@pytest.mark.integration
def test_fiftyone_uses_the_tests_database_while_a_foreign_service_runs() -> None:
    """A process that FiftyOne would attach to exists; FiftyOne still connects only to the tests' mongod."""
    foreign = foreign_service()
    try:
        assert wait_until_listed(foreign), "the stand-in service is not seen by FiftyOne's lookup"
        probe = subprocess.run(
            [sys.executable, "-c", PROBE], capture_output=True, text=True, env=dict(os.environ), timeout=120
        )
    finally:
        foreign.kill()
        foreign.wait()
    assert probe.returncode == 0, probe.stderr
    found = json.loads(probe.stdout.strip().splitlines()[-1])
    host, port = os.environ["FIFTYONE_DATABASE_URI"].split("//", 1)[1].rsplit(":", 1)
    loopback = {"127.0.0.1", "localhost"}  # CI gives localhost, a local run 127.0.0.1
    assert found["service"] is False and found["address"][1] == int(port)
    assert found["address"][0] == host or {found["address"][0], host} <= loopback


def test_a_test_server_gets_its_own_database_and_workspace(tmp_path: Path) -> None:
    env = isolated_server.server_env({"FIFTYONE_PRIVATE_DATABASE_PORT": "27017"}, tmp_path, 40003)
    assert env["ACTIVELABEL_WORKSPACE"] == str(tmp_path)
    assert env["FIFTYONE_DATABASE_URI"] == "mongodb://127.0.0.1:40003"
    assert isolation.problem(env, "mongodb://127.0.0.1:40003", ci=False) is None


def test_every_folder_a_test_server_resolves_is_inside_its_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test server must not write to the person's profile: with a long workspace path the training
    environment used to fall back to the real ``%LOCALAPPDATA%\activelabel\train-<hash>``."""
    real = {
        "LOCALAPPDATA": str(Path.home() / "AppData" / "Local"),
        "HF_HOME": str(Path.home() / ".cache" / "hf"),
    }
    workspace = tmp_path / ("a-long-folder-name-" * 12) / "ws"  # longer than the Windows path limit allows
    env = isolated_server.server_env(real, workspace, 40004)
    root = workspace.resolve()
    paths = {name: env[name] for name in isolated_server.PROFILE_VARIABLES}
    paths |= {
        name: env[name] for name in ("ACTIVELABEL_WORKSPACE", "FIFTYONE_DATABASE_DIR", "FIFTYONE_CONFIG_PATH")
    }
    for name in isolated_server.PROFILE_VARIABLES:  # what the server process sees
        monkeypatch.setenv(name, env[name])
    paths["training environment"] = str(TrainingEnvironment(workspace).path)
    for name, path in paths.items():
        assert Path(path).resolve().is_relative_to(root), f"{name} -> {path} is outside {root}"


@pytest.mark.parametrize("port", [8090, 8091])
def test_a_test_server_refuses_the_persons_ports(tmp_path: Path, port: int) -> None:
    assert isolated_server.main(["--port", str(port), "--workspace", str(tmp_path)]) == 2


def test_no_test_writes_to_the_persons_local_app_data(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The training environment of a long test workspace stays in the run's temporary folder (conftest)."""
    workspace = tmp_path / ("a-long-folder-name-" * 12) / "ws"
    run = tmp_path_factory.getbasetemp().resolve()
    assert Path(os.environ["LOCALAPPDATA"]).resolve().is_relative_to(run)
    assert Path(TrainingEnvironment(workspace).path).resolve().is_relative_to(run)
