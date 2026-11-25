"""Start an ActiveLabel server for manual or browser tests on a FiftyOne database of its own.

Usage (from the repository root)::

    python tests/helpers/isolated_server.py --port 8112 --workspace <scratch folder>

The server gets its own ``mongod`` (see :mod:`fiftyone_isolation`), a workspace of its own and FiftyOne
folders under that workspace, so it never attaches to the database service of the person's real
ActiveLabel. It refuses to start when the settings are not safe (:func:`fiftyone_isolation.problem`), and
stops the mongod when the server ends (Ctrl+C).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import fiftyone_isolation as isolation

REFUSED_PORTS = {8090, 8091}  # the person's own servers


def server_env(base: dict[str, str], workspace: Path, port: int) -> dict[str, str]:
    """The environment of the test server.

    Args:
        base: The current environment.
        workspace: The workspace folder of this server (its FiftyOne folders go under it).
        port: Port of the server's own mongod.

    Returns:
        A new environment (``base`` is not changed).
    """
    env = dict(base)
    env["ACTIVELABEL_WORKSPACE"] = str(workspace)
    isolation.apply(env, workspace / "fiftyone", port)
    return env


def main(argv: list[str]) -> int:
    """Start the database and the server; stop both when the server ends.

    Args:
        argv: Command-line arguments (``--port``, ``--workspace``, ``--lang``).

    Returns:
        The exit code (2 when the settings are refused).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--lang", default="pt")
    args = parser.parse_args(argv)
    if args.port in REFUSED_PORTS:
        print(f"Port {args.port} belongs to the person's own server: choose another.", file=sys.stderr)
        return 2
    workspace = args.workspace.resolve()
    database = isolation.OwnDatabase(workspace / "fiftyone", isolation.free_port())
    env = server_env(dict(os.environ), workspace, database.port)
    env["ACTIVELABEL_LANG"] = args.lang
    found = isolation.problem(env, database.uri, ci=False)
    if found:
        print(f"Refused: {found}", file=sys.stderr)
        return 2
    database.start()
    try:
        command = [sys.executable, "-m", "activelabel.cli", "serve", "--port", str(args.port)]
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 0
    finally:
        database.stop()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
