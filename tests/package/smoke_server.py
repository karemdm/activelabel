"""Start the installed server, create a project and fetch the pages (used by the "package" job of the CI).

Usage: ``python tests/package/smoke_server.py <port>`` with the package installed (not editable) in the
Python that runs this script. It uses a temporary workspace and stops the server at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import activelabel


def fetch(url: str, body: dict | None = None) -> tuple[int, bytes]:
    """GET (or POST a JSON body) and return the status and the bytes."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_until_up(base: str, server: subprocess.Popen) -> None:
    """Wait for the home page; fail early when the server process ends."""
    end = time.time() + 60
    while time.time() < end:
        if server.poll() is not None:
            raise SystemExit(f"the server stopped (exit code {server.returncode})")
        try:
            if fetch(base + "/")[0] == 200:
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("the server did not answer in 60 seconds")


def main() -> int:
    """Run the checks; returns the exit code."""
    port = sys.argv[1]
    base = f"http://127.0.0.1:{port}"
    installed = Path(activelabel.__file__).resolve()
    print(f"package at {installed}")
    if "site-packages" not in installed.parts:
        raise SystemExit("this is not the installed package (is it an editable install?)")
    with tempfile.TemporaryDirectory() as workspace:
        env = {**os.environ, "ACTIVELABEL_WORKSPACE": workspace}
        server = subprocess.Popen([sys.executable, "-m", "activelabel.cli", "serve", "--port", port], env=env)
        try:
            wait_until_up(base, server)
            status, body = fetch(base + "/api/projects", {"name": "Smoke", "task": "classification"})
            assert status == 200, (status, body)
            project = json.loads(body)["id"]
            failures = []
            for path in ("/", "/static/app.js", "/static/app.css", f"/projects/{project}", "/api/projects"):
                code, content = fetch(base + path)
                print(f"GET {path} -> {code} ({len(content)} bytes)")
                if code != 200 or not content:
                    failures.append(path)
            return 1 if failures else 0
        finally:
            server.terminate()
            server.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
