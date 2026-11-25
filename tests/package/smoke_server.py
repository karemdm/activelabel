"""Start the installed server, create a project and fetch the pages (used by the "package" job of the CI).

Usage: ``python tests/package/smoke_server.py <port>`` with the package installed (not editable) in the
Python that runs this script. It uses a temporary workspace and stops the server at the end. The front end
is made of native ES modules: ``/static/js/main.js`` and every module it imports (recursively) must be served
with status 200, a body and a JavaScript content type, or the browser refuses to run the page.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import activelabel

MAIN_MODULE = "/static/js/main.js"
# ``import x from '...'``, ``export {x} from '...'`` and bare ``import '...'`` (static, not ``import(...)``).
IMPORT_PATTERN = re.compile(r"""(?:\bfrom\s*|\bimport\s*)(['"])([^'"\n]+)\1""")


def fetch(url: str, body: dict | None = None) -> tuple[int, bytes, str]:
    """GET (or POST a JSON body) and return the status, the bytes and the Content-Type header.

    Args:
        url: Full URL to request.
        body: JSON body to POST; ``None`` makes a GET.

    Returns:
        The tuple ``(status, content, content_type)``; the content type is empty when the header is absent.
    """
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status, answer.read(), answer.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def relative_imports(source: str, module_path: str) -> list[str]:
    """URL paths of the relative modules that a module imports statically.

    Args:
        source: JavaScript text of the module.
        module_path: URL path of that module (for example ``/static/js/main.js``).

    Returns:
        The resolved URL paths, in order of appearance and without repeats.
    """
    found: list[str] = []
    for match in IMPORT_PATTERN.finditer(source):
        specifier = match.group(2)
        if not specifier.startswith(("./", "../")):
            continue
        path = posixpath.normpath(urllib.parse.urljoin(module_path, specifier))
        if path not in found:
            found.append(path)
    return found


def check_modules(get: Callable[[str], tuple[int, bytes, str]], entry: str = MAIN_MODULE) -> list[str]:
    """Fetch the entry module and every module it imports; print one line per module.

    Args:
        get: Function that takes a URL path and returns ``(status, content, content_type)``.
        entry: URL path of the entry module.

    Returns:
        The URL paths that failed (status other than 200, empty body or a non-JavaScript content type).
    """
    failures: list[str] = []
    seen: set[str] = set()
    pending = [entry]
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        code, content, content_type = get(path)
        print(f"GET {path} -> {code} ({len(content)} bytes, {content_type})")
        if code != 200 or not content or "javascript" not in content_type.lower():
            failures.append(path)
            continue
        pending.extend(relative_imports(content.decode("utf-8", errors="replace"), path))
    return failures


def wait_until_up(base: str, server: subprocess.Popen) -> None:
    """Wait for the home page; fail early when the server process ends.

    Args:
        base: Base URL of the server.
        server: The server process.
    """
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
            status, body, _ = fetch(base + "/api/projects", {"name": "Smoke", "task": "classification"})
            assert status == 200, (status, body)
            project = json.loads(body)["id"]
            failures = check_modules(lambda path: fetch(base + path))
            for path in ("/", "/static/app.css", f"/projects/{project}", "/api/projects"):
                code, content, _ = fetch(base + path)
                print(f"GET {path} -> {code} ({len(content)} bytes)")
                if code != 200 or not content:
                    failures.append(path)
            return 1 if failures else 0
        finally:
            server.terminate()
            server.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
