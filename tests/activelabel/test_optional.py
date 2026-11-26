"""The optional parts: FiftyOne is an extra (docs/DEVELOPMENT.md, *Light install*)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from activelabel import optional

SRC = Path(optional.__file__).resolve().parents[1]
LIGHT_SCRIPT = textwrap.dedent(
    """
    import sys

    sys.modules["fiftyone"] = None  # `import fiftyone` raises ImportError, as in the light install
    from fastapi.testclient import TestClient

    from activelabel import optional
    from activelabel.web.app import create_app

    assert not optional.FIFTYONE
    client = TestClient(create_app())
    features = client.get("/api/features").json()
    assert features["fiftyone"]["installed"] is False, features
    assert features["fiftyone"]["install"] == optional.FIFTYONE_INSTALL
    assert client.post("/api/projects", json={"name": "P", "task": "detection"}).status_code == 200
    answer = client.post("/api/projects/1/fiftyone", json={})
    assert answer.status_code == 501, answer.text
    assert answer.json()["code"] == "fiftyone_not_installed"
    assert answer.json()["params"]["install"] == optional.FIFTYONE_INSTALL
    print("LIGHT-OK")
    """
)


def test_fiftyone_is_available_in_the_development_environment() -> None:
    assert optional.FIFTYONE is True
    assert optional.fo is not None and optional.F is not None


def test_require_fiftyone_passes_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optional, "FIFTYONE", True)
    assert optional.require_fiftyone() is None


def test_require_fiftyone_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optional, "FIFTYONE", False)
    with pytest.raises(optional.FeatureMissing) as caught:
        optional.require_fiftyone()
    assert caught.value.code == "fiftyone_not_installed"
    assert caught.value.params == {"install": optional.FIFTYONE_INSTALL}
    assert optional.FIFTYONE_INSTALL in str(caught.value)


def test_version_of_an_installed_and_a_missing_distribution() -> None:
    assert optional.version("pytest")
    assert optional.version("activelabel-no-such-distribution") is None


def test_the_app_runs_without_fiftyone(tmp_path: Path) -> None:
    """A real light install: ``import fiftyone`` fails, the app starts, the FiftyOne features answer 501."""
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "ACTIVELABEL_WORKSPACE": str(tmp_path / "workspace"),
    }
    done = subprocess.run(
        [sys.executable, "-c", LIGHT_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "LIGHT-OK" in done.stdout
