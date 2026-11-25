"""Settings for the whole test run, applied before any test module (and so before FiftyOne) is imported.

FiftyOne reads its configuration once, when it is first imported, and without a database URI it attaches to
any FiftyOne database service already running for the user — such as the one of the person's real
ActiveLabel, which holds their real datasets. So, outside CI, the tests always run on a ``mongod`` of their
own (see ``tests/helpers/fiftyone_isolation.py``): the URI is set here, and the mongod starts only when a
test that uses FiftyOne (marker ``integration``) is selected. On CI the job provides its own database service
(``FIFTYONE_DATABASE_URI`` with ``CI=true``). Anything else stops the session before a test runs.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "helpers"))  # conftest loads before the pythonpath setting

import fiftyone_isolation as isolation  # noqa: E402 - needs the helpers folder on the path

CI = os.environ.get("CI", "").lower() == "true"
OWN: isolation.OwnDatabase | None = None
if not (CI and os.environ.get("FIFTYONE_DATABASE_URI")):
    OWN = isolation.OwnDatabase(isolation.TESTS_HOME, isolation.free_port())
    isolation.apply(os.environ, isolation.TESTS_HOME, OWN.port)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop the run when FiftyOne could reach a database that is not the tests' own.

    Args:
        session: The test session.

    Raises:
        pytest.UsageError: the FiftyOne settings are not safe (see ``fiftyone_isolation.problem``).
    """
    found = isolation.problem(dict(os.environ), OWN.uri if OWN else None, CI)
    if found:
        raise pytest.UsageError(f"Unsafe FiftyOne settings for tests: {found}")


def pytest_collection_finish(session: pytest.Session) -> None:
    """Start the tests' own mongod when a selected test uses FiftyOne, and check FiftyOne's resolved URI.

    Args:
        session: The test session (``session.items``: the tests selected to run).

    Raises:
        pytest.UsageError: FiftyOne was configured with another database before this file ran.
    """
    loaded = sys.modules.get("fiftyone")
    if loaded is not None and OWN is not None and getattr(loaded.config, "database_uri", None) != OWN.uri:
        raise pytest.UsageError("FiftyOne was imported with another database before the tests' own was set.")
    if OWN is not None and any(item.get_closest_marker("integration") for item in session.items):
        OWN.start()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Stop the tests' own mongod.

    Args:
        config: The pytest configuration.
    """
    if OWN is not None:
        OWN.stop()
