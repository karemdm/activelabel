"""Settings for the whole test run, applied before any test module (and so before FiftyOne) is imported.

FiftyOne reads its configuration once, when it is first imported. Unless a database is given
(``FIFTYONE_DATABASE_URI``, as the CI job does with a MongoDB service), the tests use a database and a
dataset folder of their own under the system's temporary folder, never the person's ``~/.fiftyone``
(its database holds the datasets of their real projects). A user ``config.json`` is not read either: it
could point the tests at that database.
"""

import os
import tempfile
from pathlib import Path

FIFTYONE_HOME = Path(tempfile.gettempdir()) / "activelabel-tests" / "fiftyone"


def isolated_fiftyone_env(environ: dict[str, str], home: Path) -> dict[str, str]:
    """The FiftyOne variables the tests need on top of ``environ``.

    Args:
        environ: The current environment variables.
        home: Folder that holds the test database and datasets.

    Returns:
        The variables to set (empty when a database URI is already given).
    """
    if environ.get("FIFTYONE_DATABASE_URI"):
        return {}
    return {
        "FIFTYONE_DATABASE_DIR": str(home / "db"),
        "FIFTYONE_DEFAULT_DATASET_DIR": str(home / "datasets"),
        "FIFTYONE_CONFIG_PATH": str(home / "config.json"),
        "FIFTYONE_APP_CONFIG_PATH": str(home / "app_config.json"),
        "FIFTYONE_DO_NOT_TRACK": "true",
    }


os.environ.update(isolated_fiftyone_env(dict(os.environ), FIFTYONE_HOME))
