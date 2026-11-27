"""The FiftyOne plugins of ActiveLabel and how ActiveLabel turns them on for the App it starts.

Each plugin is a folder here. ``activelabel_annotate/`` opens the image window of the App in *Annotate*
mode, so a box or a polygon point is moved by dragging it with the mouse (in *Explore* mode, the App's own
default, the labels are only shown). ActiveLabel points FiftyOne at this folder for its own process (and the
App server it starts) only: nothing is written to ``~/.fiftyone`` or to the FiftyOne configuration files of
the person.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from pathlib import Path

from activelabel.optional import fo

log = logging.getLogger(__name__)

PLUGINS_DIR = Path(__file__).resolve().parent
#: The FiftyOne version the plugins were checked with in a browser. ``activelabel_annotate`` uses an App
#: hook that is not a public API (``useModalModeController``): after an upgrade, check it again (images
#: open in Annotate mode) and update this number; the tests fail until then.
VERIFIED_FIFTYONE = "1.22.1"


def installed_fiftyone() -> str | None:
    """The installed FiftyOne version.

    Returns:
        The version, or None when FiftyOne is not installed.
    """
    try:
        return metadata.version("fiftyone")
    except metadata.PackageNotFoundError:
        return None


def use_plugins() -> None:
    """Make the App started from this process load ActiveLabel's plugins.

    The plugins folder is set in the environment of this process (the App server it starts inherits it)
    and in the FiftyOne config of this process. Another FiftyOne version than :data:`VERIFIED_FIFTYONE`
    is logged as a warning (the plugin itself stays quiet when the App changed).
    """
    version = installed_fiftyone()
    if version != VERIFIED_FIFTYONE:
        log.warning(
            "FiftyOne %s is installed; ActiveLabel's App plugins were checked with %s. If images do not "
            "open in Annotate mode, click the Annotate tab to edit.",
            version,
            VERIFIED_FIFTYONE,
        )
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(PLUGINS_DIR)
    fo.config.plugins_dir = str(PLUGINS_DIR)
