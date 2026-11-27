"""The FiftyOne plugins of ActiveLabel and how ActiveLabel turns them on for the App it starts.

Each plugin is a folder here. ``activelabel_annotate/`` opens the image window of the App in *Annotate*
mode, so a box or a polygon point is moved by dragging it with the mouse (in *Explore* mode, the App's own
default, the labels are only shown). ActiveLabel points FiftyOne at this folder for its own process (and the
App server it starts) only: nothing is written to ``~/.fiftyone`` or to the FiftyOne configuration files of
the person.
"""

from __future__ import annotations

import os
from pathlib import Path

from activelabel.optional import fo

PLUGINS_DIR = Path(__file__).resolve().parent


def use_plugins() -> None:
    """Make the App started from this process load ActiveLabel's plugins.

    The plugins folder is set in the environment of this process (the App server it starts inherits it)
    and in the FiftyOne config of this process.
    """
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(PLUGINS_DIR)
    fo.config.plugins_dir = str(PLUGINS_DIR)
