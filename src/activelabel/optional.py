"""Optional parts of ActiveLabel: the one place that imports FiftyOne (docs/DEVELOPMENT.md, *Light install*).

``pip install activelabel`` installs the core: projects, images, curation measures and the similarity map,
annotations, tasks and videos, CVAT synchronisation, datasets, training and pre-annotation (their packages
go to the separate training environment, docs/TRAINING.md). ``pip install "activelabel[fiftyone]"`` adds
FiftyOne: its App, its annotation tool, the import of CVAT projects (read with FiftyOne's CVAT reader) and
the similarity map in the App. Without it these answer ``fiftyone_not_installed`` and the rest works.

Modules that need FiftyOne import it from here; when it is missing the names are None and
:data:`FIFTYONE` is False.
"""

from __future__ import annotations

from importlib import metadata

from activelabel.errors import CodedError

try:
    import fiftyone as fo
    import fiftyone.brain as fob
    from fiftyone import ViewField as F
    from fiftyone.core.utils import to_slug
    from fiftyone.utils import cvat as fouc
except ImportError:  # the light install: FiftyOne is an extra
    fo = None
    fob = None
    F = None
    to_slug = None
    fouc = None

#: Whether FiftyOne can be used.
FIFTYONE = fo is not None
#: The command that adds FiftyOne.
FIFTYONE_INSTALL = 'pip install "activelabel[fiftyone]"'
#: What FiftyOne brings (shown with the installed parts).
FIFTYONE_PROVIDES = ("fiftyone_app", "fiftyone_annotation", "cvat_import", "similarity_in_app")


class FeatureMissing(CodedError):
    """A feature whose optional packages are not installed (the message says how to add them)."""


def require_fiftyone() -> None:
    """Stop here when FiftyOne is not installed.

    Raises:
        FeatureMissing: with the code ``fiftyone_not_installed`` and the install command as ``install``.
    """
    if not FIFTYONE:
        raise FeatureMissing(
            "fiftyone_not_installed",
            f"FiftyOne is not installed. Add it with: {FIFTYONE_INSTALL}",
            install=FIFTYONE_INSTALL,
        )


def version(distribution: str) -> str | None:
    """The installed version of a Python distribution, read without importing it.

    Args:
        distribution: name of the distribution (``fiftyone``).

    Returns:
        The version, or None when it is not installed.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
