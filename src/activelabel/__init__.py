"""activelabel: image datasets from curation to trained models."""

from importlib.metadata import PackageNotFoundError, version

FALLBACK_VERSION = "0.0.0+unknown"

try:
    # Single source of the version: the git tag, read by setuptools-scm when the package is built.
    __version__ = version("activelabel")
except PackageNotFoundError:  # running from a source folder that was never installed
    __version__ = FALLBACK_VERSION
