"""Check that a built wheel holds every file of the package (used by the "package" job of the CI).

Usage: ``python tests/package/check_wheel.py <wheel> <src/activelabel folder>``

The unit tests run on an editable install, which reads the source tree, so a data file that the build
leaves out (a ``package-data`` pattern that does not match, for example because it does not recurse into
sub-folders) is never noticed there. Here the wheel is compared with the source tree: every file of
``src/activelabel`` must be inside it, and the kinds of file the app reads at run time must be there at all.
"""

from __future__ import annotations

import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path

REQUIRED = [
    "activelabel/web/templates/*.html",
    "activelabel/web/static/js/main.js",
    "activelabel/web/static/app.css",
    "activelabel/i18n/*.json",
    "activelabel/training/workers/*.py",
]


def missing(wheel: Path, package: Path) -> list[str]:
    """Names of the files that are in the source folder but not in the wheel (and unmet REQUIRED globs)."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    absent = [
        f"{package.name}/{path.relative_to(package).as_posix()}"
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    problems = [name for name in absent if name not in names]
    problems += [f"nothing matches {glob}" for glob in REQUIRED if not any(fnmatch(n, glob) for n in names)]
    return problems


def main() -> int:
    """Command line entry: prints what is missing and returns the exit code."""
    wheel, package = Path(sys.argv[1]), Path(sys.argv[2])
    problems = missing(wheel, package)
    for problem in problems:
        print(f"missing from the wheel: {problem}")
    print(f"{wheel.name}: {'FAILED' if problems else 'every file of the package is inside'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
