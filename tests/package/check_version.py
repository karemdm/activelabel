"""Check that a built wheel carries a real version (used by the "package" CI job and the release workflow).

Usage: ``python tests/package/check_version.py <wheel> [expected version] [--no-tag-yet]``

The version comes from the git tag through setuptools-scm (``[tool.setuptools_scm]`` in pyproject.toml):

- ``0.0.0+unknown`` is the fallback of a build without any git data; it is always refused.
- ``0.0`` is what a checkout without any ``v*`` tag builds (or a shallow clone that lost the tags). It is
  refused, except with ``--no-tag-yet``: the CI passes that flag only while ``git describe --tags`` finds no
  tag, so pull requests are not blocked before the first release. The release workflow never passes it.
- With an expected version (the tag without its ``v``) the wheel must carry exactly that.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

FALLBACK = "0.0.0+unknown"
UNTAGGED = "0.0"


def wheel_version(wheel: Path) -> str:
    """Version written in the METADATA of a wheel.

    Args:
        wheel: path of the ``.whl`` file.

    Returns:
        The value of the ``Version`` field.

    Raises:
        SystemExit: the wheel has no METADATA file or no ``Version`` field.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if not names:
            raise SystemExit(f"no METADATA in {wheel}")
        text = archive.read(names[0]).decode("utf-8")
    match = re.search(r"^Version: (\S+)$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"no Version field in {wheel}")
    return match.group(1)


def problem(found: str, expected: str | None, no_tag_yet: bool) -> str | None:
    """Why a wheel version is not acceptable.

    Args:
        found: the version of the wheel.
        expected: the version it must be (the tag without its ``v``), or ``None`` for any real version.
        no_tag_yet: the repository has no ``v*`` tag yet, so ``0.0`` is accepted.

    Returns:
        A message, or ``None`` when the version is acceptable.
    """
    if found == FALLBACK or found.startswith("0.0.0"):
        return "the wheel has the fallback version: it was built without git data"
    if found == UNTAGGED and not no_tag_yet:
        return "the wheel has version 0.0: build from a checkout with the tags (fetch-depth: 0)"
    if expected is not None and found != expected:
        return f"expected version {expected}"
    return None


def main(argv: list[str]) -> int:
    """Run the check.

    Args:
        argv: the command-line arguments (wheel, optional expected version, optional ``--no-tag-yet``).

    Returns:
        Exit code: 0 when the version is acceptable, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("expected", nargs="?")
    parser.add_argument("--no-tag-yet", action="store_true", help="accept 0.0: the repository has no tag yet")
    args = parser.parse_args(argv)
    found = wheel_version(args.wheel)
    print(f"wheel version: {found}")
    message = problem(found, args.expected, args.no_tag_yet)
    if message is not None:
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
