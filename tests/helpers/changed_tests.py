"""Run only the tests that cover the changed files (the commit hook); the CI runs the whole suite.

The tests mirror ``src/activelabel`` (``src/activelabel/a/b.py`` is tested by
``tests/activelabel/a/test_b.py`` and ``test_b_*.py``), so the files a commit touches name the tests
to run. A change to what every test shares (``conftest.py``, ``tests/helpers``, ``tests/fixtures``,
``pyproject.toml``) runs the whole suite, and so does ``ACTIVELABEL_HOOK_FULL=1``.

Usage: ``python tests/helpers/changed_tests.py FILE...`` (pre-commit passes the staged files).
"""

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTEST = [sys.executable, "-m", "pytest", "-q", "-m", "not integration"]
#: Files every test depends on: a change to one of them runs the whole suite.
SHARED = ("pyproject.toml", "tests/helpers/", "tests/fixtures/")
#: Repository rules over the Python sources (cheap); they run when any Python file changes.
PYTHON_RULES = ("tests/rules/test_imports.py", "tests/rules/test_no_nested.py", "tests/rules/test_layout.py")
#: Python tests that read the page, the scripts and the texts.
FRONTEND_TESTS = ("tests/activelabel/web", "tests/activelabel/i18n", "tests/rules/test_js_modules.py")


def mirror_tests(source: Path, root: Path = ROOT) -> list[str]:
    """The test files that mirror one source module.

    Args:
        source: a Python file under ``src/activelabel``, relative to the repository.
        root: the repository folder.

    Returns:
        The existing ``test_<name>.py`` and ``test_<name>_*.py`` next to the mirror path, relative to
        ``root``.
    """
    folder = root / "tests" / Path(*source.parts[1:-1])
    found = [folder / f"test_{source.stem}.py", *sorted(folder.glob(f"test_{source.stem}_*.py"))]
    return [path.relative_to(root).as_posix() for path in found if path.is_file()]


def tests_for(changed: Iterable[str], root: Path = ROOT) -> list[str] | None:
    """The test paths to run for these changed files.

    Args:
        changed: changed file paths, relative to the repository (as pre-commit passes them).
        root: the repository folder.

    Returns:
        The sorted test paths, possibly empty (nothing to test), or ``None`` for the whole suite.
    """
    selected: set[str] = set()
    for name in changed:
        path = Path(name.replace("\\", "/"))
        text = path.as_posix()
        if path.name == "conftest.py" or text.startswith(SHARED):
            return None
        if path.suffix == ".py":
            selected.update(PYTHON_RULES)
        if text.startswith("tests/") and path.name.startswith("test_") and path.suffix == ".py":
            if (root / path).is_file():
                selected.add(text)
        elif text.startswith("src/activelabel/") and path.suffix == ".py":
            selected.update(mirror_tests(path, root))
        elif text.startswith(("src/activelabel/web/", "src/activelabel/i18n/")):
            selected.update(FRONTEND_TESTS)
    return sorted(selected)


def main(argv: list[str]) -> int:
    """Run pytest on the tests of the changed files.

    Args:
        argv: the changed file paths.

    Returns:
        pytest's exit code (0 when there is nothing to test).
    """
    selected = None if os.environ.get("ACTIVELABEL_HOOK_FULL") == "1" else tests_for(argv)
    if selected == []:
        print("no tests cover the changed files")
        return 0
    command = PYTEST + (selected or [])
    print("pytest:", "the whole suite" if selected is None else " ".join(selected))
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
