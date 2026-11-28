"""The commit hook picks the tests that mirror the changed files (tests/helpers/changed_tests.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "helpers" / "changed_tests.py"
SPEC = importlib.util.spec_from_file_location("changed_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
changed_tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(changed_tests)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small repository with a mirrored module and two of its test files.

    Args:
        tmp_path: pytest's temporary folder.

    Returns:
        The repository folder.
    """
    tests = tmp_path / "tests" / "activelabel" / "training"
    tests.mkdir(parents=True)
    for name in ("test_runner.py", "test_runner_roundtrip.py", "test_other.py"):
        (tests / name).write_text("", encoding="utf-8")
    return tmp_path


def test_a_module_runs_its_mirror_tests_and_the_rules(repo):
    selected = changed_tests.tests_for(["src/activelabel/training/runner.py"], repo)
    assert selected == sorted(
        [
            "tests/activelabel/training/test_runner.py",
            "tests/activelabel/training/test_runner_roundtrip.py",
            *changed_tests.PYTHON_RULES,
        ]
    )


def test_a_changed_test_file_runs_itself(repo):
    selected = changed_tests.tests_for(["tests/activelabel/training/test_other.py"], repo)
    assert "tests/activelabel/training/test_other.py" in selected


def test_a_deleted_test_file_is_not_passed_to_pytest(repo):
    assert "tests/activelabel/training/test_gone.py" not in changed_tests.tests_for(
        ["tests/activelabel/training/test_gone.py"], repo
    )


@pytest.mark.parametrize("name", ["tests/conftest.py", "tests/helpers/cvat_fake.py", "pyproject.toml"])
def test_shared_files_run_the_whole_suite(repo, name):
    assert changed_tests.tests_for([name], repo) is None


def test_the_page_and_the_texts_run_the_frontend_tests(repo):
    selected = changed_tests.tests_for(
        ["src/activelabel/web/static/js/cv/grid.js", "src/activelabel/i18n/pt.json"], repo
    )
    assert selected == sorted(changed_tests.FRONTEND_TESTS)


def test_docs_run_nothing(repo):
    assert changed_tests.tests_for(["docs/GUIA.pt-BR.md", "CHANGELOG.md"], repo) == []


def test_windows_separators_are_read(repo):
    assert "tests/activelabel/training/test_runner.py" in changed_tests.tests_for(
        ["src\\activelabel\\training\\runner.py"], repo
    )


def test_main_with_nothing_to_test_passes(capsys, monkeypatch):
    monkeypatch.delenv("ACTIVELABEL_HOOK_FULL", raising=False)
    assert changed_tests.main(["README.md"]) == 0
    assert "no tests" in capsys.readouterr().out


def test_main_runs_pytest_on_the_selection(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.delenv("ACTIVELABEL_HOOK_FULL", raising=False)
    monkeypatch.setattr(changed_tests.subprocess, "call", lambda command, cwd: calls.append(command) or 0)
    assert changed_tests.main(["tests/rules/test_layout.py"]) == 0
    assert calls == [changed_tests.PYTEST + sorted(changed_tests.PYTHON_RULES)]


def test_the_environment_variable_runs_the_whole_suite(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setenv("ACTIVELABEL_HOOK_FULL", "1")
    monkeypatch.setattr(changed_tests.subprocess, "call", lambda command, cwd: calls.append(command) or 0)
    assert changed_tests.main(["README.md"]) == 0
    assert calls == [changed_tests.PYTEST]


def test_the_runner_is_this_python():
    assert changed_tests.PYTEST[0] == sys.executable
