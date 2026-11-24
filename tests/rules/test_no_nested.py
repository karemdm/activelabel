"""No function or class is defined inside the body of a function.

A nested ``def``/``async def``/``class`` hides code from tests and from readers: it cannot be imported, called
or covered on its own. Move it to module level (or to a method of a class). Methods of a class and lambdas are
fine; what counts is a definition whose nearest enclosing scope is a function.

``BASELINE`` lists today's offenders as ``file:Outer.function.inner`` (path relative to ``src/activelabel``).
It may only shrink: a new offender fails the test, and so does an entry that no longer exists, so remove an
entry in the same change that moves the code out.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "activelabel"
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

BASELINE: set[str] = set()  # every nested definition is gone; keep it empty


def _nested_definitions(path: Path) -> set[str]:
    found: set[str] = set()
    relative = path.relative_to(SOURCE).as_posix()

    def visit(node: ast.AST, scopes: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*FUNCTIONS, ast.ClassDef)):
                if scopes and isinstance(scopes[-1], FUNCTIONS):
                    found.add(f"{relative}:{'.'.join(s.name for s in scopes)}.{child.name}")
                visit(child, [*scopes, child])
            else:
                visit(child, scopes)

    visit(ast.parse(path.read_text(encoding="utf-8")), [])
    return found


def _all_nested() -> set[str]:
    return {name for path in sorted(SOURCE.rglob("*.py")) for name in _nested_definitions(path)}


def test_no_new_nested_definitions():
    new = sorted(_all_nested() - BASELINE)
    assert not new, f"definitions nested in a function (move them to module level): {new}"


def test_baseline_has_no_stale_entries():
    stale = sorted(BASELINE - _all_nested())
    assert not stale, f"fixed or renamed: remove from BASELINE: {stale}"
