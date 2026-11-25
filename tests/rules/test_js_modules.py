"""The interface's ES modules (``src/activelabel/web/static/js``) stay linked, layered and tested.

The rules: every relative import names a file that exists; every module is reachable from ``main.js`` (the
only one the page loads); a module imports only from its own layer or the layers below it (``shell`` <
``integrations`` < ``cv`` and ``pages``; ``main.js`` may import anything); and every module has a mirror test
``tests/js/<same path>.test.mjs``.

``BASELINE`` lists the modules without a mirror test, as paths relative to ``static/js``. It may only shrink:
a new module without a test fails, and so does an entry whose test now exists.
"""

import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "src" / "activelabel" / "web" / "static" / "js"
TESTS = ROOT / "tests" / "js"

BASELINE = {"main.js"}  # main.js is what is left of app.js; it is tested through tests/js/*.test.mjs

# the folders each layer may import from (main.js, at the top, may import from all of them)
ALLOWED = {
    "shell": {"shell"},
    "integrations": {"shell", "integrations"},
    "cv": {"shell", "integrations", "cv"},
    "pages": {"shell", "integrations", "pages"},
}

# static ``import ... from '...'``, ``export ... from '...'`` and bare ``import '...'``
IMPORT = re.compile(r"""^\s*(?:import|export)\s*(?:[^'";]*?\bfrom\s*)?(['"])([^'"\n]+)\1""", re.M)


def modules() -> list[str]:
    """The modules, as paths relative to ``static/js``.

    Returns:
        Every ``*.js`` path under ``static/js``, sorted, with forward slashes.
    """
    return sorted(path.relative_to(JS).as_posix() for path in JS.rglob("*.js"))


def imports(module: str) -> list[str]:
    """The modules that one module imports, resolved against its folder.

    Args:
        module: Path of the module relative to ``static/js``.

    Returns:
        The imported paths relative to ``static/js`` (relative specifiers only), in order.
    """
    text = (JS / module).read_text(encoding="utf-8")
    found = [m.group(2) for m in IMPORT.finditer(text)]
    return [
        posixpath.normpath(posixpath.join(posixpath.dirname(module), s)) for s in found if s.startswith(".")
    ]


def test_every_import_is_relative_and_names_an_existing_file():
    for module in modules():
        text = (JS / module).read_text(encoding="utf-8")
        for match in IMPORT.finditer(text):
            assert match.group(2).startswith(("./", "../")), (module, match.group(2))
            assert match.group(2).endswith(".js"), (module, match.group(2))
        for target in imports(module):
            assert (JS / target).is_file(), f"{module} imports {target}, which does not exist"


def test_every_module_is_reachable_from_main():
    seen, pending = set(), ["main.js"]
    while pending:
        module = pending.pop()
        if module not in seen:
            seen.add(module)
            pending += imports(module)
    assert sorted(set(modules()) - seen) == []


def test_modules_import_only_from_their_layer_or_below():
    wrong = []
    for module in modules():
        layer = module.split("/")[0]
        if layer not in ALLOWED:
            assert module == "main.js", f"{module} is outside the folders {sorted(ALLOWED)}"
            continue
        wrong += [(module, t) for t in imports(module) if t.split("/")[0] not in ALLOWED[layer]]
    assert wrong == []


def test_every_module_has_a_mirror_test():
    missing = {m for m in modules() if not (TESTS / m.replace(".js", ".test.mjs")).is_file()}
    assert sorted(missing - BASELINE) == [], "add tests/js/<path>.test.mjs for the new module"
    assert sorted(BASELINE - missing) == [], "the module has a test (or is gone): remove it from BASELINE"
