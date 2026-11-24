"""Every module with code has a mirror test file.

The rule: ``src/activelabel/<folders>/x.py`` is tested by ``tests/activelabel/<folders>/test_x.py``. A module
that only has a docstring, imports or ``__all__`` (this includes most ``__init__.py`` files) needs no mirror.
Several tests may live in the mirror file; a test of a module that spans other files stays in the file of the
module it is about.

``BASELINE`` lists today's modules without a mirror, as paths relative to ``src/activelabel``. It may
only shrink: a new module without a mirror fails the test, and so does an entry that is fixed (the mirror
exists, or the module was removed or has no code), so remove an entry in the same change that adds the
test file.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "activelabel"
TESTS = ROOT / "tests" / "activelabel"

BASELINE = {
    "annotate/base.py",
    "annotate/registry.py",
    "annotations/coco.py",
    "annotations/context.py",
    "annotations/folders.py",
    "annotations/importer.py",
    "annotations/models.py",
    "annotations/parsing.py",
    "annotations/staging.py",
    "annotations/target.py",
    "annotations/validation.py",
    "annotations/videoformats.py",
    "annotations/xmlformats.py",
    "annotations/yolo.py",
    "curation/issues.py",
    "curation/quality.py",
    "curation/summary.py",
    "fiftyone_names.py",
    "preannotate.py",
    "sync/registry.py",
    "sync/remote.py",
    "training/catalog.py",
    "training/config.py",
    "training/runs.py",
    "training/workers/classification.py",
    "training/workers/detection.py",
    "training/workers/export_onnx.py",
    "training/workers/predict.py",
    "training/workers/segmentation.py",
    "web/annotate_api.py",
    "web/app.py",
    "web/errors.py",
    "web/models_api.py",
    "web/preannotate_api.py",
    "web/state.py",
}


def _has_code(path: Path) -> bool:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and "__all__" in ast.unparse(node):
            continue
        return True
    return False


def _mirror(path: Path) -> Path:
    relative = path.relative_to(SOURCE)
    return TESTS / relative.parent / f"test_{relative.name}"


def _without_mirror() -> set[str]:
    return {
        path.relative_to(SOURCE).as_posix()
        for path in sorted(SOURCE.rglob("*.py"))
        if _has_code(path) and not _mirror(path).exists()
    }


def test_every_module_with_code_has_a_mirror_test():
    new = sorted(_without_mirror() - BASELINE)
    assert not new, f"add tests/activelabel/<folders>/test_<name>.py for: {new}"


def test_baseline_has_no_stale_entries():
    stale = sorted(BASELINE - _without_mirror())
    assert not stale, f"now covered or removed: remove from BASELINE: {stale}"
