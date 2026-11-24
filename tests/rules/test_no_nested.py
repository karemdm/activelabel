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

BASELINE = {
    "annotate/diff.py:_compare_all.bump",
    "annotate/diff.py:_compare_all.problem",
    "annotations/videoformats.py:_read_cvat_video.size",
    "annotations/videoformats.py:_read_cvat_video.video_frame",
    "curation/issues.py:_group_duplicates.find",
    "curation/quality.py:measure_images.save",
    "curation/quality.py:measure_images.work",
    "datasets.py:assign.pick",
    "datasets.py:assign.place_video",
    "datasets.py:assign.put",
    "datasets.py:assign.score",
    "datasets.py:build_groups.find",
    "datasets.py:build_groups.union",
    "datasets.py:plan_split.attempt",
    "datasets.py:plan_split.frame_no",
    "datasets.py:plan_split.seconds",
    "sync/cvat_remote.py:_same_geometry.close",
    "sync/cvat_remote.py:labels_to_cvat.shape",
    "sync/engine.py:apply_plan.ids_of",
    "sync/engine.py:make_plan.remote_known",
    "training/catalog.py:_classification.m",
    "training/public_models.py:suggest_mapping.key",
    "training/runner.py:TrainingRunner._export.step",
    "training/runner.py:TrainingRunner._preannotate.on_event",
    "training/runner.py:TrainingRunner._preannotate.step",
    "training/runner.py:TrainingRunner._run.step",
    "training/runner.py:TrainingRunner._run_children.update",
    "training/runner.py:TrainingRunner._run_export.pump",
    "training/runner.py:TrainingRunner._train.pump",
    "training/workers/classification.py:train.checkpoint",
    "training/workers/classification.py:train.loader",
    "training/workers/classification.py:train.on_step",
    "training/workers/detection.py:train.checkpoint",
    "training/workers/detection.py:train.loader",
    "training/workers/detection.py:train.on_step",
    "training/workers/export_onnx.py:export.Logits",
    "training/workers/segmentation.py:train.checkpoint",
    "training/workers/segmentation.py:train.loader",
    "training/workers/segmentation.py:train.on_step",
    "web/annotate_api.py:pull.run",
    "web/annotate_api.py:send.run",
    "web/api.py:analyze.run",
    "web/api.py:drop_fiftyone_dataset.run",
    "web/api.py:import_folder.run",
    "web/api.py:open_fiftyone.run",
    "web/app.py:create_app.lifespan",
    "web/errors.py:install_error_handlers.api_error",
    "web/errors.py:install_error_handlers.damaged_file",
    "web/import_api.py:start_import.run",
    "web/pages.py:_bound_translate.t",
    "web/sync_api.py:apply.run",
    "web/sync_api.py:plan.run",
    "web/video_api.py:task_from_video.run",
}


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
