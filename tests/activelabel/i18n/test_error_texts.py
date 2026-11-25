"""Every error code the server can send has a translated text (``error.<code>``) in both languages."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from activelabel import i18n

SRC = Path(i18n.__file__).parents[1]
CODED = {
    "ApiError": 1,
    "AugmentationError": 0,
    "CodedError": 0,
    "ConfigError": 0,
    "ModelError": 0,
    "RemoteChanged": 1,
    "RemoteNotFound": 1,
    "SyncError": 1,
    "ToolError": 1,
}  # name -> position of the code
LEGACY_BODIES = {
    "file_damaged",
    "task_change_annotations",
    "classes_in_use",
    "no_images_added",
    "validation_error",
    "not_found",
    "method_not_allowed",
}


def _load(lang: str) -> dict[str, str]:
    return json.loads((Path(i18n.__file__).parent / f"{lang}.json").read_text(encoding="utf-8"))


def _raised_codes() -> dict[str, set[str]]:
    """Every error code written as a literal in src, with the names of the params it is raised with."""
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            at = CODED.get(name)
            if at is None or len(node.args) <= at:
                continue
            code = node.args[at]
            if isinstance(code, ast.Constant) and isinstance(code.value, str):
                found.setdefault(code.value, set()).update(k.arg for k in node.keywords if k.arg)
    return found


RAISED = _raised_codes()


def test_the_scan_finds_the_codes_of_every_module():
    """A guard on the scan itself: it sees the codes of the core, the split, training and video modules."""
    assert {
        "project_name_taken",
        "dataset_part_empty",
        "run_no_weights",
        "video_unreadable",
        "tool_error",
    } <= set(RAISED)
    assert len(RAISED) > 120


def test_every_code_has_a_text_in_both_languages():
    codes = set(RAISED) | LEGACY_BODIES
    for lang in ("pt", "en"):
        missing = sorted(c for c in codes if f"error.{c}" not in _load(lang))
        assert not missing, f"{lang}.json lacks error.<code> for: {', '.join(missing)}"


def test_the_portuguese_text_is_not_the_english_one():
    pt, en = _load("pt"), _load("en")
    same = sorted(f"error.{c}" for c in set(RAISED) | LEGACY_BODIES if pt[f"error.{c}"] == en[f"error.{c}"])
    assert not same, f"still in English in pt.json: {', '.join(same)}"


@pytest.mark.parametrize("code", sorted(RAISED))
def test_a_text_only_uses_params_its_error_carries(code: str):
    placeholders = set(re.findall(r"\{(\w+)\}", _load("en")[f"error.{code}"]))
    assert placeholders <= RAISED[code], (
        f"error.{code} uses {placeholders - RAISED[code]} that the server does not send"
    )


def test_the_validation_texts_cover_the_usual_problems():
    pt = _load("pt")
    assert {"error.v.missing", "error.v.int_parsing", "error.v.string_too_long"} <= set(pt)
