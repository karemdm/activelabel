"""Annotation files that come with a project's images: parse, check and import them as ground truth."""

from __future__ import annotations

from visionloop.annotations.folders import folder_classes
from visionloop.annotations.importer import apply
from visionloop.annotations.models import ParseResult, Problem, ValidationReport
from visionloop.annotations.parsing import FORMATS, detect_format, parse
from visionloop.annotations.staging import discard_staged, load_staged, stage
from visionloop.annotations.validation import validate

__all__ = [
    "FORMATS",
    "ParseResult",
    "Problem",
    "ValidationReport",
    "apply",
    "detect_format",
    "discard_staged",
    "folder_classes",
    "load_staged",
    "parse",
    "stage",
    "validate",
]
