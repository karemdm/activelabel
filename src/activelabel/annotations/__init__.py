"""Annotation files that come with a project's images: parse, check and import them as ground truth."""

from __future__ import annotations

from activelabel.annotations.folders import folder_classes
from activelabel.annotations.importer import apply
from activelabel.annotations.models import ParseResult, Problem, ValidationReport
from activelabel.annotations.parsing import FORMATS, detect_format, parse
from activelabel.annotations.staging import discard_staged, load_staged, stage
from activelabel.annotations.validation import validate

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
