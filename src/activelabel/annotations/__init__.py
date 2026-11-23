"""Annotation files that come with a project's images: parse, check and import them as ground truth."""

from __future__ import annotations

from activelabel.annotations.convert import conversion_possible, convert_annotations
from activelabel.annotations.folders import folder_classes
from activelabel.annotations.importer import apply
from activelabel.annotations.models import ParseResult, Problem, ValidationReport
from activelabel.annotations.parsing import FORMATS, VIDEO_ONLY, detect_format, parse
from activelabel.annotations.staging import discard_staged, load_staged, set_target, stage
from activelabel.annotations.target import VideoTarget, build_target, video_tasks
from activelabel.annotations.validation import validate

__all__ = [
    "FORMATS",
    "ParseResult",
    "Problem",
    "VIDEO_ONLY",
    "ValidationReport",
    "VideoTarget",
    "apply",
    "build_target",
    "conversion_possible",
    "convert_annotations",
    "detect_format",
    "discard_staged",
    "folder_classes",
    "load_staged",
    "parse",
    "set_target",
    "stage",
    "validate",
    "video_tasks",
]
