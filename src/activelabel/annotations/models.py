"""Result types of parsing and validating annotation files (all JSON-serialisable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from activelabel.labels import Label

MAX_EXAMPLES = 5

# code -> (level, short message). The interface translates by code; codes are stable.
PROBLEMS: dict[str, tuple[str, str]] = {
    "invalid_file": ("error", "A file could not be read or is not valid."),
    "no_images_matched": ("error", "No annotation matches an image of this project."),
    "no_labels": ("error", "No usable annotations were found."),
    "unmatched_entries": ("warning", "Some annotations refer to images that are not in the project."),
    "ambiguous_image_name": ("warning", "Several images have this name, so the annotations were skipped."),
    "duplicate_image_entry": (
        "warning",
        "The same image appears more than once; later entries were skipped.",
    ),
    "size_mismatch": ("warning", "The image size in the file differs from the uploaded image."),
    "box_clipped": ("warning", "Some boxes went outside the image and were clipped."),
    "box_degenerate": ("warning", "Some boxes have no area and were skipped."),
    "polygon_too_few_points": ("warning", "Some polygons have fewer than 3 points and were skipped."),
    "unknown_class_id": ("warning", "Some annotations use a class that is not defined and were skipped."),
    "missing_class_names": ("warning", "Class names were not found; names like class_0 are used."),
    "keypoint_count_mismatch": (
        "warning",
        "Some objects have a different number of keypoints than expected.",
    ),
    "unsupported_annotation": (
        "warning",
        "Some annotations have no box, polygon or keypoints and were skipped.",
    ),
    "unsupported_shape": ("warning", "Some shapes of an unsupported type were skipped."),
    "rle_bbox_only": ("warning", "Mask (RLE) objects were imported as boxes only."),
    "multiple_classes": ("warning", "Some images have more than one class; only the first one is kept."),
    "invalid_annotation": ("warning", "Some annotations are malformed and were skipped."),
    "frames_not_in_task": ("warning", "Some annotations are on frames that the task does not use."),
    "frame_number_missing": ("warning", "Some entries have no frame number and were skipped."),
}


@dataclass
class Problem:
    """One kind of problem, with how many times it happened and a few examples."""

    level: str  # "error" blocks the import; "warning" does not
    code: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, example: str | None = None, n: int = 1) -> None:
        """Count another occurrence (keeps up to 5 distinct examples)."""
        self.count += n
        if example and example not in self.examples and len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


@dataclass
class ParseResult:
    """What was read from the annotation files.

    Attributes:
        format: coco, yolo, voc, cvat, labelstudio, folders, cvatvideo or mot.
        labels: labels per image id. Images matched without objects have an empty list.
        classes: class names in file order (declared ones first, then by first use).
        files: names of the annotation files read.
        images_total: images offered to the parser (the project's active images).
        unmatched_count: annotation entries whose image was not found.
        unmatched: up to 20 examples of those names.
        problems: problems found while reading.
        keypoint_names: names of the keypoints, when the format has them.
    """

    format: str
    labels: dict[str, list[Label]]
    classes: list[str]
    files: list[str]
    images_total: int
    unmatched_count: int = 0
    unmatched: list[str] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    keypoint_names: list[str] = field(default_factory=list)

    @property
    def images_matched(self) -> int:
        """Images that got an entry (possibly empty)."""
        return len(self.labels)

    def all_labels(self) -> list[Label]:
        """Every label of every image."""
        return [lb for labels in self.labels.values() for lb in labels]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready summary (labels are not included, they can be large)."""
        return {
            "format": self.format,
            "classes": self.classes,
            "files": self.files,
            "images_total": self.images_total,
            "images_matched": self.images_matched,
            "labels_total": len(self.all_labels()),
            "unmatched_count": self.unmatched_count,
            "unmatched": self.unmatched,
            "problems": [p.as_dict() for p in self.problems],
            "keypoint_names": self.keypoint_names,
        }


@dataclass
class ValidationReport:
    """Check of a parse result against a project.

    Attributes:
        format: detected or given format.
        inferred_task: task the labels fit (None when there are no labels).
        project_task: task of the project.
        task_matches: inferred task equals project task.
        classes_found: classes in the files.
        classes_new: found classes missing from the project (empty when the project has none yet).
        counts: annotation_files, images_matched, images_with_labels, images_without_labels,
            labels_total, labels_per_class.
        problems: problems with level, code, message, count and examples.
        can_import: no blocking error and at least one label.
        options: actions the user can take: import, change_task_and_import, convert_and_import, discard.
        unmatched: count and examples of annotation entries without an image.
    """

    format: str
    inferred_task: str | None
    project_task: str
    task_matches: bool
    classes_found: list[str]
    classes_new: list[str]
    counts: dict[str, Any]
    problems: list[Problem]
    can_import: bool
    options: list[str]
    unmatched: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        data = asdict(self)
        data["problems"] = [p.as_dict() for p in self.problems]
        return data
