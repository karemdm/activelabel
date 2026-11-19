"""Numbers for the curation screen: counts, image lists per issue, histograms, suggestions."""

from __future__ import annotations

from typing import Any

import numpy as np

from activelabel.curation.issues import BLURRY, BRIGHT, DARK, ISSUES, LOW_CONTRAST, NEAR_DUPLICATE
from activelabel.images.manifest import ImageRecord, Manifest

BINS = 20
HISTOGRAM_METRICS = ("brightness", "contrast", "sharpness")


def _histogram(values: list[float]) -> dict[str, list[float]]:
    """``BINS`` equal bins between the minimum and maximum (a single bin of data if all equal)."""
    if not values:
        return {"edges": [], "counts": []}
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.0
    counts, edges = np.histogram(values, bins=BINS, range=(low, high))
    return {"edges": [round(float(e), 3) for e in edges], "counts": [int(c) for c in counts]}


def summary(manifest: Manifest) -> dict[str, Any]:
    """Overview of the quality of a project (excluded images are not counted in issues or charts).

    Returns:
        ``{"totals": {images, active, excluded, measured}, "issues": {name: count},
        "issue_ids": {name: [image ids]}, "histograms": {metric: {"edges", "counts"}}}``.
    """
    records = manifest.all()
    active = [r for r in records if r.status == "active"]
    issue_ids: dict[str, list[str]] = {issue: [] for issue in ISSUES}
    for record in active:
        for issue in record.issues:
            issue_ids.setdefault(issue, []).append(record.id)
    return {
        "totals": {
            "images": len(records),
            "active": len(active),
            "excluded": len(records) - len(active),
            "measured": sum(1 for r in active if r.metrics),
        },
        "issues": {issue: len(ids) for issue, ids in issue_ids.items()},
        "issue_ids": issue_ids,
        "histograms": {
            metric: _histogram([r.metrics[metric] for r in active if metric in r.metrics])
            for metric in HISTOGRAM_METRICS
        },
    }


def _worth_excluding(record: ImageRecord) -> bool:
    issues = set(record.issues)
    return NEAR_DUPLICATE in issues or (BLURRY in issues and bool(issues & {DARK, BRIGHT, LOW_CONTRAST}))


def suggest_exclusions(manifest: Manifest) -> list[str]:
    """Ids of active images the user will most likely want to remove (never excludes anything).

    Conservative: near duplicates, and blurry images that are also dark, overexposed or flat.
    """
    return [r.id for r in manifest.active() if _worth_excluding(r)]
