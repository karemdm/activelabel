"""The similarity map in the FiftyOne App: a visualization brain run (Embeddings panel) and a group field.

After the map is computed, the project's dataset gets the brain run ``BRAIN_KEY`` (the same 2-D points as the
map of the curation, so both show the same picture) and every mapped sample gets ``GROUP_FIELD``, its group
numbered from 1, to colour the points or filter the grid by group. Nothing here needs PyTorch.
"""

from __future__ import annotations

from typing import Any

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

BRAIN_KEY = "activelabel_similarity"
GROUP_FIELD = "similarity_group"


def publish(dataset: fo.Dataset, result: dict[str, Any]) -> int:
    """Write the map and the groups of ``result`` into the dataset, replacing the previous ones.

    Args:
        dataset: The project's FiftyOne dataset (samples carry ``image_id``).
        result: The map (see :func:`activelabel.curation.similarity.build_map`).

    Returns:
        How many samples were placed on the map (images no longer in the dataset are left out).
    """
    if BRAIN_KEY in dataset.list_brain_runs():
        dataset.delete_brain_run(BRAIN_KEY)
    if dataset.has_sample_field(GROUP_FIELD):
        dataset.set_values(GROUP_FIELD, [None] * len(dataset))
    sample_of = dict(zip(dataset.values("image_id"), dataset.values("id"), strict=True))
    points = result["points"]
    rows = [i for i, image_id in enumerate(points["ids"]) if image_id in sample_of]
    if not rows:
        return 0
    sample_ids = [sample_of[points["ids"][i]] for i in rows]
    dataset.set_values(
        GROUP_FIELD,
        {sid: points["group"][i] + 1 for sid, i in zip(sample_ids, rows, strict=True)},
        key_field="id",
    )
    fob.compute_visualization(
        dataset.select(sample_ids, ordered=True),  # the points follow this order
        points=np.array([points["xy"][i] for i in rows], dtype=float),
        brain_key=BRAIN_KEY,
        progress=False,
    )
    return len(rows)
