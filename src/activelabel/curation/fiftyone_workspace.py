"""A saved workspace of the FiftyOne App with the charts of the project (#623).

The workspace has the samples grid next to the Histograms panel and, once the similarity map exists
(:data:`~activelabel.curation.similarity_fiftyone.BRAIN_KEY`), the Embeddings panel. It is only listed in the
App's workspace menu: nothing is opened for the person.
"""

from __future__ import annotations

import logging

from activelabel.curation.similarity_fiftyone import BRAIN_KEY
from activelabel.optional import fo

log = logging.getLogger(__name__)

WORKSPACE = "ActiveLabel: distribuições"
SAMPLES, HISTOGRAMS, EMBEDDINGS = "Samples", "Histograms", "Embeddings"


def panel_types(dataset: fo.Dataset) -> list[str]:
    """Panels the workspace of the charts should have for this dataset.

    Args:
        dataset: the project's dataset.

    Returns:
        ``Samples`` and ``Histograms``, plus ``Embeddings`` when the similarity map exists.
    """
    types = [SAMPLES, HISTOGRAMS]
    if BRAIN_KEY in dataset.list_brain_runs():
        types.append(EMBEDDINGS)
    return types


def _space(types: list[str]) -> fo.Space:
    """The layout: the grid on the left and the chart panels as tabs on the right.

    Args:
        types: the panel types, the first one is the grid.

    Returns:
        The space to save.
    """
    state = {"brainResult": BRAIN_KEY}
    charts = [fo.Panel(type=t, state=state if t == EMBEDDINGS else {}) for t in types[1:]]
    return fo.Space(
        children=[fo.Space(children=[fo.Panel(type=types[0])]), fo.Space(children=charts)],
        orientation="horizontal",
    )


def save_workspace(dataset: fo.Dataset) -> bool:
    """Save the workspace of the charts (again only when the similarity map appeared or went away).

    Args:
        dataset: the project's dataset.

    Returns:
        True when the workspace was written, False when it was already as wanted or could not be saved
        (the App still works without it).
    """
    wanted = panel_types(dataset)
    try:
        if _flat(dataset) == wanted:
            return False
        dataset.save_workspace(
            WORKSPACE,
            _space(wanted),
            description="Samples, Histograms and Embeddings",
            overwrite=True,
        )
    except Exception:  # noqa: BLE001 - only a convenience of the App
        log.warning("Could not save the charts workspace of the FiftyOne App", exc_info=True)
        return False
    return True


def _flat(dataset: fo.Dataset) -> list[str] | None:
    """Panel types of the saved workspace, left to right.

    Args:
        dataset: the project's dataset.

    Returns:
        The types in order, or None when the workspace is not saved.
    """
    if not dataset.has_workspace(WORKSPACE):
        return None
    out: list[str] = []
    for column in dataset.load_workspace(WORKSPACE).children:
        out += [panel.type for panel in column.children]
    return out
