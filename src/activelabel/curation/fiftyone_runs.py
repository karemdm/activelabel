"""The analyses of a project recorded as runs of its FiftyOne dataset (#605, #629).

FiftyOne keeps, per dataset, the *runs* that computed something on it: brain runs (the similarity map's
``compute_visualization``, ``compute_mistakenness``, ``compute_hardness``), evaluations, annotation runs and
custom runs. ActiveLabel records the analyses it computes itself (uniqueness, regrouping, the label check)
as custom runs under stable keys (:data:`RUN_KEYS`), with the parameters and the time, so the App and
:func:`listing` show what was computed and how. A new computation replaces the run of the same key; deleting
the project deletes its dataset and every run with it.
"""

from __future__ import annotations

from typing import Any

from activelabel.storage import now

#: Stable key of the run of each analysis ActiveLabel records (letters, digits and underscores only).
RUN_KEYS = {
    "uniqueness": "activelabel_uniqueness",
    "clustering": "activelabel_clustering",
    "mistakenness": "activelabel_label_check",
}
#: Kinds of FiftyOne runs listed, with the collection methods that list them and describe one.
_KINDS = (
    ("brain", "list_brain_runs", "get_brain_info"),
    ("evaluation", "list_evaluations", "get_evaluation_info"),
    ("annotation", "list_annotation_runs", "get_annotation_info"),
    ("run", "list_runs", "get_run_info"),
)


def record(dataset: Any, analysis: str, **params: Any) -> str:
    """Record (or replace) the run of an analysis on the dataset.

    Args:
        dataset: the project's FiftyOne dataset.
        analysis: a key of :data:`RUN_KEYS` (``uniqueness``, ``clustering``, ``mistakenness``).
        **params: the parameters of the computation (JSON-ready), stored in the run's config.

    Returns:
        The run key.
    """
    key = RUN_KEYS[analysis]
    config = dataset.init_run(analysis=analysis, recorded=now(), **params)
    dataset.register_run(key, config, overwrite=True)
    return key


def _describe(info: Any) -> tuple[str, dict[str, Any]]:
    """The time and the parameters of a run, from FiftyOne's run info.

    Args:
        info: a FiftyOne ``RunInfo``.

    Returns:
        ``(timestamp in ISO format or "", config as a dict)``.
    """
    stamp = getattr(info, "timestamp", None)
    config = getattr(info, "config", None)
    serialized = config.serialize() if config is not None and hasattr(config, "serialize") else {}
    return (stamp.isoformat() if stamp is not None else ""), dict(serialized)


def listing(dataset: Any) -> list[dict[str, Any]]:
    """Every run of the dataset: brain runs, evaluations, annotation runs and custom runs.

    Args:
        dataset: the project's FiftyOne dataset.

    Returns:
        ``[{"type", "key", "timestamp", "config", "ours"}]`` sorted by type then key; ``ours`` is true for
        a run ActiveLabel records itself.
    """
    out: list[dict[str, Any]] = []
    ours = set(RUN_KEYS.values())
    for kind, list_name, info_name in _KINDS:
        for key in getattr(dataset, list_name)():
            timestamp, config = _describe(getattr(dataset, info_name)(key))
            out.append(
                {"type": kind, "key": key, "timestamp": timestamp, "config": config, "ours": key in ours}
            )
    return sorted(out, key=lambda run: (run["type"], run["key"]))


def delete(dataset: Any, key: str) -> bool:
    """Delete the run of one of ActiveLabel's analyses (the sample fields it wrote stay).

    Only the runs of :data:`RUN_KEYS` can be deleted here: the others belong to FiftyOne's own tools (an
    annotation run, for example, is tied to an annotation backend).

    Args:
        dataset: the project's FiftyOne dataset.
        key: the run key.

    Returns:
        True when the run existed and was deleted; False for an unknown key or a run that is not ours.
    """
    if key not in RUN_KEYS.values() or key not in dataset.list_runs():
        return False
    dataset.delete_run(key)
    return True
