"""Renaming a class where a project keeps class names besides its ground truth (#521).

The ground truth (``annotations.json``, pre-annotations included: they are labels with a ``score``) is renamed
by the code that renames the class. The files below keep a class list of their own and would keep the old
name:

- **Saved models** (``models/m<N>/model.json``): a trained model's ``classes`` are names for the indexes its
  weights output, so the name at an index is changed and the weights keep working. The names it was trained
  with stay in ``trained_classes`` (the interface shows them as a note). A public model's ``mapping`` (its
  categories to project classes) follows the new name.
- **Training runs** (``run.json``): the classes a run recorded, which become those of the model saved from it.
- **Pre-annotation runs** (``preannotations.json``): the classes a run was asked to write.

Dataset versions are not touched: a version is a frozen copy of what a training used, and a model trained
from it after the rename comes out with the names of the version (the pre-annotation screen marks the classes
the project no longer has).
"""

from __future__ import annotations

from pathlib import Path

from activelabel.preannotate import PreannotationStore
from activelabel.training.models import ModelStore
from activelabel.training.runs import RunStore


def case_renames(old: list[str], new: list[str]) -> dict[str, str]:
    """The classes whose name changed only in its letter case.

    Args:
        old: the class list before the change.
        new: the class list after it.

    Returns:
        ``{old name: new name}`` for each class of ``new`` that has the name of one of ``old`` apart from the
        case (``cat`` to ``Cat``).
    """
    before = {c.lower(): c for c in old}
    return {before[c.lower()]: c for c in new if c.lower() in before and before[c.lower()] != c}


def _renamed(names: list[str], renames: dict[str, str]) -> list[str]:
    """A class list with the renamed classes under their new name.

    Args:
        names: the class names.
        renames: ``{old name: new name}``.

    Returns:
        The names, in the same order.
    """
    return [renames.get(n, n) for n in names]


def _rename_models(project_dir: Path, renames: dict[str, str]) -> int:
    """Rename the classes of the saved models of the project.

    Args:
        project_dir: folder of the project.
        renames: ``{old name: new name}``.

    Returns:
        How many models changed.
    """
    store = ModelStore(project_dir)
    changed = 0
    for card in store.list():
        fields: dict[str, object] = {}
        classes = list(card.get("classes") or [])
        if _renamed(classes, renames) != classes:
            fields["classes"] = _renamed(classes, renames)
            if card.get("source") != "public":
                fields["trained_classes"] = card.get("trained_classes") or classes
        mapping = dict(card.get("mapping") or {})
        if any(v in renames for v in mapping.values()):
            fields["mapping"] = {k: renames.get(v, v) for k, v in mapping.items()}
        if fields:
            store.update(card["number"], **fields)
            changed += 1
    return changed


def _rename_runs(project_dir: Path, renames: dict[str, str]) -> int:
    """Rename the classes recorded by the training runs of the project.

    Args:
        project_dir: folder of the project.
        renames: ``{old name: new name}``.

    Returns:
        How many runs changed.
    """
    store = RunStore(project_dir)
    changed = 0
    for run in store.list():
        classes = list(run.get("classes") or [])
        if _renamed(classes, renames) != classes:
            store.update(run["number"], classes=_renamed(classes, renames))
            changed += 1
    return changed


def _rename_preannotations(project_dir: Path, renames: dict[str, str]) -> int:
    """Rename the classes chosen for the pre-annotation runs of the project.

    Args:
        project_dir: folder of the project.
        renames: ``{old name: new name}``.

    Returns:
        How many runs changed.
    """
    store = PreannotationStore(project_dir)
    changed = 0
    for run in store.list():
        classes = run.get("classes")
        if isinstance(classes, list) and _renamed(classes, renames) != classes:
            store.update(run["number"], classes=_renamed(classes, renames))
            changed += 1
    return changed


def rename_classes(project_dir: Path, renames: dict[str, str]) -> dict[str, int]:
    """Rename classes in the saved models, training runs and pre-annotation runs of a project.

    Args:
        project_dir: folder of the project.
        renames: ``{old name: new name}``.

    Returns:
        How many ``models``, training ``runs`` and ``preannotations`` changed.
    """
    if not renames:
        return {"models": 0, "runs": 0, "preannotations": 0}
    return {
        "models": _rename_models(project_dir, renames),
        "runs": _rename_runs(project_dir, renames),
        "preannotations": _rename_preannotations(project_dir, renames),
    }
