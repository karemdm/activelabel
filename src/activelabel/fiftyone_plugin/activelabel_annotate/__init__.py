"""FiftyOne plugin: delete several labels at once, and undo that deletion.

In the image window the App deletes one label at a time. The operator *Apagar rótulos* (a trash button in
the image window's toolbar, also in the operator browser) lists the boxes and polygons of ``ground_truth`` of
the image shown, with the labels already selected in the App ticked, and deletes the ticked ones in one go.
Outside the image window it works on the labels selected in the App. The labels deleted last are kept in the
dataset's execution store, so *Desfazer a remoção* (shown only while there is something to undo) puts them
back with their ids. Nothing reaches the project before *Revisar e aplicar* in ActiveLabel.

FiftyOne loads this file on its own, outside the ``activelabel`` package, so it imports nothing from it.
The App side of the plugin (Annotate mode, select tool after a deletion) is ``dist/index.umd.js``.
"""

from __future__ import annotations

import os
from typing import Any

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

FIELD = "ground_truth"  # the field ActiveLabel edits (activelabel.fiftyone_bridge.GROUND_TRUTH)
LISTS = {"Detections": "detections", "Polylines": "polylines"}  # label list attribute of each list type
STORE = "activelabel_annotate"  # execution store of the dataset
UNDO_KEY = "last_deleted"  # what the last deletion removed, for the undo
PICK_PREFIX = "label_"  # dialog field of each label: ``label_<sample id>_<label id>``
RELOAD_PAGE = "@activelabel/annotate/reload_page"  # the JS operator of dist/index.umd.js

TEXT: dict[str, dict[str, str]] = {
    "pt": {
        "delete": "Apagar rótulos",
        "undo": "Desfazer a remoção",
        "pick_title": "Quais rótulos apagar?",
        "pick": "Marque os rótulos e clique no botão Execute. Para desfazer, use Desfazer a remoção "
        "(lixeira com seta).",
        "item": "{label} n.º {n} (canto em {x}%, {y}%)",
        "none_title": "Nenhum rótulo para apagar",
        "none": "Abra uma imagem com caixas ou polígonos em ground_truth, ou selecione rótulos.",
        "nothing": "Nenhum rótulo marcado: nada foi apagado.",
        "deleted": "{n} rótulo(s) apagado(s). Para desfazer, use Desfazer a remoção.",
        "undo_title": "Pôr de volta os rótulos apagados",
        "undo_text": "{n} rótulo(s) apagado(s) por último voltam para as imagens.",
        "restored": "{n} rótulo(s) de volta.",
        "no_undo": "Não há remoção para desfazer.",
    },
    "en": {
        "delete": "Delete labels",
        "undo": "Undo the deletion",
        "pick_title": "Which labels to delete?",
        "pick": "Tick the labels and click the Execute button. To undo, use Undo the deletion "
        "(trash can with an arrow).",
        "item": "{label} no. {n} (corner at {x}%, {y}%)",
        "none_title": "No label to delete",
        "none": "Open an image with boxes or polygons in ground_truth, or select labels.",
        "nothing": "No label ticked: nothing was deleted.",
        "deleted": "{n} label(s) deleted. To undo, use Undo the deletion.",
        "undo_title": "Put the deleted labels back",
        "undo_text": "The {n} label(s) deleted last go back to their images.",
        "restored": "{n} label(s) back.",
        "no_undo": "There is no deletion to undo.",
    },
}


def say(key: str, **values: Any) -> str:
    """A text of the plugin in the language of ActiveLabel (``ACTIVELABEL_LANG``; Portuguese by default).

    Args:
        key: the name of the text.
        **values: values to put in the text.

    Returns:
        The text, in one language only.
    """
    lang = "en" if os.environ.get("ACTIVELABEL_LANG", "").strip().lower().startswith("en") else "pt"
    return TEXT[lang][key].format(**values)


def label_list(value: Any) -> list[Any] | None:
    """The list of labels of a ``ground_truth`` value (boxes or polygons).

    Args:
        value: the value of the field (``Detections``, ``Polylines`` or anything else).

    Returns:
        The list held by the value, or None when the value holds no list of boxes or polygons.
    """
    attr = LISTS.get(type(value).__name__)
    return None if attr is None else getattr(value, attr)


def corner(label: Any) -> tuple[int, int]:
    """Where a label starts on the image, in percent, to tell labels of the same class apart.

    Args:
        label: a ``Detection`` (its ``bounding_box``) or a ``Polyline`` (its first point).

    Returns:
        ``(x, y)`` in whole percents; ``(0, 0)`` when the label has no position.
    """
    box = getattr(label, "bounding_box", None)
    if box:
        return round(box[0] * 100), round(box[1] * 100)
    points = getattr(label, "points", None) or []
    first = points[0][0] if points and points[0] else None
    return (round(first[0] * 100), round(first[1] * 100)) if first else (0, 0)


def candidates(dataset: Any, current: str | None, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The labels the dialog offers: those of the image shown, or else the labels selected in the App.

    Args:
        dataset: the dataset open in the App.
        current: id of the sample shown in the image window, or None.
        selected: the labels selected in the App (``sample_id``, ``field``, ``label_id``).

    Returns:
        ``{sample_id, label_id, text, ticked}`` per label, in the order of each image's list.
    """
    chosen = {s.get("label_id") for s in selected if s.get("field") == FIELD}
    samples = [current] if current else sorted({s["sample_id"] for s in selected if s.get("field") == FIELD})
    out: list[dict[str, Any]] = []
    for sample_id in samples:
        labels = label_list(dataset[sample_id][FIELD]) or []
        if not current:
            labels = [x for x in labels if x.id in chosen]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label.label] = counts.get(label.label, 0) + 1
            x, y = corner(label)
            text = say("item", label=label.label, n=counts[label.label], x=x, y=y)
            out.append(
                {"sample_id": sample_id, "label_id": label.id, "text": text, "ticked": label.id in chosen}
            )
    return out


def delete_labels(dataset: Any, picked: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Delete labels of ``ground_truth`` and return what was deleted, to undo it.

    Args:
        dataset: the dataset.
        picked: ``(sample_id, label_id)`` of the labels to delete.

    Returns:
        ``{sample_id, kind, labels}`` per image; ``labels`` holds the deleted labels as dicts, with their
        ids.
    """
    by_sample: dict[str, set[str]] = {}
    for sample_id, label_id in picked:
        by_sample.setdefault(sample_id, set()).add(label_id)
    removed: list[dict[str, Any]] = []
    for sample_id, ids in by_sample.items():
        sample = dataset[sample_id]
        value = sample[FIELD]
        labels = label_list(value)
        if labels is None:
            continue
        gone = [x for x in labels if x.id in ids]
        if not gone:
            continue
        setattr(value, LISTS[type(value).__name__], [x for x in labels if x.id not in ids])
        sample.save()
        removed.append(
            {"sample_id": sample_id, "kind": type(value).__name__, "labels": [x.to_dict() for x in gone]}
        )
    return removed


def restore_labels(dataset: Any, removed: list[dict[str, Any]]) -> int:
    """Put back labels deleted by :func:`delete_labels`; a label whose id is there again is skipped.

    Args:
        dataset: the dataset.
        removed: what :func:`delete_labels` returned.

    Returns:
        How many labels came back.
    """
    count = 0
    for entry in removed:
        sample = dataset[entry["sample_id"]]
        kind = entry["kind"]
        value = sample[FIELD]
        if value is None:
            value = getattr(fo, kind)()
        labels = label_list(value)
        if labels is None:
            continue
        have = {x.id for x in labels}
        back = [fo.Label.from_dict(d) for d in entry["labels"]]
        back = [x for x in back if x.id not in have]
        setattr(value, LISTS[kind], labels + back)
        sample[FIELD] = value
        sample.save()
        count += len(back)
    return count


def notice(inputs: Any, title: str, text: str) -> Any:
    """A dialog that only says something.

    Args:
        inputs: the dialog's object.
        title: the title.
        text: the text.

    Returns:
        The dialog.
    """
    inputs.view("notice", types.Notice(label=title, description=text))
    return types.Property(inputs)


def show_again(ctx: Any) -> None:
    """Make the App show the changed images.

    Neither ``reload_samples`` nor ``reload_dataset`` refreshes the image window, whose Annotate mode keeps
    its own copy of the labels, so the page is reloaded (``reload_page`` of ``dist/index.umd.js``); the
    image window comes back by itself, its id being in the address.

    Args:
        ctx: the operator context.
    """
    ctx.trigger(RELOAD_PAGE)


class DeleteLabels(foo.Operator):
    """*Apagar rótulos*: tick several labels of the image shown (or of the selection) and delete them."""

    @property
    def config(self) -> Any:
        """Operator settings."""
        return foo.OperatorConfig(name="delete_labels", label=say("delete"))

    def resolve_placement(self, ctx: Any) -> Any:
        """A trash button in the image window's toolbar.

        Args:
            ctx: the operator context.

        Returns:
            The placement.
        """
        button = types.Button(label=say("delete"), icon="delete_outline", prompt=True)
        return types.Placement(types.Places.SAMPLES_VIEWER_ACTIONS, button)

    def resolve_input(self, ctx: Any) -> Any:
        """One tick box per label (those selected in the App already ticked), or a notice when there is none.

        Args:
            ctx: the operator context.

        Returns:
            The dialog.
        """
        inputs = types.Object()
        found = candidates(ctx.dataset, ctx.current_sample, list(ctx.selected_labels or []))
        if not found:
            return notice(inputs, say("none_title"), say("none"))
        inputs.view("intro", types.Notice(label=say("pick_title"), description=say("pick")))
        for item in found:
            inputs.bool(
                f"{PICK_PREFIX}{item['sample_id']}_{item['label_id']}",
                default=item["ticked"],
                label=item["text"],
            )
        return types.Property(inputs)

    def execute(self, ctx: Any) -> dict[str, Any]:
        """Delete the ticked labels, keep them for the undo and reload the images.

        Args:
            ctx: the operator context.

        Returns:
            ``{deleted}``: how many labels were deleted.
        """
        picked = [
            tuple(key[len(PICK_PREFIX) :].split("_", 1))
            for key, value in (ctx.params or {}).items()
            if key.startswith(PICK_PREFIX) and value is True
        ]
        if not picked:
            ctx.ops.notify(say("nothing"))
            return {"deleted": 0}
        removed = delete_labels(ctx.dataset, picked)
        count = sum(len(entry["labels"]) for entry in removed)
        if removed:
            ctx.store(STORE).set(UNDO_KEY, removed)
        show_again(ctx)
        ctx.ops.notify(say("deleted", n=count), variant="success")
        return {"deleted": count}


class RestoreLabels(foo.Operator):
    """*Desfazer a remoção*: put back the labels deleted last by :class:`DeleteLabels`."""

    @property
    def config(self) -> Any:
        """Operator settings."""
        return foo.OperatorConfig(name="restore_labels", label=say("undo"), dynamic=True)

    def resolve_placement(self, ctx: Any) -> Any:
        """A button in the image window's toolbar, only while there is a deletion to undo.

        Args:
            ctx: the operator context.

        Returns:
            The placement, or None when there is nothing to undo.
        """
        if not ctx.store(STORE).get(UNDO_KEY):
            return None
        button = types.Button(label=say("undo"), icon="restore_from_trash", prompt=True)
        return types.Placement(types.Places.SAMPLES_VIEWER_ACTIONS, button)

    def resolve_input(self, ctx: Any) -> Any:
        """Say how many labels come back.

        Args:
            ctx: the operator context.

        Returns:
            The dialog.
        """
        removed = ctx.store(STORE).get(UNDO_KEY) or []
        count = sum(len(entry["labels"]) for entry in removed)
        if not count:
            return notice(types.Object(), say("undo_title"), say("no_undo"))
        return notice(types.Object(), say("undo_title"), say("undo_text", n=count))

    def execute(self, ctx: Any) -> dict[str, Any]:
        """Put the labels back, forget the deletion and reload the images.

        Args:
            ctx: the operator context.

        Returns:
            ``{restored}``: how many labels came back.
        """
        store = ctx.store(STORE)
        count = restore_labels(ctx.dataset, store.get(UNDO_KEY) or [])
        store.delete(UNDO_KEY)
        show_again(ctx)
        ctx.ops.notify(say("restored", n=count), variant="success")
        return {"restored": count}


def register(p: Any) -> None:
    """Register the operators with FiftyOne.

    Args:
        p: the FiftyOne plugin context.
    """
    p.register(DeleteLabels)
    p.register(RestoreLabels)
