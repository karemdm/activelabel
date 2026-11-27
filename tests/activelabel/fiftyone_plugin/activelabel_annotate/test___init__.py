import uuid
from types import SimpleNamespace

import fiftyone as fo
import pytest

from activelabel.fiftyone_plugin import activelabel_annotate as plugin

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset(tmp_path):
    ds = fo.Dataset(f"test-activelabel-delete-{uuid.uuid4().hex[:8]}")
    boxes = [
        fo.Detection(label="cat", bounding_box=[0.1, 0.2, 0.2, 0.2]),
        fo.Detection(label="cat", bounding_box=[0.5, 0.5, 0.2, 0.2]),
        fo.Detection(label="dog", bounding_box=[0.7, 0.1, 0.1, 0.1]),
    ]
    ds.add_sample(fo.Sample(filepath=str(tmp_path / "a.png"), ground_truth=fo.Detections(detections=boxes)))
    yield ds
    ds.delete()


@pytest.fixture
def polygons(tmp_path):
    ds = fo.Dataset(f"test-activelabel-delete-{uuid.uuid4().hex[:8]}")
    poly = fo.Polyline(label="cat", points=[[(0.3, 0.4), (0.5, 0.4), (0.4, 0.6)]], closed=True, filled=True)
    ds.add_sample(fo.Sample(filepath=str(tmp_path / "b.png"), ground_truth=fo.Polylines(polylines=[poly])))
    yield ds
    ds.delete()


class Store:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def delete(self, key):
        return self.data.pop(key, None) is not None


class Ops:
    def __init__(self):
        self.calls = []

    def notify(self, message, variant="info"):
        self.calls.append((variant, message))


class Triggers(list):
    def __call__(self, uri, params=None):
        self.append(uri)


def context(dataset, params=None, current=None, selected=None, store=None):
    shared = store or Store()
    return SimpleNamespace(
        dataset=dataset,
        params=params or {},
        current_sample=current,
        selected_labels=selected or [],
        ops=Ops(),
        trigger=Triggers(),
        store=lambda name: shared,
        shared=shared,
    )


def test_the_texts_are_in_one_language(monkeypatch):
    monkeypatch.setenv("ACTIVELABEL_LANG", "en")
    assert plugin.say("delete") == "Delete labels"
    monkeypatch.delenv("ACTIVELABEL_LANG")
    assert plugin.say("delete") == "Apagar rótulos"
    assert set(plugin.TEXT["pt"]) == set(plugin.TEXT["en"])


def test_the_labels_of_the_image_shown_are_offered_with_the_selected_ones_ticked(dataset):
    sample = dataset.first()
    ids = [d.id for d in sample.ground_truth.detections]
    selected = [{"sample_id": sample.id, "field": "ground_truth", "label_id": ids[1]}]
    found = plugin.candidates(dataset, sample.id, selected)
    assert [f["label_id"] for f in found] == ids
    assert [f["ticked"] for f in found] == [False, True, False]
    assert "cat" in found[0]["text"] and "1" in found[0]["text"] and "2" in found[1]["text"]
    assert "10%" in found[0]["text"] and "20%" in found[0]["text"]


def test_outside_the_image_window_only_the_selected_labels_are_offered(dataset):
    sample = dataset.first()
    first = sample.ground_truth.detections[0].id
    found = plugin.candidates(
        dataset, None, [{"sample_id": sample.id, "field": "ground_truth", "label_id": first}]
    )
    assert [f["label_id"] for f in found] == [first]
    assert (
        plugin.candidates(dataset, None, [{"sample_id": sample.id, "field": "predictions", "label_id": "x"}])
        == []
    )


def test_a_polygon_is_offered_by_its_first_point(polygons):
    sample = polygons.first()
    found = plugin.candidates(polygons, sample.id, [])
    assert len(found) == 1 and "30%" in found[0]["text"] and "40%" in found[0]["text"]


def test_several_labels_are_deleted_at_once_and_the_undo_puts_them_back(dataset):
    sample = dataset.first()
    ids = [d.id for d in sample.ground_truth.detections]
    params = {
        f"label_{sample.id}_{ids[0]}": True,
        f"label_{sample.id}_{ids[2]}": True,
        f"label_{sample.id}_{ids[1]}": False,
    }
    ctx = context(dataset, params, current=sample.id)
    assert plugin.DeleteLabels().execute(ctx) == {"deleted": 2}
    sample.reload()
    assert [d.id for d in sample.ground_truth.detections] == [ids[1]]
    assert ctx.ops.calls[0][0] == "success"
    assert ctx.trigger == [plugin.RELOAD_PAGE]  # the App reads the images again
    undo = context(dataset, store=ctx.shared)
    assert plugin.RestoreLabels().execute(undo) == {"restored": 2}
    sample.reload()
    assert sorted(d.id for d in sample.ground_truth.detections) == sorted(ids)
    assert ctx.shared.get(plugin.UNDO_KEY) is None


def test_a_polygon_is_deleted_and_restored_as_a_polygon(polygons):
    sample = polygons.first()
    pid = sample.ground_truth.polylines[0].id
    removed = plugin.delete_labels(polygons, [(sample.id, pid)])
    sample.reload()
    assert sample.ground_truth.polylines == []
    assert plugin.restore_labels(polygons, removed) == 1
    sample.reload()
    back = sample.ground_truth.polylines[0]
    assert back.id == pid and back.closed is True and back.label == "cat"


def test_a_restore_does_not_duplicate_a_label_that_is_there_again(dataset):
    sample = dataset.first()
    first = sample.ground_truth.detections[0].id
    removed = plugin.delete_labels(dataset, [(sample.id, first)])
    assert plugin.restore_labels(dataset, removed) == 1
    assert plugin.restore_labels(dataset, removed) == 0
    sample.reload()
    assert len(sample.ground_truth.detections) == 3


def test_nothing_ticked_deletes_nothing(dataset):
    ctx = context(dataset, {"label_a_b": False}, current=dataset.first().id)
    assert plugin.DeleteLabels().execute(ctx) == {"deleted": 0}
    assert ctx.shared.get(plugin.UNDO_KEY) is None


def test_the_dialogs(dataset):
    sample = dataset.first()
    dialog = plugin.DeleteLabels().resolve_input(context(dataset, current=sample.id))
    names = [name for name in dialog.type.properties if name.startswith(plugin.PICK_PREFIX)]
    assert len(names) == 3
    empty = plugin.DeleteLabels().resolve_input(context(dataset))
    assert "notice" in empty.type.properties
    assert plugin.RestoreLabels().resolve_placement(context(dataset)) is None
    store = Store()
    store.set(plugin.UNDO_KEY, [{"sample_id": sample.id, "kind": "Detections", "labels": [{}]}])
    assert plugin.RestoreLabels().resolve_placement(context(dataset, store=store)) is not None
    assert plugin.DeleteLabels().resolve_placement(context(dataset)) is not None
    assert plugin.RestoreLabels().resolve_input(context(dataset, store=store)) is not None


def test_a_value_without_a_list_of_labels_is_left_alone(dataset):
    assert plugin.label_list(fo.Classification(label="cat")) is None
    assert plugin.corner(SimpleNamespace()) == (0, 0)


def test_register_adds_both_operators():
    registered = []
    plugin.register(SimpleNamespace(register=registered.append))
    assert registered == [plugin.DeleteLabels, plugin.RestoreLabels]
