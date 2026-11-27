import logging
from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel import fiftyone_bridge, fiftyone_names
from activelabel.annotations import edit
from activelabel.images.tags import set_tags
from activelabel.labels import Label, load_annotations
from activelabel.sync import links
from activelabel.tasks import Task
from activelabel.web import labels_api

BOX = [0.1, 0.1, 0.3, 0.3]
POSE = [[0.2, 0.2, 2], [0.4, 0.4, 1]]
THREE = [[0.1, 0.1, 2], [0.2, 0.2, 2], [0.3, 0.3, 2]]
RING = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]]
REAL_REFRESH = labels_api.refresh_fiftyone  # the autouse fixture below replaces it in the tests


@pytest.fixture(autouse=True)
def refreshes(monkeypatch):
    """The FiftyOne refresh is recorded, never run (it must not load FiftyOne)."""
    calls = []
    monkeypatch.setattr(labels_api, "refresh_fiftyone", lambda *args: calls.append(args))
    return calls


def setup(tmp_path, task="detection", classes=("cat", "dog")):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": task, "classes": list(classes)})
    paths = [make_image(tmp_path / f"in/{i}.png", seed=i) for i in range(2)]
    assert upload(client, 1, paths).json()["added"] == 2
    ids = [item["id"] for item in client.get("/api/projects/1/images").json()["items"]]
    return client, state, ids


def url(image_id):
    return f"/api/projects/1/images/{image_id}/labels"


def save(client, image_id, labels, revision=None, **extra):
    if revision is None:
        revision = client.get(url(image_id)).json()["revision"]
    return client.put(url(image_id), json={"labels": labels, "revision": revision, **extra})


def refresh(folder, labels=None):
    REAL_REFRESH(folder, 1, "img", Task.DETECTION, labels or [])


def broken(folder, project_id):
    raise RuntimeError("database down")


def test_get_without_labels_and_with_labels(tmp_path):
    client, _, ids = setup(tmp_path)
    empty = client.get(url(ids[0]))
    assert empty.status_code == 200
    assert empty.json() == {
        "labels": [],
        "classes": ["cat", "dog"],
        "revision": edit.revision([]),
        "tags": [],
    }
    saved = save(client, ids[0], [{"label": "cat", "bbox": BOX}]).json()
    got = client.get(url(ids[0])).json()
    assert got["labels"] == saved["labels"] and got["revision"] == saved["revision"]
    assert got["labels"][0]["bbox"] == BOX
    assert client.get(url(ids[1])).json()["labels"] == []


def test_unknown_image_and_project(tmp_path):
    client, _, _ = setup(tmp_path)
    r = client.get(url("nope"))
    assert r.status_code == 404 and r.json()["code"] == "image_not_found"
    assert client.get("/api/projects/9/images/x/labels").status_code == 404
    assert save(client, "nope", [], revision="x").json()["code"] == "image_not_found"


def test_revision_changes_when_the_labels_change(tmp_path):
    client, _, ids = setup(tmp_path)
    first = client.get(url(ids[0])).json()["revision"]
    second = save(client, ids[0], [{"label": "cat", "bbox": BOX}]).json()["revision"]
    third = save(client, ids[0], [{"label": "cat", "bbox": [0.2, 0.1, 0.3, 0.3]}]).json()["revision"]
    assert len({first, second, third}) == 3


def test_put_detection_is_stored_and_returns_labels_revision_classes(tmp_path, refreshes):
    client, state, ids = setup(tmp_path)
    r = save(client, ids[0], [{"label": "CAT", "bbox": BOX, "track": 3, "score": 0.8, "model": "m"}])
    assert r.status_code == 200, r.text
    body = r.json()
    only = body["labels"][0]
    assert (only["label"], only["bbox"], only["track"]) == ("cat", BOX, 3)
    assert (only["score"], only["model"]) == (0.8, "m")
    assert body["classes"] == ["cat", "dog"]
    assert body["revision"] == client.get(url(ids[0])).json()["revision"]
    stored = load_annotations(state.store.dir(1)).images[ids[0]]
    assert stored[0].label == "cat" and stored[0].track == 3
    assert len(refreshes) == 1
    project_dir, project_id, image_id, task, labels = refreshes[0]
    assert (project_id, image_id, task) == (1, ids[0], Task.DETECTION) and labels[0].label == "cat"
    assert project_dir == state.store.dir(1)


def test_put_empty_list_keeps_the_image_listed(tmp_path):
    client, state, ids = setup(tmp_path)
    save(client, ids[0], [{"label": "cat", "bbox": BOX}])
    r = save(client, ids[0], [])
    assert r.status_code == 200 and r.json()["labels"] == []
    assert load_annotations(state.store.dir(1)).images[ids[0]] == []
    listed = client.get("/api/projects/1/images").json()["items"]
    assert {item["id"] for item in listed} == set(ids)


def test_put_segmentation(tmp_path):
    client, _, ids = setup(tmp_path, "segmentation")
    r = save(client, ids[0], [{"label": "cat", "polygons": RING}])
    assert r.status_code == 200 and r.json()["labels"][0]["polygons"] == RING


def test_put_keypoints(tmp_path):
    client, _, ids = setup(tmp_path, "keypoints")
    r = save(client, ids[0], [{"label": "cat", "bbox": BOX, "keypoints": POSE}])
    assert r.status_code == 200 and r.json()["labels"][0]["keypoints"] == POSE


def test_put_classification(tmp_path):
    client, _, ids = setup(tmp_path, "classification")
    r = save(client, ids[0], [{"label": "dog"}])
    assert r.status_code == 200 and r.json()["labels"][0]["label"] == "dog"
    two = save(client, ids[1], [{"label": "dog"}, {"label": "cat"}])
    assert two.status_code == 400 and two.json()["params"]["problem"] == "multiple_classes"


def test_stale_revision_is_a_conflict(tmp_path, refreshes):
    client, _, ids = setup(tmp_path)
    stale = client.get(url(ids[0])).json()["revision"]
    current = save(client, ids[0], [{"label": "cat", "bbox": BOX}]).json()["revision"]
    refreshes.clear()
    r = save(client, ids[0], [{"label": "dog", "bbox": BOX}], revision=stale)
    assert r.status_code == 409 and r.json()["code"] == "labels_changed"
    assert r.json()["params"]["revision"] == current
    assert refreshes == []


def test_new_class_is_a_conflict_unless_added(tmp_path):
    client, state, ids = setup(tmp_path)
    r = save(client, ids[0], [{"label": "bird", "bbox": BOX}, {"label": "Fish", "bbox": BOX}])
    assert r.status_code == 409 and r.json()["code"] == "classes_new"
    assert r.json()["params"] == {"classes": ["bird", "Fish"], "count": 2}
    ok = save(client, ids[0], [{"label": "bird", "bbox": BOX}], add_classes=True)
    assert ok.status_code == 200 and ok.json()["classes"] == ["cat", "dog", "bird"]
    assert state.store.get(1).classes == ["cat", "dog", "bird"]


@pytest.mark.parametrize(
    ("task", "labels", "problem"),
    [
        ("detection", [{"label": "cat", "bbox": [0.1, 0.1, 0.0, 0.3]}], "box_invalid"),
        ("detection", [{"label": "cat", "bbox": [0.8, 0.1, 0.4, 0.3]}], "outside_image"),
        (
            "segmentation",
            [{"label": "cat", "polygons": [[[0.1, 0.1], [0.5, 0.1]]]}],
            "polygon_too_few_points",
        ),
        ("keypoints", [{"label": "cat", "keypoints": [[0.2, 0.2, 3], [0.4, 0.4, 1]]}], "shape_invalid"),
        ("keypoints", [{"label": "cat", "keypoints": [[0.2, 0.2], [0.4, 0.4]]}], "shape_invalid"),
        ("detection", [{"label": "cat", "bbox": BOX, "polygons": RING}], "geometry_not_for_task"),
        ("detection", [{"label": " ", "bbox": BOX}], "shape_invalid"),
    ],
)
def test_invalid_labels_are_a_400(tmp_path, refreshes, task, labels, problem):
    client, _, ids = setup(tmp_path, task)
    stamp = client.get(url(ids[0])).json()["revision"]
    r = save(client, ids[0], labels, revision=stamp)
    assert r.status_code == 400 and r.json()["code"] == "labels_invalid"
    params = r.json()["params"]
    assert params["problem"] == problem and params["count"] == 1 and params["index"] == 1
    assert params["problems"] == [{"index": 0, "problem": problem}]
    assert client.get(url(ids[0])).json()["revision"] == stamp and refreshes == []


def test_keypoint_count_must_match_the_other_images(tmp_path):
    client, _, ids = setup(tmp_path, "keypoints")
    assert save(client, ids[0], [{"label": "cat", "keypoints": THREE}]).status_code == 200
    r = save(client, ids[1], [{"label": "cat", "keypoints": POSE}])
    assert r.status_code == 400 and r.json()["params"]["problem"] == "keypoints_count_mismatch"


def test_put_waits_for_a_running_activity(tmp_path):
    client, state, ids = setup(tmp_path)
    state.jobs.active = lambda project_id: True  # type: ignore[method-assign]
    r = save(client, ids[0], [{"label": "cat", "bbox": BOX}])
    assert r.status_code == 409 and r.json()["code"] == "busy"
    assert load_annotations(state.store.dir(1)) is None


def test_a_put_marks_the_image_in_a_linked_project(tmp_path):
    client, state, ids = setup(tmp_path)
    folder = state.store.dir(1)
    entry = {"key": "1:0", "base": links.label_hash([])}
    links.save_link(folder, "cvat", links.new_link("http://cvat", "7", "P", {ids[0]: entry}))
    body = save(client, ids[0], [{"label": "cat", "bbox": BOX}]).json()
    saved = [Label.from_dict(item) for item in body["labels"]]
    assert links.get_link(folder, "cvat")["items"][ids[0]]["edited"] == links.label_hash(saved)


def test_refresh_without_a_recorded_dataset_never_touches_fiftyone(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", lambda *args: called.append(args))
    refresh(tmp_path)
    assert called == [] and not (tmp_path / "fiftyone.json").exists()


def test_refresh_shows_the_labels_in_the_dataset(tmp_path, monkeypatch):
    fiftyone_names.record_name(tmp_path, "pets")
    dataset = SimpleNamespace(name="pets")
    shown = []
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", lambda folder, project_id: dataset)
    monkeypatch.setattr(fiftyone_bridge, "show_project_labels", lambda *args: shown.append(args))
    labels = [Label("cat", bbox=BOX)]
    refresh(tmp_path, labels)
    assert shown == [(dataset, "img", Task.DETECTION, labels)]


def test_refresh_skips_a_dataset_that_does_not_exist(tmp_path, monkeypatch):
    fiftyone_names.record_name(tmp_path, "pets")
    shown = []
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", lambda folder, project_id: None)
    monkeypatch.setattr(fiftyone_bridge, "show_project_labels", lambda *args: shown.append(args))
    refresh(tmp_path)
    assert shown == []


def test_refresh_swallows_an_error(tmp_path, monkeypatch, caplog):
    fiftyone_names.record_name(tmp_path, "pets")
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", broken)
    with caplog.at_level(logging.WARNING):
        refresh(tmp_path)
    assert "Could not show the edited labels" in caplog.text


def test_get_returns_the_image_tags_and_put_keeps_the_label_tags(tmp_path):
    client, state, ids = setup(tmp_path)
    set_tags(state.store.dir(1), {ids[0]: ["check"]})
    assert client.get(url(ids[0])).json()["tags"] == ["check"]
    assert client.get(url(ids[1])).json()["tags"] == []
    sent = [{"label": "cat", "bbox": BOX, "tags": ["hard"]}]
    saved = save(client, ids[0], sent)
    assert saved.status_code == 200, saved.text
    assert saved.json()["labels"][0]["tags"] == ["hard"]
    assert load_annotations(state.store.dir(1)).images[ids[0]][0].tags == ["hard"]
    assert client.get(url(ids[0])).json()["labels"][0]["tags"] == ["hard"]
    plain = save(client, ids[1], [{"label": "dog", "bbox": BOX}]).json()
    assert "tags" not in plain["labels"][0]
