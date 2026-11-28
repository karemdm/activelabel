import io
import tempfile
import zipfile

from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training.config import RunConfig
from activelabel.training.runs import RunStore

BASE = "/api/projects/1/datasets"


def _setup(tmp_path, n=30):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a", "b"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(n)]
    assert upload(client, 1, paths).status_code == 200
    ids = [r.id for r in state.manifest(1).all()]
    images = {i: [Label("a" if k % 3 else "b", bbox=[0, 0, 0.5, 0.5])] for k, i in enumerate(ids)}
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["a", "b"], images, {}))
    return client, state, ids


def _listed(client, **params):
    response = client.get("/api/projects/1/images", params={"limit": 200, **params})
    assert response.status_code == 200, response.text
    return [i["id"] for i in response.json()["items"]]


def test_preview_writes_nothing_and_create_returns_summary(tmp_path):
    client, state, _ = _setup(tmp_path)
    preview = client.post(BASE + "/preview", json={"seed": 1})
    assert preview.status_code == 200
    body = preview.json()
    assert body["sizes"] == {"train": 21, "val": 6, "test": 3} and body["type"] == "detection"
    assert set(body["stats"]) == {"train", "val", "test"} and isinstance(body["warnings"], list)
    assert not (state.store.dir(1) / "datasets").exists()
    assert client.get(BASE).json() == {"versions": []}

    created = client.post(BASE, json={"seed": 1, "note": "first"})
    assert created.status_code == 200
    summary = created.json()
    assert summary["version"] == 1 and summary["note"] == "first" and summary["incomplete"] is False
    assert summary["sizes"] == body["sizes"] and summary["options"]["seed"] == 1
    client.post(BASE, json={"seed": 2})
    assert [v["version"] for v in client.get(BASE).json()["versions"]] == [2, 1]


def test_get_without_labels_and_not_found(tmp_path):
    client, _, _ = _setup(tmp_path)
    client.post(BASE, json={})
    one = client.get(f"{BASE}/1").json()
    assert "labels" not in one and one["version"] == 1 and set(one["parts"]) == {"train", "val", "test"}
    assert client.get(f"{BASE}/9").status_code == 404
    assert client.get("/api/projects/9/datasets").status_code == 404
    assert client.post("/api/projects/9/datasets/preview", json={}).status_code == 404


def test_validation_errors_are_readable(tmp_path):
    client, _, _ = _setup(tmp_path)
    for body in (
        {"split": {"train": 50, "val": 20, "test": 10}},
        {"kfold": 1},
        {"tasks": [9]},
        {"split": {"train": 0, "val": 50, "test": 50}},
    ):
        for url in (BASE + "/preview", BASE):
            response = client.post(url, json=body)
            assert response.status_code == 400 and response.json()["detail"], body
    assert client.get(BASE).json() == {"versions": []}
    client.post("/api/projects", json={"name": "Empty", "task": "detection"})
    empty = client.post("/api/projects/2/datasets/preview", json={})
    assert empty.status_code == 400 and "no images" in empty.json()["detail"]


def test_delete_needs_confirm_and_numbers_are_not_reused(tmp_path):
    client, _, _ = _setup(tmp_path)
    client.post(BASE, json={})
    client.post(BASE, json={})
    assert client.request("DELETE", f"{BASE}/2", json={}).status_code == 400
    assert client.delete(f"{BASE}/2").status_code == 400
    assert client.request("DELETE", f"{BASE}/2", json={"confirm": True}).json() == {"deleted": True}
    assert client.request("DELETE", f"{BASE}/2", json={"confirm": True}).status_code == 404
    assert client.post(BASE, json={}).json()["version"] == 3


def test_incomplete_after_image_deleted(tmp_path):
    client, _, ids = _setup(tmp_path)
    client.post(BASE, json={})
    client.post("/api/projects/1/images/delete", json={"ids": [ids[0]], "confirm": True})
    assert client.get(f"{BASE}/1").json()["incomplete"] is True
    assert client.get(BASE).json()["versions"][0]["incomplete"] is True


def test_stale_tells_that_labels_changed_after_the_version(tmp_path):
    client, state, ids = _setup(tmp_path)
    client.post(BASE, json={})
    fresh = {"changed_images": 0, "added": 0, "removed": 0, "excluded": 0, "restored": 0}
    assert client.get(f"{BASE}/1").json()["stale"] == fresh
    images = {i: [Label("a", bbox=[0, 0, 0.5, 0.5])] for i in ids}
    images[ids[0]] = [Label("b", bbox=[0, 0, 1, 1])]
    del images[ids[1]]
    save_annotations(state.store.dir(1), AnnotationSet(Task.DETECTION, ["a", "b"], images, {}))
    changed = client.get(BASE).json()["versions"][0]["stale"]
    assert changed["removed"] == 1 and changed["added"] == 0 and changed["changed_images"] >= 1
    assert client.get(f"{BASE}/1").json()["stale"] == changed
    assert client.post(BASE, json={}).json()["stale"] == fresh


def test_images_filters_by_part(tmp_path):
    client, _, ids = _setup(tmp_path)
    client.post(BASE, json={})
    parts = {p: _listed(client, dataset=1, part=p) for p in ("train", "val", "test")}
    assert [len(v) for v in parts.values()] == [21, 6, 3]
    assert sorted(i for v in parts.values() for i in v) == sorted(ids)
    assert sorted(_listed(client, dataset=1)) == sorted(ids)
    page = client.get("/api/projects/1/images", params={"dataset": 1, "part": "val", "limit": 2}).json()
    assert page["total"] == 6 and len(page["items"]) == 2
    # existing filters keep working, and combine
    assert len(_listed(client, task=1)) == 30
    assert sorted(_listed(client, task=1, dataset=1, part="test")) == sorted(parts["test"])
    assert client.get("/api/projects/1/images", params={"dataset": 9}).status_code == 404
    assert client.get("/api/projects/1/images", params={"dataset": 1, "part": "x"}).status_code == 400
    assert client.get("/api/projects/1/images", params={"dataset": 1, "fold": 0}).status_code == 400
    assert client.get("/api/projects/1/images", params={"part": "train"}).status_code == 400
    assert client.get("/api/projects/1/images", params={"fold": 0}).status_code == 400


def test_images_filters_by_fold(tmp_path):
    client, _, ids = _setup(tmp_path)
    created = client.post(BASE, json={"kfold": 3, "split": {"train": 0, "val": 0, "test": 10}})
    assert created.json()["sizes"]["folds"] == [9, 9, 9] and created.json()["sizes"]["test"] == 3
    val = _listed(client, dataset=1, part="val", fold=1)
    train = _listed(client, dataset=1, part="train", fold=1)
    test = _listed(client, dataset=1, part="test")
    assert (len(val), len(train), len(test)) == (9, 18, 3)
    assert not set(val) & set(train) and not (set(val) | set(train)) & set(test)
    assert _listed(client, dataset=1, fold=1) == val
    assert sorted(val + train + test) == sorted(ids)
    for params in ({"part": "train"}, {"part": "val", "fold": 3}, {"part": "val", "fold": -1}):
        response = client.get("/api/projects/1/images", params={"dataset": 1, **params})
        assert response.status_code == 400, params


def test_video_options_are_passed_and_reported(tmp_path):
    client, state, ids = _setup(tmp_path, n=40)
    changes = {i: {"video": "v1", "frame": k, "time": k / 2} for k, i in enumerate(ids[:36])}
    state.manifest(1).update(changes)
    body = {"seed": 1, "videos": "auto", "gap_seconds": 0.5}
    preview = client.post(BASE + "/preview", json=body).json()
    assert preview["options"]["videos"] == "auto" and preview["options"]["gap_seconds"] == 0.5
    (video,) = preview["videos"]
    assert video["video"] == "v1" and video["frames"] == 36 and video["split"] is True
    assert preview["sizes"]["val"] > 0 and preview["sizes"]["test"] > 0
    assert any(w["code"] == "video_split" for w in preview["warnings"])
    whole = client.post(BASE + "/preview", json={**body, "videos": "whole"}).json()
    assert whole["options"]["videos"] == "whole" and whole["videos"][0]["split"] is False
    created = client.post(BASE, json={**body, "note": "x"}).json()
    assert created["videos"] == preview["videos"] and created["options"]["gap_seconds"] == 0.5
    assert client.get(f"{BASE}/1").json()["videos"] == preview["videos"]
    for bad in ({"videos": "x"}, {"gap_seconds": -1}):
        assert client.post(BASE + "/preview", json=bad).status_code == 400
    default = client.post(BASE + "/preview", json={}).json()["options"]
    assert default["videos"] == "auto" and default["gap_seconds"] == 0.5


def test_empty_training_or_validation_part_is_refused_unless_allowed(tmp_path):
    client, _, _ = _setup(tmp_path, n=2)
    preview = client.post(BASE + "/preview", json={"seed": 1}).json()
    assert "val" in preview["empty_parts"] and preview["groups"] == 2
    refused = client.post(BASE, json={"seed": 1})
    assert refused.status_code == 409 and "validation" in refused.json()["detail"]
    assert client.get(BASE).json() == {"versions": []}
    allowed = client.post(BASE, json={"seed": 1, "allow_empty": True})
    assert allowed.status_code == 200 and allowed.json()["sizes"]["val"] == 0
    asked_for_none = client.post(BASE, json={"seed": 1, "split": {"train": 100, "val": 0, "test": 0}})
    assert asked_for_none.status_code == 200  # nothing was asked for the empty parts


def test_preview_tells_the_largest_kfold(tmp_path):
    client, _, _ = _setup(tmp_path, n=30)
    body = client.post(BASE + "/preview", json={"seed": 1}).json()
    assert body["groups"] == 30 and body["max_kfold"] == 10 and body["empty_parts"] == []
    few, _, _ = _setup(tmp_path / "few", n=4)[0], None, None
    small = few.post(BASE + "/preview", json={"seed": 1}).json()
    assert small["max_kfold"] == 2  # two classes of 2 images: the test part takes one of each (#347)
    assert few.post(BASE + "/preview", json={"kfold": 5}).status_code == 400


def test_runs_in_the_messages_are_named_like_on_the_screens(tmp_path):
    """#548: the screens call a run ``#1``, so the error text does too (#653: through the API, not the private
    helper)."""
    client, state, _ = _setup(tmp_path)
    assert client.post(BASE, json={"seed": 1}).status_code == 200
    runs = RunStore(state.store.dir(1))
    for _ in range(2):
        runs.create(RunConfig(dataset=1, model="resnet18", params={"epochs": 2}, metric="accuracy"))
    reply = client.request("DELETE", BASE + "/1", json={"confirm": True})
    assert reply.status_code == 409 and reply.json()["code"] == "dataset_in_use"
    assert "Training run(s) #1, #2 use this version" in reply.json()["detail"]
    assert reply.json()["params"]["runs"] == [1, 2]


def test_export_downloads_a_zip_of_the_version(tmp_path):
    client, _, _ = _setup(tmp_path)
    client.post(BASE, json={"seed": 1})
    check = client.get(f"{BASE}/1/export/check", params={"format": "yolo"})
    assert check.status_code == 200
    assert check.json()["groups"] == {"train": 21, "val": 6, "test": 3} and check.json()["images"] == 30

    response = client.get(f"{BASE}/1/export", params={"format": "yolo"})
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    assert 'filename="P-v1-yolo.zip"' in response.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert "data.yaml" in names and sum(n.startswith("images/") for n in names) == 30
    assert sum(n.startswith("labels/") for n in names) == 30

    labels_only = client.get(f"{BASE}/1/export", params={"format": "coco", "images": "false"})
    names = zipfile.ZipFile(io.BytesIO(labels_only.content)).namelist()
    assert sorted(names) == [f"annotations/instances_{p}.json" for p in ("test", "train", "val")]


def test_export_refusals_are_coded_and_a_failed_export_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    client, _, _ = _setup(tmp_path)
    client.post(BASE, json={})
    for url in (f"{BASE}/1/export", f"{BASE}/1/export/check"):
        bad = client.get(url, params={"format": "voc"})
        assert bad.status_code == 400 and bad.json()["code"] == "export_format_invalid"
        assert client.get(url.replace("/1/", "/9/"), params={"format": "yolo"}).status_code == 404
    assert client.get("/api/projects/9/datasets/1/export", params={"format": "yolo"}).status_code == 404
    assert list((tmp_path / "tmp").iterdir()) == []  # the download leaves nothing behind either
    client.get(f"{BASE}/1/export", params={"format": "yolo"})
    assert list((tmp_path / "tmp").iterdir()) == []
