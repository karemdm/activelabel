from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task

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
