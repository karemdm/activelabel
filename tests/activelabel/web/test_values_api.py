"""Editing the values of one image of a regression project (#619)."""

import pytest
from api_client import make_client, upload
from sample_images import make_image

from activelabel.labels import load_annotations


@pytest.fixture
def scene(tmp_path):
    client, state = make_client(tmp_path)
    body = {"name": "R", "task": "regression", "classes": ["age", "height"]}
    assert client.post("/api/projects", json=body).status_code == 200
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(2)]
    assert upload(client, 1, paths).status_code == 200
    return client, state, [r.id for r in state.manifest(1).all()]


def put(client, image_id, values, project=1):
    return client.put(f"/api/projects/{project}/images/{image_id}/values", json={"values": values})


def test_a_value_is_set_changed_and_cleared(scene):
    client, state, ids = scene
    assert put(client, ids[0], {"height": 1.7, "age": 31}).json()["values"] == {"age": 31.0, "height": 1.7}
    assert put(client, ids[0], {"age": 32.5}).json()["values"] == {"age": 32.5, "height": 1.7}  # others stay
    assert put(client, ids[0], {"height": None}).json()["values"] == {"age": 32.5}
    saved = load_annotations(state.store.dir(1))
    assert saved is not None and saved.classes == ["age", "height"]
    assert [(lb.label, lb.value) for lb in saved.images[ids[0]]] == [("age", 32.5)]
    listed = client.get("/api/projects/1/images").json()["items"]
    assert {i["id"]: i["values"] for i in listed} == {ids[0]: {"age": 32.5}, ids[1]: {}}


def test_a_bad_request_is_refused_with_a_code(scene):
    client, _, ids = scene
    assert put(client, ids[0], {"weight": 3}).json()["code"] == "target_unknown"
    assert put(client, "nope", {"age": 3}).json()["code"] == "image_not_found"
    assert put(client, ids[0], {"age": "x"}).status_code == 422
    assert client.put(
        f"/api/projects/1/images/{ids[0]}/values",
        content='{"values": {"age": NaN}}',
        headers={"content-type": "application/json"},
    ).status_code in (400, 422)


def test_only_a_regression_project_has_values(tmp_path):
    client, state = make_client(tmp_path)
    assert (
        client.post(
            "/api/projects", json={"name": "C", "task": "classification", "classes": ["a"]}
        ).status_code
        == 200
    )
    assert upload(client, 1, [make_image(tmp_path / "s" / "a.png", seed=1, fmt="PNG")]).status_code == 200
    image_id = state.manifest(1).all()[0].id
    reply = put(client, image_id, {"a": 1})
    assert reply.status_code == 400 and reply.json()["code"] == "values_not_regression"
