import pytest
from ann_helpers import make_project

from activelabel.annotate import sessions
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task


@pytest.fixture
def proj(tmp_path):
    store, project, manifest = make_project(tmp_path, ["a.png", "b.png", "c.png"], classes=["cat"])
    ids = [r.id for r in manifest.all()]
    manifest.update({ids[2]: {"issues": ["blurry"]}})
    manifest.set_status([ids[1]], "excluded", "x")
    save_annotations(
        store.dir(project.id),
        AnnotationSet(Task.DETECTION, ["cat"], {ids[0]: [Label("cat", bbox=[0, 0, 1, 1])]}, {}),
    )
    return store.dir(project.id), manifest, ids


def test_scopes(proj):
    _, manifest, ids = proj
    assert sessions.select_images(manifest, "all") == [ids[0], ids[2]]
    assert sessions.select_images(manifest, "unlabeled") == [ids[2]]
    assert sessions.select_images(manifest, "selected", [ids[1], ids[2], "zz"]) == [ids[2]]
    assert sessions.select_images(manifest, "issue", issue="blurry") == [ids[2]]


@pytest.mark.parametrize(
    ("scope", "ids", "issue"),
    [("selected", ["zz"], ""), ("issue", [], "dark"), ("issue", [], ""), ("nope", [], "")],
)
def test_empty_or_bad_selection(proj, scope, ids, issue):
    with pytest.raises(ValueError, match=r"\w"):
        sessions.select_images(proj[1], scope, ids, issue)


def test_lifecycle(proj):
    folder, _, ids = proj
    first = sessions.create(folder, "fiftyone", "all", ids[:1], [{"label": "x", "url": "u"}], {"view": "v"})
    second = sessions.create(folder, "cvat", "all", ids, [], {})
    assert first["status"] == "open" and len(first["id"]) == 8
    assert {s["id"] for s in sessions.list_sessions(folder)} == {first["id"], second["id"]}
    sessions.update(folder, first["id"], status="pulled")
    assert sessions.get(folder, first["id"])["status"] == "pulled"
    assert "image_ids" not in sessions.public(first)
    sessions.close(folder, first["id"])
    with pytest.raises(sessions.SessionNotFound):
        sessions.get(folder, first["id"])
    with pytest.raises(ValueError):
        sessions.create(folder, "cvat", "all", [], [], {})
