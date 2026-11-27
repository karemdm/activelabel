import pytest
from ann_helpers import make_project

from activelabel.annotate import sessions
from activelabel.errors import CodedError
from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.tasks_store import TaskStore


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


def test_task_and_job_scopes_choose_the_active_images_of_that_task_or_job(proj):
    folder, manifest, ids = proj
    task = TaskStore(folder).create("T", 1, ids)  # three jobs of one image; the second image is excluded
    assert sessions.select_images(manifest, "task", task=task.id) == [ids[0], ids[2]]
    assert sessions.select_images(manifest, "task", task=task.id, job=3) == [ids[2]]
    for task_id, job, code in [
        (task.id, 2, "no_images_match"),
        (task.id, 9, "job_not_found"),
        (99, None, "task_not_found"),
    ]:
        with pytest.raises(CodedError) as error:
            sessions.select_images(manifest, "task", task=task_id, job=job)
        assert error.value.code == code
    with pytest.raises(CodedError, match="task_required|Choose a task"):
        sessions.select_images(manifest, "task")


def test_a_session_records_its_task_and_job_and_shows_them(proj):
    folder, _, ids = proj
    session = sessions.create(folder, "fiftyone", "task", ids[:1], [], {}, task=4, job=2)
    assert (session["task"], session["job"]) == (4, 2)
    assert sessions.public(session)["task"] == 4
    assert "task" not in sessions.create(folder, "fiftyone", "all", ids[:1], [], {})


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


def test_protected_images_are_those_of_open_fiftyone_sessions(proj):
    project_dir, _, ids = proj
    fo_open = sessions.create(project_dir, "fiftyone", "all", [ids[0]], [], {})
    applied = sessions.create(project_dir, "fiftyone", "all", [ids[2]], [], {})
    sessions.update(project_dir, applied["id"], status=sessions.APPLIED)
    sessions.create(project_dir, "cvat", "all", [ids[1]], [], {})
    assert sessions.protected_images(project_dir) == {ids[0]}
    sessions.update(project_dir, fo_open["id"], status=sessions.PULLED, base={ids[0]: "h"})
    assert sessions.protected_images(project_dir) == {ids[0]}
    assert "base" not in sessions.public(sessions.get(project_dir, fo_open["id"]))
    sessions.close(project_dir, fo_open["id"])
    assert sessions.protected_images(project_dir) == set()


def test_find_open_matches_the_tool_scope_and_exact_images(tmp_path):
    made = sessions.create(tmp_path, "fiftyone", "all", ["a", "b"], [], {})
    assert sessions.find_open(tmp_path, "fiftyone", "all", ["b", "a"])["id"] == made["id"]
    assert sessions.find_open(tmp_path, "fiftyone", "all", ["a"]) is None
    assert sessions.find_open(tmp_path, "fiftyone", "selected", ["a", "b"]) is None
    assert sessions.find_open(tmp_path, "fiftyone", "all", ["a", "b"], task=1) is None
    sessions.update(tmp_path, made["id"], status=sessions.PULLED)
    assert sessions.find_open(tmp_path, "fiftyone", "all", ["a", "b"]) is None


def test_public_hides_the_image_tags(proj):
    folder, _, ids = proj
    session = sessions.create(folder, "fiftyone", "all", ids[:1], [], {})
    updated = sessions.update(folder, session["id"], status="pulled", image_tags={ids[0]: ["check"]})
    assert updated["image_tags"] == {ids[0]: ["check"]}
    assert "image_tags" not in sessions.public(updated)
    assert "image_tags" not in sessions.public(sessions.get(folder, session["id"]))
