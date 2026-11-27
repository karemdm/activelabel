"""The annotation-tool API: settings of the tools and the helpers of the session routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from api_client import make_client, upload
from fastapi.testclient import TestClient
from sample_images import make_image

from activelabel.annotate import registry, sessions
from activelabel.annotate.base import Pulled, ToolSettings
from activelabel.images.tags import load_tags, set_tags
from activelabel.labels import AnnotationSet, Label, annotations_lock, load_annotations, save_annotations
from activelabel.sync.links import label_hash
from activelabel.tasks import Task
from activelabel.web import annotate_api
from activelabel.web.errors import ApiError


@pytest.fixture(autouse=True)
def _no_real_keyring(fake_keyring) -> None:
    """No test of this file may reach the real system keyring."""


def _client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)[0]


def test_unknown_tool_is_a_404_with_a_code(tmp_path: Path) -> None:
    response = _client(tmp_path).delete("/api/annotation-tools/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "tool_unknown"


def test_list_tools_has_every_tool_without_secrets(tmp_path: Path) -> None:
    tools = _client(tmp_path).get("/api/annotation-tools").json()
    assert {t["tool"] for t in tools} == set(annotate_api.TOOLS)
    assert all("secret" not in t for t in tools)


def test_saving_a_tool_needs_a_url(tmp_path: Path) -> None:
    response = _client(tmp_path).put("/api/annotation-tools/cvat", json={"url": "", "username": "me"})
    assert response.status_code == 400
    assert response.json()["code"] == "url_required"


def test_forget_tool_answers_removed(tmp_path: Path) -> None:
    assert _client(tmp_path).delete("/api/annotation-tools/fiftyone").json() == {"removed": True}


def test_unconfigured_tool_is_refused(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    try:
        annotate_api._settings(state, "cvat")
    except ApiError as error:
        assert (error.status_code, error.code) == (400, "tool_not_configured")
    else:
        raise AssertionError("an unconfigured tool must be refused")


def test_local_hashes_follow_the_labels(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    project = state.store.create("Cats", "", "classification", ["cat"])
    project_dir = state.store.dir(project.id)
    with annotations_lock(project_dir):
        save_annotations(project_dir, AnnotationSet(project.task, ["cat"], {"a": [Label("cat")]}, {}))
    first = annotate_api._local_hashes(project_dir, ["a", "b"])
    assert set(first) == {"a", "b"} and first["a"] != first["b"]


def test_sending_to_fiftyone_without_it_installed_has_the_code_the_interface_translates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Light install (#218): "Send to FiftyOne" answered an English 400 without a code, so the Portuguese
    interface showed English; now it is the 501 ``fiftyone_not_installed`` of the other FiftyOne parts."""
    monkeypatch.setattr(registry, "FIFTYONE", False)
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["a"]})
    response = client.post("/api/projects/1/annotate", json={"tool": "fiftyone", "scope": "all"})
    assert response.status_code == 501
    assert response.json()["code"] == "fiftyone_not_installed" and "install" in response.json()["params"]


class TaggedTool:
    """A tool that gives back a fixed pull: a tagged label and the tags of two images."""

    def __init__(self, labels: dict[str, list[Label]], tags: dict[str, list[str]] | None) -> None:
        self.labels = labels
        self.tags = tags

    def pull(self, dataset, image_ids, project, settings, ref) -> Pulled:
        return Pulled(self.labels, set(image_ids), tags=self.tags)

    def applied(self, dataset, image_ids, project, labels) -> None:
        """Nothing to bring in line: there is no real tool."""


class Quiet:
    """A progress handle that reports nowhere."""

    def update(self, **kwargs) -> None:
        """Ignore the report."""


def _tagged_session(tmp_path: Path, tags: dict[str, list[str]] | None):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "detection", "classes": ["cat"]})
    paths = [make_image(tmp_path / f"in/{i}.png", seed=i) for i in range(2)]
    assert upload(client, 1, paths).json()["added"] == 2
    ids = [item["id"] for item in client.get("/api/projects/1/images").json()["items"]]
    folder = state.store.dir(1)
    session = sessions.create(folder, "fiftyone", "all", ids, [], {})
    labels = {ids[0]: [Label("cat", bbox=[0.1, 0.1, 0.3, 0.3], tags=["hard"])], ids[1]: []}
    return client, state, folder, ids, session["id"], TaggedTool(labels, None if tags is None else tags)


def test_pull_keeps_the_image_tags_in_the_session_and_apply_stores_them(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, None)
    tool.tags = {ids[0]: ["check", "excluded"], ids[1]: []}
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    monkeypatch.setattr(annotate_api, "_tool", lambda name: tool)
    project = state.project(1)
    result = annotate_api._pull_job(
        state, project, tool, ToolSettings("fiftyone"), sessions.get(folder, sid), sid, Quiet()
    )
    assert result["diff"]["tags"] == {"added": 2, "removed": 0}  # the image tag and the label tag
    assert "image_tags" not in result["session"]  # never shown by the API
    assert sessions.get(folder, sid)["image_tags"] == {ids[0]: ["check", "excluded"], ids[1]: []}
    assert load_tags(folder) == {}  # nothing is stored before the apply
    reply = client.post(f"/api/projects/1/annotate/{sid}/apply")
    assert reply.status_code == 200, reply.text
    assert reply.json()["applied"]["tags"] == {"added": 2, "removed": 0}
    assert load_tags(folder) == {ids[0]: ["check"]}
    assert load_annotations(folder).images[ids[0]][0].tags == ["hard"]


def test_a_pull_of_a_tool_without_image_tags_changes_no_stored_tags(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, None)
    set_tags(folder, {ids[1]: ["keep"]})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    monkeypatch.setattr(annotate_api, "_tool", lambda name: tool)
    annotate_api._pull_job(
        state, state.project(1), tool, ToolSettings("fiftyone"), sessions.get(folder, sid), sid, Quiet()
    )
    assert not sessions.get(folder, sid).get("image_tags")
    assert client.post(f"/api/projects/1/annotate/{sid}/apply").status_code == 200
    assert load_tags(folder) == {ids[1]: ["keep"]}


class MovedTool(TaggedTool):
    """A tool that says only the first image was edited, and gives the origin of the labels of both."""

    def __init__(
        self, labels: dict[str, list[Label]], origins: dict[str, list[str | None]], edited: str
    ) -> None:
        super().__init__(labels, None)
        self.origins = origins
        self.edited = edited

    def pull(self, dataset, image_ids, project, settings, ref) -> Pulled:
        return Pulled(self.labels, {self.edited}, origins=self.origins)


def test_origins_pair_a_moved_box_and_are_kept_only_for_the_edited_images(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, _ = _tagged_session(tmp_path, None)
    cat = Label("cat", bbox=[0.1, 0.1, 0.2, 0.2])
    other = Label("cat", bbox=[0.5, 0.5, 0.2, 0.2])
    save_annotations(folder, AnnotationSet(Task.DETECTION, ["cat"], {ids[0]: [cat], ids[1]: [other]}, {}))
    moved = Label("cat", bbox=[0.7, 0.7, 0.2, 0.2])  # far from where it was: no overlap with the old box
    tool = MovedTool(
        {ids[0]: [moved], ids[1]: [moved]},
        {ids[0]: [label_hash([cat])], ids[1]: [label_hash([other])]},
        edited=ids[0],
    )
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    monkeypatch.setattr(annotate_api, "_tool", lambda name: tool)
    result = annotate_api._pull_job(
        state, state.project(1), tool, ToolSettings("fiftyone"), sessions.get(folder, sid), sid, Quiet()
    )
    diff = result["diff"]
    assert (diff["added"], diff["removed"], diff["edited"], diff["images_changed"]) == (0, 0, 1, 1)
    assert "origins" not in result["session"]  # never shown by the API
    assert sessions.get(folder, sid)["origins"] == {
        ids[0]: [label_hash([cat])]
    }  # the other image was not edited
    reply = client.post(f"/api/projects/1/annotate/{sid}/apply")
    assert reply.status_code == 200, reply.text
    applied = reply.json()["applied"]
    assert (applied["added"], applied["removed"], applied["edited"]) == (0, 0, 1)
    saved = load_annotations(folder).images
    assert saved[ids[0]][0].bbox == moved.bbox and saved[ids[1]] == [other]


def _auto_pull(state, folder, sid, tool):
    return annotate_api._pull_job(
        state, state.project(1), tool, ToolSettings("fiftyone"), sessions.get(folder, sid), sid, Quiet(), True
    )


def test_a_pull_from_the_app_applies_at_once_and_the_apply_can_be_undone(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    result = _auto_pull(state, folder, sid, tool)
    assert result["applied"]["images_changed"] == 1 and "review" not in result
    assert result["session"]["status"] == "applied"
    assert load_annotations(folder).images[ids[0]][0].label == "cat"
    undone = client.post(f"/api/projects/1/annotate/{sid}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json()["session"]["status"] == "pulled"  # can be applied again, or discarded
    annotations = load_annotations(folder)
    assert not (annotations and annotations.images.get(ids[0]))  # the project is as before the apply
    again = client.post(f"/api/projects/1/annotate/{sid}/undo")
    assert again.status_code == 409 and again.json()["code"] == "undo_not_applied"
    assert client.post(f"/api/projects/1/annotate/{sid}/apply").status_code == 200


def test_an_apply_cannot_be_undone_after_the_annotations_changed(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    _auto_pull(state, folder, sid, tool)
    with annotations_lock(folder):
        current = load_annotations(folder)
        current.images[ids[1]] = [Label("cat", bbox=[0.5, 0.5, 0.2, 0.2])]
        save_annotations(folder, current)
    reply = client.post(f"/api/projects/1/annotate/{sid}/undo")
    assert reply.status_code == 409 and reply.json()["code"] == "undo_changed_since"


def test_a_project_that_asks_for_a_review_keeps_the_pull_for_the_review(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    assert client.get("/api/projects/1/annotate/prefs").json() == {"review_first": False}
    assert client.put("/api/projects/1/annotate/prefs", json={"review_first": True}).json() == {
        "review_first": True
    }
    result = _auto_pull(state, folder, sid, tool)
    assert result["review"] == "asked" and "applied" not in result
    assert sessions.get(folder, sid)["status"] == "pulled"


@pytest.mark.parametrize(
    ("summary", "why"),
    [
        ({"images_changed": 0}, "nothing_changed"),
        ({"images_changed": 1, "problems": [{"code": "box_invalid"}]}, "problems"),
        ({"images_changed": 1, "problems": [], "unknown_classes": ["dog"]}, "unknown_classes"),
        ({"images_changed": 2, "problems": [], "unknown_classes": []}, ""),
    ],
)
def test_what_needs_a_review(summary, why) -> None:
    assert annotate_api._needs_review(summary) == why


def test_an_apply_that_fails_leaves_the_edits_for_the_review(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)

    def changed(*args):
        raise ApiError(409, "annotations_changed", "changed")

    monkeypatch.setattr(annotate_api, "_apply_pulled", changed)
    result = _auto_pull(state, folder, sid, tool)
    assert result["review"] == "annotations_changed" and "applied" not in result


def test_a_session_whose_safety_copy_is_gone_cannot_be_undone(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    with annotations_lock(folder):  # the project has annotations before the apply: a safety copy is made
        save_annotations(folder, AnnotationSet(Task.DETECTION, ["cat"], {ids[1]: []}, {}))
    result = _auto_pull(state, folder, sid, tool)
    (folder / result["applied"]["backup"]).unlink()
    reply = client.post(f"/api/projects/1/annotate/{sid}/undo")
    assert reply.status_code == 409 and reply.json()["code"] == "undo_not_applied"


def test_an_applied_session_is_read_again_only_by_a_send_from_the_app(tmp_path, monkeypatch) -> None:
    client, state, folder, ids, sid, tool = _tagged_session(tmp_path, {})
    monkeypatch.setattr(annotate_api, "open_dataset", lambda project, state: None)
    _auto_pull(state, folder, sid, tool)
    monkeypatch.setattr(annotate_api, "_session_tool", lambda name: tool)
    plain = client.post(f"/api/projects/1/annotate/{sid}/pull")
    assert plain.status_code == 409 and plain.json()["code"] == "session_finished"
    again = client.post(f"/api/projects/1/annotate/{sid}/pull?apply=auto")
    assert again.status_code == 200, again.text  # the image was edited again after the send: read it again
