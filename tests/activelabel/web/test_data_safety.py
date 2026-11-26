"""Malformed or hostile input never damages data or answers 500 (found by a QA pass)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from api_client import make_client, upload
from make_video import make_video
from sample_images import make_image

from activelabel.curation import issues
from activelabel.images.manifest import Manifest
from activelabel.images.video import VideoInfo, extract_frames, resolve_selection
from activelabel.tasks_store import TaskStore


def _project(client, name, task="detection", classes=("cat",)):
    return client.post("/api/projects", json={"name": name, "task": task, "classes": list(classes)}).json()[
        "id"
    ]


def test_delete_ignores_ids_of_other_projects(tmp_path):
    client, state = make_client(tmp_path)
    one, two = _project(client, "A"), _project(client, "B")
    upload(client, two, [make_image(tmp_path / "x.jpg", seed=3, fmt="JPEG")])
    victim = Manifest(state.store.dir(two)).all()[0]
    thumb = state.store.dir(two) / "thumbs" / f"{victim.id}.jpg"
    assert client.get(f"/api/projects/{two}/images/{victim.id}/thumb").status_code == 200
    assert thumb.exists()
    sneaky = [f"../../{two}/thumbs/{victim.id}", f"../../{two}/images/{victim.id}"]
    response = client.post(f"/api/projects/{one}/images/delete", json={"ids": sneaky, "confirm": True})
    assert response.json() == {"deleted": 0}
    assert thumb.exists() and Manifest(state.store.dir(two)).file_path(victim).exists()
    too_long = client.post(f"/api/projects/{one}/images/exclude", json={"ids": [], "reason": "x" * 501})
    assert too_long.status_code == 422


@pytest.mark.parametrize(
    "body",
    ['{"min_side": NaN}', '{"min_side": -5}', '{"phash_distance": 99.5}', '{"blur_relative": Infinity}'],
)
def test_bad_thresholds_are_refused(tmp_path, body):
    client, _ = make_client(tmp_path)
    pid = _project(client, "A")
    url = f"/api/projects/{pid}/curation/thresholds"
    response = client.put(url, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400, response.text
    assert client.get(f"/api/projects/{pid}/curation").status_code == 200
    assert client.put(url, json={"min_side": 100}).json()["thresholds"]["min_side"] == 100


def test_saved_bad_thresholds_fall_back_to_defaults(tmp_path):
    (tmp_path / "curation.json").write_text('{"thresholds": {"min_side": NaN, "dark": 50, "x": 1}}')
    loaded = issues.load_thresholds(tmp_path)
    assert loaded.min_side == issues.Thresholds().min_side and loaded.dark == 50.0


def test_reset_makes_duplicates_not_analysed(tmp_path):
    client, state = make_client(tmp_path)
    pid = _project(client, "A")
    upload(client, pid, [make_image(tmp_path / f"{k}.png", seed=k, fmt="PNG") for k in range(2)])
    path = issues.curation_path(state.store.dir(pid))
    path.write_text(json.dumps({"duplicate_of": {}, "thresholds": {"dark": 50}}))
    assert client.delete(f"/api/projects/{pid}/curation").status_code == 200
    assert json.loads(path.read_text()) == {"format_version": 1, "thresholds": {"dark": 50}}


def test_seconds_between_frames_must_be_reasonable():
    info = VideoInfo("x", "v.mp4", 30.0, 300, 10.0, 64, 48, 1)
    for bad in (1e308, float("inf"), float("nan"), 0, -1):
        with pytest.raises(ValueError):
            resolve_selection(info, every_seconds=bad)
    assert resolve_selection(info, every_seconds=1) == (0, 299, 30)


def test_frames_are_saved_on_a_path_with_accents(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=6)
    target = tmp_path / "pasta ção açaí"
    target.mkdir()
    frames = extract_frames(video, 0, 5, 2, target)
    assert [i for i, _ in frames] == [0, 2, 4] and all(p.stat().st_size > 0 for _, p in frames)


@pytest.mark.parametrize(
    "files",
    [
        {"a.txt": "nan 0.5 0.5 0.2 0.2\n", "classes.txt": "cat\n"},
        {"a.txt": "inf 0.5 0.5 0.2 0.2\n", "classes.txt": "cat\n"},
        {"a.txt": "0 0.5 0.5 0.2 0.2 0.1 0.1\n", "data.yaml": "names: [cat]\nkpt_shape: [17, 0]\n"},
        {
            "a.xml": "<annotation><filename>a.jpg</filename><size><width>nan</width>"
            "<height>abc</height></size>"
            "<object><name>cat</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
            "</object></annotation>"
        },
        {
            "c.json": '{"images": [{"id": 1, "file_name": "a.jpg", "width": 1e999, "height": 10}],'
            ' "annotations": [], "categories": [{"id": 1, "name": "cat"}]}'
        },
        {"gt.txt": "inf,1,1,1,1,1,1,1\n"},
    ],
)
def test_malformed_numbers_in_annotation_files_are_not_a_500(tmp_path, files):
    client, _ = make_client(tmp_path)
    pid = _project(client, "A")
    upload(client, pid, [make_image(tmp_path / "a.jpg", seed=1, fmt="JPEG")])
    payload = [("files", (name, text.encode())) for name, text in files.items()]
    response = client.post(f"/api/projects/{pid}/annotations", files=payload)
    assert response.status_code in (200, 400), response.text


def test_applying_an_unknown_staging_is_404(tmp_path):
    client, _ = make_client(tmp_path)
    pid = _project(client, "A")
    response = client.post(
        f"/api/projects/{pid}/annotations/0123456789abcdef/apply", json={"action": "import"}
    )
    assert response.status_code == 404


def test_a_damaged_zip_entry_is_rejected_not_a_500(tmp_path):
    client, _ = make_client(tmp_path)
    pid = _project(client, "A")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("good.png", make_image(tmp_path / "g.png", seed=2, fmt="PNG").read_bytes())
        archive.writestr("bad.png", make_image(tmp_path / "b.png", seed=4, fmt="PNG").read_bytes() * 3)
    data = bytearray(buffer.getvalue())
    info = zipfile.ZipFile(io.BytesIO(bytes(data))).getinfo("bad.png")
    start = info.header_offset + 30 + len(info.filename) + len(info.extra)
    for k in range(start + 20, start + 60):  # scramble the compressed bytes of one entry
        data[k] ^= 0x5A
    response = client.post(f"/api/projects/{pid}/images", files=[("files", ("pics.zip", bytes(data)))])
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["added"] == 1 and [r["reason"] for r in report["rejected"]] == ["corrupt"]


def test_deleting_a_task_while_training_changes_nothing(tmp_path):
    client, state = make_client(tmp_path)
    pid = _project(client, "A")
    for k in range(2):
        upload(client, pid, [make_image(tmp_path / f"{k}.png", seed=10 + k, fmt="PNG")])
    tasks = TaskStore(state.store.dir(pid))
    before = [(t.id, t.images) for t in tasks.load()]
    state.runner.has_active = lambda project_id: True  # a training run is going
    response = client.request("DELETE", f"/api/projects/{pid}/tasks/1", json={"confirm": True})
    assert response.status_code == 409
    assert [(t.id, t.images) for t in tasks.load()] == before
    state.runner.has_active = lambda project_id: False
    assert client.request("DELETE", f"/api/projects/{pid}/tasks/1", json={"confirm": True}).status_code == 200
    upload(client, pid, [make_image(tmp_path / "n.png", seed=30, fmt="PNG")])
    assert [t.id for t in tasks.load()] == [2, 3]  # no task id used up by the delete


def test_manifest_delete_saves_even_if_a_file_is_locked(tmp_path, monkeypatch):
    client, state = make_client(tmp_path)
    pid = _project(client, "A")
    upload(client, pid, [make_image(tmp_path / f"{k}.png", seed=40 + k, fmt="PNG") for k in range(2)])
    manifest = Manifest(state.store.dir(pid))
    ids = [r.id for r in manifest.all()]
    real_unlink = type(tmp_path).unlink

    def locked(self, missing_ok=False):
        if self.name.startswith(ids[0]):
            raise PermissionError("in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(type(tmp_path), "unlink", locked)
    assert manifest.delete(ids) == 2
    assert manifest.all() == []
