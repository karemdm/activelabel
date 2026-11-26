from types import SimpleNamespace

import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FAKE_EMBED, FakeEnv, wait_for

from activelabel import optional
from activelabel.curation import similarity, similarity_job
from activelabel.training.runner import TrainingRunner

FINISHED = ("done", "failed", "cancelled")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A project with 9 images and a runner whose embeddings worker is the fake one (no torch)."""
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(9)]
    assert upload(client, 1, paths).status_code == 200
    (tmp_path / "embed.py").write_text(FAKE_EMBED)
    runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: tmp_path / "train.py",
        poll=0.05,
    )
    state.runner = runner
    published = []
    monkeypatch.setattr(similarity_job, "_publish", lambda *args: published.append(args) or "")
    yield client, state, runner, published
    runner.shutdown()


def finished(state):
    return lambda: (
        (s := similarity.SimilarityStore(state.store.dir(1)).read()).get("status") in FINISHED and s
    )


def test_the_map_is_computed_in_the_queue_and_saved(setup):
    client, state, runner, published = setup
    queued = similarity_job.start(runner, state.project(1), state.store.dir(1))
    assert queued["status"] == "queued" and queued["job"]
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    assert sorted(g["size"] for g in done["groups"]) == [3, 3, 3]  # the fake worker makes three groups
    assert done["backbone"] == similarity_job.BACKBONE and done["unreadable"] == []
    assert runner.environment.ensured == [list(similarity_job.PACKAGES)]
    ids, vectors, _ = similarity.SimilarityStore(state.store.dir(1)).load_vectors()
    assert len(ids) == 9 and vectors.shape == (9, 8)
    assert len(published) == 1
    assert list(runner.tmp_root.iterdir()) == []  # the temporary folder is removed
    active = {r.id for r in state.manifest(1).active()}
    assert similarity_job.state(runner, state.store.dir(1), active)["outdated"] is False


def test_unreadable_images_are_listed_and_not_outdated(setup):
    client, state, runner, _ = setup
    similarity_job.start(runner, state.project(1), state.store.dir(1), device="skip")
    done = wait_for(finished(state))
    assert len(done["unreadable"]) == 1 and len(done["points"]["ids"]) == 8
    active = {r.id for r in state.manifest(1).active()}
    assert similarity_job.state(runner, state.store.dir(1), active)["outdated"] is False


def test_a_worker_error_is_kept_and_the_previous_map_stays(setup):
    client, state, runner, _ = setup
    similarity_job.start(runner, state.project(1), state.store.dir(1))
    wait_for(finished(state))
    similarity_job.start(runner, state.project(1), state.store.dir(1), device="fail")
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "boom" in failed["error"]
    assert len(failed["points"]["ids"]) == 9  # the last good map is still shown


def test_cancel_stops_the_worker(setup):
    client, state, runner, _ = setup
    similarity_job.start(runner, state.project(1), state.store.dir(1), device="slow")
    wait_for(lambda: similarity.SimilarityStore(state.store.dir(1)).read().get("phase") == "embedding")
    similarity_job.cancel(runner, 1, state.store.dir(1))
    assert wait_for(finished(state))["status"] == "cancelled"


def test_cancel_while_queued_never_starts(setup):
    client, state, runner, _ = setup
    gate: list[bool] = []
    blocker = runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    similarity_job.start(runner, state.project(1), state.store.dir(1))
    assert similarity_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert similarity.SimilarityStore(state.store.dir(1)).read()["status"] == "cancelled"
    assert similarity_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"  # nothing to cancel


def test_a_computation_left_by_a_previous_start_is_interrupted(setup):
    client, state, runner, _ = setup
    similarity.SimilarityStore(state.store.dir(1)).update(status="running", job="gone")
    shown = similarity_job.state(runner, state.store.dir(1), set())
    assert shown["status"] == "failed" and shown["error"] == "interrupted"
    assert similarity_job.state(runner, state.store.dir(2), set()) == {}


def test_no_image_fails_with_a_message(setup, tmp_path):
    client, state, runner, _ = setup
    client.post("/api/projects", json={"name": "Empty", "task": "classification", "classes": ["a"]})
    similarity_job.start(runner, state.project(2), state.store.dir(2))
    store = similarity.SimilarityStore(state.store.dir(2))
    done = wait_for(lambda: store.read().get("status") in FINISHED and store.read())
    assert done["status"] == "failed" and "no image" in done["error"]


def database_down(*args):
    raise RuntimeError("database down")


def test_publish_reports_a_fiftyone_failure_without_losing_the_map(monkeypatch, tmp_path):
    monkeypatch.setattr(similarity_job.fiftyone_bridge, "sync_dataset", database_down)
    assert similarity_job._publish(SimpleNamespace(id=1), None, tmp_path, {}) == "database down"


def test_images_to_embed_skips_missing_files(setup):
    client, state, runner, _ = setup
    manifest = state.manifest(1)
    first = manifest.active()[0]
    manifest.file_path(first).unlink()
    items = similarity_job.images_to_embed(manifest)
    assert len(items) == 8 and first.id not in {i["id"] for i in items}


# -- the global queue (#604): lane of the device and its cancel button --------------------------


def test_the_map_waits_in_the_lane_of_its_device(setup):
    client, state, runner, _ = setup
    data = similarity_job.start(runner, state.project(1), state.store.dir(1), device="slow")
    job = runner.tool_job(data["job"])
    assert job is not None and job.kind == similarity_job.KIND and job.device == "slow"
    similarity_job.cancel(runner, 1, state.store.dir(1))  # the fake "slow" device never ends by itself
    wait_for(finished(state))


def test_the_queue_cancel_button_kills_the_running_worker(setup):
    client, state, runner, _ = setup
    data = similarity_job.start(runner, state.project(1), state.store.dir(1), device="slow")
    wait_for(lambda: similarity.SimilarityStore(state.store.dir(1)).read().get("phase") == "embedding")
    assert runner.cancel_job(data["job"]) is True
    assert wait_for(finished(state))["status"] == "cancelled"


def test_the_queue_cancel_button_on_a_waiting_map(setup):
    client, state, runner, _ = setup
    gate: list[bool] = []
    blocker = runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    data = similarity_job.start(runner, state.project(1), state.store.dir(1))
    assert runner.cancel_job(data["job"]) is True
    assert similarity.SimilarityStore(state.store.dir(1)).read()["status"] == "cancelled"
    assert runner.tool_job(data["job"]).status == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")


def test_publish_without_fiftyone_gives_a_readable_message(monkeypatch, tmp_path):
    monkeypatch.setattr(similarity_job.optional, "FIFTYONE", False)
    monkeypatch.setattr(similarity_job.fiftyone_bridge, "sync_dataset", database_down)  # must not be called
    message = similarity_job._publish(SimpleNamespace(id=1), None, tmp_path, {})
    assert "FiftyOne is not installed" in message and optional.FIFTYONE_INSTALL in message


def test_an_image_without_its_file_is_listed_so_the_new_map_is_not_outdated(setup):
    """Found in a real run: an image whose file was deleted was left out silently, so a map just computed
    said "the images changed since this map"; it is now listed with the unreadable ones."""
    client, state, runner, _ = setup
    manifest = state.manifest(1)
    first = manifest.active()[0]
    manifest.file_path(first).unlink()
    similarity_job.start(runner, state.project(1), state.store.dir(1))
    done = wait_for(finished(state))
    assert done["status"] == "done" and done["unreadable"] == [first.id]
    active = {r.id for r in state.manifest(1).active()}
    assert similarity_job.state(runner, state.store.dir(1), active)["outdated"] is False
