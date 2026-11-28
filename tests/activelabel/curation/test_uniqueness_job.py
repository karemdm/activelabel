import uuid
from types import SimpleNamespace

import fiftyone as fo
import numpy as np
import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FakeEnv, wait_for

from activelabel import fiftyone_bridge
from activelabel.curation import fiftyone_runs, similarity, uniqueness, uniqueness_job
from activelabel.errors import CodedError
from activelabel.training.runner import TrainingRunner

FINISHED = ("done", "failed", "cancelled")


def slow_scores(vectors, check):
    """Scores that never end by themselves: only a cancel stops them."""
    while True:
        check(0, len(vectors))
        wait_for(lambda: True)


def broken_scores(vectors, check):
    raise ZeroDivisionError("division by zero")


def database_down(*args):
    raise RuntimeError("database down")


@pytest.fixture
def setup(tmp_path):
    """A project with 6 images and a runner without a worker (the job runs in its thread)."""
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(6)]
    assert upload(client, 1, paths).status_code == 200
    runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: tmp_path / "train.py",
        poll=0.05,
    )
    state.runner = runner
    yield client, state, runner
    runner.shutdown()


@pytest.fixture
def published(monkeypatch):
    """Replaces the FiftyOne publish by a recorder (the calls are listed)."""
    calls = []
    monkeypatch.setattr(uniqueness_job, "_publish", lambda *args: calls.append(args) or "")
    return calls


def save_vectors(state, count=None):
    """Save fake embeddings for the first ``count`` active images (all by default)."""
    ids = [r.id for r in state.manifest(1).active()][:count]
    vectors = np.random.default_rng(1).normal(size=(len(ids), 5))
    similarity.SimilarityStore(state.store.dir(1)).save_vectors(ids, vectors, "fake")
    return ids


def finished(state):
    return lambda: (
        (s := uniqueness.UniquenessStore(state.store.dir(1)).read()).get("status") in FINISHED and s
    )


def block(runner):
    """Occupy the CPU lane (the job's own: ``auto`` would be mps on a Mac) until the gate gets an item."""
    gate: list[bool] = []
    blocker = runner.enqueue_tool(
        1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30), device="cpu"
    )
    return gate, blocker


def test_the_scores_cover_the_active_images_only(setup, published):
    client, state, runner = setup
    ids = save_vectors(state)
    state.manifest(1).set_status([ids[0]], "excluded")
    queued = uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    assert queued["status"] == "queued" and queued["job"]
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    assert set(done["scores"]) == set(ids[1:]) and done["missing"] == 0
    assert max(done["scores"].values()) == pytest.approx(1.0) and done["backbone"] == "fake"
    assert done["phase"] == "" and done["fiftyone_error"] == ""
    assert len(published) == 1


def test_images_without_embeddings_are_counted_as_missing(setup, published):
    client, state, runner = setup
    ids = save_vectors(state, 4)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    done = wait_for(finished(state))
    assert set(done["scores"]) == set(ids) and done["missing"] == 2


def test_without_the_similarity_map_it_needs_it(setup):
    client, state, runner = setup
    with pytest.raises(CodedError) as caught:
        uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    assert caught.value.code == "similarity_map_needed"
    assert uniqueness.UniquenessStore(state.store.dir(1)).read() == {}


def test_embeddings_removed_after_queueing_fail_the_job(setup, published):
    client, state, runner = setup
    save_vectors(state)
    gate, _ = block(runner)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    similarity.SimilarityStore(state.store.dir(1)).clear()
    gate.append(True)
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "similarity map" in failed["error"].lower()


def test_cancel_while_queued_never_starts(setup, published):
    client, state, runner = setup
    save_vectors(state)
    gate, blocker = block(runner)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    assert uniqueness_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert uniqueness.UniquenessStore(state.store.dir(1)).read()["status"] == "cancelled"
    assert uniqueness_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"  # nothing to cancel
    assert published == []


def test_the_queue_cancel_button_on_a_waiting_job(setup, published):
    client, state, runner = setup
    save_vectors(state)
    gate, blocker = block(runner)
    data = uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    job = runner.tool_job(data["job"])
    assert job.kind == uniqueness_job.KIND and job.device == "cpu"
    assert runner.cancel_job(data["job"]) is True
    assert uniqueness.UniquenessStore(state.store.dir(1)).read()["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")


def test_cancelling_a_running_job_stops_it_between_chunks(setup, published, monkeypatch):
    client, state, runner = setup
    save_vectors(state)
    monkeypatch.setattr(uniqueness_job.uniqueness, "scores", slow_scores)
    data = uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    wait_for(lambda: uniqueness.UniquenessStore(state.store.dir(1)).read().get("status") == "running")
    assert runner.cancel_job(data["job"]) is True
    assert wait_for(finished(state))["status"] == "cancelled"
    assert published == []


def test_a_scoring_error_is_kept(setup, published, monkeypatch):
    client, state, runner = setup
    save_vectors(state)
    monkeypatch.setattr(uniqueness_job.uniqueness, "scores", broken_scores)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and failed["error"] and failed["phase"] == ""


def test_a_job_left_by_a_previous_start_is_interrupted(setup):
    client, state, runner = setup
    uniqueness.UniquenessStore(state.store.dir(1)).update(status="running", job="gone")
    shown = uniqueness_job.state(runner, state.store.dir(1))
    assert shown["status"] == "failed" and shown["error"] == "interrupted"
    assert uniqueness_job.state(runner, state.store.dir(2)) == {}


def test_publish_reports_a_fiftyone_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(uniqueness_job.fiftyone_bridge, "sync_dataset", database_down)
    assert uniqueness_job._publish(SimpleNamespace(id=1), None, tmp_path, {}) == "database down"


def test_without_fiftyone_the_scores_stay_in_activelabel(monkeypatch, tmp_path):
    monkeypatch.setattr(uniqueness_job.optional, "FIFTYONE", False)
    message = uniqueness_job._publish(SimpleNamespace(id=1), None, tmp_path, {"a": 1.0})
    assert "FiftyOne is not installed" in message and uniqueness_job.optional.FIFTYONE_INSTALL in message


def test_a_publish_failure_keeps_the_scores(setup, monkeypatch):
    client, state, runner = setup
    save_vectors(state)
    monkeypatch.setattr(uniqueness_job.fiftyone_bridge, "sync_dataset", database_down)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    done = wait_for(finished(state))
    assert done["status"] == "done" and len(done["scores"]) == 6
    assert done["fiftyone_error"] == "database down"


@pytest.mark.integration
def test_the_scores_are_written_to_the_fiftyone_field(setup, monkeypatch):
    client, state, runner = setup
    name = f"test-uniqueness-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    save_vectors(state)
    uniqueness_job.start(runner, state.project(1), state.store.dir(1))
    try:
        done = wait_for(finished(state), timeout=120)
        assert done["status"] == "done" and done["fiftyone_error"] == "", done
        dataset = fo.load_dataset(name)
        by_image = dict(zip(dataset.values("image_id"), dataset.values(uniqueness_job.FIELD), strict=True))
        assert by_image == done["scores"]
        assert fiftyone_runs.RUN_KEYS["uniqueness"] in dataset.list_runs()
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
