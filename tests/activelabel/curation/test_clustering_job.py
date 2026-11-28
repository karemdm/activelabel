import uuid
from types import SimpleNamespace

import fiftyone as fo
import numpy as np
import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FAKE_EMBED, FakeEnv, wait_for

from activelabel import fiftyone_bridge
from activelabel.curation import (
    clustering_job,
    fiftyone_runs,
    similarity,
    similarity_fiftyone,
    similarity_job,
)
from activelabel.errors import CodedError
from activelabel.training.runner import TrainingRunner

FINISHED = ("done", "failed", "cancelled")
COUNT = 12
REAL_PUBLISH = similarity_job._publish
REAL_EXISTING = fiftyone_bridge.existing_dataset


def never_cluster(vectors, method, groups, min_size):
    """A grouping that must not run (the job was cancelled while it waited)."""
    raise AssertionError("the grouping must not run")


def broken_cluster(vectors, method, groups, min_size):
    raise ZeroDivisionError("division by zero")


def database_down(*args):
    raise RuntimeError("database down")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A project with 12 images and a runner without a worker (the job runs in its thread)."""
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(COUNT)]
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
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", lambda project_dir, pid: None)  # no FiftyOne run
    yield client, state, runner, published
    runner.shutdown()


def save_map(state, outlier=False):
    """Save a finished map: two far blobs (and, with ``outlier``, one image far from both)."""
    ids = [r.id for r in state.manifest(1).active()]
    rng = np.random.default_rng(2)
    vectors = rng.normal(0, 0.1, (len(ids), 8))
    vectors[: len(ids) // 2, 0] += 10
    vectors[len(ids) // 2 :, 1] += 10
    if outlier:
        vectors[-1] = 60.0
    store = similarity.SimilarityStore(state.store.dir(1))
    store.save_vectors(ids, vectors, "fake")
    store.update(status="done", **similarity.build_map(ids, vectors))
    return ids


def entry(state):
    return similarity.SimilarityStore(state.store.dir(1)).read().get("clustering", {})


def finished(state):
    return lambda: (s := entry(state)).get("status") in FINISHED and s


def block(runner):
    """Occupy the CPU lane (the job's own: ``auto`` would be mps on a Mac) until the gate gets an item."""
    gate: list[bool] = []
    blocker = runner.enqueue_tool(
        1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30), device="cpu"
    )
    return gate, blocker


def test_the_groups_change_and_the_points_stay(setup):
    client, state, runner, published = setup
    save_map(state)
    before = similarity.SimilarityStore(state.store.dir(1)).read()["points"]
    queued = clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=3)
    assert queued["status"] == "queued" and queued["job"] and queued["method"] == "kmeans"
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    assert done["noise"] == 0 and done["fiftyone_error"] == "" and done["groups"] == 3
    data = similarity.SimilarityStore(state.store.dir(1)).read()
    assert data["points"]["ids"] == before["ids"] and data["points"]["xy"] == before["xy"]
    assert sorted(set(data["points"]["group"])) == [0, 1, 2]
    sizes = [g["size"] for g in data["groups"]]
    assert sizes == sorted(sizes, reverse=True) and sum(sizes) == COUNT
    assert len(published) == 1 and published[0][3]["points"] == data["points"]


def test_hdbscan_counts_its_noise_and_leaves_it_out_of_the_groups(setup):
    client, state, runner, published = setup
    save_map(state, outlier=True)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "hdbscan", min_size=4)
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    data = similarity.SimilarityStore(state.store.dir(1)).read()
    assert done["noise"] == 1 and data["points"]["group"][-1] == similarity.NOISE
    assert sum(g["size"] for g in data["groups"]) == COUNT - 1


def test_a_publish_failure_keeps_the_groups(setup, monkeypatch):
    client, state, runner, published = setup
    save_map(state)
    monkeypatch.setattr(similarity_job, "_publish", REAL_PUBLISH)
    monkeypatch.setattr(similarity_job.fiftyone_bridge, "sync_dataset", database_down)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    done = wait_for(finished(state))
    assert done["status"] == "done" and done["fiftyone_error"] == "database down"
    assert len(similarity.SimilarityStore(state.store.dir(1)).read()["groups"]) == 2


def test_without_fiftyone_no_run_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(clustering_job.optional, "FIFTYONE", False)
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", database_down)  # never reached
    clustering_job._record(SimpleNamespace(id=1), tmp_path, {"method": "auto"}, 3)


def test_a_failure_recording_the_run_is_only_logged(setup, monkeypatch):
    client, state, runner, published = setup
    save_map(state)
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", lambda project_dir, pid: "dataset")
    monkeypatch.setattr(clustering_job.fiftyone_runs, "record", database_down)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    done = wait_for(finished(state))
    assert done["status"] == "done" and done["fiftyone_error"] == ""
    assert len(similarity.SimilarityStore(state.store.dir(1)).read()["groups"]) == 2


def test_without_the_map_it_needs_it(setup):
    client, state, runner, published = setup
    with pytest.raises(CodedError) as caught:
        clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=3)
    assert caught.value.code == "similarity_map_needed"
    assert entry(state) == {}


@pytest.mark.parametrize(
    ("method", "groups", "min_size", "code"),
    [
        ("dbscan", 0, 5, "cluster_method_unknown"),
        ("kmeans", 1, 5, "cluster_groups_range"),
        ("kmeans", 51, 5, "cluster_groups_range"),
        ("hdbscan", 0, 1, "cluster_min_size_small"),
    ],
)
def test_a_method_or_number_out_of_range_is_refused_before_queueing(setup, method, groups, min_size, code):
    client, state, runner, published = setup
    save_map(state)
    with pytest.raises(CodedError) as caught:
        clustering_job.start(runner, state.project(1), state.store.dir(1), method, groups, min_size)
    assert caught.value.code == code and caught.value.params  # the numbers the translated text needs
    assert entry(state) == {}


def test_the_embeddings_removed_after_queueing_fail_the_job(setup):
    client, state, runner, published = setup
    save_map(state)
    gate, _ = block(runner)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    similarity.SimilarityStore(state.store.dir(1)).vectors_path.unlink()
    gate.append(True)
    failed = wait_for(lambda: (s := entry(state)).get("status") in FINISHED and s)
    assert failed["status"] == "failed" and "similarity map" in failed["error"].lower()


def test_cancel_while_queued_never_starts(setup, monkeypatch):
    client, state, runner, published = setup
    save_map(state)
    monkeypatch.setattr(clustering_job.similarity, "cluster", never_cluster)
    gate, blocker = block(runner)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    assert clustering_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert entry(state)["status"] == "cancelled"
    assert clustering_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"  # nothing to cancel
    assert published == []


def test_the_queue_cancel_button_on_a_waiting_job(setup):
    client, state, runner, published = setup
    save_map(state)
    gate, blocker = block(runner)
    data = clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    job = runner.tool_job(data["job"])
    assert job.kind == clustering_job.KIND and job.device == "cpu"
    assert runner.cancel_job(data["job"]) is True
    assert entry(state)["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert published == []


def test_a_grouping_error_is_kept_and_the_groups_stay(setup, monkeypatch):
    client, state, runner, published = setup
    save_map(state)
    before = similarity.SimilarityStore(state.store.dir(1)).read()["groups"]
    monkeypatch.setattr(clustering_job.similarity, "cluster", broken_cluster)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and failed["error"]
    assert similarity.SimilarityStore(state.store.dir(1)).read()["groups"] == before
    assert published == []


def test_a_job_left_by_a_previous_start_is_interrupted(setup):
    client, state, runner, published = setup
    similarity.SimilarityStore(state.store.dir(1)).update(clustering={"status": "running", "job": "gone"})
    shown = clustering_job.state(runner, state.store.dir(1))
    assert shown["status"] == "failed" and shown["error"] == "interrupted"
    assert clustering_job.state(runner, state.store.dir(2)) == {}


def test_a_new_similarity_map_resets_the_regrouping(setup):
    client, state, runner, published = setup
    save_map(state)
    clustering_job.start(runner, state.project(1), state.store.dir(1), "kmeans", groups=2)
    assert wait_for(finished(state))["status"] == "done"
    similarity_job.start(runner, state.project(1), state.store.dir(1))
    store = similarity.SimilarityStore(state.store.dir(1))
    done = wait_for(lambda: (s := store.read()).get("status") == "done" and s)
    assert done["clustering"] == {}
    assert clustering_job.state(runner, state.store.dir(1)) == {}


@pytest.mark.integration
def test_the_noise_has_no_group_in_the_fiftyone_field(setup, monkeypatch):
    client, state, runner, published = setup
    name = f"test-clustering-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(similarity_job, "_publish", REAL_PUBLISH)
    monkeypatch.setattr(fiftyone_bridge, "existing_dataset", REAL_EXISTING)
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    save_map(state, outlier=True)
    try:
        clustering_job.start(runner, state.project(1), state.store.dir(1), "hdbscan", min_size=4)
        done = wait_for(finished(state), timeout=120)
        assert done["status"] == "done" and done["noise"] == 1 and done["fiftyone_error"] == "", done
        dataset = fo.load_dataset(name)
        assert similarity_fiftyone.BRAIN_KEY in dataset.list_brain_runs()
        assert dataset.values(similarity_fiftyone.GROUP_FIELD).count(None) == 1
        config = dataset.get_run_info(fiftyone_runs.RUN_KEYS["clustering"]).config.serialize()
        assert config["method"] == "hdbscan" and config["min_size"] == 4 and config["images"] == COUNT
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
