import json
import uuid

import fiftyone as fo
import pytest
from training_fakes import saved_model_project, tmp_dirs, wait_for

from activelabel import fiftyone_bridge
from activelabel.curation import mistakenness_job
from activelabel.errors import CodedError
from activelabel.labels import load_annotations, save_annotations
from activelabel.training.models import ModelNotFound

FINISHED = ("done", "failed", "cancelled")


class FakeScore:
    """Replaces ``mistakenness_fiftyone.score``: remembers its arguments, scores the predicted images."""

    def __init__(self):
        self.calls = []

    def __call__(self, dataset, task, predictions, logits, classes):
        self.calls.append((dataset, task, predictions, logits, classes))
        return {i: {"mistakenness": 0.5, "missing": 0, "spurious": 0, "hardness": 0.25} for i in predictions}


@pytest.fixture
def project(tmp_path):
    """A project with 8 labelled images and a saved model m1, whose runner has fake workers."""
    client, state, runner = saved_model_project(tmp_path)
    yield client, state, runner
    runner.shutdown()


@pytest.fixture
def setup(project, monkeypatch):
    """The same project with FiftyOne replaced: the dataset is a string and the scoring a recorder."""
    client, state, runner = project
    score = FakeScore()
    monkeypatch.setattr(mistakenness_job.mistakenness_fiftyone, "score", score)
    monkeypatch.setattr(mistakenness_job.fiftyone_bridge, "sync_dataset", lambda *args: "dataset")
    return client, state, runner, score


def current(state):
    return mistakenness_job.store(state.store.dir(1)).read()


def finished(state):
    return lambda: (s := current(state)).get("status") in FINISHED and s


def start(state, runner, device="auto", model=1):
    return mistakenness_job.start(runner, state.project(1), state.store.dir(1), model, device)


def block(runner):
    """Occupy the lane until the returned gate gets an item."""
    gate: list[bool] = []
    blocker = runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    return gate, blocker


def test_the_labelled_images_are_the_active_ones_that_have_labels(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    ids = sorted(r.id for r in state.manifest(1).active())
    annotations = load_annotations(project_dir)
    del annotations.images[ids[0]]
    save_annotations(project_dir, annotations)
    state.manifest(1).set_status([ids[1]], "excluded")
    items = mistakenness_job.labelled_images(project_dir)
    assert sorted(i["id"] for i in items) == ids[2:]
    assert all(i["path"].endswith(".png") for i in items)


def test_the_check_predicts_then_scores_the_labelled_images(setup):
    client, state, runner, score = setup
    queued = start(state, runner)
    assert queued["status"] == "queued" and queued["job"] and queued["model"] == 1
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    assert len(done["scores"]) == 8 and done["checked"] == 8
    assert done["phase"] == "" and done["unreadable"] == []
    assert next(iter(done["scores"].values())) == {
        "mistakenness": 0.5,
        "missing": 0,
        "spurious": 0,
        "hardness": 0.25,
    }
    ((dataset, task, predictions, logits, classes),) = score.calls
    assert dataset == "dataset" and task.value == "classification" and classes == ["a", "b"]
    assert len(predictions) == 8 and set(logits) == set(predictions)  # a classification asks for the logits
    labelled = [v for v in predictions.values() if v]
    assert len(labelled) == 7 and all([lab.label for lab in v] == ["a"] for v in labelled)  # none for image 1
    spec = json.loads((state.store.dir(1) / "models" / "m1" / "predict_spec.json").read_text())
    assert spec["logits"] is True and spec["device"] == "auto" and spec["keep"] is None
    assert spec["threshold"] == mistakenness_job.CONFIDENCE and len(spec["items"]) == 8
    assert "torch" in runner.environment.ensured[-1]
    assert (state.store.dir(1) / "mistakenness.log").exists()
    wait_for(lambda: tmp_dirs(runner) == [])
    assert mistakenness_job.state_of(runner, state.store.dir(1))["status"] == "done"


def test_the_job_is_of_the_chosen_device_lane(setup):
    client, state, runner, _ = setup
    job = runner.tool_job(start(state, runner, "cpu")["job"])
    assert job.kind == mistakenness_job.KIND and job.device == "cpu"
    assert wait_for(finished(state))["status"] == "done"


def test_images_the_worker_cannot_read_are_listed(setup):
    client, state, runner, score = setup
    start(state, runner, "skip")
    done = wait_for(finished(state))
    assert done["status"] == "done" and len(done["unreadable"]) == 1
    assert done["unreadable"][0] not in done["scores"]


def test_without_labelled_images_it_does_not_start(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    annotations = load_annotations(project_dir)
    annotations.images.clear()
    save_annotations(project_dir, annotations)
    with pytest.raises(CodedError) as caught:
        start(state, runner)
    assert caught.value.code == "no_labelled_images"
    assert current(state) == {}


def test_an_unknown_model_does_not_start(setup):
    client, state, runner, _ = setup
    with pytest.raises(ModelNotFound):
        start(state, runner, model=9)
    assert current(state) == {}


def test_labels_removed_after_queueing_fail_the_job(setup):
    client, state, runner, score = setup
    gate, _ = block(runner)
    start(state, runner)
    project_dir = state.store.dir(1)
    annotations = load_annotations(project_dir)
    annotations.images.clear()
    save_annotations(project_dir, annotations)
    gate.append(True)
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "label" in failed["error"].lower() and score.calls == []


def test_a_worker_failure_is_kept(setup):
    client, state, runner, score = setup
    start(state, runner, "fail")
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "too strict" in failed["error"]
    assert failed["phase"] == "" and score.calls == []
    wait_for(lambda: tmp_dirs(runner) == [])


def test_cancel_while_queued_never_starts(setup):
    client, state, runner, score = setup
    gate, blocker = block(runner)
    start(state, runner)
    assert mistakenness_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert current(state)["status"] == "cancelled"
    assert (
        mistakenness_job.cancel(runner, 1, state.store.dir(1))["status"] == "cancelled"
    )  # nothing to cancel
    assert score.calls == []


def test_the_queue_cancel_button_on_a_waiting_job(setup):
    client, state, runner, score = setup
    gate, blocker = block(runner)
    data = start(state, runner)
    assert runner.cancel_job(data["job"]) is True
    assert current(state)["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert score.calls == []


def test_cancelling_a_running_check_kills_its_worker(setup):
    client, state, runner, score = setup
    data = start(state, runner, "slow")
    wait_for(lambda: current(state).get("phase") == "predicting")
    assert runner.cancel_job(data["job"]) is True
    cancelled = wait_for(finished(state))
    assert cancelled["status"] == "cancelled" and cancelled["phase"] == "" and score.calls == []
    wait_for(lambda: tmp_dirs(runner) == [])


def test_cancelling_a_running_check_from_its_page(setup):
    client, state, runner, score = setup
    start(state, runner, "slow")
    wait_for(lambda: current(state).get("phase") == "predicting")
    mistakenness_job.cancel(runner, 1, state.store.dir(1))
    assert wait_for(finished(state))["status"] == "cancelled"


def test_the_previous_scores_stay_when_a_new_check_is_cancelled(setup):
    client, state, runner, _ = setup
    start(state, runner)
    first = wait_for(finished(state))
    gate, _ = block(runner)
    start(state, runner)
    mistakenness_job.cancel(runner, 1, state.store.dir(1))
    gate.append(True)
    assert current(state)["status"] == "cancelled" and current(state)["scores"] == first["scores"]


def test_a_job_left_by_a_previous_start_is_interrupted(setup):
    client, state, runner, _ = setup
    mistakenness_job.store(state.store.dir(1)).update(status="running", job="gone")
    shown = mistakenness_job.state_of(runner, state.store.dir(1))
    assert shown["status"] == "failed" and shown["error"] == "interrupted"
    assert mistakenness_job.state_of(runner, state.store.dir(2)) == {}


@pytest.mark.integration
def test_the_predictions_are_compared_in_fiftyone(project, monkeypatch):
    client, state, runner = project
    name = f"test-mistakenness-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    start(state, runner)
    try:
        done = wait_for(finished(state), timeout=180)
        assert done["status"] == "done", done
        assert len(done["scores"]) == 8 and all(s["hardness"] is not None for s in done["scores"].values())
        assert fo.load_dataset(name).has_sample_field("hardness")
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
