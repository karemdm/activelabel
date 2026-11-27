import uuid
from types import SimpleNamespace

import fiftyone as fo
import pytest
from training_fakes import saved_model_project, tmp_dirs, wait_for

from activelabel import fiftyone_bridge
from activelabel.curation import evaluation_job, fiftyone_runs
from activelabel.datasets import DatasetNotFound, DatasetStore
from activelabel.errors import CodedError
from activelabel.labels import load_annotations, save_annotations
from activelabel.tasks import Task
from activelabel.training.models import ModelNotFound

FINISHED = ("done", "failed", "cancelled")


class FakeEvaluate:
    """Replaces ``evaluation_fiftyone.evaluate``: remembers its arguments and answers a small result."""

    def __init__(self):
        self.calls = []
        self.recorded = []

    def __call__(self, dataset, task, model, predictions):
        self.calls.append((dataset, task, model, predictions))
        return {
            "kind": "classification",
            "images": len(predictions),
            "metrics": {"accuracy": 0.5, "macro_f1": 0.4},
            "classes": [],
            "confusion": {"classes": [], "other": False, "missing": False, "matrix": []},
            "worst": [],
            "fields": ["eval_m1_pred", "eval_m1"],
        }


@pytest.fixture
def project(tmp_path):
    """A project with 8 labelled images, a version 1 (test and val: 2 images each) and a saved model m1."""
    client, state, runner = saved_model_project(tmp_path)
    yield client, state, runner
    runner.shutdown()


@pytest.fixture
def setup(project, monkeypatch):
    """The same project with FiftyOne replaced: the dataset is a string and the evaluation a recorder."""
    client, state, runner = project
    evaluate = FakeEvaluate()
    monkeypatch.setattr(evaluation_job.evaluation_fiftyone, "evaluate", evaluate)
    monkeypatch.setattr(evaluation_job.fiftyone_bridge, "sync_dataset", lambda *args: "dataset")
    monkeypatch.setattr(
        evaluation_job.fiftyone_runs,
        "record",
        lambda *args, **kwargs: evaluate.recorded.append((args, kwargs)),
    )
    return client, state, runner, evaluate


def entry(state, model=1):
    return evaluation_job.entry_of(state.store.dir(1), model)


def finished(state, model=1):
    return lambda: (e := entry(state, model)).get("status") in FINISHED and e


def start(state, runner, device="auto", model=1, **where):
    return evaluation_job.start(runner, state.project(1), state.store.dir(1), model, device, **where)


def block(runner):
    """Occupy the lane until the returned gate gets an item."""
    gate: list[bool] = []
    blocker = runner.enqueue_tool(1, "block", lambda progress: wait_for(lambda: progress.check() or gate, 30))
    return gate, blocker


def test_the_default_target_is_the_test_part_of_the_version_of_the_model(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    test = DatasetStore(project_dir).image_ids(1, "test")
    assert evaluation_job.target(project_dir, {"dataset": 1}, None, None, None) == (1, "test", None, test)


def test_without_a_test_part_it_is_the_validation_part(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    split = {"train": 80, "val": 20, "test": 0}
    assert client.post("/api/projects/1/datasets", json={"split": split}).status_code == 200
    val = DatasetStore(project_dir).image_ids(2, "val")
    assert evaluation_job.target(project_dir, {"dataset": 2}, None, None, None) == (2, "val", None, val)


def test_a_version_that_was_deleted_gives_way_to_the_newest_one(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    assert client.post("/api/projects/1/datasets", json={}).status_code == 200
    assert evaluation_job.target(project_dir, {"dataset": 7}, None, None, None)[0] == 2
    assert evaluation_job.target(project_dir, {}, None, "val", None)[:2] == (2, "val")
    assert evaluation_job.target(project_dir, {"dataset": 2}, 1, "train", None)[:2] == (1, "train")


def test_a_kfold_version_is_evaluated_on_its_fold_zero_without_a_test_part(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    made = client.post(
        "/api/projects/1/datasets", json={"kfold": 2, "split": {"train": 0, "val": 0, "test": 0}}
    )
    assert made.status_code == 200, made.text
    folds = DatasetStore(project_dir).read(2)["folds"]
    assert evaluation_job.target(project_dir, {}, 2, None, None) == (2, "val", 0, folds[0])
    assert evaluation_job.target(project_dir, {}, 2, "val", 1)[3] == folds[1]


def test_without_a_version_there_is_nothing_to_evaluate_on(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    with pytest.raises(DatasetNotFound):
        evaluation_job.target(project_dir, {}, 5, None, None)
    DatasetStore(project_dir).delete(1)
    with pytest.raises(CodedError) as caught:
        evaluation_job.target(project_dir, {"dataset": 1}, None, None, None)
    assert caught.value.code == "no_dataset_version"


def test_the_images_to_evaluate_were_reviewed_and_are_active(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    ids = DatasetStore(project_dir).image_ids(1)
    annotations = load_annotations(project_dir)
    del annotations.images[ids[0]]
    save_annotations(project_dir, annotations)
    state.manifest(1).set_status([ids[1]], "excluded")
    items = evaluation_job.evaluable(project_dir, Task.CLASSIFICATION, ids)
    assert [i["id"] for i in items] == ids[2:]
    assert all(i["path"].endswith(".png") for i in items)


def test_an_object_image_reviewed_with_no_objects_counts(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    ids = DatasetStore(project_dir).image_ids(1)
    annotations = load_annotations(project_dir)
    annotations.images[ids[0]] = []
    save_annotations(project_dir, annotations)
    assert ids[0] in [i["id"] for i in evaluation_job.evaluable(project_dir, Task.DETECTION, ids)]
    assert ids[0] not in [i["id"] for i in evaluation_job.evaluable(project_dir, Task.CLASSIFICATION, ids)]


def test_the_evaluation_predicts_then_evaluates_the_part(setup):
    client, state, runner, evaluate = setup
    test = DatasetStore(state.store.dir(1)).image_ids(1, "test")
    queued = start(state, runner)
    assert queued["status"] == "queued" and queued["job"]
    assert (queued["version"], queued["part"], queued["images"]) == (1, "test", 2)
    done = wait_for(finished(state))
    assert done["status"] == "done", done.get("error")
    assert done["phase"] == "" and done["result"]["images"] == 2
    assert (done["result"]["version"], done["result"]["part"], done["result"]["fold"]) == (1, "test", None)
    assert done["result"]["unreadable"] == [] and done["result"]["evaluated"]
    ((dataset, task, model, predictions),) = evaluate.calls
    assert dataset == "dataset" and task is Task.CLASSIFICATION and model == 1
    assert set(predictions) == set(test)
    spec = (state.store.dir(1) / "models" / "m1" / "predict_spec.json").read_text()
    assert '"threshold": 0.05' in spec and '"logits": false' in spec
    assert "torch" in runner.environment.ensured[-1]
    assert (state.store.dir(1) / "evaluation.log").exists()
    ((args, kwargs),) = evaluate.recorded
    assert args[1] == "evaluation" and kwargs["suffix"] == "_m1" and kwargs["eval_key"] == "eval_m1"
    assert kwargs["part"] == "test" and kwargs["images"] == 2
    wait_for(lambda: tmp_dirs(runner) == [])
    shown = evaluation_job.state_of(runner, state.store.dir(1), 1)
    assert shown["status"] == "done" and shown["active"] is None


def test_the_part_and_the_device_lane_can_be_chosen(setup):
    client, state, runner, evaluate = setup
    queued = start(state, runner, "cpu", part="val")
    job = runner.tool_job(queued["job"])
    assert job.kind == evaluation_job.KIND and job.device == "cpu" and queued["part"] == "val"
    done = wait_for(finished(state))
    assert done["status"] == "done" and done["result"]["part"] == "val"
    assert set(evaluate.calls[0][3]) == set(DatasetStore(state.store.dir(1)).image_ids(1, "val"))


def test_images_the_worker_cannot_read_are_listed(setup):
    client, state, runner, evaluate = setup
    start(state, runner, "skip")
    done = wait_for(finished(state))
    assert done["status"] == "done" and len(done["result"]["unreadable"]) == 1
    assert done["result"]["unreadable"][0] not in evaluate.calls[0][3]


def test_a_keypoints_or_regression_project_is_refused(setup):
    client, state, runner, _ = setup
    for task in (Task.KEYPOINTS, Task.REGRESSION):
        with pytest.raises(CodedError) as caught:
            evaluation_job.start(runner, SimpleNamespace(id=1, task=task), state.store.dir(1), 1)
        assert caught.value.code == "evaluation_task_unsupported"
        assert caught.value.params == {"task": task.value}
    assert entry(state) == {}


def test_an_unknown_model_does_not_start(setup):
    client, state, runner, _ = setup
    with pytest.raises(ModelNotFound):
        start(state, runner, model=9)
    assert entry(state, 9) == {}


def test_a_part_without_reviewed_images_does_not_start(setup):
    client, state, runner, _ = setup
    project_dir = state.store.dir(1)
    annotations = load_annotations(project_dir)
    annotations.images.clear()
    save_annotations(project_dir, annotations)
    with pytest.raises(CodedError) as caught:
        start(state, runner)
    assert caught.value.code == "no_images_in_part" and caught.value.params == {"version": 1, "part": "test"}
    assert entry(state) == {}


def test_only_one_evaluation_runs_at_a_time(setup):
    client, state, runner, _ = setup
    evaluation_job._patch(state.store.dir(1), 2, status="running", job="x")
    with pytest.raises(CodedError) as caught:
        start(state, runner)
    assert caught.value.code == "evaluation_running" and caught.value.params == {"model": 2}
    assert entry(state) == {}


def test_the_state_of_another_model_tells_which_one_is_running(setup):
    client, state, runner, _ = setup
    gate, _ = block(runner)
    start(state, runner)
    assert evaluation_job.state_of(runner, state.store.dir(1), 2) == {"active": 1}
    gate.append(True)
    wait_for(finished(state))
    assert evaluation_job.state_of(runner, state.store.dir(1), 2) == {"active": None}


def test_labels_removed_after_queueing_fail_the_job(setup):
    client, state, runner, evaluate = setup
    gate, _ = block(runner)
    start(state, runner)
    project_dir = state.store.dir(1)
    annotations = load_annotations(project_dir)
    annotations.images.clear()
    save_annotations(project_dir, annotations)
    gate.append(True)
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "reviewed" in failed["error"] and evaluate.calls == []


def test_a_worker_failure_is_kept(setup):
    client, state, runner, evaluate = setup
    start(state, runner, "fail")
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "too strict" in failed["error"]
    assert failed["phase"] == "" and evaluate.calls == []
    wait_for(lambda: tmp_dirs(runner) == [])


def database_down(*args, **kwargs):
    raise RuntimeError("database down")


def test_a_failure_in_fiftyone_is_kept(setup, monkeypatch):
    client, state, runner, evaluate = setup
    monkeypatch.setattr(evaluation_job.evaluation_fiftyone, "evaluate", database_down)
    start(state, runner)
    failed = wait_for(finished(state))
    assert failed["status"] == "failed" and "database down" in failed["error"] and "result" not in failed


def test_cancel_while_queued_never_starts(setup):
    client, state, runner, evaluate = setup
    gate, blocker = block(runner)
    start(state, runner)
    assert evaluation_job.cancel(runner, 1, state.store.dir(1), 1)["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert entry(state)["status"] == "cancelled"
    assert (
        evaluation_job.cancel(runner, 1, state.store.dir(1), 1)["status"] == "cancelled"
    )  # nothing to cancel
    assert evaluate.calls == []


def test_the_queue_cancel_button_on_a_waiting_job(setup):
    client, state, runner, evaluate = setup
    gate, blocker = block(runner)
    data = start(state, runner)
    assert runner.cancel_job(data["job"]) is True
    assert entry(state)["status"] == "cancelled"
    gate.append(True)
    wait_for(lambda: runner.tool_job(blocker).status == "done")
    assert evaluate.calls == []


def test_cancelling_a_running_evaluation_kills_its_worker(setup):
    client, state, runner, evaluate = setup
    data = start(state, runner, "slow")
    wait_for(lambda: entry(state).get("phase") == "predicting")
    assert runner.cancel_job(data["job"]) is True
    cancelled = wait_for(finished(state))
    assert cancelled["status"] == "cancelled" and cancelled["phase"] == "" and evaluate.calls == []
    wait_for(lambda: tmp_dirs(runner) == [])


def test_cancelling_a_running_evaluation_from_its_page(setup):
    client, state, runner, evaluate = setup
    start(state, runner, "slow")
    wait_for(lambda: entry(state).get("phase") == "predicting")
    evaluation_job.cancel(runner, 1, state.store.dir(1), 1)
    assert wait_for(finished(state))["status"] == "cancelled"


def test_the_previous_result_stays_when_a_new_evaluation_is_cancelled(setup):
    client, state, runner, _ = setup
    start(state, runner)
    first = wait_for(finished(state))
    gate, _ = block(runner)
    start(state, runner)
    evaluation_job.cancel(runner, 1, state.store.dir(1), 1)
    gate.append(True)
    assert entry(state)["status"] == "cancelled" and entry(state)["result"] == first["result"]


def test_a_job_left_by_a_previous_start_is_interrupted(setup):
    client, state, runner, _ = setup
    evaluation_job._patch(state.store.dir(1), 1, status="running", job="gone")
    shown = evaluation_job.state_of(runner, state.store.dir(1), 1)
    assert shown["status"] == "failed" and shown["error"] == "interrupted" and shown["active"] is None
    assert evaluation_job.state_of(runner, state.store.dir(2), 1) == {"active": None}


def test_a_failure_recording_the_run_is_only_logged(setup, monkeypatch):
    client, state, runner, evaluate = setup
    monkeypatch.setattr(evaluation_job.fiftyone_runs, "record", database_down)
    start(state, runner)
    done = wait_for(finished(state))
    assert done["status"] == "done" and done["result"]["images"] == 2


@pytest.mark.integration
def test_the_model_is_evaluated_in_fiftyone(project, monkeypatch):
    client, state, runner = project
    name = f"test-evaluation-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(fiftyone_bridge, "dataset_name", lambda project_dir, pid: name)
    start(state, runner)
    try:
        done = wait_for(finished(state), timeout=180)
        assert done["status"] == "done", done
        result = done["result"]
        assert result["images"] == 2 and result["kind"] == "classification"
        assert set(result["metrics"]) == {"accuracy", "macro_f1"} and result["confusion"]["matrix"]
        dataset = fo.load_dataset(name)
        assert "eval_m1" in dataset.list_evaluations() and len(dataset.load_saved_view("evaluation m1")) == 2
        key = fiftyone_runs.RUN_KEYS["evaluation"] + "_m1"
        config = dataset.get_run_info(key).config.serialize()
        assert config["images"] == 2 and config["part"] == "test" and config["eval_key"] == "eval_m1"
        assert fiftyone_runs.is_ours(key) and fiftyone_runs.delete(dataset, key)
    finally:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
