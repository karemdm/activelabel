import json
import threading
import time

import pytest

from activelabel.annotate.base import ToolError
from activelabel.jobs import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    JobManager,
    database_failure,
    readable_error,
)
from activelabel.scheduler import CPU, Limits, Scheduler
from activelabel.textcheck import message_of


def wait(job, timeout=5.0):
    end = time.time() + timeout
    while job.status not in (DONE, FAILED, CANCELLED) and time.time() < end:
        time.sleep(0.01)
    return job


def test_job_runs_and_reports():
    jobs = JobManager()

    def work(progress):
        for i in range(3):
            progress.update(i + 1, 3, f"step {i + 1}")
        return {"n": 3}

    job = wait(jobs.submit(1, "demo", work))
    assert (job.status, job.done, job.total, job.result) == (DONE, 3, 3, {"n": 3})
    assert jobs.list(1) == [job] and jobs.list(2) == []


def test_failure_is_captured():
    jobs = JobManager()
    job = wait(jobs.submit(1, "demo", lambda progress: 1 / 0))
    assert job.status == FAILED and "division by zero" in job.error


def test_cancel_running_and_queued():
    jobs = JobManager()
    started = threading.Event()

    def slow(progress):
        started.set()
        while True:
            progress.check()
            time.sleep(0.01)

    running = jobs.submit(1, "slow", slow)
    queued = jobs.submit(1, "next", lambda progress: {"ran": True})
    started.wait(2)
    assert {j.id for j in jobs.active(1)} == {running.id, queued.id}
    jobs.cancel(queued.id)
    jobs.cancel(running.id)
    assert wait(running).status == CANCELLED
    assert wait(queued).status == CANCELLED and queued.result == {}


def test_clear_removes_only_finished_jobs_of_the_project():
    jobs = JobManager()
    started = threading.Event()
    wait(jobs.submit(1, "a", lambda progress: {}))
    other = wait(jobs.submit(2, "b", lambda progress: {}))
    wait(jobs.submit(1, "c", lambda progress: 1 / 0))

    def slow(progress):
        started.set()
        while True:
            progress.check()
            time.sleep(0.01)

    running = jobs.submit(1, "slow", slow)
    queued = jobs.submit(1, "next", lambda progress: {})
    started.wait(2)
    assert jobs.clear(1) == 2
    assert {j.id for j in jobs.list(1)} == {running.id, queued.id}
    assert jobs.list(2) == [other]
    jobs.cancel(running.id)
    jobs.cancel(queued.id)
    wait(running)
    wait(queued)
    assert jobs.clear(1) == 2 and jobs.clear(1) == 0


def test_fiftyone_database_errors_are_readable():
    class ServiceListenTimeout(Exception):
        pass

    message = readable_error(ServiceListenTimeout("DatabaseService failed to bind to port"))
    assert message.startswith("The FiftyOne database did not respond") and "ServiceListenTimeout" in message
    assert readable_error(ValueError("x")) == "x"


def test_error_message_has_no_class_name_and_the_result_has_no_traceback(caplog):
    jobs = JobManager()
    with caplog.at_level("ERROR", logger="activelabel.jobs"):
        job = wait(jobs.submit(1, "demo", lambda progress: int("x")))
    assert job.status == FAILED and "ValueError" not in job.error and "invalid literal" in job.error
    assert job.result == {}  # nothing technical reaches the interface
    assert "ValueError" in caplog.text  # the traceback goes to the log


def test_a_coded_error_keeps_its_code_and_params_in_the_job():
    def fail(progress):
        raise ToolError("CVAT error 500.", "cvat_error", status=500)

    job = wait(JobManager().submit(1, "demo", fail))
    assert (job.error, job.error_code, job.error_params) == ("CVAT error 500.", "cvat_error", {"status": 500})
    assert job.as_dict()["error_code"] == "cvat_error"
    plain = wait(JobManager().submit(1, "demo", lambda progress: 1 / 0))
    assert (plain.error_code, plain.error_params) == ("", {})


def test_error_text_has_no_quotes_from_a_key_error():
    assert readable_error(KeyError("The image is gone")) == "The image is gone"
    assert message_of(ValueError("A project named 'x'")) == "A project named 'x'"
    assert message_of(KeyError(7)) == "7"
    assert readable_error(RuntimeError()) == "The activity failed (no details were given)."


class ServerSelectionTimeoutError(Exception):
    """Same name as pymongo's error when the database does not answer."""


def test_database_failure_recognises_the_fiftyone_database_errors():
    assert database_failure(ServerSelectionTimeoutError("Timeout: 30s"))
    assert database_failure(OSError("mongod exited"))
    assert not database_failure(ValueError("x"))


# -- the global queue ------------------------------------------------------------------


class Gate:
    """A job function that records its start and holds until released."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log, self.name = log, name
        self.started, self.release = threading.Event(), threading.Event()

    def __call__(self, progress):
        self.log.append(self.name)
        self.started.set()
        self.release.wait(5)
        return {"name": self.name}


def never(progress):
    raise AssertionError("a cancelled job must not run")


def test_a_job_has_the_light_lane_by_default_and_the_device_it_was_given():
    jobs = JobManager()
    light = wait(jobs.submit(1, "demo", lambda progress: {}))
    heavy = wait(jobs.submit(1, "demo", lambda progress: {}, device="cuda:1"))
    assert (light.device, heavy.device) == ("light", "cuda:1")


def test_managers_that_share_a_scheduler_wait_in_the_same_lane():
    shared = Scheduler()
    first, second = JobManager(scheduler=shared), JobManager(scheduler=shared)
    log: list[str] = []
    a, b = Gate(log, "a"), Gate(log, "b")
    job_a = first.submit(1, "x", a, device=CPU)
    job_b = second.submit(2, "y", b, device=CPU)
    assert a.started.wait(2)
    time.sleep(0.1)
    assert log == ["a"] and job_b.status == QUEUED and shared.position(job_b.id) == 1
    a.release.set()
    assert b.started.wait(2)
    b.release.set()
    assert wait(job_a).status == DONE and wait(job_b).status == DONE and log == ["a", "b"]


def test_different_lanes_of_a_shared_scheduler_run_together():
    shared = Scheduler()
    first, second = JobManager(scheduler=shared), JobManager(scheduler=shared)
    log: list[str] = []
    a, b = Gate(log, "a"), Gate(log, "b")
    job_a = first.submit(1, "x", a, device="cuda:0")
    job_b = second.submit(1, "y", b, device=CPU)
    assert a.started.wait(2) and b.started.wait(2)
    assert job_a.status == RUNNING and job_b.status == RUNNING
    a.release.set()
    b.release.set()
    wait(job_a)
    wait(job_b)


def test_cancelling_a_queued_job_cancels_it_at_once_and_it_never_runs():
    jobs = JobManager()
    log: list[str] = []
    blocker = Gate(log, "blocker")
    running = jobs.submit(1, "x", blocker, device=CPU)
    queued = jobs.submit(1, "y", never, device=CPU)
    assert blocker.started.wait(2)
    assert jobs.cancel(queued.id).status == CANCELLED  # no waiting for the lane to free up
    assert queued.finished and jobs.scheduler.position(queued.id) is None
    blocker.release.set()
    assert wait(running).status == DONE
    assert jobs.active(1) == [] and queued.status == CANCELLED and queued.started == ""


def test_cancelling_a_job_that_is_starting_stops_it_before_it_runs(monkeypatch):
    jobs = JobManager()
    log: list[str] = []
    blocker = Gate(log, "blocker")
    running = jobs.submit(1, "x", blocker, device=CPU)
    queued = jobs.submit(1, "y", never, device=CPU)
    assert blocker.started.wait(2)
    monkeypatch.setattr(jobs.scheduler, "discard", lambda job: False)  # the scheduler had just started it
    answer = jobs.cancel(queued.id)
    assert answer.status == QUEUED and answer.message.startswith("Cancelling")
    blocker.release.set()
    assert wait(running).status == DONE and wait(queued).status == CANCELLED and queued.result == {}


def test_cancelling_a_finished_job_changes_nothing():
    jobs = JobManager()
    job = wait(jobs.submit(1, "x", lambda progress: {"n": 1}))
    assert jobs.cancel(job.id).status == DONE
    with pytest.raises(KeyError):
        jobs.cancel("nothing")


def test_shutdown_keeps_what_was_queued_in_queue_json(tmp_path):
    path = tmp_path / "queue.json"
    jobs = JobManager(scheduler=Scheduler(path, Limits(cpu=1)))
    blocker = Gate([], "blocker")
    running = jobs.submit(1, "training", blocker, {"number": 1}, device=CPU)
    waiting = jobs.submit(1, "training", never, {"number": 2}, device=CPU)
    assert blocker.started.wait(2)
    before = json.loads(path.read_text(encoding="utf-8"))
    jobs.shutdown()
    blocker.release.set()
    wait(running)
    time.sleep(0.1)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after == before
    assert [(e["id"], e["status"]) for e in after["jobs"]] == [
        (running.id, "running"),
        (waiting.id, "queued"),
    ]
    assert waiting.status == QUEUED  # it never started
