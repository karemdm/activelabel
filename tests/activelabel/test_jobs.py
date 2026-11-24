import threading
import time

from activelabel.jobs import CANCELLED, DONE, FAILED, JobManager


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
    from activelabel.jobs import readable_error

    class ServiceListenTimeout(Exception):
        pass

    message = readable_error(ServiceListenTimeout("DatabaseService failed to bind to port"))
    assert message.startswith("The FiftyOne database did not respond") and "ServiceListenTimeout" in message
    assert readable_error(ValueError("x")) == "x"


def test_error_message_has_no_class_name_but_traceback_keeps_it():
    jobs = JobManager()
    job = wait(jobs.submit(1, "demo", lambda progress: int("x")))
    assert job.status == FAILED and "ValueError" not in job.error and "invalid literal" in job.error
    assert "ValueError" in job.result["traceback"]


def test_error_text_has_no_quotes_from_a_key_error():
    from activelabel.jobs import readable_error
    from activelabel.textcheck import message_of

    assert readable_error(KeyError("The image is gone")) == "The image is gone"
    assert message_of(ValueError("A project named 'x'")) == "A project named 'x'"
    assert message_of(KeyError(7)) == "7"
    assert readable_error(RuntimeError()) == "The activity failed (no details were given)."
