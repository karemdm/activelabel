import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

from activelabel.scheduler import (
    CPU,
    LIGHT,
    Limits,
    Scheduler,
    device_lane,
    remaining_seconds,
    seconds_between,
)
from activelabel.storage import now


@dataclass
class FakeJob:
    id: str
    device: str = CPU
    project_id: int = 1
    kind: str = "training"
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    done: int = 0
    total: int = 0
    message: str = ""
    created: str = field(default_factory=now)
    started: str = ""


class Probe:
    """Records which jobs started, and holds each one until it is released."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.started: list[str] = []
        self.gates: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def add(self, job: FakeJob) -> FakeJob:
        self.gates[job.id] = threading.Event()
        self.scheduler.add(job, partial(self._run, job))
        return job

    def _run(self, job: FakeJob) -> None:
        with self.lock:
            self.started.append(job.id)
        self.gates[job.id].wait(5)
        self.scheduler.finished(job)

    def release(self, *ids: str) -> None:
        for job_id in ids:
            self.gates[job_id].set()

    def wait_started(self, count: int) -> list[str]:
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with self.lock:
                if len(self.started) >= count:
                    return list(self.started)
            time.sleep(0.01)
        raise AssertionError(f"only {self.started} started")

    def settle(self) -> list[str]:
        time.sleep(0.1)  # nothing else is expected to start
        with self.lock:
            return list(self.started)


def boom() -> None:
    raise RuntimeError("listener failed")


def test_limits_of_each_lane_is_at_least_one():
    limits = Limits(cpu=2, gpu=3, light=0)
    assert (limits.of(CPU), limits.of("cuda:1"), limits.of("mps"), limits.of(LIGHT)) == (2, 3, 3, 1)
    assert Limits(cpu=-4, gpu=0).of(CPU) == 1 and Limits(gpu=0).of("cuda:0") == 1


def test_device_lane_resolves_auto():
    assert device_lane("auto", gpus=2) == "cuda:0"
    assert device_lane("", gpus=1, mps=True) == "cuda:0"
    assert device_lane("auto", mps=True) == "mps"
    assert device_lane("auto") == CPU and device_lane("") == CPU
    assert device_lane("cuda:1", gpus=0) == "cuda:1" and device_lane("cpu", gpus=4) == "cpu"


def test_seconds_between_and_remaining_seconds():
    assert seconds_between("2026-01-01T00:00:00+00:00", "2026-01-01T00:01:30+00:00") == 90.0
    assert seconds_between("2026-01-01T00:01:30+00:00", "2026-01-01T00:00:00+00:00") == 0.0
    assert seconds_between("", "x") == 0.0
    assert seconds_between("2026-01-01T00:00:00", "2026-01-01T00:00:05+00:00") == 0.0  # naive vs aware
    running = FakeJob("a", status="running", done=1, total=4, started="2026-01-01T00:00:10+00:00")
    assert remaining_seconds(running, "2026-01-01T00:00:20+00:00") == 30.0
    assert remaining_seconds(FakeJob("b", status="running", total=4, started="x"), "y") is None
    assert remaining_seconds(FakeJob("c", status="queued", done=1, total=4, started="x"), "y") is None
    assert remaining_seconds(FakeJob("d", status="running", done=1, total=0, started="x"), "y") is None
    assert remaining_seconds(FakeJob("e", status="running", done=1, total=4), "y") is None


def test_jobs_of_a_lane_start_in_the_order_they_were_added():
    probe = Probe(Scheduler())
    for name in "abc":
        probe.add(FakeJob(name))
    assert probe.wait_started(1) == ["a"]
    assert probe.settle() == ["a"]
    probe.release("a")
    assert probe.wait_started(2) == ["a", "b"]
    probe.release("b")
    assert probe.wait_started(3) == ["a", "b", "c"]
    probe.release("c")


def test_a_lane_runs_as_many_as_its_limit():
    probe = Probe(Scheduler(limits=Limits(cpu=2)))
    for name in "abc":
        probe.add(FakeJob(name))
    assert sorted(probe.wait_started(2)) == ["a", "b"]
    assert probe.settle() == ["a", "b"]
    probe.release("a")
    assert probe.wait_started(3)[-1] == "c"
    probe.release("b", "c")


def test_lanes_are_independent():
    probe = Probe(Scheduler())
    probe.add(FakeJob("gpu", device="cuda:0"))
    probe.add(FakeJob("cpu"))
    probe.add(FakeJob("gpu2", device="cuda:0"))
    probe.add(FakeJob("other", device="cuda:1"))
    assert sorted(probe.wait_started(3)) == ["cpu", "gpu", "other"]
    assert "gpu2" not in probe.settle()
    probe.release("gpu", "cpu", "other", "gpu2")


def test_a_later_job_never_jumps_ahead_inside_a_lane():
    probe = Probe(Scheduler(limits=Limits(cpu=2)))
    for name in "abcd":
        probe.add(FakeJob(name))
    probe.wait_started(2)
    probe.release("b")  # b ends first: c starts, d keeps waiting for its turn
    assert probe.wait_started(3)[2] == "c"
    assert probe.settle() == ["a", "b", "c"]
    probe.release("a", "c")
    assert probe.wait_started(4)[3] == "d"
    probe.release("d")


def test_discard_removes_a_waiting_job_only():
    probe = Probe(Scheduler())
    running, waiting, last = FakeJob("a"), FakeJob("b"), FakeJob("c")
    for job in (running, waiting, last):
        probe.add(job)
    probe.wait_started(1)
    assert probe.scheduler.discard(running) is False  # already running
    assert probe.scheduler.discard(FakeJob("zzz")) is False
    assert probe.scheduler.discard(waiting) is True
    assert probe.scheduler.position("b") is None and probe.scheduler.position("c") == 1
    probe.release("a")
    assert probe.wait_started(2) == ["a", "c"]
    probe.release("c")


def test_position_counts_inside_the_lane_of_the_job():
    probe = Probe(Scheduler())
    for job in (FakeJob("a"), FakeJob("g", device="cuda:0"), FakeJob("b"), FakeJob("h", device="cuda:0")):
        probe.add(job)
    probe.wait_started(2)
    scheduler = probe.scheduler
    assert scheduler.position("b") == 1 and scheduler.position("h") == 1
    assert scheduler.position("a") is None and scheduler.position("nothing") is None
    probe.add(FakeJob("c"))
    assert scheduler.position("c") == 2
    probe.release("a", "g", "b", "h", "c")


def test_snapshot_shape_and_filter():
    scheduler = Scheduler(limits=Limits(cpu=1, gpu=2))
    probe = Probe(scheduler)
    started = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(timespec="seconds")
    probe.add(FakeJob("a", project_id=1, status="running", done=1, total=4, started=started))
    probe.add(FakeJob("b", project_id=2, params={"number": 7}))
    probe.add(FakeJob("c", project_id=1, device="cuda:0"))
    probe.wait_started(2)
    snap = scheduler.snapshot()
    assert snap["limits"] == {"cpu": 1, "gpu": 2, "light": 1}
    lanes = {lane["device"]: lane for lane in snap["lanes"]}
    assert lanes[CPU] == {"device": CPU, "limit": 1, "running": 1, "queued": 1}
    assert lanes["cuda:0"] == {"device": "cuda:0", "limit": 2, "running": 1, "queued": 0}
    items = {i["id"]: i for i in snap["items"]}
    assert items["a"]["status"] == "running" and items["a"]["position"] is None
    assert items["a"]["eta_seconds"] is not None and items["a"]["eta_seconds"] > 0
    assert items["b"]["status"] == "queued" and items["b"]["position"] == 1
    assert items["b"]["params"] == {"number": 7} and items["b"]["project_id"] == 2
    assert items["b"]["wait_seconds"] is not None and items["b"]["wait_seconds"] > 0
    assert snap["interrupted"] == []
    mine = scheduler.snapshot(1)
    assert {i["id"] for i in mine["items"]} == {"a", "c"}
    assert {lane["device"]: lane["queued"] for lane in mine["lanes"]}[CPU] == 1  # lanes count every project
    probe.release("a", "b", "c")


def test_wait_estimate_is_unknown_without_progress_or_on_a_wide_lane():
    probe = Probe(Scheduler())
    probe.add(FakeJob("a"))  # running without any progress yet
    probe.add(FakeJob("b"))
    probe.wait_started(1)
    items = {i["id"]: i for i in probe.scheduler.snapshot()["items"]}
    assert items["a"]["eta_seconds"] is None and items["b"]["wait_seconds"] is None
    probe.release("a", "b")
    wide = Probe(Scheduler(limits=Limits(cpu=2)))
    wide.add(FakeJob("x", status="running", done=1, total=2, started="2026-01-01T00:00:00+00:00"))
    wide.add(FakeJob("y"))
    wide.add(FakeJob("z"))
    wide.wait_started(2)
    assert {i["id"]: i for i in wide.scheduler.snapshot()["items"]}["z"]["wait_seconds"] is None
    wide.release("x", "y", "z")


def test_listeners_are_called_and_a_failing_one_does_not_break_the_queue():
    scheduler = Scheduler()
    calls: list[int] = []
    scheduler.listen(boom)
    scheduler.listen(partial(calls.append, 1))
    probe = Probe(scheduler)
    probe.add(FakeJob("a"))
    probe.wait_started(1)
    assert len(calls) >= 1
    probe.release("a")
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) >= 2  # also after finished(), even though the first listener raised


def test_close_stops_dispatch_and_writing(tmp_path):
    path = tmp_path / "queue.json"
    scheduler = Scheduler(path)
    probe = Probe(scheduler)
    probe.add(FakeJob("a"))
    probe.add(FakeJob("b"))
    probe.wait_started(1)
    before = path.read_text(encoding="utf-8")
    scheduler.close()
    probe.release("a")
    assert probe.settle() == ["a"]  # b never starts
    probe.add(FakeJob("c"))  # ignored after close
    assert probe.settle() == ["a"] and scheduler.position("c") is None
    assert path.read_text(encoding="utf-8") == before  # still lists a running and b queued
    probe.release("b", "c")


def test_queue_json_lists_running_and_queued_in_order(tmp_path):
    path = tmp_path / "queue.json"
    probe = Probe(Scheduler(path))
    probe.add(FakeJob("a", params={"number": 1}))
    probe.add(FakeJob("b", device="cuda:0"))
    probe.add(FakeJob("c"))
    probe.wait_started(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format_version"] >= 1
    assert [(e["id"], e["status"]) for e in data["jobs"]] == [
        ("a", "running"),
        ("b", "running"),
        ("c", "queued"),
    ]
    assert data["jobs"][0]["params"] == {"number": 1} and data["jobs"][2]["device"] == CPU
    probe.release("a", "b", "c")


def test_saved_is_read_once_and_running_entries_are_interrupted(tmp_path):
    path = tmp_path / "queue.json"
    entries = [
        {"id": "r", "project_id": 1, "kind": "training", "device": "cpu", "status": "running", "params": {}},
        {"id": "q", "project_id": 2, "kind": "export", "device": "cpu", "status": "queued", "params": {}},
        "junk",
    ]
    path.write_text(json.dumps({"jobs": entries}), encoding="utf-8")
    scheduler = Scheduler(path)
    assert [e["id"] for e in scheduler.saved()] == ["r", "q"]
    path.write_text(json.dumps({"jobs": []}), encoding="utf-8")  # a second read does not look at the file
    assert [e["id"] for e in scheduler.saved()] == ["r", "q"]
    snap = scheduler.snapshot()
    assert [(e["id"], e["status"]) for e in snap["interrupted"]] == [("r", "interrupted")]
    assert scheduler.snapshot(2)["interrupted"] == [] and len(scheduler.snapshot(1)["interrupted"]) == 1


def test_saved_without_a_path_or_with_a_damaged_file_is_empty(tmp_path):
    assert Scheduler().saved() == []
    damaged = tmp_path / "queue.json"
    damaged.write_text("{not json", encoding="utf-8")
    scheduler = Scheduler(damaged)
    assert scheduler.saved() == [] and scheduler.snapshot()["interrupted"] == []
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"jobs": "nope"}), encoding="utf-8")
    assert Scheduler(wrong).saved() == []
    assert Scheduler(tmp_path / "missing.json").saved() == []


def test_a_write_failure_is_logged_and_the_queue_goes_on(tmp_path):
    directory = tmp_path / "queue.json"
    directory.mkdir()  # cannot be written as a file
    probe = Probe(Scheduler(directory))
    probe.add(FakeJob("a"))
    assert probe.wait_started(1) == ["a"]
    probe.release("a")
