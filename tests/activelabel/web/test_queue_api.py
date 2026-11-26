"""The work queue API, and the jobs of the web layer that survive a restart."""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from api_client import make_client, upload
from fastapi.testclient import TestClient
from sample_images import make_image
from training_fakes import wait_for

from activelabel.config import Settings
from activelabel.jobs import Job, Progress
from activelabel.scheduler import CPU
from activelabel.storage import write_json
from activelabel.web import api, video_api
from activelabel.web.app import create_app
from activelabel.web.state import AppState

ITEM_FIELDS = {
    "id",
    "project_id",
    "project_name",
    "kind",
    "device",
    "params",
    "status",
    "position",
    "done",
    "total",
    "message",
    "created",
    "started",
    "eta_seconds",
    "wait_seconds",
}


def add_project(client: TestClient, tmp_path: Path, name: str, seed: int) -> int:
    """Create a classification project with one image; returns its id."""
    project = client.post("/api/projects", json={"name": name, "task": "classification"}).json()
    image = make_image(tmp_path / f"{name}.png", seed=seed)
    assert upload(client, project["id"], [image]).status_code == 200
    return int(project["id"])


@pytest.fixture
def lane() -> Iterator[threading.Event]:
    """Gate that releases the job holding the processor lane (always opened at the end)."""
    gate = threading.Event()
    yield gate
    gate.set()


def hold_cpu(state: AppState, gate: threading.Event, project_id: int = 0) -> Job:
    """Occupy the processor lane with a job that waits for ``gate``; returns it once it runs."""

    def work(progress: Progress) -> dict[str, Any]:
        gate.wait(30)
        return {}

    job = state.jobs.submit(project_id, "blocker", work, device=CPU)
    wait_for(lambda: job.status == "running")
    return job


def saved(project_id: int, kind: str, status: str, params: dict[str, Any], job_id: str) -> dict[str, Any]:
    """An entry as ``queue.json`` keeps it."""
    return {
        "id": job_id,
        "project_id": project_id,
        "kind": kind,
        "device": "cpu",
        "params": params,
        "status": status,
        "created": "2026-01-01T00:00:00",
        "started": "2026-01-01T00:00:01" if status == "running" else "",
    }


def test_queue_lists_every_project_or_one_with_names(tmp_path: Path, lane: threading.Event) -> None:
    client, state = make_client(tmp_path)
    first = add_project(client, tmp_path, "Cats", 1)
    second = add_project(client, tmp_path, "Dogs", 2)
    blocker = hold_cpu(state, lane, first)
    a = client.post(f"/api/projects/{first}/curation/analyze", json={"recompute": True}).json()
    b = client.post(f"/api/projects/{second}/curation/analyze", json={}).json()
    other = state.jobs.submit(0, "import", lambda progress: None)  # light lane, no project
    ghost = state.jobs.submit(99, "ghost", lambda progress: None, device=CPU)  # a deleted project

    queue = client.get("/api/queue").json()
    assert set(queue) == {"limits", "lanes", "items", "interrupted"}
    assert queue["limits"] == {"cpu": 1, "gpu": 1, "light": 1}
    by_id = {i["id"]: i for i in queue["items"]}
    assert {blocker.id, a["id"], b["id"], ghost.id} <= set(by_id)
    assert all(ITEM_FIELDS <= set(i) for i in queue["items"])
    assert (by_id[a["id"]]["project_name"], by_id[b["id"]]["project_name"]) == ("Cats", "Dogs")
    assert by_id[ghost.id]["project_name"] == "" and by_id[ghost.id]["project_id"] == 99
    assert by_id[blocker.id]["status"] == "running" and by_id[blocker.id]["position"] is None
    first_item = by_id[a["id"]]
    assert (first_item["status"], first_item["position"], first_item["device"]) == ("queued", 1, "cpu")
    assert first_item["kind"] == "analyze" and first_item["params"] == {"recompute": True}
    assert by_id[b["id"]]["position"] == 2 and by_id[b["id"]]["params"] == {"recompute": False}
    cpu = next(info for info in queue["lanes"] if info["device"] == "cpu")
    assert cpu["limit"] == 1 and cpu["running"] == 1 and cpu["queued"] == 3
    if other.id in by_id:  # a light job may already have run: if it is listed, it has no project name
        assert by_id[other.id]["project_name"] == ""

    only = client.get("/api/queue", params={"project": second}).json()
    assert [i["id"] for i in only["items"]] == [b["id"]]
    assert only["items"][0]["project_name"] == "Dogs"
    assert only["lanes"] == queue["lanes"]  # the lanes still count every project


def test_cancel_a_queued_item_returns_the_whole_queue(tmp_path: Path, lane: threading.Event) -> None:
    client, state = make_client(tmp_path)
    first = add_project(client, tmp_path, "Cats", 1)
    second = add_project(client, tmp_path, "Dogs", 2)
    blocker = hold_cpu(state, lane, first)
    a = client.post(f"/api/projects/{first}/curation/analyze", json={}).json()
    b = client.post(f"/api/projects/{second}/curation/analyze", json={}).json()

    response = client.post(f"/api/queue/{a['id']}/cancel")
    assert response.status_code == 200
    items = response.json()["items"]
    assert a["id"] not in [i["id"] for i in items] and {blocker.id, b["id"]} <= {i["id"] for i in items}
    assert next(i for i in items if i["id"] == b["id"])["position"] == 1
    listed = client.get(f"/api/projects/{first}/jobs").json()
    assert next(j for j in listed if j["id"] == a["id"])["status"] == "cancelled"


def test_cancel_an_unknown_id_is_a_404_with_a_code(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    response = client.post("/api/queue/nothing/cancel")
    assert response.status_code == 404
    assert response.json()["code"] == "activity_not_found"


def test_cancel_asks_the_runner_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state = make_client(tmp_path)
    asked: list[str] = []
    monkeypatch.setattr(state.runner, "cancel_job", lambda job_id: asked.append(job_id) or True)
    assert client.post("/api/queue/abc/cancel").status_code == 200
    assert asked == ["abc"]


def test_the_queue_and_the_runner_share_one_scheduler(tmp_path: Path) -> None:
    _, state = make_client(tmp_path)
    assert state.runner.scheduler is state.scheduler is state.jobs.scheduler
    assert state.scheduler.path == tmp_path / "ws" / "queue.json"
    assert state.scheduler.limits.of("cpu") == 1


def test_the_settings_set_the_lane_limits(tmp_path: Path) -> None:
    state = AppState(Settings(workspace=tmp_path / "ws", queue_cpu=3, queue_gpu=2))
    assert (state.scheduler.limits.cpu, state.scheduler.limits.gpu) == (3, 2)


def test_an_injected_job_manager_brings_its_scheduler(tmp_path: Path) -> None:
    other = AppState(Settings(workspace=tmp_path / "one"))
    state = AppState(Settings(workspace=tmp_path / "two"), jobs=other.jobs)
    assert state.scheduler is other.scheduler


def test_restart_queues_the_saved_analysis_again(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    project = add_project(client, tmp_path, "Cats", 1)
    write_json(
        state.scheduler.path,
        {
            "jobs": [
                saved(project, "analyze", "queued", {"recompute": True}, "old1"),
                saved(77, "analyze", "queued", {"recompute": False}, "old2"),  # the project is gone
                saved(project, "unknown_kind", "queued", {}, "old3"),
            ]
        },
    )
    with TestClient(create_app(Settings(workspace=tmp_path / "ws"))) as restarted:
        jobs = wait_for(
            lambda: [
                j for j in restarted.get(f"/api/projects/{project}/jobs").json() if j["status"] == "done"
            ]
        )
        assert [j["kind"] for j in jobs] == ["analyze"]
        assert jobs[0]["params"] == {"recompute": True} and jobs[0]["device"] == "cpu"
        assert jobs[0]["result"]["measured"] == 1
        assert restarted.get("/api/projects/77/jobs").status_code == 404
        assert restarted.get("/api/queue").json()["interrupted"] == []


def test_restart_lists_what_was_running_as_interrupted(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    project = add_project(client, tmp_path, "Cats", 1)
    write_json(
        state.scheduler.path,
        {"jobs": [saved(project, "analyze", "running", {"recompute": False}, "gone1")]},
    )
    with TestClient(create_app(Settings(workspace=tmp_path / "ws"))) as restarted:
        queue = restarted.get("/api/queue").json()
        assert queue["items"] == []
        assert [(e["id"], e["status"], e["project_name"]) for e in queue["interrupted"]] == [
            ("gone1", "interrupted", "Cats")
        ]
        assert restarted.get("/api/queue", params={"project": project + 1}).json()["interrupted"] == []


def test_restore_analyze_skips_a_project_without_images(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    empty = client.post("/api/projects", json={"name": "Empty", "task": "classification"}).json()["id"]
    api.restore_analyze(state, saved(empty, "analyze", "queued", {"recompute": False}, "x"))
    api.restore_analyze(state, saved(404, "analyze", "queued", {}, "y"))
    assert state.scheduler.snapshot()["items"] == []


def test_restore_video_task_skips_what_is_gone(tmp_path: Path) -> None:
    client, state = make_client(tmp_path)
    project = add_project(client, tmp_path, "Cats", 1)
    request = {"video": "nope", "start": 0, "stop": None, "step": 1, "every_seconds": None}
    video_api.restore_video_task(state, saved(project, "video_task", "queued", {"request": request}, "v1"))
    video_api.restore_video_task(state, saved(404, "video_task", "queued", {"request": request}, "v2"))
    video_api.restore_video_task(state, saved(project, "video_task", "queued", {"request": 5}, "v3"))
    assert state.scheduler.snapshot()["items"] == []
