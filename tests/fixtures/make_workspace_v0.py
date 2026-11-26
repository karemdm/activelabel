"""Build a small legacy workspace ("version 0") with the stores of the current code.

The workspace stands for what a user already has on disk. ``tests/rules/test_legacy_workspace.py`` opens it
with the code under test, so a change that stops reading old data fails there. Images are generated here:
nothing binary is committed. The current stores write ``format_version``; it is removed at the end, as
version 0 had none.

It holds one project per type (names with accents and spaces) with images, tasks and jobs, annotations, a
curation analysis, a data split version, a finished training run with a tiny fake weights file, a saved
model, a public model card, a fake CVAT link and a fake open FiftyOne session.

Run it by hand with ``python tests/fixtures/make_workspace_v0.py <empty folder>``.
When the format of the stored data changes on purpose, keep this script building the OLD format (do not
update it to the new one): that is what the test protects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "helpers"), str(HERE.parents[1] / "src")]

from api_client import make_client, upload  # noqa: E402
from sample_images import make_image  # noqa: E402

from activelabel.annotate import sessions  # noqa: E402
from activelabel.fiftyone_names import record_name  # noqa: E402
from activelabel.labels import AnnotationSet, Label, save_annotations  # noqa: E402
from activelabel.storage import FORMAT_KEY, write_json  # noqa: E402
from activelabel.sync.links import new_link, save_link  # noqa: E402
from activelabel.tasks import Task  # noqa: E402
from activelabel.training.config import RunConfig  # noqa: E402
from activelabel.training.models import ModelStore  # noqa: E402
from activelabel.training.runs import RunStore  # noqa: E402

IMAGES = 8
TYPES: dict[str, tuple[str, list[str]]] = {
    "classification": ("Classificação de flores", ["rosa vermelha", "margarida"]),
    "detection": ("Detecção de pessoas e carros", ["pessoa", "carro"]),
    "segmentation": ("Segmentação de céu e chão", ["céu", "chão"]),
    "keypoints": ("Pontos da mão — teste", ["mão"]),
}


def _label(kind: str, name: str, k: int) -> Label:
    """One label of the shape the project type needs."""
    shift = 0.05 * (k % 5)
    if kind == "classification":
        return Label(name)
    if kind == "detection":
        return Label(name, bbox=[0.1 + shift, 0.1, 0.3, 0.4])
    if kind == "segmentation":
        return Label(name, polygons=[[[0.1 + shift, 0.1], [0.6, 0.1], [0.6, 0.7], [0.1, 0.7]]])
    return Label(name, keypoints=[[0.2 + shift, 0.2, 2], [0.5, 0.4, 2], [0.7, 0.8, 1]])


def _wait_for_job(client: Any, project_id: int, job_id: str) -> None:
    end = time.time() + 60
    while time.time() < end:
        jobs = client.get(f"/api/projects/{project_id}/jobs").json()
        if next(j for j in jobs if j["id"] == job_id)["status"] in ("done", "failed", "cancelled"):
            return
        time.sleep(0.05)
    raise TimeoutError("the analysis did not finish")


def _add_project(client: Any, state: Any, source: Path, kind: str) -> int:
    """A project with images (some in a task with jobs), annotations and a curation analysis."""
    name, classes = TYPES[kind]
    project = client.post("/api/projects", json={"name": name, "task": kind, "classes": classes}).json()
    pid = project["id"]
    paths = [make_image(source / kind / f"foto {k} ação.png", seed=k, fmt="PNG") for k in range(IMAGES)]
    names = [p.name for p in paths]
    assert upload(client, pid, paths[:4], names[:4]).status_code == 200
    files = [("files", (p.name, p.read_bytes())) for p in paths[4:]]
    task = client.post(
        f"/api/projects/{pid}/tasks", files=files, data={"name": "Tarefa número 1", "job_size": "2"}
    )
    assert task.status_code == 200, task.text
    ids = [r.id for r in state.manifest(pid).all()]
    images = {i: [_label(kind, classes[k % len(classes)], k)] for k, i in enumerate(ids)}
    save_annotations(state.store.dir(pid), AnnotationSet(Task(kind), classes, images, {}))
    job = client.post(f"/api/projects/{pid}/curation/analyze", json={}).json()
    _wait_for_job(client, pid, job["id"])
    assert client.post(f"/api/projects/{pid}/datasets", json={}).status_code == 200
    return int(pid)


def _add_finished_run(state: Any, project_id: int) -> None:
    """A finished run with a tiny fake weights file, saved as model 1 (the way an old version left it)."""
    runs = RunStore(state.store.dir(project_id))
    run = runs.create(RunConfig(dataset=1, model="resnet18", params={"epochs": 2}, metric="accuracy"))
    folder = runs.path(run["number"])
    (folder / "best.pt").write_bytes(b"fake weights")
    (folder / "last.pt").write_bytes(b"fake weights")
    (folder / "log.txt").write_text("epoch 1\nepoch 2\n", encoding="utf-8")
    (folder / "metrics.jsonl").write_text(
        '{"event": "epoch", "epoch": 1, "train_loss": 1.0, "val_accuracy": 0.5}\n'
        '{"event": "epoch", "epoch": 2, "train_loss": 0.5, "val_accuracy": 0.75}\n',
        encoding="utf-8",
    )
    write_json(folder / "results.json", {"best_epoch": 2, "val": {"accuracy": 0.75}, "test": None})
    runs.update(
        run["number"],
        status="done",
        started="2026-01-01T10:00:00+00:00",
        finished="2026-01-01T10:05:00+00:00",
        epochs=2,
        epoch=2,
        best={"epoch": 2, "value": 0.75, "best_fold_value": None},
        classes=TYPES["classification"][1],
        device={"cuda": False, "torch": "2.0.0"},
        packages={"torch": "2.0.0"},
    )
    ModelStore(state.store.dir(project_id)).save_run(run["number"], "primeiro modelo")


def _add_public_model(state: Any, project_id: int) -> None:
    classes = TYPES["detection"][1]
    ModelStore(state.store.dir(project_id)).add_public(
        "coco_fasterrcnn_mobilenet_v3", {"person": "pessoa", "car": "carro"}, classes, "modelo público"
    )


def _add_remote_traces(state: Any, project_id: int) -> None:
    """A link with CVAT (nothing is contacted) and an open FiftyOne session."""
    project_dir = state.store.dir(project_id)
    images = [r.id for r in state.manifest(project_id).all()]
    items = {i: {"remote_id": str(100 + k), "key": f"foto {k} ação.png"} for k, i in enumerate(images)}
    save_link(project_dir, "cvat", new_link("http://cvat.invalid", "7", "Projeto CVAT ação", items))
    record_name(project_dir, f"activelabel-{project_id}-detecção")
    sessions.create(
        project_dir, "fiftyone", "all", images, [], {"dataset": f"activelabel-{project_id}-detecção"}
    )


def _as_version_0(workspace: Path) -> None:
    """Remove ``format_version`` from every file: version 0 had no such field (format.json did not exist)."""
    for path in workspace.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and FORMAT_KEY in data:
            del data[FORMAT_KEY]
            text = json.dumps(data, ensure_ascii=False, indent=1)
            path.write_text(text, encoding="utf-8", newline="\n")
    assert not (workspace / "format.json").exists()


def build(base: Path) -> Path:
    """Build the workspace in ``base / "ws"`` and return it (``base`` must be empty or new)."""
    client, state = make_client(base)
    source = base / "sources"
    ids = {kind: _add_project(client, state, source, kind) for kind in TYPES}
    _add_finished_run(state, ids["classification"])
    _add_public_model(state, ids["detection"])
    _add_remote_traces(state, ids["detection"])
    state.jobs.shutdown()
    _as_version_0(base / "ws")
    return base / "ws"


def main() -> None:
    """Command line: build the workspace in the folder given."""
    base = Path(sys.argv[1]).resolve()
    print(build(base))


if __name__ == "__main__":
    main()
