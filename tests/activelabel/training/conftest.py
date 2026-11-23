import pytest
from api_client import make_client, upload
from sample_images import make_image
from training_fakes import FAKE_WORKER, FakeEnv

from activelabel.labels import AnnotationSet, Label, save_annotations
from activelabel.tasks import Task
from activelabel.training.runner import TrainingRunner


@pytest.fixture
def setup(tmp_path):
    client, state = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["a", "b"]})
    paths = [make_image(tmp_path / "src" / f"{k}.png", seed=k, fmt="PNG") for k in range(20)]
    assert upload(client, 1, paths).status_code == 200
    ids = [r.id for r in state.manifest(1).all()]
    images = {i: [Label("a" if k % 2 else "b")] for k, i in enumerate(ids)}
    save_annotations(state.store.dir(1), AnnotationSet(Task.CLASSIFICATION, ["a", "b"], images, {}))
    assert client.post("/api/projects/1/datasets", json={}).status_code == 200
    assert client.post("/api/projects/1/datasets", json={"kfold": 2}).status_code == 200
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    runner = TrainingRunner(
        tmp_path / "ws",
        state.store.dir,
        lambda: [1],
        environment=FakeEnv(tmp_path / "ws"),
        scripts=lambda kind: script,
        poll=0.05,
    )
    state.runner = runner
    yield client, state, runner
    runner.shutdown()
