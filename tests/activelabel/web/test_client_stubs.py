import pytest
from api_client import make_client

from activelabel.training.runner import TrainingRunner


def test_the_shared_client_never_starts_runs_or_pre_annotations(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test started a real run")

    monkeypatch.setattr(TrainingRunner, "enqueue", refuse)
    monkeypatch.setattr(TrainingRunner, "enqueue_preannotation", refuse)
    _, state = make_client(tmp_path)
    assert state.runner.enqueue(1, 1) is None
    assert state.runner.enqueue_preannotation(1, 1) is None
    with pytest.raises(AssertionError):  # a runner built by hand is still the real one
        TrainingRunner(tmp_path / "ws", state.store.dir, lambda: [1]).enqueue_preannotation(1, 1)
