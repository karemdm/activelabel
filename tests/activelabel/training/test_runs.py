import pytest

from activelabel.training.config import RunConfig
from activelabel.training.runs import RunBusy, RunNotFound, RunStore


def _cfg(**kw):
    return RunConfig(**{"dataset": 1, "model": "resnet18", **kw})


def test_run_store_numbers_and_files(tmp_path):
    store = RunStore(tmp_path)
    assert store.list() == []
    first = store.create(_cfg())
    assert first["number"] == 1 and first["status"] == "queued" and first["children"] == []
    assert (tmp_path / "runs" / "r1" / "config.yaml").exists()
    assert store.config(1) == _cfg()
    store.create(_cfg())
    store.delete(2)
    assert store.create(_cfg())["number"] == 3  # never reused
    assert [r["number"] for r in store.list()] == [3, 1]

    store.update(1, status="running", started="t")
    assert store.get(1)["status"] == "running" and store.get(1)["created"] == first["created"]
    with pytest.raises(RunBusy):
        store.delete(1)
    with pytest.raises(ValueError):
        store.update(1, status="weird")
    store.update(1, status="done")
    assert store.metrics(1) == [] and store.results(1) is None
    folder = tmp_path / "runs" / "r1"
    (folder / "metrics.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n{"epo', encoding="utf-8")
    (folder / "results.json").write_text('{"best": 2}', encoding="utf-8")
    assert store.metrics(1) == [{"epoch": 1}, {"epoch": 2}] and store.results(1) == {"best": 2}
    store.delete(1)
    with pytest.raises(RunNotFound):
        store.get(1)
