import numpy as np
import pytest

from activelabel.curation import uniqueness


class Stop(Exception):
    """Raised by a check callback to stop the computation."""


def cloud(rows=40, dims=6, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, dims))


def test_fewer_than_two_rows_score_zero():
    assert uniqueness.scores(np.zeros((0, 4))).tolist() == []
    assert uniqueness.scores(np.ones((1, 4))).tolist() == [0.0]


def test_a_twin_scores_lower_than_an_outlier_and_the_most_unique_is_one():
    rows = np.vstack([np.tile([1.0, 0.0, 0.0], (5, 1)) + 0.001, [[0.0, 1.0, 0.0]]])
    values = uniqueness.scores(rows)
    assert values[5] == pytest.approx(1.0) and values.max() == pytest.approx(1.0)
    assert values[:5].max() < values[5]
    assert ((values >= 0) & (values <= 1)).all()


def test_two_rows_use_a_single_neighbour():
    values = uniqueness.scores(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert values.tolist() == [1.0, 1.0]


def test_chunked_computation_equals_one_block(monkeypatch):
    rows = cloud()
    whole = uniqueness.scores(rows)
    monkeypatch.setattr(uniqueness, "BLOCK", 7 * len(rows))  # 7 rows per chunk
    assert np.allclose(uniqueness.scores(rows), whole)


def test_check_is_called_after_every_chunk(monkeypatch):
    monkeypatch.setattr(uniqueness, "BLOCK", 10 * 40)
    calls = []
    uniqueness.scores(cloud(), lambda done, total: calls.append((done, total)))
    assert calls == [(10, 40), (20, 40), (30, 40), (40, 40)]


def stop_check(done, total):
    raise Stop


def test_check_can_stop_the_computation():
    with pytest.raises(Stop):
        uniqueness.scores(cloud(), stop_check)


def test_the_store_reads_updates_and_clears(tmp_path):
    store = uniqueness.UniquenessStore(tmp_path)
    assert store.read() == {}
    assert store.update(status="queued") == {"status": "queued"}
    assert store.update(job="j1") == {"status": "queued", "job": "j1"}
    assert uniqueness.UniquenessStore(tmp_path).read()["job"] == "j1"
    assert store.lock is uniqueness.UniquenessStore(tmp_path).lock
    store.clear()
    store.clear()  # nothing left: no error
    assert store.read() == {}


def test_a_state_that_is_not_an_object_reads_as_empty(tmp_path):
    (tmp_path / uniqueness.STATE_FILE).write_text("[1, 2]")
    assert uniqueness.UniquenessStore(tmp_path).read() == {}


def test_identical_images_all_score_zero():
    assert not uniqueness.scores(np.ones((6, 3))).any()
