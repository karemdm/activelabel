import threading

from activelabel.curation.state_store import StateStore, _lock


def test_nothing_is_saved_at_first(tmp_path):
    assert StateStore(tmp_path, "x.json").read() == {}


def test_update_merges_fields_and_keeps_the_others(tmp_path):
    store = StateStore(tmp_path, "x.json")
    assert store.update(status="queued", n=1) == {"status": "queued", "n": 1}
    assert store.update(status="done") == {"status": "done", "n": 1}
    assert StateStore(tmp_path, "x.json").read() == {"status": "done", "n": 1}


def test_the_files_of_one_project_are_independent(tmp_path):
    StateStore(tmp_path, "a.json").update(v=1)
    assert StateStore(tmp_path, "b.json").read() == {}


def test_clear_deletes_the_state_and_tolerates_none(tmp_path):
    store = StateStore(tmp_path, "x.json")
    store.update(v=1)
    store.clear()
    assert store.read() == {} and not store.path.exists()
    store.clear()


def test_a_file_that_is_not_an_object_reads_as_empty(tmp_path):
    (tmp_path / "x.json").write_text("[1, 2]", encoding="utf-8")
    assert StateStore(tmp_path, "x.json").read() == {}


def test_stores_of_the_same_file_share_one_reentrant_lock(tmp_path):
    first, second = StateStore(tmp_path, "x.json"), StateStore(tmp_path, "x.json")
    assert first.lock is second.lock is _lock(tmp_path / "x.json")
    assert first.lock is not StateStore(tmp_path, "y.json").lock
    with first.lock, second.lock:  # re-entrant: update inside a held lock does not deadlock
        first.update(v=1)


def test_concurrent_updates_do_not_lose_fields(tmp_path):
    store = StateStore(tmp_path, "x.json")
    threads = [threading.Thread(target=store.update, kwargs={f"k{n}": n}) for n in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.read() == {f"k{n}": n for n in range(20)}
