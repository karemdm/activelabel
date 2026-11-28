"""Images edited in the App and not in the project yet (#700): which sessions count, and the cache."""

from pathlib import Path
from typing import Any

import pytest

from activelabel import fiftyone_bridge
from activelabel.annotate import edited, sessions
from activelabel.tasks import Task


@pytest.fixture(autouse=True)
def _clean_cache():
    edited._cache.clear()
    yield
    edited._cache.clear()


def _session(folder: Path, ids: list[str], tool: str = "fiftyone", status: str = sessions.OPEN) -> str:
    session = sessions.create(folder, tool, "all", ids, [], {})
    if status != sessions.OPEN:
        sessions.update(folder, session["id"], status=status)
    return session["id"]


class FakeFiftyOne:
    """Stands in for the stamp and the sample scan of the bridge, and counts the scans."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, edited_ids: set[str]) -> None:
        self.stamp = "1"
        self.edited_ids = edited_ids
        self.scans = 0
        monkeypatch.setattr(fiftyone_bridge, "edit_stamp", lambda dataset, ids: self.stamp)
        monkeypatch.setattr(fiftyone_bridge, "edited_images", self.scan)

    def scan(self, dataset: Any, ids: list[str], task: Task) -> set[str]:
        self.scans += 1
        return self.edited_ids & set(ids)


def test_tracked_lists_the_fiftyone_sessions_that_may_hold_edits(tmp_path: Path) -> None:
    keep = [_session(tmp_path, ["a"]), _session(tmp_path, ["b"], status=sessions.PULLED)]
    keep.append(_session(tmp_path, ["c"], status=sessions.APPLIED))
    _session(tmp_path, ["d"], status=sessions.DISCARDED)
    _session(tmp_path, ["e"], tool="cvat")
    assert edited.tracked(tmp_path) == {keep[0]: ["a"], keep[1]: ["b"], keep[2]: ["c"]}


def test_no_session_means_nothing_edited_and_the_dataset_is_not_opened(tmp_path: Path) -> None:
    def forbidden() -> Any:
        raise AssertionError("the dataset must not be opened")

    assert edited.summary(tmp_path, forbidden, Task.DETECTION) == {"edited_count": 0, "edited": {}}


def test_without_fiftyone_nothing_is_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(tmp_path, ["a"])
    monkeypatch.setattr(edited, "FIFTYONE", False)
    assert edited.summary(tmp_path, lambda: pytest.fail("no dataset"), Task.DETECTION)["edited_count"] == 0


def test_a_missing_dataset_counts_nothing(tmp_path: Path) -> None:
    _session(tmp_path, ["a"])
    assert edited.summary(tmp_path, lambda: None, Task.DETECTION) == {"edited_count": 0, "edited": {}}


def test_summary_counts_images_once_and_each_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _session(tmp_path, ["a", "b"])
    second = _session(tmp_path, ["b", "c"], status=sessions.PULLED)
    third = _session(tmp_path, ["d"])
    FakeFiftyOne(monkeypatch, {"b", "c"})
    result = edited.summary(tmp_path, lambda: object(), Task.DETECTION)
    assert result == {"edited_count": 2, "edited": {first: 1, second: 2}}
    assert third not in result["edited"]


def test_the_samples_are_scanned_again_only_when_the_stamp_or_the_images_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, ["a", "b"])
    fake = FakeFiftyOne(monkeypatch, {"a"})
    for _ in range(3):
        assert edited.summary(tmp_path, lambda: object(), Task.DETECTION)["edited_count"] == 1
    assert fake.scans == 1
    fake.stamp, fake.edited_ids = "2", {"a", "b"}  # a sample was saved
    assert edited.summary(tmp_path, lambda: object(), Task.DETECTION)["edited_count"] == 2
    assert fake.scans == 2
    _session(tmp_path, ["c"])  # a new session: other images
    edited.summary(tmp_path, lambda: object(), Task.DETECTION)
    assert fake.scans == 3


def test_projects_do_not_share_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    _session(one, ["a"])
    _session(two, ["a"])
    FakeFiftyOne(monkeypatch, {"a"})
    assert edited.summary(one, lambda: object(), Task.DETECTION)["edited_count"] == 1
    monkeypatch.setattr(fiftyone_bridge, "edited_images", lambda *a: set())
    assert edited.summary(two, lambda: object(), Task.DETECTION)["edited_count"] == 0


def test_a_failure_is_logged_and_counts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _session(tmp_path, ["a"])

    def broken() -> Any:
        raise RuntimeError("database down")

    assert edited.summary(tmp_path, broken, Task.DETECTION) == {"edited_count": 0, "edited": {}}
    assert "Could not count the images edited" in caplog.text
