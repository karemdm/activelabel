"""Tests of the format migrations: nothing written for old data, backups, additions only, idempotence."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from activelabel import migrations
from activelabel.migrations import Migration, MigrationError, migrate, removed_or_changed
from activelabel.storage import DamagedFile

TASKS = '{\n "next_id": 1,\n "tasks": [\n  {\n   "name": "a"\n  }\n ]\n}'
PROJECT = '{\n "name": "Cats"\n}'
IMAGES = '[{"id": 1, "name": "a.jpg"}]'
STEP_FILES = ("projects/*/tasks.json", "projects/*/project.json")


def add_field(data: Any) -> Any:
    return {**data, "extra": data.get("extra", 0)}


def drop_next_id(data: Any) -> Any:
    return {k: v for k, v in data.items() if k != "next_id"}


def add_to_list(data: Any) -> Any:
    return [*data, {"id": 2}]


def make_workspace(root: Path) -> Path:
    for rel, text in (
        ("projects/1/project.json", PROJECT),
        ("projects/1/tasks.json", TASKS),
        ("projects/1/images.json", IMAGES),
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_migrations_leave_an_old_workspace_untouched(tmp_path):
    root = make_workspace(tmp_path)
    before = snapshot(root)
    assert migrate(root) is None
    assert snapshot(root) == before


def test_missing_workspace_returns_none(tmp_path):
    assert migrate(tmp_path / "nope") is None


def test_step_backs_up_and_adds_a_field(tmp_path):
    root = make_workspace(tmp_path)
    step = Migration(2, "adds extra", STEP_FILES, add_field)
    backup = migrate(root, (step,))
    assert backup is not None and backup.parent == root / migrations.BACKUP_DIR
    assert (backup / "projects/1/tasks.json").read_text(encoding="utf-8") == TASKS
    assert (backup / "projects/1/project.json").read_text(encoding="utf-8") == PROJECT
    tasks = load(root / "projects/1/tasks.json")
    assert tasks["format_version"] == 2 and tasks["extra"] == 0
    assert tasks["next_id"] == 1 and tasks["tasks"] == [{"name": "a"}]
    assert load(root / "projects/1/project.json") == {"name": "Cats", "extra": 0}
    assert load(root / migrations.WORKSPACE_FILE)["format_version"] == 2
    assert (root / "projects/1/images.json").read_text(encoding="utf-8") == IMAGES
    after = snapshot(root)
    assert migrate(root, (step,)) is None
    assert snapshot(root) == after


def test_a_second_backup_of_the_same_second_gets_its_own_folder(tmp_path, monkeypatch):
    root = make_workspace(tmp_path)
    monkeypatch.setattr(migrations, "now", lambda: "2026-01-01T00:00:00+00:00")
    first = migrations._back_up(root, [root / "projects/1/tasks.json"], 0, 2)
    second = migrations._back_up(root, [root / "projects/1/tasks.json"], 0, 2)
    assert first != second and second.name.endswith("-2")


def test_a_file_already_at_the_step_version_is_skipped(tmp_path):
    root = make_workspace(tmp_path)
    done = '{"format_version": 2, "next_id": 1, "tasks": []}'
    (root / "projects/1/tasks.json").write_text(done, encoding="utf-8")
    migrate(root, (Migration(2, "adds extra", STEP_FILES, add_field),))
    assert (root / "projects/1/tasks.json").read_text(encoding="utf-8") == done
    assert load(root / "projects/1/project.json")["extra"] == 0


def test_a_step_that_removes_data_is_refused_and_writes_nothing(tmp_path):
    root = make_workspace(tmp_path)
    before = snapshot(root)
    step = Migration(2, "bad", ("projects/*/tasks.json",), drop_next_id)
    with pytest.raises(MigrationError) as info:
        migrate(root, (step,))
    assert info.value.code == "format_migration_refused"
    assert info.value.params["file"] == "projects/1/tasks.json" and info.value.params["where"] == "next_id"
    after = snapshot(root)
    assert all(after[name] == value for name, value in before.items())
    assert not (root / migrations.WORKSPACE_FILE).exists()


def test_refused_step_changes_no_file_even_when_another_file_was_fine(tmp_path):
    root = make_workspace(tmp_path)
    second = root / "projects/2/project.json"
    second.parent.mkdir(parents=True)
    second.write_text('{"next_id": 1}', encoding="utf-8")
    # project.json files are fine to change; the tasks.json of project 1 is refused in the same step
    step = Migration(2, "bad", ("projects/*/project.json", "projects/*/tasks.json"), drop_next_id)
    with pytest.raises(MigrationError):
        migrate(root, (step,))
    assert (root / "projects/1/tasks.json").read_text(encoding="utf-8") == TASKS
    assert (root / "projects/1/project.json").read_text(encoding="utf-8") == PROJECT
    assert second.read_text(encoding="utf-8") == '{"next_id": 1}'


def test_removed_or_changed_reports_the_first_place():
    old = {"tasks": [{"name": "a"}], "n": 1, "format_version": 1}
    assert removed_or_changed(old, {**old, "new": 1, "format_version": 2}) is None
    assert removed_or_changed(old, {"tasks": [{"name": "b"}], "n": 1}) == "tasks > 0 > name"
    assert removed_or_changed(old, {"tasks": [{"name": "a"}]}) == "n"
    assert removed_or_changed(old, {"tasks": [], "n": 1}) == "tasks"
    assert removed_or_changed([1], [1, 2]) == "(whole file)"
    assert removed_or_changed(1, True) == "(whole file)"
    assert removed_or_changed(1, 1.0) == "(whole file)"
    assert removed_or_changed(1, 1) is None
    assert removed_or_changed({"a": 1}, {"a": True}) == "a"
    assert removed_or_changed({"a": 1}, {"a": [1]}) == "a"


def test_newer_workspace_is_left_alone_with_a_warning(tmp_path, caplog):
    root = make_workspace(tmp_path)
    (root / migrations.WORKSPACE_FILE).write_text('{"format_version": 9}', encoding="utf-8")
    before = snapshot(root)
    with caplog.at_level(logging.WARNING, logger="activelabel.migrations"):
        assert migrate(root, (Migration(2, "x", STEP_FILES, add_field),)) is None
    assert "newer than this version" in caplog.text
    assert snapshot(root) == before


def test_damaged_format_file_raises(tmp_path):
    (tmp_path / migrations.WORKSPACE_FILE).write_text("{oops", encoding="utf-8")
    with pytest.raises(DamagedFile):
        migrate(tmp_path)


def test_compact_files_stay_compact_and_readable_ones_indented(tmp_path):
    root = make_workspace(tmp_path)
    (root / "projects/1/images.json").write_text('{"images": [1]}', encoding="utf-8")
    step = Migration(2, "adds extra", ("projects/*/images.json", "projects/*/tasks.json"), add_field)
    migrate(root, (step,))
    assert "\n" not in (root / "projects/1/images.json").read_text(encoding="utf-8").strip()
    assert "\n" in (root / "projects/1/tasks.json").read_text(encoding="utf-8").strip()


def test_a_list_that_grows_is_refused(tmp_path):
    root = make_workspace(tmp_path)
    with pytest.raises(MigrationError) as info:
        migrate(root, (Migration(2, "adds one", ("projects/*/images.json",), add_to_list),))
    assert info.value.params["where"] == "(whole file)"
    assert (root / "projects/1/images.json").read_text(encoding="utf-8") == IMAGES


def test_a_second_step_backs_up_the_workspace_marker_too(tmp_path):
    root = make_workspace(tmp_path)
    (root / migrations.WORKSPACE_FILE).write_text('{"format_version": 1}', encoding="utf-8")
    backup = migrate(root, (Migration(2, "adds extra", STEP_FILES, add_field),))
    assert backup is not None
    assert (backup / migrations.WORKSPACE_FILE).read_text(encoding="utf-8") == '{"format_version": 1}'
    assert load(root / migrations.WORKSPACE_FILE)["format_version"] == 2


def test_chained_steps_end_at_the_last_version(tmp_path):
    root = make_workspace(tmp_path)
    steps = (Migration(2, "adds extra", STEP_FILES, add_field), Migration(3, "declares only"))
    assert migrate(root, steps) is not None
    assert load(root / migrations.WORKSPACE_FILE)["format_version"] == 3
    assert load(root / "projects/1/tasks.json")["format_version"] == 2
    assert migrate(root, steps) is None


def test_a_step_without_change_writes_nothing(tmp_path):
    root = make_workspace(tmp_path)
    before = snapshot(root)
    assert migrate(root, (Migration(2, "declares only"),)) is None
    assert snapshot(root) == before
