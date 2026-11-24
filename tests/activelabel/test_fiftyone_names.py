from activelabel.fiftyone_names import record_name, recorded_name


def test_the_name_is_recorded_and_read_back(tmp_path):
    assert recorded_name(tmp_path) is None
    record_name(tmp_path, "activelabel-1")
    assert recorded_name(tmp_path) == "activelabel-1"


def test_a_damaged_record_counts_as_none(tmp_path):
    (tmp_path / "fiftyone.json").write_text('{"dataset": 5}', encoding="utf-8")
    assert recorded_name(tmp_path) is None
