import pytest
from worker_loader import load_worker


@pytest.fixture(scope="module")
def module():
    return load_worker("export_onnx")


def test_export_worker_key_check(module):
    module.check_keys(["backbone.bn1.num_batches_tracked"], ["aux_classifier.4.weight"])
    with pytest.raises(ValueError, match="do not match"):
        module.check_keys(["head.weight"], [])
    with pytest.raises(ValueError, match="do not match"):
        module.check_keys([], ["extra.weight"])
    assert module.KINDS == ("classification", "semantic")


def test_emit_writes_one_json_line(module, capsys):
    module.emit("log", message="hi")
    assert capsys.readouterr().out.strip() == '{"event": "log", "message": "hi"}'


def test_main_reports_a_bad_spec_as_an_error_event(module, tmp_path, capsys):
    assert module.main(["export_onnx.py", str(tmp_path / "missing.json")]) == 1
    assert '"event": "error"' in capsys.readouterr().out
