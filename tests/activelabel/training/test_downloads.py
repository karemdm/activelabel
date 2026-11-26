from pathlib import Path

from activelabel.training.downloads import clean, downloaded
from activelabel.training.environment import remove_tree


def test_downloaded_lists_weight_files_only(tmp_path: Path) -> None:
    (tmp_path / "torch" / "hub").mkdir(parents=True)
    (tmp_path / "torch" / "hub" / "r18.pth").write_bytes(b"x" * 10)
    (tmp_path / "hf" / "model.safetensors").parent.mkdir()
    (tmp_path / "hf" / "model.safetensors").write_bytes(b"x" * 5)
    (tmp_path / "yolov8n.PT").write_bytes(b"x" * 3)
    (tmp_path / "data.json").write_text("{}")
    names = [f["name"] for f in downloaded(tmp_path)]
    assert names == ["hf/model.safetensors", "torch/hub/r18.pth", "yolov8n.PT"]
    assert downloaded(tmp_path / "missing") == []


def test_clean_measures_then_deletes(tmp_path: Path) -> None:
    tmp = tmp_path / "job"
    tmp.mkdir()
    (tmp / "w.pt").write_bytes(b"x" * 7)
    report = clean(tmp, remove_tree)
    assert report == {"files": [{"name": "w.pt", "bytes": 7}], "bytes": 7, "deleted": True}
    assert not tmp.exists()


def test_clean_reports_a_folder_that_stayed(tmp_path: Path) -> None:
    (tmp_path / "w.bin").write_bytes(b"x")
    report = clean(tmp_path, lambda _path: None)  # the removal failed (it only logs)
    assert report["deleted"] is False and report["bytes"] == 1
