"""Workers see a phone photo as it is shown (EXIF rotation applied), the frame its labels are drawn in."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

WORKERS = Path(__file__).parents[3] / "src" / "activelabel" / "training" / "workers"


def _worker(name):
    spec = importlib.util.spec_from_file_location(f"{name}_worker_exif", WORKERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rotated(tmp_path):
    """Stored 40 wide x 20 high, shown 20 x 40 (EXIF orientation 6, as phones write)."""
    path = tmp_path / "phone.jpg"
    image = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(path, exif=exif)
    return path


def test_classification_worker(rotated):
    seen = []
    dataset = _worker("classification").ImageDataset([{"path": str(rotated), "label": 0}], seen.append)
    dataset[0]
    assert seen[0].size == (20, 40)


def test_detection_worker(rotated):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    dataset = _worker("detection").DetectionDataset(
        [{"path": str(rotated), "objects": []}], "detection", 0, False, 0
    )
    tensor, _ = dataset[0]
    assert tuple(tensor.shape) == (3, 40, 20)
