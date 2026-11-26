import json

import numpy as np
import pytest
from PIL import Image
from worker_loader import load_worker


@pytest.fixture(scope="module")
def module():
    return load_worker("embed")


def test_rows_are_scaled_to_length_one(module):
    out = module.normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.tolist() == [[0.6, 0.8], [0.0, 0.0]]


def test_the_input_is_a_normalised_square(module):
    tensor = module.to_tensor(Image.new("RGB", (50, 30), (124, 116, 104)))
    assert tuple(tensor.shape) == (3, module.SIZE, module.SIZE)
    assert float(tensor.abs().max()) < 0.05  # the ImageNet mean becomes about zero


def test_an_unknown_backbone_is_refused(module):
    with pytest.raises(ValueError, match="Unknown backbone"):
        module.build_backbone("nope")


def test_embed_writes_one_row_per_readable_image(module, tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        module, "build_backbone", lambda name: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(4))
    )
    good = tmp_path / "a.png"
    Image.new("RGB", (20, 20), (10, 200, 30)).save(good)
    spec = {
        "items": [{"id": "a", "path": str(good)}, {"id": "b", "path": str(tmp_path / "missing.png")}],
        "device": "cpu",
        "output": str(tmp_path / "out"),
    }
    totals = module.embed(spec)
    assert totals == {"images": 1, "dims": 4, "backbone": module.DEFAULT_BACKBONE, "unreadable": 1}
    assert json.loads((tmp_path / "out" / "ids.json").read_text()) == ["a"]
    vectors = np.load(tmp_path / "out" / "vectors.npy")
    assert vectors.shape == (1, 4) and np.isclose(np.linalg.norm(vectors[0]), 1)
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()]
    assert events == ["progress", "skipped", "progress"]
