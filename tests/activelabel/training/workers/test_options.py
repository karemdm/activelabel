import numpy as np
import pytest
from worker_loader import load_worker


@pytest.fixture(scope="module")
def o():
    return load_worker("options")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


# -- resize policy (#316) ------------------------------------------------------


def test_layout_of_each_policy(o):
    assert o.fit_layout(200, 100, 50, "stretch") == (50, 50, 0, 0)
    assert o.fit_layout(200, 100, 50, "letterbox") == (50, 25, 0, 12)
    assert o.fit_layout(200, 100, 50, "center_crop") == (100, 50, -25, 0)
    with pytest.raises(ValueError, match="Unknown resize policy"):
        o.fit_layout(10, 10, 5, "zoom")
    with pytest.raises(ValueError, match="Unknown resize policy"):
        o.Fit(5, "zoom")


def test_fit_image_bands_and_crop(o):
    from PIL import Image

    photo = Image.new("RGB", (200, 100), (255, 0, 0))
    boxed = o.fit_image(photo, 50, "letterbox")
    assert boxed.size == (50, 50)
    assert boxed.getpixel((25, 2)) == o.FILL and boxed.getpixel((25, 25)) == (255, 0, 0)
    cropped = o.fit_image(photo, 50, "center_crop")
    assert cropped.size == (50, 50) and cropped.getpixel((0, 0)) == (255, 0, 0)
    assert o.fit_image(photo, 50, "stretch").size == (50, 50)
    mask = o.fit_image(Image.new("L", (200, 100), 3), 50, "letterbox", Image.Resampling.NEAREST, 0)
    assert mask.getpixel((25, 2)) == 0 and mask.getpixel((25, 25)) == 3
    assert "letterbox" in repr(o.Fit(50, "letterbox")) and o.Fit(50, "stretch")(photo).size == (50, 50)


@pytest.mark.parametrize("policy", ["stretch", "letterbox", "center_crop"])
def test_box_goes_in_and_comes_back(o, policy):
    box = np.array([[60.0, 30.0, 120.0, 70.0]])
    target = {"boxes": box.astype(np.float32), "labels": np.array([1])}
    inside = o.fit_target(target, 200, 100, 50, policy)
    assert inside["boxes"].shape == (1, 4) and (inside["boxes"] <= 50).all()
    back = o.unfit_boxes(inside["boxes"], 200, 100, 50, policy)
    assert back == pytest.approx(box, abs=1e-3)


def test_letterbox_prediction_returns_to_the_photo(o):
    # a 400 x 200 photo in a 100 px letterbox: scale 0.25, 25 px band on top
    predicted = np.array([[10.0, 35.0, 30.0, 45.0]])  # input pixels
    assert o.unfit_boxes(predicted, 400, 200, 100, "letterbox")[0] == pytest.approx([40, 40, 120, 80])
    points = np.array([[[10.0, 35.0, 2.0]]])
    assert o.unfit_points(points, 400, 200, 100, "letterbox")[0, 0] == pytest.approx([40, 40, 2])
    assert o.unfit_points(np.zeros((0, 3, 3)), 400, 200, 100, "letterbox").shape == (0, 3, 3)


def test_crop_drops_objects_cut_away_and_masks_follow(o):
    target = {
        "boxes": np.array([[0, 0, 20, 20], [90, 40, 110, 60]], dtype=np.float32),
        "labels": np.array([1, 2]),
    }
    inside = o.fit_target(target, 200, 100, 50, "center_crop")
    assert inside["labels"].tolist() == [2]
    masks = np.zeros((1, 100, 200), dtype=np.uint8)
    masks[0, 40:60, 90:110] = 1
    moved = o.fit_target({**target, "masks": masks, "labels": np.array([1])}, 200, 100, 50, "letterbox")
    assert moved["masks"].shape == (1, 50, 50) and moved["boxes"][0] == pytest.approx([22, 22, 28, 28], abs=1)


def test_unfit_map_back_to_the_photo_grid(o, torch):
    maps = torch.zeros(1, 1, 50, 50)
    maps[:, :, 12:37, :] = 1.0  # the photo part of a 200 x 100 letterbox
    back = o.unfit_map(maps, 200, 100, 50, "letterbox", (40, 20))
    assert back.shape == (1, 1, 20, 40) and float(back.min()) == pytest.approx(1.0, abs=0.05)
    cut = o.unfit_map(torch.ones(1, 2, 50, 50), 200, 100, 50, "center_crop", (40, 20), fill=-1.0)
    assert float(cut[0, 0, 10, 0]) == -1.0 and float(cut[0, 0, 10, 20]) == pytest.approx(1.0)
    same = o.unfit_map(torch.ones(1, 1, 50, 50), 200, 100, 50, "stretch", (40, 20))
    assert same.shape == (1, 1, 20, 40)
