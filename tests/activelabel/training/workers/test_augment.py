"""Augmentation of the training workers: the labels must stay on the objects the image shows."""

import random

import numpy as np
import pytest
from PIL import Image
from worker_loader import load_worker

from activelabel.training import augmentations

SEEDS = (0, 1, 2, 3)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WIDTH, HEIGHT = 96, 72
RECT = (24, 18, 72, 54)  # x1, y1, x2, y2 of the white rectangle
DOTS = {"red": (30, 24), "green": (64, 46)}  # centres of the 3 x 3 coloured dots
CHANNEL = {"red": 0, "green": 1}


def entry(name, **params):
    """A technique entry as the runner resolves it."""
    return {"name": name, "params": params}


GEOMETRIC = {
    "hflip": entry("hflip", p=1.0),
    "vflip": entry("vflip", p=1.0),
    "rotation": entry("rotation", degrees=10.0),
    "scale_jitter": entry("scale_jitter", scale_min=0.8, scale_max=1.1),
    "translate": entry("translate", fraction=0.1),
    "random_resized_crop": entry("random_resized_crop", scale_min=0.6),
}
CASES = [(name, seed) for name in GEOMETRIC for seed in SEEDS]
IDS = [f"{name}-{seed}" for name, seed in CASES]


class EdgeRng:
    """Random source that always answers the largest value (a maximal, certain transform)."""

    def random(self):
        return 0.0

    def uniform(self, low, high):
        return high

    def randint(self, low, high):
        return high


@pytest.fixture(scope="module")
def aug():
    return load_worker("augment")


def rectangle_image():
    """Black image with the white rectangle."""
    image = Image.new("RGB", (WIDTH, HEIGHT))
    image.paste((255, 255, 255), RECT)
    return image


def white_region(image):
    """Boolean array of the white pixels of an image."""
    return np.asarray(image).min(axis=2) > 127


def white_extent(image):
    """``[x1, y1, x2, y2]`` around the white pixels, or ``None``."""
    ys, xs = np.nonzero(white_region(image))
    if not len(xs):
        return None
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def rectangle_target(kind="box"):
    """Detection target of the rectangle (with its mask for ``instance``)."""
    target = {
        "boxes": np.array([RECT], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    if kind == "instance":
        mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
        mask[0, RECT[1] : RECT[3], RECT[0] : RECT[2]] = 1
        target["masks"] = mask
    return target


def dots_image():
    """Black image with a red and a green 3 x 3 dot."""
    image = Image.new("RGB", (WIDTH, HEIGHT))
    for colour, (x, y) in DOTS.items():
        rgb = tuple(255 if i == CHANNEL[colour] else 0 for i in range(3))
        image.paste(rgb, (x - 1, y - 1, x + 2, y + 2))
    return image


def dots_target():
    """Target with one object whose points 0 and 1 sit on the red and the green dot."""
    points = [[x + 0.5, y + 0.5, 2.0] for x, y in DOTS.values()]
    return {
        "boxes": np.array([[20, 15, 70, 50]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "keypoints": np.array([points], dtype=np.float32),
    }


def dot_centre(image, colour):
    """Centroid of a coloured dot (pixels weighted by their channel), or ``None`` when it is gone."""
    data = np.asarray(image).astype(np.float64)
    weight = data[:, :, CHANNEL[colour]] - data[:, :, 2]
    weight = np.clip(weight, 0, None)
    if weight.sum() < 100:
        return None
    ys, xs = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    return (
        float((weight * (xs + 0.5)).sum() / weight.sum()),
        float((weight * (ys + 0.5)).sum() / weight.sum()),
    )


def iou(a, b):
    """Intersection over union of two boolean arrays."""
    union = (a | b).sum()
    return (a & b).sum() / union if union else 1.0


# ---------------------------------------------------------------------------
# Geometric techniques keep the labels on the objects


@pytest.mark.parametrize(("name", "seed"), CASES, ids=IDS)
def test_box_matches_the_moved_rectangle(aug, name, seed):
    image, target = aug.augment_objects(
        rectangle_image(), rectangle_target(), [GEOMETRIC[name]], random.Random(seed)
    )
    extent = white_extent(image)
    if not len(target["boxes"]):
        assert extent is None or (extent[2] - extent[0]) * (extent[3] - extent[1]) < 0.3 * 48 * 36
        return
    assert image.size == (WIDTH, HEIGHT)
    assert target["boxes"].shape == (1, 4) and target["labels"].tolist() == [1]
    assert np.abs(np.array(extent) - target["boxes"][0]).max() <= 1.5


@pytest.mark.parametrize(("name", "seed"), CASES, ids=IDS)
def test_instance_mask_matches_the_moved_rectangle(aug, name, seed):
    image, target = aug.augment_objects(
        rectangle_image(), rectangle_target("instance"), [GEOMETRIC[name]], random.Random(seed)
    )
    if not len(target["boxes"]):
        assert target["masks"].shape == (0, HEIGHT, WIDTH)
        return
    mask = target["masks"][0] > 0
    assert iou(mask, white_region(image)) >= 0.9
    assert target["boxes"][0].tolist() == aug.mask_box(mask)


@pytest.mark.parametrize(("name", "seed"), CASES, ids=IDS)
def test_keypoints_stay_on_their_dots(aug, name, seed):
    image, target = aug.augment_objects(dots_image(), dots_target(), [GEOMETRIC[name]], random.Random(seed))
    for index, colour in enumerate(DOTS):
        centre = dot_centre(image, colour)
        point = target["keypoints"][0, index] if len(target["keypoints"]) else None
        if point is None or point[2] == 0:
            continue  # the point left the image (or the whole object did)
        assert centre is not None
        assert np.abs(np.array(centre) - point[:2]).max() <= 2.0


@pytest.mark.parametrize(("name", "seed"), CASES, ids=IDS)
def test_semantic_mask_matches_the_moved_rectangle(aug, name, seed):
    mask = Image.new("L", (WIDTH, HEIGHT))
    mask.paste(1, RECT)
    image, moved = aug.augment_pair(rectangle_image(), mask, [GEOMETRIC[name]], random.Random(seed))
    assert image.size == moved.size == (WIDTH, HEIGHT)
    assert {v for _, v in moved.getcolors()} <= {0, 1}  # nearest: no invented class values
    assert iou(np.asarray(moved) == 1, white_region(image)) >= 0.9


def test_hflip_swaps_the_keypoint_pairs(aug):
    plain = entry("hflip", p=1.0)
    paired = entry("hflip", p=1.0, flip_pairs=[[0, 1]])
    image, target = aug.augment_objects(dots_image(), dots_target(), [paired], random.Random(0))
    green, red = dot_centre(image, "green"), dot_centre(image, "red")
    # the point 0 ("left") is now the mirrored old point 1, which sits on the green dot
    assert np.abs(np.array(green) - target["keypoints"][0, 0, :2]).max() <= 2.0
    assert np.abs(np.array(red) - target["keypoints"][0, 1, :2]).max() <= 2.0
    image, target = aug.augment_objects(dots_image(), dots_target(), [plain], random.Random(0))
    assert np.abs(np.array(dot_centre(image, "red")) - target["keypoints"][0, 0, :2]).max() <= 2.0


def test_points_pushed_outside_are_missing(aug):
    keypoints = dots_target()["keypoints"]
    shift = np.array([[1, 0, 40], [0, 1, 0], [0, 0, 1]], dtype=float)
    moved = aug.move_keypoints(keypoints, shift, WIDTH, HEIGHT, [])
    assert moved[0, 0].tolist() == [pytest.approx(70.5), pytest.approx(24.5), 2.0]
    assert moved[0, 1].tolist() == [0.0, 0.0, 0.0]  # x = 104.5 is outside


def test_missing_points_stay_missing(aug):
    keypoints = dots_target()["keypoints"]
    keypoints[0, 0] = 0
    assert aug.move_keypoints(keypoints, np.eye(3), WIDTH, HEIGHT, [])[0, 0].tolist() == [0.0, 0.0, 0.0]


def test_move_keypoints_with_no_objects(aug):
    out = aug.move_keypoints(np.zeros((0, 3, 3)), np.eye(3), WIDTH, HEIGHT, [[0, 1]])
    assert out.shape == (0, 3, 3)


# ---------------------------------------------------------------------------
# Objects that leave the image, identity, photometric techniques


def test_objects_moved_out_of_the_image_are_dropped(aug):
    boxes = np.array([[30, 20, 50, 40], [88, 10, 94, 20]], dtype=np.float32)
    target = {"boxes": boxes, "labels": np.array([1, 2], dtype=np.int64)}
    # the largest shift (down and right): half the width and the height
    _, out = aug.augment_objects(rectangle_image(), target, [entry("translate", fraction=0.5)], EdgeRng())
    assert out["labels"].tolist() == [1]
    assert out["boxes"].tolist() == [[78.0, 56.0, 96.0, 72.0]]


def test_masks_moved_out_of_the_image_are_dropped(aug):
    target = {
        "boxes": np.array([[80, 5, 94, 15]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "masks": np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8),
    }
    target["masks"][0, 5:15, 80:94] = 1
    _, out = aug.augment_objects(rectangle_image(), target, [entry("translate", fraction=0.5)], EdgeRng())
    assert out["labels"].shape == (0,) and out["masks"].shape == (0, HEIGHT, WIDTH)
    assert out["boxes"].shape == (0, 4)


@pytest.mark.parametrize("entries", [[], [entry("hflip", p=0.0)], [entry("translate", fraction=0.0)]])
def test_nothing_applied_leaves_image_and_target(aug, entries):
    image, target = rectangle_image(), rectangle_target("instance")
    out_image, out = aug.augment_objects(image, target, entries, random.Random(0))
    assert out_image is image  # no resampling at all
    assert all(np.array_equal(out[k], target[k]) for k in target)
    mask = Image.new("L", (WIDTH, HEIGHT))
    out_image, out_mask = aug.augment_pair(image, mask, entries, random.Random(0))
    assert out_image is image and out_mask is mask


PHOTOMETRIC = [
    entry("color_jitter", brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    entry("blur", p=1.0, sigma_max=2.0),
    entry("grayscale", p=1.0),
]


@pytest.mark.parametrize("technique", PHOTOMETRIC, ids=lambda t: t["name"])
def test_photometric_techniques_do_not_touch_the_target(aug, technique):
    image = dots_image()
    target = dots_target()
    out_image, out = aug.augment_objects(image, target, [technique], random.Random(0))
    assert out_image.size == image.size
    assert all(np.array_equal(out[k], target[k]) for k in target)


def test_blur_and_grayscale_change_the_pixels(aug):
    image = dots_image()
    blurred = aug.photometric(image, [entry("blur", p=1.0, sigma_max=2.0)], random.Random(0))
    assert not np.array_equal(np.asarray(blurred), np.asarray(image))
    grey = np.asarray(aug.photometric(image, [entry("grayscale", p=1.0)], random.Random(0)))
    assert (grey[:, :, 0] == grey[:, :, 1]).all() and (grey[:, :, 1] == grey[:, :, 2]).all()
    kept = aug.photometric(image, [entry("grayscale", p=0.0)], random.Random(0))
    assert np.array_equal(np.asarray(kept), np.asarray(image))


@pytest.mark.parametrize("name", ["mixup", "cutmix", "randaugment", "trivialaugment", "random_erasing", "x"])
def test_object_workers_refuse_classification_only_techniques(aug, name):
    with pytest.raises(aug.AugmentError, match=name):
        aug.check_objects([entry(name)])
    with pytest.raises(aug.AugmentError):
        aug.augment_objects(rectangle_image(), rectangle_target(), [entry(name)])
    with pytest.raises(aug.AugmentError):
        aug.augment_pair(rectangle_image(), Image.new("L", (WIDTH, HEIGHT)), [entry(name)])


def test_object_workers_accept_the_object_presets(aug):
    for kind in ("detection", "instance", "keypoints", "semantic"):
        for level in augmentations.PRESETS_NAMES:
            aug.check_objects(augmentations.preset(level, kind))


# ---------------------------------------------------------------------------
# Classification


CLASSIFICATION_TECHNIQUES = [
    t.name for t in augmentations.TECHNIQUES if "classification" in t.kinds and not t.batch
]


@pytest.mark.parametrize("name", CLASSIFICATION_TECHNIQUES)
def test_classification_transform_of_each_technique(aug, name):
    entries = augmentations.validate([entry(name)], "classification")
    tensor = aug.classification_transform(entries, 32, MEAN, STD)(rectangle_image())
    assert tuple(tensor.shape) == (3, 32, 32)


@pytest.mark.parametrize("level", augmentations.PRESETS_NAMES)
def test_classification_transform_of_each_preset(aug, level):
    entries = augmentations.preset(level, "classification")
    tensor = aug.classification_transform(entries, 32, MEAN, STD)(rectangle_image())
    assert tuple(tensor.shape) == (3, 32, 32)


def test_classification_transform_order(aug):
    steps = aug.classification_transform([entry("hflip", p=0.5)], 32, MEAN, STD).transforms
    assert isinstance(steps[0], aug.transforms.Resize)  # no random_resized_crop: resize first
    crop = aug.classification_transform([entry("random_resized_crop", scale_min=0.5)], 32, MEAN, STD)
    assert isinstance(crop.transforms[0], aug.transforms.RandomResizedCrop)
    assert not any(isinstance(s, aug.transforms.Resize) for s in crop.transforms)
    erasing = aug.classification_transform([entry("random_erasing", p=0.5)], 32, MEAN, STD).transforms
    assert isinstance(erasing[-2], aug.transforms.Normalize)
    assert isinstance(erasing[-1], aug.transforms.RandomErasing)
    assert isinstance(erasing[-3], aug.transforms.ToTensor)


def test_classification_transform_skips_batch_mixes(aug):
    steps = aug.classification_transform([entry("mixup", alpha=0.2)], 32, MEAN, STD).transforms
    assert [type(s).__name__ for s in steps] == ["Resize", "ToTensor", "Normalize"]


def test_classification_refuses_unknown_technique(aug):
    with pytest.raises(aug.AugmentError, match="bogus"):
        aug.classification_transform([entry("bogus")], 32, MEAN, STD)


def test_mix_settings(aug):
    entries = [entry("hflip", p=0.5), entry("mixup", alpha=0.2), entry("cutmix", alpha=1.0)]
    assert aug.mix_settings(entries) == [("mixup", 0.2), ("cutmix", 1.0)]
    assert aug.mix_settings([entry("hflip", p=0.5)]) == []


def batch_of_constants(aug):
    """Four images, the ``i``-th filled with ``i + 1``."""
    values = aug.torch.arange(1, 5, dtype=aug.torch.float32)
    return values.view(4, 1, 1, 1).repeat(1, 3, 8, 10)


@pytest.mark.parametrize("seed", SEEDS)
def test_mixup_blends_each_image_with_its_partner(aug, seed):
    aug.torch.manual_seed(seed)
    images = batch_of_constants(aug)
    mixed, order, lam = aug.mix_batch(images, [("mixup", 0.4)], random.Random(seed))
    assert 0.0 <= lam <= 1.0 and sorted(order.tolist()) == [0, 1, 2, 3]
    assert aug.torch.allclose(mixed, lam * images + (1.0 - lam) * images[order])


@pytest.mark.parametrize("seed", SEEDS)
def test_cutmix_pastes_a_patch_of_the_partner(aug, seed):
    aug.torch.manual_seed(seed)
    images = batch_of_constants(aug)
    mixed, order, lam = aug.mix_batch(images, [("cutmix", 1.0)], random.Random(seed))
    for k, partner in enumerate(order.tolist()):
        own = (mixed[k, 0] == images[k, 0]).float().mean().item()
        pasted = (mixed[k, 0] == images[partner, 0]).float().mean().item()
        if partner != k:
            assert own == pytest.approx(lam) and own + pasted == pytest.approx(1.0)
    assert 0.0 <= lam <= 1.0


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.9, 1.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_cut_box_stays_inside(aug, lam, seed):
    top, left, bottom, right = aug.cut_box(40, 60, lam, random.Random(seed))
    assert 0 <= top <= bottom <= 40 and 0 <= left <= right <= 60
    if lam == 1.0:
        assert top == bottom and left == right  # nothing to cut


# ---------------------------------------------------------------------------
# Small helpers


@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 1.5, 5.0])
def test_blur_kernel_is_odd_and_holds_three_sigmas(aug, sigma):
    size = aug.blur_kernel(sigma)
    assert size >= 3 and size % 2 == 1 and size // 2 >= 3 * sigma


@pytest.mark.parametrize("min_scale", [0.3, 0.6])
@pytest.mark.parametrize("seed", range(20))
def test_crop_box_is_inside_and_big_enough(aug, min_scale, seed):
    left, top, right, bottom = aug.crop_box(100, 80, min_scale, random.Random(seed))
    assert 0 <= left < right <= 100 and 0 <= top < bottom <= 80
    assert (right - left) * (bottom - top) / (100 * 80) >= min_scale - 0.02


@pytest.mark.parametrize("min_scale", [1.0, 1.5])
def test_crop_box_keeps_the_whole_image_at_full_scale(aug, min_scale):
    assert aug.crop_box(100, 80, min_scale, random.Random(0)) == (0, 0, 100, 80)


def test_mirrors(aug):
    flip = np.array([[-1.0, 0, 10], [0, 1, 0], [0, 0, 1]])
    assert aug.mirrors(flip) and not aug.mirrors(np.eye(3)) and aug.mirrors(flip @ flip @ flip)
    assert not aug.mirrors(flip @ flip)


def test_describe(aug):
    assert aug.describe([]) == "none"
    text = aug.describe([entry("hflip", p=0.5), entry("trivialaugment"), entry("blur", p=0.2, sigma_max=1.5)])
    assert text == "hflip (p=0.5), trivialaugment, blur (p=0.2, sigma_max=1.5)"


def test_flip_pairs_and_params(aug):
    entries = [entry("hflip", p=1.0, flip_pairs=[[0, 1], [2, 3]]), entry("blur", p=0.2, sigma_max=1.0)]
    assert aug.flip_pairs(entries) == [[0, 1], [2, 3]]
    assert aug.flip_pairs([entry("vflip", p=1.0)]) == []
    assert aug.params_of(entries)["blur"] == {"p": 0.2, "sigma_max": 1.0}
