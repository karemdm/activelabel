import numpy as np
import pytest
from PIL import Image
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


# -- learning rate schedule (#317) -----------------------------------------------


def test_warmup_ramps_then_the_schedule_takes_over(o):
    # 10 steps per epoch, 4 epochs, 1 epoch of warmup
    values = [o.lr_factor("none", step, 10, 4, 1) for step in range(40)]
    assert values[0] == pytest.approx(o.WARMUP_START) and values[5] == pytest.approx(0.5, abs=0.01)
    assert values[10:] == [1.0] * 30
    cosine = [o.lr_factor("cosine", step, 10, 4, 1) for step in range(40)]
    assert cosine[10] == pytest.approx(1.0) and cosine[25] == pytest.approx(0.5, abs=0.05)
    assert cosine[39] < 0.01 and cosine == sorted(cosine[:10]) + sorted(cosine[10:], reverse=True)
    assert o.lr_factor("cosine", 0, 10, 4, 0) == 1.0  # no warmup


def test_step_onecycle_and_plateau_factors(o):
    assert [o.lr_factor("step", e * 10, 10, 9, 0) for e in (0, 2, 3, 6)] == pytest.approx([1, 1, 0.1, 0.01])
    cycle = [o.lr_factor("onecycle", step, 10, 10, 5) for step in range(100)]  # the warmup is ignored
    assert cycle[0] == pytest.approx(o.ONECYCLE_LOW) and max(cycle) == pytest.approx(1.0)
    assert cycle.index(max(cycle)) == 30 and cycle[-1] < 0.001
    assert o.lr_factor("plateau", 50, 10, 10, 0, plateau=0.01) == 0.01
    with pytest.raises(ValueError, match="Unknown learning rate schedule"):
        o.lr_factor("linear", 0, 10, 10, 0)


class _Optimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.1}, {"lr": 1.0}]


def test_schedule_sets_every_group_and_plateau_waits(o):
    opt = _Optimizer()
    schedule = o.LrSchedule(opt, "plateau", epochs=5, steps_per_epoch=2, warmup_epochs=1)
    schedule.advance()
    assert opt.param_groups[0]["lr"] == pytest.approx(
        0.1 * o.WARMUP_START
    ) and schedule.lr() == pytest.approx(o.WARMUP_START)
    for _ in range(3):
        schedule.advance()
    assert schedule.lr() == 1.0
    for score in (0.5, 0.6, 0.6):
        schedule.end_epoch(score)
    assert schedule.plateau == 1.0
    schedule.end_epoch(0.4)  # second epoch without improvement
    schedule.advance()
    assert schedule.plateau == pytest.approx(o.PLATEAU_FACTOR) and schedule.lr() == pytest.approx(0.1)
    other = o.LrSchedule(_Optimizer(), "cosine", 5, 2)
    other.end_epoch(0.1)
    assert other.plateau == 1.0
    with pytest.raises(ValueError):
        o.LrSchedule(_Optimizer(), "bogus", 5, 2)


# -- mixed precision (#318) -------------------------------------------------------


def test_amp_only_on_cuda(o):
    assert o.amp_enabled("auto", "cuda") == (True, "") and o.amp_enabled("on", "cuda:1") == (True, "")
    assert o.amp_enabled("auto", "cpu") == (False, "") and o.amp_enabled("off", "cuda") == (False, "")
    used, note = o.amp_enabled("on", "mps")
    assert not used and "CUDA" in note
    with pytest.raises(ValueError, match="Unknown amp"):
        o.amp_enabled("half", "cpu")


def test_full_precision_step_and_non_finite_loss(o, torch):
    layer = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.weight.detach().clone()
    plain = o.Precision(False)
    with plain.autocast():
        loss = layer(torch.ones(1, 2)).sum()
    plain.step(loss, optimizer)
    assert not torch.equal(before, layer.weight)
    with pytest.raises(RuntimeError, match="not finite: try a lower"):
        plain.step(torch.tensor(float("nan")), optimizer)
    mixed = o.Precision(False)
    mixed.enabled = True  # the message names mixed precision (no CUDA needed to check it)
    with pytest.raises(RuntimeError, match="mixed precision"):
        mixed.step(torch.tensor(float("inf")), optimizer)


# -- device choice (#320) -----------------------------------------------------------


class _Backend:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class _Cuda(_Backend):
    def __init__(self, count):
        super().__init__(count > 0)
        self.count = count

    def device_count(self):
        return self.count


class _Torch:
    def __init__(self, gpus=0, mps=False):
        self.cuda = _Cuda(gpus)
        self.backends = type("Backends", (), {"mps": _Backend(mps)})()


def test_pick_device_auto_and_explicit(o):
    assert o.pick_device(_Torch(gpus=1)) == "cuda" and o.pick_device(_Torch(mps=True), "") == "mps"
    assert o.pick_device(_Torch()) == "cpu" and o.pick_device(_Torch(gpus=2), "cpu") == "cpu"
    assert (
        o.pick_device(_Torch(gpus=2), "cuda:1") == "cuda:1"
        and o.pick_device(_Torch(mps=True), "mps") == "mps"
    )


@pytest.mark.parametrize(
    ("fake", "choice", "message"),
    [
        (_Torch(gpus=1), "cuda:1", "GPU 1 does not exist"),
        (_Torch(), "cuda:0", "No CUDA GPU"),
        (_Torch(), "mps", "Apple GPU"),
        (_Torch(gpus=1), "gpu", "Unknown device"),
    ],
)
def test_pick_device_refuses_what_is_not_there(o, fake, choice, message):
    with pytest.raises(o.DeviceError, match=message):
        o.pick_device(fake, choice)


# -- out of GPU memory (#329) -------------------------------------------------------


class _Gpu:
    """A training step that runs out of memory above ``fits`` images."""

    def __init__(self, fits):
        self.fits, self.tried, self.cleared = fits, [], 0

    def step(self, batch):
        self.tried.append(batch)
        if batch > self.fits:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def clear(self):
        self.cleared += 1


class _Events:
    def __init__(self):
        self.seen = []

    def __call__(self, event, **fields):
        self.seen.append((event, fields))


def test_batch_is_halved_until_it_fits_and_each_halving_warns(o):
    gpu, events = _Gpu(fits=3), _Events()
    assert o.fit_batch(gpu.step, 16, events, clear=gpu.clear) == 2
    assert gpu.tried == [16, 8, 4, 2] and gpu.cleared == 3
    assert [f["params"] for _, f in events.seen] == [
        {"from": 16, "to": 8},
        {"from": 8, "to": 4},
        {"from": 4, "to": 2},
    ]
    assert {f["code"] for _, f in events.seen} == {"batch_halved"}
    assert events.seen[0][0] == "warning" and "batch of 16 images" in events.seen[0][1]["message"]
    assert o.fit_batch(_Gpu(fits=99).step, 8, events) == 8


def test_no_batch_fits_or_another_error(o):
    with pytest.raises(o.OutOfMemory, match="Not even a batch of 2"):
        o.fit_batch(_Gpu(fits=1).step, 8, _Events(), smallest=2)
    with pytest.raises(ValueError, match="not memory"):
        o.fit_batch(_broken, 8, _Events())
    assert o.is_out_of_memory(RuntimeError("CUDA out of memory")) and not o.is_out_of_memory(ValueError("x"))


def _broken(batch):
    raise ValueError("not memory")


class _Loader:
    """Gives one random batch of the size asked for and remembers the sizes."""

    def __init__(self, torch):
        self.torch, self.sizes = torch, []

    def __call__(self, batch_size):
        self.sizes.append(batch_size)
        return [self.torch.randn(batch_size, 3)]


def _loss(model, batch):
    return model(batch).sum()


def test_probe_leaves_weights_statistics_and_gradients_as_they_were(o, torch):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader_of = _Loader(torch)
    o.probe(4, model, loader_of, _loss, o.Precision(False), optimizer)
    assert loader_of.sizes == [4]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


# -- coded failures (#552) ------------------------------------------------------------


def test_failures_carry_a_code_and_params(o):
    with pytest.raises(o.OutOfMemory) as caught:
        o.fit_batch(_Gpu(fits=0).step, 2, _Events())
    assert o.coded(caught.value) == {"code": "out_of_memory", "params": {"smallest": 1}}
    with pytest.raises(o.DeviceError) as missing:
        o.pick_device(_Torch(gpus=1), "cuda:3")
    assert o.coded(missing.value) == {"code": "device_unavailable", "params": {"device": "cuda:3"}}
    with pytest.raises(o.DeviceError) as bad:
        o.pick_device(_Torch(), "gpu")
    assert o.coded(bad.value)["code"] == "device_invalid"
    assert o.coded(ValueError("plain")) == {}
    assert o.CodedFailure("x").code == "training_failed"


# -- class imbalance (#147) ----------------------------------------------------


def test_class_weights_none_and_empty_give_no_weights(o):
    assert o.class_weights([5, 1], "none") is None
    assert o.class_weights([0, 0], "inverse") is None
    with pytest.raises(ValueError, match="Unknown class weights"):
        o.class_weights([1, 1], "median")


def test_inverse_class_weights_are_capped_and_have_mean_one(o):
    weights = o.class_weights([100, 50, 1], "inverse")
    assert sum(weights) / 3 == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert max(weights) / min(weights) == pytest.approx(o.WEIGHT_CAP)  # 100x rarer, capped at 10x
    assert o.class_weights([10, 10], "inverse") == pytest.approx([1.0, 1.0])
    absent = o.class_weights([10, 0, 5], "inverse")
    assert absent[1] == max(absent)  # a class with no example takes the heaviest weight


def test_focal_loss_discounts_easy_examples(o, torch):
    logits = torch.tensor([[4.0, 0.0], [0.0, 0.5]])
    target = torch.tensor([0, 0])
    plain = torch.nn.functional.cross_entropy(logits, target)
    assert float(o.FocalLoss(0.0)(logits, target)) == pytest.approx(float(plain))
    assert 0 < float(o.FocalLoss(2.0)(logits, target)) < float(plain)
    weighted = o.build_loss("focal", [1.0, 3.0], 0.1, 2.0)(logits, target)
    assert float(weighted) > 0


def test_dice_loss_is_near_zero_for_a_perfect_prediction(o, torch):
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1
    perfect = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float() * 40.0
    assert float(o.DiceLoss()(perfect, target)) < 0.05
    background_only = torch.zeros(1, 2, 4, 4)
    background_only[:, 0] = 40.0
    assert float(o.DiceLoss()(background_only, target)) > 0.3
    assert float(o.DiceLoss(torch.tensor([1.0, 3.0]))(background_only, target)) > float(
        o.DiceLoss()(background_only, target)
    )


def test_build_loss_gives_each_kind(o, torch):
    logits = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    for name in ("cross_entropy", "focal", "dice", "cross_entropy_dice"):
        loss = o.build_loss(name, [1.0, 2.0, 0.5])(logits, target)
        assert loss.ndim == 0 and float(loss) > 0
    both = o.build_loss("cross_entropy_dice")(logits, target)
    assert float(both) > float(o.build_loss("dice")(logits, target))
    with pytest.raises(ValueError, match="Unknown loss"):
        o.build_loss("hinge")
