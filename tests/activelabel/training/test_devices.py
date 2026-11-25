import pytest

from activelabel.errors import CodedError
from activelabel.training import devices
from activelabel.training.devices import DeviceError, available, check, check_here, choices
from activelabel.training.environment import TrainingEnvironment


class Env:
    """An environment that says what the computer has, without looking at it."""

    def __init__(self, gpus, mps):
        self.gpus, self.mps = gpus, mps

    def gpu_count(self):
        return self.gpus

    def mps_hint(self):
        return self.mps


def explode(self):
    raise AssertionError("the computer must not be looked at")


def test_choices():
    assert choices(0, False) == ["auto", "cpu"]
    assert choices(2, False) == ["auto", "cpu", "cuda:0", "cuda:1"]
    assert choices(1, True) == ["auto", "cpu", "cuda:0", "mps"]
    assert choices(0, True) == ["auto", "cpu", "mps"]


def test_available_uses_the_environment():
    assert available(Env(2, True)) == ["auto", "cpu", "cuda:0", "cuda:1", "mps"]
    assert available(Env(0, False)) == ["auto", "cpu"]


@pytest.mark.parametrize("choice", ["auto", "cpu", "cuda:1", "mps"])
def test_check_accepts_what_is_offered(choice):
    assert check(choice, ["auto", "cpu", "cuda:0", "cuda:1", "mps"]) == choice


@pytest.mark.parametrize("empty", [None, ""])
def test_check_empty_means_auto(empty):
    assert check(empty, ["auto", "cpu"]) == "auto"


@pytest.mark.parametrize("bad", ["gpu1", "cuda", "cuda:", "cuda:x", "CPU", "cuda:-1", " cpu", 5, 0, ["cpu"]])
def test_check_refuses_names_that_are_not_devices(bad):
    with pytest.raises(DeviceError) as error:
        check(bad, ["auto", "cpu", "cuda:0"])
    assert error.value.code == "device_invalid"


def test_check_refuses_a_device_this_computer_lacks():
    with pytest.raises(DeviceError) as error:
        check("cuda:3", ["auto", "cpu", "cuda:0"])
    assert error.value.code == "device_unavailable"
    assert "cuda:3" in str(error.value) and "cuda:0" in str(error.value)
    with pytest.raises(DeviceError) as error:
        check("mps", ["auto", "cpu"])
    assert error.value.code == "device_unavailable"


def test_device_error_is_a_coded_error():
    assert issubclass(DeviceError, CodedError)


@pytest.mark.parametrize(
    ("choice", "stored"), [("auto", "auto"), ("cpu", "cpu"), (None, "auto"), ("", "auto")]
)
def test_check_here_does_not_look_for_gpus_for_auto_and_cpu(tmp_path, monkeypatch, choice, stored):
    monkeypatch.setattr(TrainingEnvironment, "gpu_count", explode)
    monkeypatch.setattr(TrainingEnvironment, "mps_hint", staticmethod(explode))
    assert check_here(choice, tmp_path) == stored


def test_check_here_invalid_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingEnvironment, "gpu_count", lambda self: 0)
    monkeypatch.setattr(TrainingEnvironment, "mps_hint", staticmethod(lambda: False))
    with pytest.raises(DeviceError) as error:
        check_here("gpu1", tmp_path)
    assert error.value.code == "device_invalid"


def test_check_here_looks_at_the_computer_for_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingEnvironment, "gpu_count", lambda self: 2)
    monkeypatch.setattr(TrainingEnvironment, "mps_hint", staticmethod(lambda: False))
    assert check_here("cuda:1", tmp_path) == "cuda:1"
    with pytest.raises(DeviceError) as error:
        check_here("cuda:2", tmp_path)
    assert error.value.code == "device_unavailable"
    with pytest.raises(DeviceError) as error:
        check_here("mps", tmp_path)
    assert error.value.code == "device_unavailable"
    monkeypatch.setattr(TrainingEnvironment, "gpu_count", lambda self: 0)
    with pytest.raises(DeviceError):
        check_here("cuda:0", tmp_path)


def test_module_constants():
    assert devices.AUTO == "auto" and devices.ALWAYS == ("auto", "cpu")
