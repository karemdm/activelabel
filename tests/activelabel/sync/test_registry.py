import pytest

from activelabel.annotate.base import ToolError
from activelabel.sync import registry
from activelabel.sync.cvat_remote import CvatRemote


def test_cvat_is_the_only_remote():
    assert isinstance(registry.get_remote("cvat"), CvatRemote)
    with pytest.raises(ToolError, match="cannot be synchronised"):
        registry.get_remote("fiftyone")


def test_overrides_win(monkeypatch):
    fake = object()
    monkeypatch.setitem(registry.OVERRIDES, "cvat", fake)
    assert registry.get_remote("cvat") is fake
