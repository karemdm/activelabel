import pytest

from activelabel.annotate import registry
from activelabel.annotate.base import ToolError
from activelabel.annotate.fiftyone_tool import FiftyOneTool


def test_known_tool_and_unknown_tool():
    assert isinstance(registry.get_tool("fiftyone"), FiftyOneTool)
    with pytest.raises(ToolError, match="not available"):
        registry.get_tool("nope")


def test_overrides_win(monkeypatch):
    fake = object()
    monkeypatch.setitem(registry.OVERRIDES, "fiftyone", fake)
    assert registry.get_tool("fiftyone") is fake
