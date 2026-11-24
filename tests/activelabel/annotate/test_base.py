import pytest

from activelabel.annotate.base import Sent, ToolError, ToolSettings, need_dataset


def test_need_dataset_refuses_none():
    with pytest.raises(ToolError, match="FiftyOne first"):
        need_dataset(None)
    need_dataset(object())


def test_the_secret_is_never_shown():
    settings = ToolSettings("cvat", "http://x", "user", "s3cret-value")
    assert "s3cret-value" not in repr(settings)
    assert Sent([{"label": "a", "url": "u"}], {"view": "v"}).ref == {"view": "v"}
