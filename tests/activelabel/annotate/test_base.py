import pytest

from activelabel.annotate.base import Pulled, Sent, ToolError, ToolSettings, need_dataset
from activelabel.labels import Label


def test_need_dataset_refuses_none():
    with pytest.raises(ToolError, match="FiftyOne first"):
        need_dataset(None)
    need_dataset(object())


def test_the_secret_is_never_shown():
    settings = ToolSettings("cvat", "http://x", "user", "s3cret-value")
    assert "s3cret-value" not in repr(settings)
    assert Sent([{"label": "a", "url": "u"}], {"view": "v"}).ref == {"view": "v"}


def test_sent_keeps_no_edits_by_default():
    assert Sent([], {}).kept == 0 and Sent([], {}, kept=2).kept == 2


def test_pulled_is_the_labels_plus_what_the_tool_knows():
    pulled = Pulled({"a": [Label("cat")], "b": []}, {"a"}, {"a": "h"})
    assert dict(pulled) == {"a": [Label("cat")], "b": []}
    assert pulled.edited == {"a"} and pulled.written == {"a": "h"}
    assert Pulled({}, set()).written == {}


def test_pulled_tags_are_the_image_tags_or_none():
    assert Pulled({}, set()).tags is None
    source = {"a": ["check"]}
    pulled = Pulled({"a": []}, set(), tags=source)
    assert pulled.tags == {"a": ["check"]}
    source["b"] = ["x"]
    assert pulled.tags == {"a": ["check"]}  # a copy
