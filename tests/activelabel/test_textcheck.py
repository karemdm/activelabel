import pytest

from activelabel.textcheck import has_control_characters, message_of


@pytest.mark.parametrize("text", ["cat", "gato é bom", "日本語", "a b", ""])
def test_ordinary_text_has_no_control_characters(text):
    assert not has_control_characters(text)


@pytest.mark.parametrize("text", ["a\x00b", "esc\x1b[0m", "tab\there", "line\nbreak", "cr\rx"])
def test_control_characters_are_found(text):
    assert has_control_characters(text)


def test_line_breaks_and_tabs_can_be_allowed_but_nothing_else():
    assert not has_control_characters("a\nb\tc\r", allow_newlines=True)
    assert has_control_characters("a\x00b\n", allow_newlines=True)


def test_a_key_error_shows_its_message_without_quotes():
    assert message_of(KeyError("The image is gone")) == "The image is gone"
    assert str(KeyError("The image is gone")) == "'The image is gone'"


def test_other_exceptions_show_their_text():
    assert message_of(ValueError("bad")) == "bad"
    assert message_of(KeyError(7)) == "7"
    assert message_of(KeyError("a", "b")) == str(KeyError("a", "b"))
    assert message_of(RuntimeError()) == ""
