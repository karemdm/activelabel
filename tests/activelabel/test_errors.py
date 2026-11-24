import pytest

from activelabel.errors import CodedError


def test_a_coded_error_keeps_its_code_message_and_params():
    error = CodedError("project_name_taken", "A project named 'Cats' already exists.", name="Cats")
    assert error.code == "project_name_taken"
    assert str(error) == "A project named 'Cats' already exists."
    assert error.params == {"name": "Cats"}


def test_params_are_optional():
    assert CodedError("x", "y").params == {}


def test_it_is_a_value_error_so_general_handlers_keep_working():
    with pytest.raises(ValueError, match="bad input"):
        raise CodedError("bad", "bad input")
