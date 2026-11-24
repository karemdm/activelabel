"""Text helpers shared by the places that take free text from the user or show an error to them."""

from __future__ import annotations

import unicodedata


def has_control_characters(text: str, allow_newlines: bool = False) -> bool:
    """Whether ``text`` holds a control character (NUL, escape...), which no name or note needs.

    Args:
        text: the text to check.
        allow_newlines: accept line breaks and tabs (for multi-line descriptions).
    """
    allowed = "\n\r\t" if allow_newlines else ""
    return any(unicodedata.category(ch) == "Cc" and ch not in allowed for ch in text)


def message_of(exc: BaseException) -> str:
    """The text of an exception for people: a ``KeyError`` message without the quotes ``str`` adds."""
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)
