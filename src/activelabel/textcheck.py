"""Checks shared by the places that take free text from the user (names, descriptions)."""

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
