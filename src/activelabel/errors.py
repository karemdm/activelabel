"""Errors with a stable code, so the interface can show them in its own language.

The ``message`` is English (for the API and the logs); the web layer sends ``code`` and ``params``
next to it, and the interface translates by code, filling the text with the params.
"""

from __future__ import annotations

from typing import Any


class CodedError(ValueError):
    """A rejected input (HTTP 400 at the web layer) with a code the interface can translate.

    A ``ValueError``, so code that handles bad input in general keeps working.

    Attributes:
        code: stable identifier, e.g. ``project_name_taken``.
        params: values the translated text needs, e.g. ``{"name": "Cats"}``.
    """

    def __init__(self, code: str, message: str, **params: Any) -> None:
        """Build the error.

        Args:
            code: stable identifier, e.g. ``project_name_taken``.
            message: English text for the API and the logs.
            **params: values the translated text needs.
        """
        super().__init__(message)
        self.code = code
        self.params = params
