"""The locale files give each key once: json.load keeps the last of two equal keys, silently (#561)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from activelabel import i18n

I18N_DIR = Path(i18n.__file__).parent
LOCALES = sorted(I18N_DIR.glob("*.json"), key=lambda p: p.name)


class _KeyLog:
    """An ``object_pairs_hook`` that remembers the keys seen twice in one JSON object."""

    def __init__(self) -> None:
        self.repeats: list[str] = []

    def __call__(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Record the repeated keys of one object and build it the way json.load does.

        Args:
            pairs: the ``(key, value)`` pairs of one JSON object, in file order.

        Returns:
            The object as json.load builds it (the last value of a key wins).
        """
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                self.repeats.append(key)
            seen.add(key)
        return dict(pairs)


def find_repeats(path: Path) -> list[str]:
    """The keys that a JSON file gives more than once.

    Args:
        path: the JSON file to read (UTF-8).

    Returns:
        The repeated keys, in the order they appear in the file (empty when none).
    """
    log = _KeyLog()
    json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=log)
    return log.repeats


def test_locale_files_are_found():
    names = {p.name for p in LOCALES}
    assert {"en.json", "pt.json"} <= names


@pytest.mark.parametrize("path", LOCALES, ids=lambda p: p.name)
def test_no_key_is_repeated(path):
    """A key given twice would be overwritten in silence: each key appears once per file."""
    assert find_repeats(path) == []
