"""Every interface text (``i18n/en.json``) is used by the interface, so a text whose code went missing shows.

A key counts as used when it is written in quotes in a ``static/js`` module or a template (with or without
the ``_one``/``_other`` plural suffix), or when it starts with a prefix that the JavaScript builds with a
template literal (for example ``tr.p.`` or ``error.``). Keys that only the server writes, or that a table
of the server names, go in ``SERVER_KEYS`` with the reason.

Issue #590: two fixes of the module split vanished in a merge and no test failed; their texts were still here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "activelabel"
WEB = SRC / "web"

# keys that the interface does not write itself (keep each with its reason)
SERVER_KEYS: dict[str, str] = {}

QUOTED = re.compile(r"['\"`]([A-Za-z0-9_]+(?:\.[A-Za-z0-9_\-]+)+)['\"`]")
BUILT = re.compile(r"`([A-Za-z0-9_]+\.[A-Za-z0-9_.]*)\$\{")
PLURAL = re.compile(r"_(one|other)$")


def interface_text() -> str:
    """The JavaScript modules and the templates, as one text.

    Returns:
        Every ``static/js/**/*.js`` and ``templates/*.html`` file, joined.
    """
    files = sorted((WEB / "static" / "js").rglob("*.js")) + sorted((WEB / "templates").glob("*.html"))
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def unused_keys() -> list[str]:
    """The keys of ``en.json`` that the interface never names.

    Returns:
        The keys, in the order of the file, without the ones in ``SERVER_KEYS``.
    """
    keys = json.loads((SRC / "i18n" / "en.json").read_text(encoding="utf-8"))
    text = interface_text()
    quoted = set(QUOTED.findall(text))
    prefixes = tuple(set(BUILT.findall(text)))
    return [
        key
        for key in keys
        if key not in SERVER_KEYS
        and key not in quoted
        and PLURAL.sub("", key) not in quoted
        and not key.startswith(prefixes)
    ]


def test_every_interface_text_is_used_by_the_interface() -> None:
    assert unused_keys() == [], "a text no module uses: restore the code that showed it, or remove the text"


def test_the_server_keys_exist() -> None:
    keys = json.loads((SRC / "i18n" / "en.json").read_text(encoding="utf-8"))
    assert sorted(set(SERVER_KEYS) - set(keys)) == []
