"""Interface translations (English and Brazilian Portuguese).

Messages live in ``en.json`` and ``pt.json`` (flat ``key -> text``, same keys in both).
Texts may contain ``{name}`` placeholders. English is the fallback for missing keys.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

SUPPORTED = ("en", "pt")
DEFAULT = "en"

_DIR = Path(__file__).parent
_CACHE: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}
_CACHE_LOCK = threading.Lock()


def _load(lang: str) -> dict[str, str]:
    """Messages of one language file, read again whenever the file changes.

    A running server keeps up with an update of the code (``git pull``): the page and its
    JavaScript are read from disk on each request, so the texts must be too.

    Args:
        lang: language code, the name of the file without ``.json``.

    Returns:
        The messages (key to text) of that language only.
    """
    path = _DIR / f"{lang}.json"
    info = path.stat()
    stamp = (info.st_mtime_ns, info.st_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(lang)
        if cached and cached[0] == stamp:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = {str(k): str(v) for k, v in data.items()}
        _CACHE[lang] = (stamp, loaded)
        return loaded


def messages(lang: str) -> dict[str, str]:
    """All messages of a language, with English filling any missing key.

    Args:
        lang: ``en`` or ``pt`` (anything else gives English).

    Returns:
        A new dict that is safe to modify.
    """
    merged = dict(_load(DEFAULT))
    if lang in SUPPORTED and lang != DEFAULT:
        merged.update(_load(lang))
    return merged


def translate(key: str, lang: str, **values: Any) -> str:
    """Text of ``key`` in ``lang`` with ``{name}`` placeholders replaced.

    Falls back to English, then to the key itself. Unknown placeholders are left as they are.

    Args:
        key: message key.
        lang: ``en`` or ``pt``.
        **values: what replaces each ``{name}`` placeholder.

    Returns:
        The text.
    """
    text = messages(lang).get(key, key)
    return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), text)


def _from_accept_language(header: str | None) -> str | None:
    """First supported language of an Accept-Language header, by quality.

    Args:
        header: the header value, e.g. ``pt-BR,pt;q=0.9,en;q=0.8``; None or empty when absent.

    Returns:
        ``en`` or ``pt``, or None when the header names no supported language.
    """
    if not header:
        return None
    ranked: list[tuple[float, int, str]] = []
    for position, part in enumerate(header.split(",")):
        pieces = part.strip().split(";")
        tag = pieces[0].strip().lower().split("-")[0]
        quality = 1.0
        for extra in pieces[1:]:
            name, _, value = extra.strip().partition("=")
            if name == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        if tag in SUPPORTED and quality > 0:
            ranked.append((-quality, position, tag))
    return min(ranked)[2] if ranked else None


def pick_language(
    query: str | None = None,
    cookie: str | None = None,
    accept_language: str | None = None,
    default: str = DEFAULT,
) -> str:
    """Choose the interface language.

    Order: ``?lang=`` in the URL, the remembered cookie, the browser's Accept-Language,
    the configured default, English.

    Args:
        query: the ``lang`` parameter of the URL.
        cookie: the language remembered in the cookie.
        accept_language: the Accept-Language header of the browser.
        default: the configured language, used when nothing above names a supported one.

    Returns:
        ``en`` or ``pt``.
    """
    for candidate in (query, cookie):
        if candidate and candidate.lower() in SUPPORTED:
            return candidate.lower()
    from_browser = _from_accept_language(accept_language)
    if from_browser:
        return from_browser
    return default if default in SUPPORTED else DEFAULT
