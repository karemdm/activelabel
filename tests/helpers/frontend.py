"""One way for the tests to read the front-end JavaScript, whatever files it is split into."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from activelabel.web import pages

STATIC = Path(pages.STATIC_DIR)


@lru_cache(maxsize=1)
def js_source() -> str:
    """Read all the JavaScript of the interface as one text.

    Returns:
        The text of every ``*.js`` file under the static folder (subfolders included), sorted by path and
        joined with a newline.
    """
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(STATIC.rglob("*.js")))
