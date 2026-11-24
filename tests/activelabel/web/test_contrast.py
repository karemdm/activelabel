"""The colour tokens of ``app.css`` meet WCAG AA contrast in the light and the dark theme.

Every token is written as ``--name: light-dark(#light, #dark);`` in the ``:root`` block; the test reads
those values and checks the pairs the interface really draws: text on the page and on cards, the area
accents (brand, Dados, Modelo) and the status colours as text, as fills under ``--on-accent`` and on their
soft backgrounds. 4.5:1 is the AA minimum for normal text, 3:1 for focus rings and other UI lines.
"""

import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parents[3] / "src" / "activelabel" / "web" / "static" / "app.css").read_text(
    encoding="utf-8"
)
TOKEN = re.compile(r"--([\w-]+):\s*light-dark\((#[0-9a-fA-F]{6}),\s*(#[0-9a-fA-F]{6})\)")
TOKENS = {name: (light, dark) for name, light, dark in TOKEN.findall(CSS)}
THEMES = (0, 1)  # index into (light, dark)
ACCENTS = ("brand", "dados", "modelo")
STATUS = ("ok", "warn", "danger")


def _luminance(colour: str) -> float:
    """Relative luminance of a ``#rrggbb`` colour (WCAG 2.x).

    Args:
        colour: the colour as ``#rrggbb``.

    Returns:
        The relative luminance, from 0 (black) to 1 (white).
    """
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _ratio(first: str, second: str) -> float:
    """WCAG contrast ratio between two token colours of one theme.

    Args:
        first: one colour as ``#rrggbb``.
        second: the other colour as ``#rrggbb``.

    Returns:
        The contrast ratio, from 1 to 21.
    """
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _pairs() -> list[tuple[str, str, float]]:
    """The (foreground, background, minimum ratio) token pairs the interface draws.

    Returns:
        One tuple per pair, with token names without the leading ``--``.
    """
    pairs = [(fg, bg, 4.5) for fg in ("text", "muted") for bg in ("bg", "surface", "surface-2")]
    for accent in ACCENTS:
        pairs += [
            ("on-accent", accent, 4.5),  # primary button, current step number, active chip
            (accent, "surface", 4.5),  # links, group titles, outlined accents
            (accent, f"{accent}-soft", 4.5),  # text on the current step pill
            ("text", f"{accent}-soft", 4.5),
        ]
    for status in STATUS:
        pairs += [(status, "surface", 4.5), (status, f"{status}-soft", 4.5), ("text", f"{status}-soft", 4.5)]
    pairs += [("focus", "bg", 3.0), ("focus", "surface", 3.0)]
    return pairs


def test_every_checked_token_is_defined_for_both_themes():
    needed = {name for fg, bg, _ in _pairs() for name in (fg, bg)}
    assert not needed - TOKENS.keys(), f"missing light-dark() tokens: {sorted(needed - TOKENS.keys())}"


@pytest.mark.parametrize(("fg", "bg", "minimum"), _pairs())
@pytest.mark.parametrize("theme", THEMES, ids=("light", "dark"))
def test_token_pairs_meet_wcag_aa(fg: str, bg: str, minimum: float, theme: int):
    ratio = _ratio(TOKENS[fg][theme], TOKENS[bg][theme])
    assert ratio >= minimum, f"--{fg} on --{bg} is {ratio:.2f}:1 (needs {minimum}:1)"


def _mix(colour: str, base: str, share: float) -> str:
    """``color-mix(in srgb, colour share, base)`` as the browser computes it.

    Args:
        colour: the mixed-in colour as ``#rrggbb``.
        base: the base colour as ``#rrggbb``.
        share: the share of ``colour``, from 0 to 1.

    Returns:
        The mixed colour as ``#rrggbb``.
    """
    channels = (
        round(int(colour[i : i + 2], 16) * share + int(base[i : i + 2], 16) * (1 - share)) for i in (1, 3, 5)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("theme", THEMES, ids=("light", "dark"))
def test_data_series_tints_stay_apart(accent: str, theme: int):
    """#335: train (the area colour) and validation (a 55% tint) are told apart from each other and from the
    card; test is a hatch of the area colour. Every segment also has its written label, so 2:1 is enough here
    (the bar is not the only carrier of the information, WCAG 1.4.11)."""
    assert "--area) 55%, var(--surface))" in CSS
    area, surface = TOKENS[accent][theme], TOKENS["surface"][theme]
    tint = _mix(area, surface, 0.55)
    assert _ratio(tint, area) >= 2.0 and _ratio(tint, surface) >= 2.0, (accent, tint)


def test_data_series_and_best_values_do_not_use_status_colours():
    for rule in (
        ".ds-val",
        ".ds-test",
        ".tr-line.s2",
        ".tr-best",
        ".md-table td.best",
        ".tr-rank tr.best th",
    ):
        body = re.search(re.escape(rule) + r"[^{]*\{([^}]*)\}", CSS)
        assert body and not re.search(r"--(ok|warn|danger)", body.group(1)), rule
