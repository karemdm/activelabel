"""The visual system of the interface (issue #228): area colours, step markers, activities pill, type.

These are contract checks on ``app.css`` and ``app.js``: the look itself is checked in a browser.
"""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[3] / "src" / "activelabel" / "web" / "static"
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")


def test_the_project_page_sets_its_area_from_the_current_step():
    assert "document.documentElement.dataset.area" in JS
    assert "'data-group': group" in JS
    assert re.search(r':root\[data-area="data"\]\s*\{[^}]*--area: var\(--dados\)', CSS)
    assert re.search(r':root\[data-area="model"\]\s*\{[^}]*--area: var\(--modelo\)', CSS)
    assert "--primary: var(--area)" in CSS


def test_the_area_fades_and_the_fade_is_off_with_reduced_motion():
    assert re.search(r'@property --area \{[^}]*syntax: "<color>"', CSS)
    assert re.search(r":root \{ transition: --area ", CSS)
    assert re.search(r"prefers-reduced-motion: reduce\) \{ \* \{ transition: none !important", CSS)


def test_cards_are_flat_and_only_overlays_have_a_shadow():
    card = re.search(r"^\.card \{([^}]*)\}", CSS, re.M)
    assert card and "box-shadow" not in card.group(1)
