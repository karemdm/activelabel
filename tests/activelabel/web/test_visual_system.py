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


def test_step_markers_carry_the_state_in_the_shape():
    """Done is an outlined check in the area colour, the current step is the only filled circle."""
    done = re.search(r"^\.step\.is-done \.step-num \{([^}]*)\}", CSS, re.M)
    assert done and "background" not in done.group(1) and "var(--ok" not in done.group(1)
    assert re.search(r"^\.step-num \{[^}]*background: transparent", CSS, re.M)
    assert re.search(r"^\.step\.current \.step-num \{[^}]*background: var\(--group", CSS, re.M)
    assert "state === 'done' ? icon('check', 16) : state === 'attention' ? '!'" in JS
    assert "t(`step.state_${state}`)" in JS  # the state as hidden text for screen readers


def test_activities_is_a_pill_where_failures_stay_and_successes_fade():
    """#262: the pill is the default; it never covers the last row; screen readers hear the changes."""
    assert "return { min: true, closed:" in JS
    assert re.search(r"\.jobs-pill\.is-failed \{[^}]*var\(--danger\)", CSS)
    assert re.search(r"\.jobs-pill\.fades \{[^}]*animation: jobs-fade", CSS)
    assert "state !== 'failed' || jobsSeen.has(" in JS  # a failure fades only after it was opened
    assert re.search(r"\.page:has\(\.jobs-bar:not\(\[hidden\]\)\) \{ padding-bottom", CSS)
    assert "el('p', { class: 'sr-only', role: 'status' })" in JS
    assert re.search(r"prefers-reduced-motion[^\n]*\.jobs-pill\.fades \{ animation: none", CSS)


def test_type_has_no_uppercase_and_uses_mono_for_ids_and_metrics():
    """VISUAL 2.4: kickers are colour and spacing (no uppercase); ids, counts and metrics are mono."""
    assert "text-transform: uppercase" not in CSS
    assert re.search(r"--mono: ui-monospace", CSS)
    title = re.search(r"^\.step-group-title \{([^}]*)\}", CSS, re.M)
    assert title and "--mono" not in title.group(1)  # kickers and step summaries stay in the sans font
    mono = re.search(r"^\.project-id, ([^{]*)\{ font-family: var\(--mono\)", CSS, re.M)
    assert mono and ".thumb-name" in mono.group(1) and ".step-status" not in mono.group(1)
    assert re.search(r"^\.stat-tile dd \{[^}]*font-family: var\(--mono\)", CSS, re.M)


def test_running_badges_and_info_toasts_are_neutral():
    """VISUAL 2.3: the area colour is for place, not for state; running is told by its word."""
    for selector in (r"\.status-running, \.status-queued", r"\.tr-badge\.running", r"\.toast-info"):
        rule = re.search(rf"^{selector} \{{([^}}]*)\}}", CSS, re.M)
        assert rule and "--primary" not in rule.group(1) and "--area" not in rule.group(1), selector


def test_the_dialog_head_with_its_close_button_stays_in_view():
    """#333: the head (title and X) is sticky at the top of the scrolling body, like the foot."""
    rule = re.search(r"^\.dialog-body > \.dlg-head, \.dialog-body > \.tr-detail-head \{([^}]*)\}", CSS, re.M)
    assert rule and "position: sticky" in rule.group(1) and "top: -1.25rem" in rule.group(1)
    assert "background: var(--surface)" in rule.group(1)
    assert "head.append(title, el('button'" in JS  # every dialog opened by openModal gets the head and X
