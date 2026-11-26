"""The visual system of the interface (issue #228): area colours, step markers, activities pill, type.

These are contract checks on ``app.css`` and the JavaScript (``static/js``): the look itself is checked in a
browser.
"""

import json
import re
from pathlib import Path

from frontend import js_source

STATIC = Path(__file__).resolve().parents[3] / "src" / "activelabel" / "web" / "static"
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
JS = js_source()


def test_the_project_page_takes_the_colour_of_its_task_type():
    """#621: inside a project the accent is the task colour; the home stays blue; Dados and Modelo are told
    apart by their step groups and titles, not by a colour."""
    assert "document.documentElement.dataset.task = p.task" in JS
    assert "dataset.area" not in JS and "data-area" not in CSS and "--dados" not in CSS
    assert "'data-group': group" in JS
    page = r":root\[data-task\] \{ --area: var\(--task, var\(--brand\)\); --area-soft: var\(--task-soft,"
    assert re.search(page, CSS)
    assert re.search(r"^  --area: var\(--brand\);", CSS, re.M)  # home and lists
    assert "--primary: var(--area)" in CSS
    assert re.search(r"\.topbar \{[^}]*border-bottom: 3px solid var\(--area\)", CSS)


def test_every_task_type_has_its_colour_its_icon_and_its_badge():
    """#621: each task of ``TASKS`` (names.js) sets ``--task`` for the page, its badge and its picker card;
    the badge and the home card use it, and the badge always carries the icon and the name."""
    names = (STATIC / "js" / "cv" / "names.js").read_text(encoding="utf-8")
    tasks = re.findall(r"'(\w+)'", re.search(r"export const TASKS = \[([^\]]*)\]", names).group(1))
    dom = (STATIC / "js" / "shell" / "dom.js").read_text(encoding="utf-8")
    for task in tasks:
        rule = (
            rf':root\[data-task="{task}"\], \.task-{task} \{{ '
            rf"--task: var\(--task-{task}\); --task-soft: var\(--task-{task}-soft\); \}}"
        )
        assert re.search(rule, CSS), task
        assert re.search(rf"^  {task}: '<", dom, re.M), f"no icon for {task}"
    assert "class: `badge task-badge task-${task}` }, icon(task, 14), taskName(task)" in JS
    assert "class: `project-card card task-${p.task}`" in JS
    assert "class: `task-card task-${task}" in JS
    badge = re.search(r"^\.task-badge \{([^}]*)\}", CSS, re.M)
    assert badge and "var(--task-soft" in badge.group(1) and "color: var(--task," in badge.group(1)
    assert re.search(r"^\.project-card \{[^}]*inset 0 3px 0 var\(--task, var\(--border\)\)", CSS, re.M)


def test_the_area_fades_and_the_fade_is_off_with_reduced_motion():
    assert re.search(r'@property --area \{[^}]*syntax: "<color>"', CSS)
    assert re.search(r":root \{ transition: --area ", CSS)
    assert re.search(r"prefers-reduced-motion: reduce\) \{ \* \{ transition: none !important", CSS)


def test_cards_are_flat_and_only_overlays_have_a_shadow():
    card = re.search(r"^\.card \{([^}]*)\}", CSS, re.M)
    assert card and "box-shadow" not in card.group(1)


def test_step_markers_carry_the_state_in_the_shape():
    """Done is an outlined check in the task colour, the current step is the only filled circle."""
    done = re.search(r"^\.step\.is-done \.step-num \{([^}]*)\}", CSS, re.M)
    assert done and "background" not in done.group(1) and "var(--ok" not in done.group(1)
    assert re.search(r"^\.step-num \{[^}]*background: transparent", CSS, re.M)
    assert re.search(r"^\.step\.current \.step-num \{[^}]*background: var\(--primary\)", CSS, re.M)
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


def test_phone_targets_are_44_px_wide_as_well_as_high():
    """#334: icon buttons, pager, language, brand, tabs, task names, summaries, checkboxes and chip X."""
    block = CSS[CSS.index("every target is at least 44x44 px (#334)") :]
    for selector in (
        ".icon-btn.small",
        ".tr-del",
        ".pager .btn",
        ".lb-nav",
        ".topnav a",
        ".task-title",
        "summary",
    ):
        assert selector in block, selector
    assert re.search(r"\.thumb-check \{ padding: 10px", block)  # 22px box + 2 x 10px + border = 44px
    assert re.search(r"\.chip-x \{ padding: 16px; margin: -", block)


def test_phone_segmented_controls_stack_and_scrollers_show_their_hidden_side():
    """#337 items 1-3: .seg stacks below 480 px; scrollers show an edge shadow; first column sticks."""
    block = CSS[CSS.index("/* #337: parts that did not fit a phone */") :]
    assert re.search(r"\.seg, [^{]*\{ display: flex; flex-direction: column", block)
    assert re.search(r"\.step-tabs, \.tr-table-wrap, [^{]*\{[^}]*no-repeat local", block)
    assert re.search(r"\.tr-table-wrap th:first-child, [^{]*\{ position: sticky; left: 0", block)
    # #636: the split table scrolls sideways instead of squeezing its first column to one letter per line
    assert ".ds-table { width: max-content; min-width: 100%; }" in block
    assert re.search(
        r'\.ds-table th\[scope="row"\], \.ds-table thead th:first-child \{ min-width: 8rem;', block
    )


def test_long_image_names_keep_their_end():
    """#338: a thumbnail name is cut at the start, so frame numbers and extensions stay visible."""
    assert re.search(r"^\.thumb-name \{ direction: rtl; text-align: left; \}", CSS, re.M)
    assert r'.thumb-name::before, .thumb-name::after { content: "\200E"; }' in CSS  # LTR marks
    assert "el('span', { class: 'thumb-name', title: img.name }" in JS  # the full name stays in the title


def test_curation_histograms_fill_the_row_and_limits_show_their_state():
    """#340 items 2, 3 and alignment."""
    block = CSS[CSS.index("/* #340 (Curation)") :]
    assert re.search(r"\.hist \{ flex: 1 1 240px", block)
    assert re.search(
        r'\.btn\[aria-controls="cur-limits"\]\[aria-expanded="true"\] \{ background: var\(--area-soft\)',
        block,
    )
    assert re.search(r"\.th-grid \.field\.inline \{[^}]*justify-content: flex-end", block)
    assert "'aria-controls': 'cur-limits'" in JS


def test_cards_use_the_whole_row_and_delete_is_not_alone_on_a_phone():
    """#341 items 4 and 5."""
    block = CSS[CSS.index("/* #341:") :]
    assert re.search(
        r"\.tr-cards \{ grid-template-columns: repeat\(auto-fit, minmax\(min\(100%, 340px\)", block
    )
    assert re.search(r"\.ds-cards \{ grid-template-columns: repeat\(auto-fit,", block)
    assert re.search(r"\.tr-actions > \.tr-del \{ grid-column: 2; grid-row: 1;", block)
    assert re.search(r"\.ds-del \{ position: absolute;", block)


def test_small_ux_fixes_of_the_audit():
    """#342 (CSS and wording): project types 2 x 2, the FiftyOne send button sized like the others, the class
    instruction in the help line (it was cut in the placeholder) and the annotation tile renamed."""
    assert re.search(r"\.task-cards:not\(\.tool-cards\) \{ grid-template-columns: repeat\(2,", CSS)
    assert re.search(r"\.send-row \.btn\.big \{ min-height: 40px; font-size: inherit;", CSS)
    i18n = STATIC.parent.parent / "i18n"
    pt = json.loads((i18n / "pt.json").read_text(encoding="utf-8"))
    en = json.loads((i18n / "en.json").read_text(encoding="utf-8"))
    assert pt["ann.tile_labels"] == "Objetos anotados" and en["ann.tile_labels"] == "Annotated objects"
    assert "Enter" not in pt["classes.placeholder"] and "Enter" in pt["new.classes_hint"]
    assert "Enter" not in en["classes.placeholder"] and "Enter" in en["new.classes_hint"]
