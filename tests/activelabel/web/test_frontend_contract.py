"""What the front end (app.js) relies on, checked without a JS runner: its own files and the API it reads."""

from __future__ import annotations

import re
from pathlib import Path

from api_client import make_client, upload
from sample_images import make_image

from activelabel.web import pages

STATIC = Path(pages.STATIC_DIR)
SRC = STATIC.parents[1]
JS = (STATIC / "app.js").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


def test_the_page_does_not_patch_the_dom():
    """Optional children are skipped by ``fill``/``add``/``el``, not by changing ``Element.prototype``."""
    assert "Element.prototype" not in JS
    assert ".replaceChildren(" not in re.sub(r"const fill = .*?\n  };", "", JS, flags=re.S)


def test_toasts_are_shown_inside_an_open_dialog():
    """A modal dialog hides everything behind it, so error toasts must be placed in it."""
    assert "dialog:modal" in JS
    assert re.search(r"function toastBox\(\)", JS)


def test_unused_rules_are_gone():
    assert ".annotate-box" not in CSS
    assert not re.search(r"^\.center\b", CSS, re.M)


def test_progress_message_shapes_match_what_the_server_writes():
    """Messages with a tool name are matched by shape in app.js; each one the server writes must match."""
    shapes = [re.compile(p) for p in re.findall(r"\[/(\^[^/]+\$)/, 'jobmsg\.\w+'\]", JS)]
    assert len(shapes) == 5
    written: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        written |= {
            m.replace("{tool}", "cvat").replace("{body.tool}", "cvat").replace("{session['tool']}", "cvat")
            for m in re.findall(r'progress\.update\(message=f"([^"]+)"', text)
        }
    assert {"Reading cvat", "Writing to cvat", "Sending to cvat", "Reading annotations from cvat"} <= written
    for message in sorted(written):
        assert any(s.match(message) for s in shapes), message


def test_the_annotation_summary_has_the_fields_the_screen_reads(tmp_path):
    """loadGt reads exactly these fields (it used to also try older names the API no longer sends)."""
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification", "classes": ["cat"]})
    assert upload(client, 1, [make_image(tmp_path / "in/a.png", seed=1)]).json()["added"] == 1
    for field in ("labels_per_class", "labels_total", "images_annotated", "images_preannotated"):
        assert re.search(rf"data\.{field}\b", JS), field
    assert client.get("/api/projects/1/annotations").json() == {"loaded": False}


def test_a_missing_comparison_is_the_only_stale_one(tmp_path):
    """Applying an unknown comparison answers 404 (stale); 409 means a job is running, which is not stale."""
    client, _ = make_client(tmp_path)
    client.post("/api/projects", json={"name": "P", "task": "classification"})
    assert client.post("/api/projects/1/sync/cvat/unknown/apply", json={}).status_code == 404
    assert "err.status === 404) st.error = 'stale'" in JS
    assert "err.status === 409) st.error = 'stale'" not in JS


def test_destructive_dialogs_start_on_cancel():
    """Enter must not confirm a deletion: dialogs with a destructive answer focus Cancel first."""
    assert "(input || (danger ? cancelBtn : okBtn)).focus()" in JS
    assert "cancelBtn.focus();  // Enter must never pick an answer" in JS
    assert "okBtn.focus()" not in JS


def test_steps_are_history_entries_and_errors_have_no_prefix():
    assert "history[ctx.shown ? 'pushState' : 'replaceState']" in JS
    assert "toast(String(err && err.message ? err.message : err), 'error', 10000)" in JS
    assert "t('common.error')} ${err" not in JS


def test_a_project_deleted_elsewhere_is_reported():
    """The jobs poll answers 404 for a deleted project: the page says so instead of staying silent."""
    assert re.search(r"err\.status === 404\) \{ showMissing\(\)", JS)
    assert "project.gone_title" in JS


def test_curation_actions_come_before_the_image_grid():
    """The actions of the step (analyze, FiftyOne, limits, reset) are a bar above the grid."""
    start = JS.index("function curationPanel()")
    panel = JS[start : JS.index("// ====", start)]
    mounted = panel[panel.index("add(node,") :]
    assert mounted.index("actionsBar") < mounted.index("grid.node")
    assert "fo-box" not in panel and "fo-box" not in CSS
    assert "quiet-danger cur-reset" in panel  # "reset analysis" is quiet and apart from the safe actions


def test_every_dialog_opens_through_the_one_pattern():
    """Dialogs get the same head (title and X) when they open, and their foot stays in view."""
    opened = re.findall(r"(\w+)\.showModal\(\)", JS)
    assert set(opened) <= {"dlg", "lightbox", "newDlg"}  # the others go through openModal()
    assert opened.count("dlg") == 1  # only inside openModal()
    assert re.search(r"\.dialog-body \.form-actions\s*\{[^}]*position: sticky", CSS)


def test_the_step_bar_marks_each_step_and_keeps_the_current_one_in_view():
    """A step shows done, attention or busy besides its number; a scrolling bar centres the current one."""
    for state in ("done", "attention", "busy"):
        assert f"step.state_{state}" in (SRC / "i18n" / "en.json").read_text(encoding="utf-8")
        assert f".step.is-{state}" in CSS
    assert "stepper.scrollLeft" in JS


def test_the_class_counts_show_the_biggest_few_with_search_and_see_all():
    """The annotation summary does not list every class: top rows, a search and a toggle."""
    assert "BARS_SHOWN" in JS and "ann.show_all" in JS and "stat-tiles" in JS


def test_a_dataset_version_leads_on_to_training_and_groups_its_warnings():
    """A card offers "train with this version" (the dialog opens on it) and one line for all its warnings."""
    assert "panel.startWith" in JS and "ctx.panels.training.startWith(v.version)" in JS
    assert "warnSummary(" in JS and ".ds-warn-group" in CSS


def test_the_training_settings_are_closed_with_a_line_for_what_is_not_the_default():
    """The hyperparameters sit in a closed section whose line lists only the changed values."""
    assert "changedLine()" in JS and "tr.params_default" in JS
    assert re.search(r"\.tr-adv\s*\{", CSS)


def test_empty_lists_carry_their_action_and_models_do_not_need_images():
    """Empty data splits, runs and models put the action in the empty state; step 6 is not gated by images."""
    assert JS.count("emptyState(") >= 4  # the helper and its three users
    assert "step !== 'models' && !hasImages()" in JS
