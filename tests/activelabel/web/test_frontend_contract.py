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


def test_model_cards_and_the_public_mapping_stay_short_and_lead_to_pre_annotation():
    """The class list opens on demand, and a model card can open pre-annotation on that model."""
    assert "md.mapping_summary" in JS and "md.classes_n" in JS
    assert "preannotateWith(m.number)" in JS and "panel.preannotateWith" in JS


def test_the_import_report_replaces_the_drop_zone_and_offers_other_files_when_nothing_is_usable():
    """The report is seen first; with nothing usable the way on is another file, not only Discard."""
    assert "drop.hidden = true" in JS and "drop.hidden = false" in JS
    assert "ann.choose_other" in JS and "ann.nothing_usable" in JS


def test_the_project_form_has_two_columns_and_the_cards_show_coverage():
    """New and edit project share the two-column form; a project card shows annotation coverage as a bar."""
    template = (SRC / "web" / "templates" / "new_project.html").read_text(encoding="utf-8")
    assert 'class="form-cols"' in template and "form-cols" in JS and ".form-cols" in CSS
    assert "project.coverage" in JS


def test_the_project_header_shows_the_classes_on_one_line_with_see_all():
    """The header keeps the classes to one line; the full list opens in a dialog."""
    assert "classLine(p.classes)" in JS and "classes.see_all" in JS
    assert re.search(r"\.class-line-chips\s*\{[^}]*flex-wrap: nowrap", CSS)


def test_split_warnings_of_the_label_checks_have_texts():
    """Warnings the split adds for label content (#112, #113) and the duplicates cause of a big group."""
    import json
    from pathlib import Path

    i18n = Path(__file__).parents[3] / "src" / "activelabel" / "i18n"
    for lang in ("en", "pt"):
        texts = json.loads((i18n / f"{lang}.json").read_text(encoding="utf-8"))
        for code in ("labels_without_geometry", "multiple_classes", "keypoints_count_mismatch"):
            for form in ("one", "other"):
                assert f"ds.w_{code}_{form}" in texts and f"ds.b_{code}_{form}" in texts, (lang, code, form)
        assert "ds.w_group_drift_duplicates" in texts and "ds.warn_g_labels" in texts


def test_api_errors_are_translated_by_code_and_never_by_the_server_text_alone():
    """``api()`` keeps the code and params; the text comes from ``error.<code>`` (server text: fallback)."""
    body = JS[JS.index("function errorMessage") :]
    body = body[: body.index("\n  }\n")]
    assert "`error.${code}` in I18N" in body
    assert body.index("`error.${code}` in I18N") < body.index("String(detail)")
    api = JS[JS.index("async function api(") :]
    api = api[: api.index("\n  }\n")]
    assert "errorMessage(data, response.status)" in api
    assert "err.code" in api and "err.params" in api
    assert "detailText" not in JS


def test_every_error_param_the_page_names_is_a_param_the_server_sends():
    """The ``ERROR_PARAMS`` table of app.js only names params that some error really carries."""
    table = JS[JS.index("const ERROR_PARAMS") :]
    named = set(re.findall(r"^    (\w+):", table[: table.index("\n  };")], re.M))
    sent: set[str] = set()
    for path in SRC.rglob("*.py"):
        sent |= set(re.findall(r"\b(\w+)=", path.read_text(encoding="utf-8")))
    assert named <= sent, named - sent


def test_the_step_bar_contains_its_hidden_texts():
    """``.sr-only`` is absolutely positioned: without a positioned ``.stepper`` its texts escape the bar's
    scroll box and make the whole page scroll sideways on a phone."""
    rule = re.search(r"^\.stepper \{([^}]*)\}", CSS, re.M)
    assert rule, "the .stepper rule is gone"
    assert "position: relative" in rule.group(1)
    assert "overflow-x: auto" in rule.group(1)
