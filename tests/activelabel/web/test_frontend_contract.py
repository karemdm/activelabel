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


def test_a_stale_version_is_flagged_on_its_card_and_in_the_training_dialog():
    """The API's ``stale`` counts become a warning (icon and text) with a "new version from this" action; the
    training dialog warns too, and the warning does not disable the start button."""
    api = (SRC / "datasets.py").read_text(encoding="utf-8")
    assert '"stale"' in api and "changed_images" in api and "changed_images" in JS
    card = JS[
        JS.index("function versionCard") : JS.index("function draw()", JS.index("function versionCard"))
    ]
    assert "staleCount(v)" in card and "ds-new-from" in card and "openNew(null, v)" in card
    assert "drawStale()" in JS and "tr-dataset-stale" in JS
    assert ".ds-stale" in CSS
    for lang in ("pt", "en"):
        texts = (SRC / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        for key in (
            "ds.stale_one",
            "ds.stale_other",
            "ds.new_from",
            "ds.stale_train_one",
            "ds.stale_train_other",
        ):
            assert f'"{key}"' in texts, (lang, key)


def test_the_split_preview_ranks_classes_by_rarity_and_marks_zero_cells():
    """Rare classes come first; a 0 in a part with images has an icon and a hidden text; each cell shows
    the class's share; a missing-class warning says whether the class could have been in every part."""
    api = (SRC / "datasets.py").read_text(encoding="utf-8")
    assert '"available"' in api and '"avoidable"' in api
    assert "avoidable" in JS and "available" in JS and "ds-zero" in JS and "ds.zero_sr" in JS
    assert "Number(b.zero) - Number(a.zero)" in JS and "a.sum - b.sum" in JS
    assert ".ds-zero" in CSS and ".ds-share" in CSS
    for lang in ("pt", "en"):
        texts = (SRC / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        for key in (
            "ds.w_avoidable_one",
            "ds.w_unavoidable_other",
            "ds.w_unavoidable_some_one",
            "ds.zero_sr",
        ):
            assert f'"{key}"' in texts, (lang, key)


def test_the_safety_gap_is_a_switch_and_shared_video_parts_are_flagged():
    """The gap is an on-by-default switch (off sends 0 and warns), "new version from this" restores it, and
    the ``video_parts_shared`` warning and the per-video ``gap_seconds`` are shown."""
    assert "ds-gap-on" in JS and "gapBox.checked ? parseFloat(gapInput.value) : 0" in JS
    assert "gapBox.checked = gapSeconds > 0" in JS
    assert "video_parts_shared" in JS and "gap_seconds" in JS and "sharedAlert(" in JS
    assert ".ds-shared" in CSS and ".ds-gap-off" in CSS
    for lang in ("pt", "en"):
        texts = (SRC / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        for key in ("ds.gap_switch", "ds.gap_off_warn", "ds.video_gap_s", "ds.w_video_parts_shared"):
            assert f'"{key}"' in texts, (lang, key)


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


def test_the_split_screen_uses_the_protections_the_api_offers():
    """The names app.js reads and sends exist in the API: the preview's ``empty_parts`` and ``max_kfold``, the
    ``allow_empty`` and ``confirm_runs`` flags, and the codes that ask for them."""
    api = (SRC / "datasets.py").read_text(encoding="utf-8") + (SRC / "web" / "datasets_api.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "empty_parts",
        "max_kfold",
        "allow_empty",
        "confirm_runs",
        "dataset_used_by_runs",
        "kfold_too_few_groups",
    ):
        assert name in api, name
        assert name in JS, name
    start = JS.index("function versionCard")
    card = JS[start : JS.index("function draw()", start)]
    assert "trainBlockReason" in card
    assert "disabled: !!v.incomplete || !!noTrain" in card


def test_no_text_says_segmentation_pre_annotation_comes_later():
    """Pre-annotation of segmentation exists: the "no usable model" message says what is needed instead."""
    assert "kind_later" not in JS
    assert "pre.none_supported" in JS
    for lang in ("pt", "en"):
        texts = (SRC / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        assert "kind_later" not in texts
        assert not re.search(r"segmentation models comes later|segmentação virá depois", texts)


def test_a_run_whose_model_was_deleted_cannot_be_saved_again():
    """Deleting a model takes the run's weights: the dialog says so (and that the project loses its current
    model), and the run's save button is disabled with the reason, also after a refused save."""
    assert "ctx.weightless.add(m.run)" in JS
    assert "run_no_weights" in JS
    assert "r.has_weights === false" in JS  # the list says so, also after a reload
    card = JS[JS.index("const noWeights") :]
    assert "saveButton.disabled = true" in card[:400]
    assert "md.no_weights" in card[:1500]
    remove = JS[JS.index("async function removeModel") :]
    assert "md.delete_text_current" in remove[:600]


def test_every_select_has_the_field_look():
    """A select with no class of its own (the task chooser of the import) is styled like the other fields."""
    rule = re.search(r"^select \{([^}]*)\}", CSS, re.M)
    assert rule, "there is no base rule for select"
    assert "border: 1px solid var(--border)" in rule.group(1)
    assert "background: var(--surface)" in rule.group(1)


def test_the_public_model_button_is_hidden_when_the_type_has_no_public_model():
    """The models step asks the catalogue once and drops "add a public model" when it is empty."""
    models = JS[JS.index("let publicAvailable") :]
    assert "public-catalog" in models[:600]
    assert "publicBtn.hidden = publicAvailable === false" in models
    assert "md.empty_no_public" in models


def test_zero_is_plural_in_every_language():
    """``tn`` picks the singular text only for exactly 1 (Portuguese's own rule would also give it to 0)."""
    plural = JS[JS.index("const tn =") :]
    assert "n === 1 ? '_one' : '_other'" in plural[:200]
    assert "PluralRules" not in JS


def test_the_portuguese_texts_keep_task_and_say_no_english_for_boxes():
    """Task (the unit of images) stays "task", never "tarefa"; "caixas delimitadoras" has no English gloss."""
    pt = (SRC / "i18n" / "pt.json").read_text(encoding="utf-8")
    assert "bounding box" not in pt.lower()
    assert not re.search(r"tarefa", pt, re.I)


def test_changing_only_the_step_in_the_address_closes_the_open_windows():
    """A hashchange to another step closes the open dialogs (an upload that cannot be cancelled stays)."""
    handler = re.search(r"addEventListener\('hashchange', \(\) => \{.*?\n    \}\);", JS, re.S)
    assert handler and "closeOpenDialogs();" in handler.group(0)
    assert "dispatchEvent(new Event('cancel', { cancelable: true }))" in JS


def test_the_new_task_dialog_says_cancel_until_something_was_done():
    """Like the other dialogs, its foot is "Cancel"; only a finished upload or video turns it into "Close"."""
    button = re.search(r"id: 'new-task-close'.*?\n", JS).group(0)
    assert "t('common.cancel')" in button
    stage = re.search(r"function setStage\(stage\) \{.*?\n      \}", JS, re.S).group(0)
    assert "closeBtn.textContent = t(stage === 'done' ? 'common.close' : 'common.cancel')" in stage


def test_the_viewer_shows_the_class_of_a_classification_image():
    box = JS[JS.index("function openLightbox") :]
    assert "t('lightbox.classes')" in box[:3500] and "img.classes" in box[:3500]


def test_empty_strips_are_gone_from_task_cards_sessions_and_home():
    """Task cards fill the row, session buttons stay together, empty text wraps balanced (#421)."""
    assert re.search(r"\.task-list \{[^}]*flex-wrap: wrap", CSS)
    assert "danger-outline { margin-left: auto" not in CSS
    assert re.search(r"\.empty p \{[^}]*text-wrap: balance", CSS)


def test_task_grid_has_no_holes_and_a_long_name_stays_in_two_lines():
    """Rows are filled (flex-grow), an open card takes its own row, a name is clamped (#462)."""
    assert re.search(r"\.task-list > \.task-item \{[^}]*flex: 1 1", CSS)
    assert re.search(r"\.task-item:has\(\.task-toggle\[aria-expanded=\"true\"\]\) \{ flex-basis: 100%", CSS)
    assert re.search(r"\.task-title \{[^}]*-webkit-line-clamp: 2", CSS)
    assert "align-items: start; margin: 0" not in CSS
