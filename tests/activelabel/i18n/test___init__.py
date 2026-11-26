from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from frontend import js_source

from activelabel import i18n
from activelabel.annotate import diff

I18N_DIR = Path(i18n.__file__).parent


def load(lang: str) -> dict[str, str]:
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def test_same_keys_in_both_languages():
    en, pt = load("en"), load("pt")
    assert set(en) == set(pt)
    assert all(en.values()) and all(pt.values())


def test_placeholders_match_between_languages():
    en, pt = load("en"), load("pt")
    for key in en:
        assert set(re.findall(r"\{(\w+)\}", en[key])) == set(re.findall(r"\{(\w+)\}", pt[key])), key


def test_translate_and_placeholders():
    assert i18n.translate("projects.title", "pt") == "Projetos"
    assert i18n.translate("classes.remove", "en", name="car") == "Remove class car"
    assert i18n.translate("classes.remove", "pt", name="car") == "Remover a classe car"
    assert "{name}" in i18n.translate("classes.remove", "en")  # unknown placeholder stays


def test_fallbacks():
    assert i18n.translate("projects.title", "xx") == "Projects"
    assert i18n.translate("no.such.key", "pt") == "no.such.key"
    assert i18n.messages("xx") == i18n.messages("en")


@pytest.mark.parametrize(
    ("query", "cookie", "accept", "default", "expected"),
    [
        ("pt", "en", "en", "en", "pt"),
        (None, "pt", "en", "en", "pt"),
        (None, None, "pt-BR,pt;q=0.9,en;q=0.8", "en", "pt"),
        (None, None, "fr, en;q=0.5, pt;q=0.9", "en", "pt"),
        (None, None, "fr", "pt", "pt"),
        ("zz", "yy", None, "en", "en"),
        (None, None, None, "klingon", "en"),
        (None, None, "pt;q=0", "en", "en"),
    ],
)
def test_pick_language(query, cookie, accept, default, expected):
    assert i18n.pick_language(query, cookie, accept, default) == expected


def test_messages_follow_changes_to_the_files(tmp_path, monkeypatch):
    """A running server picks up new texts after an update (the screen code is read from disk too)."""
    (tmp_path / "en.json").write_text(json.dumps({"a": "A"}), encoding="utf-8")
    (tmp_path / "pt.json").write_text(json.dumps({"a": "Á"}), encoding="utf-8")
    monkeypatch.setattr(i18n, "_DIR", tmp_path)
    monkeypatch.setattr(i18n, "_CACHE", {})
    assert i18n.messages("pt") == {"a": "Á"}
    (tmp_path / "pt.json").write_text(json.dumps({"a": "Á", "b": "Bê"}), encoding="utf-8")
    stamp = (tmp_path / "pt.json").stat().st_mtime_ns + 10**9
    os.utime(tmp_path / "pt.json", ns=(stamp, stamp))
    assert i18n.messages("pt") == {"a": "Á", "b": "Bê"}
    assert i18n.messages("en") == {"a": "A"}  # English filled from its own file, unchanged


def test_every_key_the_screen_uses_exists():
    """Keys written literally in the JS and the templates (dynamic ones are checked by their tests)."""
    web = Path(i18n.__file__).parents[1] / "web"
    js = js_source()
    html = "".join(p.read_text(encoding="utf-8") for p in (web / "templates").glob("*.html"))
    used = set(re.findall(r"(?<![\w.])tr?\(\s*'([A-Za-z0-9_.]+)'", js)) | set(
        re.findall(r"(?<![\w.])t\(\s*['\"]([A-Za-z0-9_.]+)['\"]", html)
    )
    assert len(used) > 100
    assert sorted(used - set(load("en"))) == []


SRC = Path(i18n.__file__).parents[1]


def test_every_plural_key_has_both_forms():
    """``tn('key', n)`` reads ``key_one`` and ``key_other``: a missing form would show the raw key."""
    js = js_source()
    keys = set(re.findall(r"(?<![\w.])tn\(\s*'([A-Za-z0-9_.]+)'", js))
    assert len(keys) > 20
    en = load("en")
    missing = [f"{k}{s}" for k in keys for s in ("_one", "_other") if f"{k}{s}" not in en]
    assert missing == []


def test_every_job_kind_the_server_starts_is_named():
    """The activity bar names a job by ``job.kind.<kind>``: every kind submitted anywhere needs one."""
    kinds: set[str] = set()
    for path in SRC.rglob("*.py"):
        kinds |= set(re.findall(r"jobs\.submit\(\s*[\w.]+,\s*\"(\w+)\"", path.read_text(encoding="utf-8")))
    assert {"training", "export", "preannotate", "video_task", "analyze"} <= kinds
    for lang in ("en", "pt"):
        messages = load(lang)
        missing = [k for k in sorted(kinds) if f"job.kind.{k}" not in messages and k != "sync_plan"]
        assert missing == [], lang


def test_progress_messages_are_translated():
    """Fixed progress messages the server writes are looked up as ``jobmsg.<text>``; the ones that carry a
    tool name are matched by shape in the JavaScript (``jobmsg.reading``, ``jobmsg.writing``...)."""
    texts: set[str] = set()
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        texts |= set(re.findall(r"update\(.*message=\"([^\"{]+)\"", source))
        texts |= set(re.findall(r"update\(\d+, \d+, \"([^\"{]+)\"", source))
    texts.add("Cancelling…")  # written by the job queue itself
    assert "Measuring images" in texts
    for lang in ("en", "pt"):
        messages = load(lang)
        listed = {"Listing CVAT data"}  # shape "Listing {tool} data"
        missing = [m for m in sorted(texts - listed) if f"jobmsg.{m}" not in messages]
        assert missing == [], lang
    for key in (
        "jobmsg.listing",
        "jobmsg.reading",
        "jobmsg.reading_annotations_from",
        "jobmsg.writing",
        "jobmsg.sending",
    ):
        assert "{tool}" in load("en")[key]


def test_every_import_problem_code_has_a_text():
    """Problems found while reading annotations are shown by ``problem.<code>``."""
    codes: set[str] = set()
    for path in (SRC / "importing").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        codes |= set(re.findall(r"_problem\(\s*\"(\w+)\"", source))
        codes |= set(re.findall(r"Problem\(\s*\"\w+\",\s*\"(\w+)\"", source))
    assert {"converted_to_box", "images_missing", "dropped_other_kind"} <= codes
    for lang in ("en", "pt"):
        messages = load(lang)
        assert [c for c in sorted(codes) if f"problem.{c}" not in messages] == [], lang


def test_every_rejection_reason_has_a_text():
    """Files refused on upload carry a short reason; the screen shows it as ``reason.<reason>``."""
    source = (SRC / "images" / "ingest.py").read_text(encoding="utf-8")
    reasons = set(re.findall(r"reject\([^,]+,\s*\"([a-z ]+)\"\)", source))
    reasons |= {"too small"}
    assert {"corrupt", "unsupported", "unsafe path", "archive is too large when expanded"} <= reasons
    for lang in ("en", "pt"):
        messages = load(lang)
        missing = [r for r in sorted(reasons) if f"reason.{r.replace(' ', '_')}" not in messages]
        assert missing == [], lang
    assert "reason.too_many_entries" in load("en")  # "too many entries (more than N)" carries a number


def test_session_scopes_have_a_label():
    """A session shows its scope as ``annotate.scope.<scope>``."""
    sessions = (SRC / "annotate" / "sessions.py").read_text(encoding="utf-8")
    scopes = set(re.findall(r"scope == \"(\w+)\"", sessions))
    assert {"all", "unlabeled", "selected", "preannotated", "issue"} <= scopes
    assert [s for s in sorted(scopes) if f"annotate.scope.{s}" not in load("en")] == []


def test_every_problem_of_a_pulled_session_has_a_text():
    """The problems of a diff (``annotate/diff.py``) are shown by ``problem.<code>``, not by their message."""

    for lang in ("en", "pt"):
        messages = load(lang)
        assert [c for c in sorted(diff.PROBLEM_MESSAGES) if f"problem.{c}" not in messages] == [], lang


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"


def test_fold_is_not_translated_in_portuguese():
    """#536: the owner keeps the term "fold" in Portuguese; "dobra(s)" must not come back."""
    texts = [f"pt.json {key}: {value}" for key, value in load("pt").items()]
    texts += [
        f"{path.name}: {line}"
        for path in DOCS_DIR.glob("*.pt-BR.md")
        for line in path.read_text("utf-8").splitlines()
    ]
    assert not [text for text in texts if re.search(r"\bdobras?\b", text, re.IGNORECASE)]


def test_no_text_names_label_studio_any_more():
    """Label Studio support was removed; the list of accepted formats still named "Label Studio JSON"."""
    assert not [k for lang in ("pt", "en") for k, v in load(lang).items() if "label studio" in v.lower()]


def test_help_texts_state_what_the_code_does():
    """Issue #545: split, transfer/finetune, rotation and mAP texts follow the code and the dataset guide."""
    for lang in ("en", "pt"):
        texts = load(lang)
        assert "backbone" in texts["tr.start_transfer_help"].lower()
        assert "only the last layer" not in texts["tr.start_transfer_help"]
        assert "só a última camada" not in texts["tr.start_transfer_help"]
        assert "every layer" not in texts["tr.start_finetune_help"]
        assert "todas as camadas" not in texts["tr.start_finetune_help"]
        assert "50-95" in texts["md.public_stats"]
        assert "accuracy" not in texts["md.public_stats"].lower()
        assert "precisão" not in texts["md.public_stats"].lower()
        assert "variedade" not in texts["ds.videos_help"] and "variety" not in texts["ds.videos_help"]
    assert "enclosing box" in load("en")["tr.augh.rotation"]
    assert "envolvente" in load("pt")["tr.augh.rotation"]
