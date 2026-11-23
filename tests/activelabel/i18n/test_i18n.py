from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from activelabel import i18n

I18N_DIR = Path(i18n.__file__).parent


def load(lang: str) -> dict[str, str]:
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def test_same_keys_in_both_languages():
    en, pt = load("en"), load("pt")
    assert set(en) == set(pt)
    assert all(en.values()) and all(pt.values())


def test_placeholders_match_between_languages():
    import re

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
    """Keys written literally in app.js and the templates (dynamic ones are checked by their tests)."""
    web = Path(i18n.__file__).parents[1] / "web"
    js = (web / "static" / "app.js").read_text(encoding="utf-8")
    html = "".join(p.read_text(encoding="utf-8") for p in (web / "templates").glob("*.html"))
    used = set(re.findall(r"(?<![\w.])tr?\(\s*'([A-Za-z0-9_.]+)'", js)) | set(
        re.findall(r"(?<![\w.])t\(\s*['\"]([A-Za-z0-9_.]+)['\"]", html)
    )
    assert len(used) > 100
    assert sorted(used - set(load("en"))) == []
