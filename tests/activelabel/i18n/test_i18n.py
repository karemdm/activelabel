from __future__ import annotations

import json
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
