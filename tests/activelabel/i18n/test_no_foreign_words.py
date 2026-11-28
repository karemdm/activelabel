"""The pt texts hold no common English UI words and the en texts no common Portuguese words (#222).

Some English words stay on purpose in the Portuguese texts: the technical terms of the project (task, job,
frame, grid, batch, epoch, ...) and the tool, format and model names. They are in ``ALLOWED``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from activelabel import i18n

I18N_DIR = Path(i18n.__file__).parent

#: Plain English words that should not show up in a Portuguese text (not cognates, not Portuguese words).
COMMON_EN = frozenset(
    "the and with your from this not are is file click save cancel delete error loading upload select "
    "back next close open search start stop please you can will all none or to of in on by be has have "
    "new set choose updated failed was were yes images image training project settings export import view "
    "edit add update create list items value reset confirm warning help it its an at if when until after "
    "before each every both also some these those which while log".split()
)

#: Plain Portuguese words that should not show up in an English text (not cognates, not English words).
COMMON_PT = frozenset(
    "de para com não uma um que da em dos das nos nas ao pelo pela você seu sua são está também depois "
    "antes mas porque pode arquivo arquivos salvar excluir cancelar carregando imagem imagens pontos sem "
    "ainda já onde quando todos todas nenhum nenhuma entre sobre este esta isso aqui então clique escolha "
    "selecione tente erro projeto projetos tarefa tarefas treino rótulo modelo modelos versão nome valor "
    "valores padrão fechar abrir voltar próximo sair enviando anotações anotação".split()
)

#: English terms that stay as they are in the Portuguese texts, on purpose.
ALLOWED = frozenset(
    {
        # the project's own terms (Task and Job are the CVAT names too)
        "task",
        "tasks",
        "job",
        "jobs",
        "fold",
        "folds",
        # ML and video jargon Brazilian practitioners keep in English
        "frame",
        "frames",
        "grid",
        "batch",
        "epoch",
        "epochs",
        "seed",
        "dataset",
        "datasets",
        "checkpoint",
        "checkpoints",
        "embedding",
        "embeddings",
        "backbone",
        "fine",
        "tuning",
        "download",
        "log",
        # tool, format and model names
        "cvat",
        "fiftyone",
        "yolo",
        "coco",
        "onnx",
        "resnet",
        "retinanet",
        "deeplab",
        "segformer",
        "rcnn",
        "mask",
        "faster",
        "keypointrcnn",
        "ultralytics",
        # key names
        "enter",
        "backspace",
    }
)

#: Literal code values quoted inside one message: they are not translated there.
ALLOWED_IN_KEY = {
    "error.fold_invalid": frozenset({"all"}),
    "error.start_in_params": frozenset({"start"}),
    "error.direction_invalid": frozenset({"both", "pull", "push"}),
    # the names of the FiftyOne screen, which is in English
    "annotate.fo_mouse_hint": frozenset({"create", "delete"}),
}


def load(lang: str) -> dict[str, str]:
    """The texts of one language file.

    Args:
        lang: the language code, ``"en"`` or ``"pt"``.

    Returns:
        The key and text pairs of ``<lang>.json``.
    """
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def foreign_words(key: str, text: str, common: frozenset[str]) -> list[str]:
    """The common words of the other language that a text uses without an excuse.

    Args:
        key: the message key, to look up the literals allowed in it.
        text: the message text. Placeholders such as ``{name}`` are left out.
        common: the common words of the other language, in lower case.

    Returns:
        The offending words, in the order they appear (empty when the text is clean).
    """
    cleaned = re.sub(r"\{[^}]*\}", " ", text)
    allowed = ALLOWED | ALLOWED_IN_KEY.get(key, frozenset())
    words = re.findall(r"[^\W\d_]+", cleaned)
    return [w for w in words if w.lower() in common and w.lower() not in allowed]


def test_pt_has_no_common_english_words():
    """A Portuguese text uses Portuguese words, except for the terms kept in English on purpose."""
    problems = {}
    for key, text in load("pt").items():
        found = foreign_words(key, text, COMMON_EN)
        if found:
            problems[key] = found
    assert not problems


def test_en_has_no_common_portuguese_words():
    """An English text uses English words; a Portuguese word there is a text left untranslated."""
    problems = {}
    for key, text in load("en").items():
        found = foreign_words(key, text, COMMON_PT)
        if found:
            problems[key] = found
    assert not problems


@pytest.mark.parametrize(
    ("key", "text", "common", "expected"),
    [
        ("x", "A grade de {n}", COMMON_EN, []),
        ("x", "Save the file", COMMON_EN, ["Save", "the", "file"]),
        ("x", "Salve o arquivo", COMMON_EN, []),
        ("error.start_in_params", "Use 'start' aqui", COMMON_EN, []),
        ("x", "Use 'start' aqui", COMMON_EN, ["start"]),
    ],
)
def test_checker_knows_its_exceptions(key, text, common, expected):
    """The checker flags the words it should and passes the allowed terms and placeholders."""
    assert foreign_words(key, text, common) == expected
