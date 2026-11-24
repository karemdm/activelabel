import pytest

from activelabel.training.category_names import PT_TO_COCO, coco_name, name_key
from activelabel.training.public_models import COCO_CATEGORIES


def test_name_key_ignores_case_accents_and_separators():
    assert name_key("Guarda-chuva") == "guarda chuva"
    assert name_key("  Cão_Grande ") == "cao grande"


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("gato", "cat"),
        ("Cão", "dog"),
        ("pessoa", "person"),
        ("carro", "car"),
        ("gatos", "cat"),
        ("ônibus", "bus"),
    ],
)
def test_portuguese_names_stand_for_coco_categories(name, category):
    assert coco_name(name) == category


def test_unknown_names_have_no_translation():
    assert coco_name("solda") is None and coco_name("cat") is None


def test_every_translation_points_to_a_real_coco_category():
    assert set(PT_TO_COCO.values()) <= set(COCO_CATEGORIES)
