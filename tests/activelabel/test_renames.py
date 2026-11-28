"""Renaming a class reaches the files that keep class names besides the ground truth."""

from activelabel.preannotate import PreannotationStore
from activelabel.renames import case_renames, rename_classes
from activelabel.storage import write_json
from activelabel.training.config import RunConfig
from activelabel.training.models import ModelStore
from activelabel.training.runs import RunStore


def _model(project_dir, number, card):
    folder = ModelStore(project_dir).path(number)
    folder.mkdir(parents=True)
    write_json(folder / "model.json", {"number": number, "created": "t", "run": None, **card})


def test_case_renames_finds_only_changes_of_letter_case():
    assert case_renames(["cat", "dog"], ["Cat", "dog", "bird"]) == {"cat": "Cat"}
    assert case_renames(["cat"], ["kitten"]) == {}
    assert case_renames(["Cat"], ["Cat"]) == {}


def test_nothing_to_rename_touches_nothing(tmp_path):
    assert rename_classes(tmp_path, {}) == {"models": 0, "runs": 0, "preannotations": 0}
    assert not list(tmp_path.iterdir())


def test_a_trained_model_keeps_its_indexes_and_remembers_the_trained_names(tmp_path):
    _model(tmp_path, 1, {"classes": ["gato", "cão"], "model": "resnet18"})
    _model(tmp_path, 2, {"classes": ["cão"], "model": "resnet18"})
    counts = rename_classes(tmp_path, {"gato": "felino"})
    store = ModelStore(tmp_path)
    assert counts["models"] == 1
    first = store.get(1)
    assert first["classes"] == ["felino", "cão"] and first["trained_classes"] == ["gato", "cão"]
    assert "trained_classes" not in store.get(2)  # a model without the class is not touched
    rename_classes(tmp_path, {"felino": "gatinho"})  # a second rename keeps the names it was trained with
    card = store.get(1)
    assert card["classes"] == ["gatinho", "cão"] and card["trained_classes"] == ["gato", "cão"]


def test_a_public_model_mapping_follows_the_new_name(tmp_path):
    _model(
        tmp_path,
        1,
        {"source": "public", "classes": ["gato"], "mapping": {"cat": "gato", "kitten": "gato", "dog": "cão"}},
    )
    rename_classes(tmp_path, {"gato": "felino"})
    card = ModelStore(tmp_path).get(1)
    assert card["mapping"] == {"cat": "felino", "kitten": "felino", "dog": "cão"}
    assert card["classes"] == ["felino"] and "trained_classes" not in card


def test_runs_and_preannotation_runs_follow_the_new_name(tmp_path):
    runs = RunStore(tmp_path)
    number = runs.create(RunConfig(dataset=1, model="resnet18"))["number"]
    other = runs.create(RunConfig(dataset=1, model="resnet18"))["number"]
    runs.update(number, classes=["gato", "cão"])
    pre = PreannotationStore(tmp_path)
    chosen = pre.create(classes=["gato"])["number"]
    everything = pre.create(classes=None)["number"]
    counts = rename_classes(tmp_path, {"gato": "felino"})
    assert counts == {"models": 0, "runs": 1, "preannotations": 1}
    assert runs.get(number)["classes"] == ["felino", "cão"] and not runs.get(other).get("classes")
    assert pre.get(chosen)["classes"] == ["felino"] and pre.get(everything)["classes"] is None
