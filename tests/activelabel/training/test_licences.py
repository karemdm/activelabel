import json

import pytest

from activelabel.errors import CodedError
from activelabel.training.catalog import get_model
from activelabel.training.licences import (
    LICENCES,
    LicenceNotAccepted,
    LicenceStore,
    licence_of_model,
    licences_of_packages,
)


def test_only_ultralytics_needs_a_licence():
    assert licence_of_model(get_model("yolov8n")) is LICENCES["ultralytics"]
    assert licence_of_model(get_model("resnet18")) is None
    assert licence_of_model(get_model("fasterrcnn_resnet50_fpn")) is None
    assert LICENCES["ultralytics"].name == "AGPL-3.0"


@pytest.mark.parametrize(
    ("packages", "expected"),
    [
        (["ultralytics"], ["ultralytics"]),
        (["Ultralytics==8.4.218", "pillow"], ["ultralytics"]),
        (["ultralytics[export]>=8", "ultralytics"], ["ultralytics"]),
        (["torch", "timm", "numpy"], []),
        ([], []),
    ],
)
def test_licences_of_packages(packages, expected):
    assert [lic.id for lic in licences_of_packages(packages)] == expected


def test_acceptance_is_stored_with_its_date_in_the_workspace_settings(tmp_path):
    store = LicenceStore(tmp_path)
    assert not store.is_accepted("ultralytics") and store.accepted_at("ultralytics") is None
    date = store.accept("ultralytics")
    assert store.is_accepted("ultralytics") and store.accepted_at("ultralytics") == date
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["licences"]["ultralytics"] == {"accepted_at": date, "name": "AGPL-3.0"}
    assert LicenceStore(tmp_path).is_accepted("ultralytics")  # read back by another instance


def test_accepting_again_keeps_the_first_date_and_other_settings(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    store = LicenceStore(tmp_path)
    first = store.accept("ultralytics")
    assert store.accept("ultralytics") == first
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["theme"] == "dark"
    with pytest.raises(KeyError):
        store.accept("unknown")


def test_a_settings_file_that_is_not_a_mapping_counts_as_not_accepted(tmp_path):
    (tmp_path / "settings.json").write_text("[1]", encoding="utf-8")
    assert not LicenceStore(tmp_path).is_accepted("ultralytics")


def test_require_raises_a_coded_error_until_accepted(tmp_path):
    store = LicenceStore(tmp_path)
    with pytest.raises(LicenceNotAccepted) as caught:
        store.require([LICENCES["ultralytics"]])
    assert isinstance(caught.value, CodedError)
    assert caught.value.code == "licence_not_accepted"
    assert caught.value.params == {"licence": "ultralytics", "name": "AGPL-3.0"}
    store.require([])  # nothing to accept
    store.accept("ultralytics")
    store.require([LICENCES["ultralytics"]])


def test_status_lists_every_licence(tmp_path):
    store = LicenceStore(tmp_path)
    [before] = store.status()
    assert before["id"] == "ultralytics" and before["accepted"] is False and before["accepted_at"] is None
    assert before["url"].startswith("https://") and before["package"] == "ultralytics"
    date = store.accept("ultralytics")
    [after] = store.status()
    assert after["accepted"] is True and after["accepted_at"] == date
