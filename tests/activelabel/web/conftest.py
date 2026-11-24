import pytest
from import_fakes import FakeKeyring
from sync_fakes import FakeRemote

from activelabel.annotate import credentials
from activelabel.sync import registry
from activelabel.web import api


@pytest.fixture(autouse=True)
def remotes(monkeypatch):
    """In-memory remote projects instead of the real adapters (never reach a server)."""
    fakes = {"cvat": FakeRemote("cvat")}
    for tool, fake in fakes.items():
        monkeypatch.setitem(registry.OVERRIDES, tool, fake)
    return fakes


@pytest.fixture(autouse=True)
def no_fiftyone_cleanup(monkeypatch):
    """Deleting a project must not start FiftyOne in the tests (the cleanup has its own tests)."""
    monkeypatch.setattr(api, "drop_fiftyone_dataset", lambda *args: None)


@pytest.fixture
def fake_keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "keyring", fake)
    return fake
