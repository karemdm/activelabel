import pytest
from import_fakes import FakeKeyring

from activelabel.annotate import credentials


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "keyring", fake)
    return fake


@pytest.fixture(autouse=True)
def remotes(monkeypatch):
    """In-memory remote projects instead of the real adapters (never reach a server)."""
    from sync_fakes import FakeRemote

    from activelabel.sync import registry

    fakes = {"cvat": FakeRemote("cvat"), "labelstudio": FakeRemote("labelstudio")}
    for tool, fake in fakes.items():
        monkeypatch.setitem(registry.OVERRIDES, tool, fake)
    return fakes
