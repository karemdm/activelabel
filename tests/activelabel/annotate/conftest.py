import pytest

from activelabel.annotate import credentials


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, key):
        return self.store.get((service, key))

    def set_password(self, service, key, secret):
        self.store[(service, key)] = secret

    def delete_password(self, service, key):
        if (service, key) not in self.store:
            raise KeyError(key)
        del self.store[(service, key)]


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
