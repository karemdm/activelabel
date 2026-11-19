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
