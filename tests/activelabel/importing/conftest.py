import pytest
from import_fakes import FakeKeyring

from activelabel.annotate import credentials


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "keyring", fake)
    return fake
