"""Fakes for the import tests: an HTTP session with a route table, image bytes, a keyring."""

import copy
from pathlib import Path

from sample_images import make_image


class FakeResp:
    """Stand-in for ``requests.Response``."""

    def __init__(self, status=200, body=None, content=b"", content_type="application/json"):
        self.status_code = status
        self._body = body
        self.content = content
        self.headers = {"Content-Type": content_type}

    def json(self):
        return copy.deepcopy(self._body) if self._body is not None else {}

    def iter_content(self, size):
        yield self.content


def image_resp(path: Path) -> FakeResp:
    return FakeResp(200, content=path.read_bytes(), content_type="image/png")


class FakeSession:
    """Replaces ``requests.Session``: answers from ``routes``.

    ``routes`` maps ``(METHOD, path prefix)`` to a ``FakeResp`` or a function ``(kwargs) -> FakeResp``;
    the first matching prefix in insertion order wins (longest first is up to the caller).
    """

    base = ""

    def __init__(self, routes, base=""):
        self.routes, self.calls, self.headers, self.base = routes, [], {}, base

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def request(self, method, url, **kw):
        path = url[len(self.base) :] if url.startswith(self.base) else url
        self.calls.append((method, path, kw))
        for (m, prefix), reply in self.routes.items():
            if m == method and path.startswith(prefix):
                return reply(kw) if callable(reply) else reply
        return FakeResp(404)


def make_images(folder: Path, count: int) -> list[Path]:
    """PNG files with different content (so different ids)."""
    return [make_image(folder / f"img{i}.png", seed=i + 1, fmt="PNG") for i in range(count)]


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, key):
        return self.store.get((service, key))

    def set_password(self, service, key, secret):
        self.store[(service, key)] = secret

    def delete_password(self, service, key):
        del self.store[(service, key)]
