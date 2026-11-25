"""Accept only calls made to this server by its own interface (or by a script on this machine).

The server has no login. Without this check, any page open in the same browser could call it: change or
delete projects (CSRF) or, by pointing a domain of its own at ``127.0.0.1`` (DNS rebinding), read it.
:class:`LocalOnly` refuses, with a coded HTTP 403:

* ``host_not_allowed``: a ``Host`` header that does not name this server: ``127.0.0.1``, ``localhost`` or
  ``[::1]``, the address the connection arrived at (the interface the server listens on, when it serves
  another one on purpose with ``--host``) or a name of ``ACTIVELABEL_ALLOWED_HOSTS``, always with the
  server's port;
* ``origin_not_allowed``: a request that changes data (any method but GET, HEAD and OPTIONS) whose
  ``Origin`` (or, without it, ``Referer``) is another site, or the opaque origin ``null``.

A script (``curl``, Python ``requests``) sends no ``Origin`` nor ``Referer``, so it keeps working as long
as it calls the server by the address it listens on (``http://127.0.0.1:8090``).
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from activelabel.web.errors import ApiError

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def split_host(value: str, scheme: str = "http") -> tuple[str, int] | None:
    """The name and the port of a ``Host`` value (``127.0.0.1:8090``, ``[::1]:8090``, ``localhost``).

    Args:
        value: The value of the header (or the address part of an origin).
        scheme: Scheme of the request, for the default port when the value has none.

    Returns:
        The name in lower case without brackets and the port; None when the value is not a valid address.
    """
    try:
        parts = urlsplit(f"//{value.strip()}")
        name, port = parts.hostname, parts.port
    except ValueError:  # a port that is not a number or out of range
        return None
    if not name or parts.username is not None or parts.path or parts.query or parts.fragment:
        return None
    return name.lower(), port if port is not None else DEFAULT_PORTS.get(scheme, 80)


def origin_address(value: str) -> tuple[str, int] | None:
    """The name and the port of an ``Origin`` or ``Referer`` value (``http://127.0.0.1:8090/path``).

    Args:
        value: The value of the header.

    Returns:
        The name and port of the site the request came from; None for ``null`` or an invalid value.
    """
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return None
    if parts.scheme not in DEFAULT_PORTS or not parts.netloc:
        return None
    return split_host(parts.netloc, parts.scheme)


class LocalOnly:
    """ASGI middleware: refuse requests that do not come to this server from its own pages or a script.

    Attributes:
        app: The application it protects.
        names: Host names accepted besides the loopback ones and the address of the connection.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str] = ()) -> None:
        """Wrap the application.

        Args:
            app: The application to protect.
            allowed_hosts: Extra host names accepted (``ACTIVELABEL_ALLOWED_HOSTS``), with any case.
        """
        self.app = app
        self.names = frozenset(name.strip().strip("[]").lower() for name in allowed_hosts if name.strip())

    def accepts(self, address: tuple[str, int] | None, server: tuple[str, int] | None) -> bool:
        """Whether an address names this server.

        Args:
            address: Name and port from a header (None when the header was invalid).
            server: Address and port the connection arrived at, as the ASGI server reports it (None when
                it does not; then only the name is checked).

        Returns:
            True for a loopback name, the connection's own address or an allowed name, on the server's port.
        """
        if address is None:
            return False
        name, port = address
        own = {server[0].strip("[]").lower()} if server else set()
        if name not in LOOPBACK | self.names | own:
            return False
        return server is None or port == server[1]

    def problem(self, scope: Scope) -> ApiError | None:
        """The reason to refuse a request, if any.

        Args:
            scope: The ASGI scope of an HTTP or WebSocket request.

        Returns:
            A coded 403 error (``host_not_allowed``, ``origin_not_allowed``); None to let the request go on.
        """
        headers = Headers(scope=scope)
        raw_server = scope.get("server")
        server = (str(raw_server[0]), int(raw_server[1])) if raw_server and raw_server[1] else None
        host = headers.get("host", "")
        if not self.accepts(split_host(host, scope.get("scheme", "http")), server):
            return ApiError(
                403,
                "host_not_allowed",
                f"This server only answers at its own address, not at '{host}'. "
                "Open it at http://127.0.0.1 with its port, or add the name to ACTIVELABEL_ALLOWED_HOSTS.",
                host=host,
            )
        if scope["type"] == "http" and scope.get("method", "GET") in SAFE_METHODS:
            return None
        origin = headers.get("origin")
        source = origin if origin is not None else headers.get("referer")
        if source is not None and not self.accepts(origin_address(source), server):
            return ApiError(
                403,
                "origin_not_allowed",
                f"A change was refused because it came from another site ({source}). "
                "Only the pages of this server, or a script that sends no Origin, may change data.",
                origin=source,
            )
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Answer 403 for a refused request; pass anything else to the application.

        Args:
            scope: The ASGI scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        error = self.problem(scope)
        if error is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        response = JSONResponse(
            {"detail": error.detail, "code": error.code, "params": error.params}, status_code=403
        )
        await response(scope, receive, send)
