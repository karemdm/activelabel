"""Settings of the annotation tools: URL and user in ``<workspace>/tools.json``, secret in the keyring.

The secret (password or API token) is never written to a file, logged or returned.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import keyring

from activelabel.annotate.base import TOOLS, ToolError, ToolSettings
from activelabel.errors import CodedError
from activelabel.storage import read_json, write_json

SERVICE = "activelabel"
NEEDS = {
    "fiftyone": [],
    "cvat": ["url", "username", "secret"],
}
_lock = threading.RLock()


def _file(workspace: Path) -> Path:
    return workspace / "tools.json"


def _key(tool: str, url: str) -> str:
    return f"{tool}:{url}"


def _check_tool(tool: str) -> None:
    if tool not in TOOLS:
        raise ToolError(f"Unknown annotation tool: {tool}")


def _get_secret(tool: str, url: str) -> str:
    if not url:
        return ""
    try:
        return str(keyring.get_password(SERVICE, _key(tool, url)) or "")
    except Exception:  # noqa: BLE001 - no usable keyring: behave as "no secret"
        return ""


def load_settings(workspace: Path, tool: str) -> ToolSettings:
    """Settings of a tool, with the secret from the keyring when present."""
    _check_tool(tool)
    entry = (read_json(_file(workspace), {}) or {}).get(tool, {})
    url, username = str(entry.get("url", "")), str(entry.get("username", ""))
    return ToolSettings(tool, url, username, _get_secret(tool, url))


def check_url(url: str) -> None:
    """Refuse a tool URL that is not ``http://`` or ``https://`` with a server name (empty is allowed).

    Raises:
        CodedError: ``url_invalid``.
    """
    if not url:
        return
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CodedError(
            "url_invalid", "The URL must start with http:// or https:// and name a server.", url=url[:100]
        )


def save_settings(workspace: Path, tool: str, url: str, username: str, secret: str | None) -> ToolSettings:
    """Save URL and user to ``tools.json`` and, when given, the secret to the keyring.

    Raises:
        ToolError: unknown tool or the keyring is not usable.
        CodedError: the URL is not an http(s) address (see :func:`check_url`).
    """
    _check_tool(tool)
    url = url.strip().rstrip("/")
    check_url(url)
    with _lock:
        if secret:
            try:
                keyring.set_password(SERVICE, _key(tool, url), secret)
            except Exception as exc:  # noqa: BLE001
                raise ToolError(
                    f"Could not store the secret in the system keyring ({type(exc).__name__})."
                ) from exc
        data: dict[str, Any] = read_json(_file(workspace), {}) or {}
        data[tool] = {"url": url, "username": username.strip()}
        write_json(_file(workspace), data)
    return load_settings(workspace, tool)


def forget(workspace: Path, tool: str) -> bool:
    """Remove the settings of a tool and its secret. Returns whether anything was removed."""
    _check_tool(tool)
    with _lock:
        data: dict[str, Any] = read_json(_file(workspace), {}) or {}
        entry = data.pop(tool, None)
        if entry is not None:
            write_json(_file(workspace), data)
        url = str((entry or {}).get("url", ""))
        removed = entry is not None
        if url:
            try:
                keyring.delete_password(SERVICE, _key(tool, url))
                removed = True
            except Exception:  # noqa: BLE001, S110 - nothing stored
                pass
        return removed


def is_configured(settings: ToolSettings) -> bool:
    """Whether the tool has what it needs to be used."""
    if settings.tool == "fiftyone":
        return True
    if not settings.url or not settings.secret:
        return False
    return settings.tool != "cvat" or bool(settings.username)


def public_view(settings: ToolSettings) -> dict[str, Any]:
    """What the API shows: no secret."""
    return {
        "tool": settings.tool,
        "configured": is_configured(settings),
        "url": settings.url,
        "username": settings.username,
        "needs": list(NEEDS.get(settings.tool, [])),
    }
