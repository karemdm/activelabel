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
    """Path of ``tools.json`` in the workspace.

    Args:
        workspace: the workspace folder.

    Returns:
        The path of the settings file.
    """
    return workspace / "tools.json"


def _key(tool: str, url: str) -> str:
    """Name under which the secret is kept in the keyring.

    Args:
        tool: name of the tool.
        url: URL of the tool.

    Returns:
        ``<tool>:<url>``.
    """
    return f"{tool}:{url}"


def _check_tool(tool: str) -> None:
    """Refuse a tool name that is not one of :data:`TOOLS`.

    Args:
        tool: name of the tool.

    Raises:
        ToolError: unknown tool.
    """
    if tool not in TOOLS:
        raise ToolError(f"Unknown annotation tool: {tool}")


def _get_secret(tool: str, url: str) -> str:
    """Secret of a tool from the keyring.

    Args:
        tool: name of the tool.
        url: URL of the tool.

    Returns:
        The secret, or an empty string when there is none, the URL is empty or no keyring works.
    """
    if not url:
        return ""
    try:
        return str(keyring.get_password(SERVICE, _key(tool, url)) or "")
    except Exception:  # noqa: BLE001 - no usable keyring: behave as "no secret"
        return ""


def server_url(workspace: Path, tool: str) -> str:
    """The server address saved for a tool, without reading the keyring.

    Args:
        workspace: the workspace folder holding ``tools.json``.
        tool: name of the tool.

    Returns:
        The address without a trailing slash, or an empty text when none was saved.
    """
    entry = (read_json(_file(workspace), {}) or {}).get(tool, {})
    return str(entry.get("url", "") if isinstance(entry, dict) else "").rstrip("/")


def load_settings(workspace: Path, tool: str) -> ToolSettings:
    """Settings of a tool, with the secret from the keyring when present.

    Args:
        workspace: the workspace folder holding ``tools.json``.
        tool: name of the tool.

    Returns:
        The settings (empty URL and user when nothing was saved).

    Raises:
        ToolError: unknown tool.
    """
    _check_tool(tool)
    entry = (read_json(_file(workspace), {}) or {}).get(tool, {})
    url, username = str(entry.get("url", "")), str(entry.get("username", ""))
    return ToolSettings(tool, url, username, _get_secret(tool, url))


def check_url(url: str) -> None:
    """Refuse a tool URL that is not ``http://`` or ``https://`` with a server name, or has spaces in it.

    Empty is allowed. Spaces around the URL are the caller's to trim (see :func:`save_settings`). A user or
    password inside the URL (``http://user:pw@host``) is refused: the URL is saved in plain text and shown
    back, while the secret belongs in the keyring.

    Args:
        url: the URL typed by the user.

    Raises:
        CodedError: ``url_invalid``, or ``url_has_credentials`` (the URL is not echoed back).
    """
    if not url:
        return
    parts = urlsplit(url)
    if "@" in parts.netloc:
        raise CodedError(
            "url_has_credentials", "Put the user and the password in their own fields, not in the URL."
        )
    if parts.scheme not in ("http", "https") or not parts.hostname or any(ch.isspace() for ch in url):
        raise CodedError(
            "url_invalid", "The URL must start with http:// or https:// and name a server.", url=url[:100]
        )


def save_settings(workspace: Path, tool: str, url: str, username: str, secret: str | None) -> ToolSettings:
    """Save URL and user to ``tools.json`` and, when given, the secret to the keyring.

    Args:
        workspace: the workspace folder holding ``tools.json``.
        tool: name of the tool.
        url: URL of the tool (trimmed, trailing ``/`` removed).
        username: user name (trimmed).
        secret: password or API token for the keyring; None or empty keeps the stored one.

    Returns:
        The settings as saved, with the secret from the keyring.

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
    """Remove the settings of a tool and its secret.

    Args:
        workspace: the workspace folder holding ``tools.json``.
        tool: name of the tool.

    Returns:
        Whether anything was removed.

    Raises:
        ToolError: unknown tool.
    """
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
    """Whether the tool has what it needs to be used.

    Args:
        settings: settings of the tool.

    Returns:
        True when the tool needs nothing (FiftyOne) or has a URL, a secret and, for CVAT, a user.
    """
    if settings.tool == "fiftyone":
        return True
    if not settings.url or not settings.secret:
        return False
    return settings.tool != "cvat" or bool(settings.username)


def public_view(settings: ToolSettings) -> dict[str, Any]:
    """What the API shows: no secret.

    Args:
        settings: settings of the tool.

    Returns:
        Tool name, whether it is configured, URL, user and the list of fields it needs.
    """
    return {
        "tool": settings.tool,
        "configured": is_configured(settings),
        "url": settings.url,
        "username": settings.username,
        "needs": list(NEEDS.get(settings.tool, [])),
    }
