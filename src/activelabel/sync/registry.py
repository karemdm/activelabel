"""Remote project adapters by tool name."""

from __future__ import annotations

import importlib
from typing import Any

from activelabel.annotate.base import ToolError
from activelabel.sync.remote import RemoteProject

OVERRIDES: dict[str, RemoteProject] = {}  # tests register fakes here
_LAZY = {
    "cvat": ("activelabel.sync.cvat_remote", "CvatRemote"),
    "labelstudio": ("activelabel.sync.labelstudio_remote", "LabelStudioRemote"),
}


def get_remote(tool: str) -> RemoteProject:
    """The remote project adapter of ``tool`` (``cvat`` or ``labelstudio``).

    Raises:
        ToolError: the tool cannot be synchronised or its adapter is not available.
    """
    if tool in OVERRIDES:
        return OVERRIDES[tool]
    if tool not in _LAZY:
        raise ToolError(f"The tool '{tool}' cannot be synchronised (use cvat or labelstudio).")
    module, cls = _LAZY[tool]
    try:
        factory: Any = getattr(importlib.import_module(module), cls)
        remote: RemoteProject = factory()
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"Synchronisation with '{tool}' is not available.") from exc
    return remote
