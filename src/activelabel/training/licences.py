"""Licences a person must accept before a training package is installed or used (docs/TRAINING.md).

Ultralytics (the YOLO models) is licensed AGPL-3.0, which asks more of the people who distribute or offer a
service built on it than the other training packages do. ActiveLabel therefore installs it, and trains with
it, only after the person accepted that licence. The acceptance (the licence id, its name and the date) is
kept in the workspace settings, ``<workspace>/settings.json``, under ``licences``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activelabel.errors import CodedError
from activelabel.storage import now, read_json, write_json
from activelabel.training.catalog import ModelSpec

SETTINGS_FILE = "settings.json"


@dataclass(frozen=True)
class Licence:
    """A licence that needs an explicit acceptance.

    Attributes:
        id: stable identifier (also the family that needs it).
        name: licence name (SPDX).
        package: the distribution it covers.
        url: where its text is.
    """

    id: str
    name: str
    package: str
    url: str


LICENCES: dict[str, Licence] = {
    "ultralytics": Licence(
        "ultralytics", "AGPL-3.0", "ultralytics", "https://www.gnu.org/licenses/agpl-3.0.html"
    ),
}


class LicenceNotAccepted(CodedError):
    """A package or model needs a licence the person has not accepted yet (code ``licence_not_accepted``)."""

    def __init__(self, licence: Licence) -> None:
        """Build the error for a licence.

        Args:
            licence: the licence that is missing.
        """
        super().__init__(
            "licence_not_accepted",
            f"{licence.package} is licensed {licence.name}: accept the licence before installing or "
            "training with it.",
            licence=licence.id,
            name=licence.name,
        )


def licence_of_model(model: ModelSpec) -> Licence | None:
    """The licence a model needs accepted.

    Args:
        model: catalogue entry.

    Returns:
        The licence of its family, or ``None`` when the family needs none.
    """
    return LICENCES.get(model.family)


def licences_of_packages(packages: Iterable[str]) -> list[Licence]:
    """The licences the given requirements need accepted.

    Args:
        packages: pip requirements (``ultralytics``, ``timm``...).

    Returns:
        One licence per requirement that has one, without repeats.
    """
    by_package = {lic.package: lic for lic in LICENCES.values()}
    found: dict[str, Licence] = {}
    for requirement in packages:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""
        if name in by_package:
            found[name] = by_package[name]
    return list(found.values())


class LicenceStore:
    """The licence acceptances of a workspace."""

    def __init__(self, workspace: Path | str) -> None:
        """Locate the settings of a workspace; nothing is written yet.

        Args:
            workspace: folder of the app's data.
        """
        self.path = Path(workspace) / SETTINGS_FILE

    def _read(self) -> dict[str, Any]:
        """The whole settings file.

        Returns:
            The settings (empty when there is no file or it is not a mapping).
        """
        data = read_json(self.path, {})
        return data if isinstance(data, dict) else {}

    def accepted_at(self, licence_id: str) -> str | None:
        """When a licence was accepted.

        Args:
            licence_id: a key of :data:`LICENCES`.

        Returns:
            The ISO date of the acceptance, or ``None`` when it was not accepted.
        """
        entry = (self._read().get("licences") or {}).get(licence_id)
        return str(entry["accepted_at"]) if isinstance(entry, dict) and entry.get("accepted_at") else None

    def is_accepted(self, licence_id: str) -> bool:
        """Whether a licence was accepted.

        Args:
            licence_id: a key of :data:`LICENCES`.

        Returns:
            ``True`` when the settings hold its acceptance.
        """
        return self.accepted_at(licence_id) is not None

    def accept(self, licence_id: str) -> str:
        """Record the acceptance of a licence (the first date is kept when it was accepted before).

        Args:
            licence_id: a key of :data:`LICENCES`.

        Returns:
            The date of the acceptance.

        Raises:
            KeyError: unknown licence.
        """
        licence = LICENCES[licence_id]
        existing = self.accepted_at(licence_id)
        if existing:
            return existing
        data = self._read()
        date = now()
        entries = dict(data.get("licences") or {})
        entries[licence_id] = {
            "accepted_at": date,
            "name": licence.name,
        }
        write_json(self.path, {**data, "licences": entries})
        return date

    def require(self, licences: Iterable[Licence]) -> None:
        """Check that every licence was accepted.

        Args:
            licences: the licences a use needs.

        Raises:
            LicenceNotAccepted: the first licence that was not accepted.
        """
        for licence in licences:
            if not self.is_accepted(licence.id):
                raise LicenceNotAccepted(licence)

    def status(self) -> list[dict[str, Any]]:
        """Every licence with its state, for the API.

        Returns:
            ``{id, name, package, url, accepted, accepted_at}`` per licence.
        """
        return [
            {
                "id": lic.id,
                "name": lic.name,
                "package": lic.package,
                "url": lic.url,
                "accepted": self.is_accepted(lic.id),
                "accepted_at": self.accepted_at(lic.id),
            }
            for lic in LICENCES.values()
        ]
