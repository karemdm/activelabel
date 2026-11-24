"""HTML pages of the interface. The data comes from the JSON API (``/api``) via JavaScript."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from activelabel.i18n import DEFAULT, SUPPORTED, messages, pick_language, translate

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
LANG_COOKIE = "al-lang"
ONE_YEAR = 365 * 24 * 3600

router = APIRouter()
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def compute_asset_version() -> str:
    """Short hash of the static files, so a changed file always gets a new URL."""
    digest = hashlib.sha256()
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(STATIC_DIR).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


_VERSION: dict[str, Any] = {"stamp": None, "value": ""}
_VERSION_LOCK = threading.Lock()


def asset_version() -> str:
    """:func:`compute_asset_version`, computed again only when a static file changed.

    The static files are served from disk, so after an update of the code (``git pull``) a running
    server already serves the new ones; their URLs must change too, or browsers keep old copies.
    """
    stamp = tuple(
        (p.as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(STATIC_DIR.rglob("*"))
        if p.is_file()
    )
    with _VERSION_LOCK:
        if _VERSION["stamp"] != stamp:
            _VERSION.update(stamp=stamp, value=compute_asset_version())
        return str(_VERSION["value"])


def _bound_translate(lang: str) -> Callable[..., str]:
    def t(key: str, **values: Any) -> str:
        return translate(key, lang, **values)

    return t


def render(request: Request, template: str, status_code: int = 200, **extra: Any) -> HTMLResponse:
    """Render a template with the language, translations and asset version in its context."""
    settings = getattr(request.app.state, "settings", None)
    default = getattr(settings, "language", DEFAULT)
    query = request.query_params.get("lang")
    lang = pick_language(
        query, request.cookies.get(LANG_COOKIE), request.headers.get("accept-language"), default
    )
    context = {
        "lang": lang,
        "languages": SUPPORTED,
        "t": _bound_translate(lang),
        "messages": messages(lang),
        "asset_version": asset_version(),
        "vl": {"lang": lang, "locale": "pt-BR" if lang == "pt" else "en", **extra.pop("vl", {})},
        **extra,
    }
    response = templates.TemplateResponse(request, template, context, status_code=status_code)
    if query and query.lower() in SUPPORTED:
        response.set_cookie(LANG_COOKIE, lang, max_age=ONE_YEAR, samesite="lax")
    return response


@router.get("/", response_class=HTMLResponse)
def projects_page(request: Request) -> HTMLResponse:
    """List of projects."""
    return render(request, "projects.html")


@router.get("/projects/new", response_class=HTMLResponse)
def new_project_page(request: Request) -> HTMLResponse:
    """Form to create a project."""
    return render(request, "new_project.html")


@router.get("/projects/import", response_class=HTMLResponse)
def import_project_page(request: Request) -> HTMLResponse:
    """Import a project from CVAT."""
    return render(request, "import_project.html")


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: str) -> HTMLResponse:
    """Page of one project (images, annotations, curation)."""
    if not (project_id.isascii() and project_id.isdigit()):
        return render(request, "notfound.html", status_code=404)
    return render(request, "project.html", project_id=int(project_id), vl={"projectId": int(project_id)})
