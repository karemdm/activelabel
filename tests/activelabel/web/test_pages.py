from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from activelabel.i18n import messages
from activelabel.web import pages

WEB = Path(pages.__file__).parent
ISSUES = ["blurry", "dark", "bright", "low_contrast", "small", "odd_aspect", "grayscale", "near_duplicate"]
TASKS = ["classification", "detection", "segmentation", "keypoints"]


def make_client(language: str | None = None) -> TestClient:
    app = FastAPI()
    if language:
        app.state.settings = SimpleNamespace(language=language)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=str(pages.STATIC_DIR)), name="static")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return make_client()


@pytest.mark.parametrize("path", ["/", "/projects/new", "/projects/import", "/projects/7"])
@pytest.mark.parametrize("lang", ["en", "pt"])
def test_pages_render_in_both_languages(client, path, lang):
    response = client.get(path, params={"lang": lang})
    assert response.status_code == 200
    assert f'<html lang="{lang}">' in response.text
    assert pages.translate("nav.projects", lang) in response.text
    data = re.search(r"window\.I18N = (\{.*?\}); window\.VL", response.text, re.S)
    assert data and json.loads(data.group(1)) == messages(lang)


def test_import_page_and_entry_point(client):
    html = client.get("/projects/import", params={"lang": "en"}).text
    assert 'data-page="import"' in html and 'id="import-root"' in html
    assert 'href="/projects/import"' in client.get("/", params={"lang": "en"}).text


def test_project_page_knows_its_id(client):
    html = client.get("/projects/42").text
    assert 'data-project-id="42"' in html
    assert '"projectId": 42' in html or '"projectId":42' in html


@pytest.mark.parametrize("path", ["/projects/abc", "/projects/1x", "/projects/-1", "/projects/%C2%B2"])
def test_non_numeric_project_is_a_friendly_404(client, path):
    response = client.get(path, params={"lang": "pt"})
    assert response.status_code == 404
    assert "Página não encontrada" in response.text


def test_asset_version_in_urls(client):
    html = client.get("/").text
    assert f"/static/app.css?v={pages.ASSET_VERSION}" in html
    assert f"/static/app.js?v={pages.ASSET_VERSION}" in html
    assert re.fullmatch(r"[0-9a-f]{10}", pages.ASSET_VERSION)
    assert pages.compute_asset_version() == pages.ASSET_VERSION
    assert client.get("/static/app.js").status_code == 200


def test_language_cookie_and_default(client):
    first = client.get("/", params={"lang": "pt"})
    assert "al-lang=pt" in first.headers["set-cookie"]
    assert '<html lang="pt">' in client.get("/").text  # remembered by the cookie
    assert '<html lang="en">' in client.get("/", params={"lang": "en"}).text

    configured = make_client("pt")
    assert '<html lang="pt">' in configured.get("/").text
    assert "set-cookie" not in configured.get("/").headers


def test_language_switch_links(client):
    html = client.get("/").text
    assert 'href="?lang=en"' in html and 'href="?lang=pt"' in html
    assert 'id="theme-toggle"' in html


def test_every_translation_key_in_templates_and_js_exists():
    en = messages("en")
    used: set[str] = set()
    for template in (WEB / "templates").glob("*.html"):
        used |= set(re.findall(r"\bt\(\s*'([\w.]+)'", template.read_text(encoding="utf-8")))
    js = (WEB / "static" / "app.js").read_text(encoding="utf-8")
    used |= set(re.findall(r"\bt\(\s*['\"`]([\w.]+)['\"`]", js))
    for key in re.findall(r"\btn\(\s*['\"`]([\w.]+)['\"`]", js):
        used |= {f"{key}_one", f"{key}_other"}
    assert used
    assert sorted(used - set(en)) == []


def test_dynamic_translation_keys_exist():
    en = messages("en")
    wanted = [f"task.{t}" for t in TASKS] + [f"task.{t}.help" for t in TASKS]
    wanted += [f"issue.{i}" for i in ISSUES] + [f"issue.{i}.help" for i in ISSUES]
    wanted += [f"step.{s}" for s in ("images", "annotations", "curation")]
    wanted += [f"ann.mode_{m}" for m in ("merge", "replace")] + [
        f"ann.mode_{m}.help" for m in ("merge", "replace")
    ]
    wanted += [f"job.status.{s}" for s in ("queued", "running", "done", "failed", "cancelled")]
    wanted += [f"metric.{m}" for m in ("brightness", "contrast", "sharpness")]
    wanted += [f"reason.{r}" for r in ("corrupt", "unsupported", "too_small")]
    wanted += [f"format.{f}" for f in ("coco", "yolo", "voc", "cvat", "labelstudio", "folders")]
    wanted += [f"import.tool.{x}.help" for x in ("cvat", "labelstudio")]
    wanted += [f"import.kind.{x}" for x in ("project", "task")]
    wanted += ["jobmsg.listing", "jobmsg.Downloading images", "jobmsg.Reading annotations"]
    wanted += ["jobmsg.Creating the project", "job.kind.import_project"]
    wanted += ["job.kind.import_folder", "job.kind.analyze", "job.kind.fiftyone"]
    wanted += ["job.kind.annotate_send", "job.kind.annotate_pull"]
    wanted += ["job.kind.sync_plan", "job.kind.sync_plan_generic", "job.kind.sync_apply"]
    tools = ("fiftyone", "cvat", "labelstudio")
    wanted += [f"annotate.tool.{x}" for x in tools] + [f"annotate.tool.{x}.help" for x in tools]
    wanted += [f"annotate.setup_help.{x}" for x in ("cvat", "labelstudio")]
    wanted += [f"annotate.scope.{x}" for x in ("all", "unlabeled", "selected", "issue")]
    wanted += [f"annotate.status.{x}" for x in ("open", "pulled", "applied", "discarded", "closed")]
    try:
        from activelabel.annotations.models import PROBLEMS

        wanted += [f"problem.{code}" for code in PROBLEMS]
    except ImportError:
        pass
    try:
        from dataclasses import asdict

        from activelabel.curation.issues import Thresholds

        wanted += [f"threshold.{name}" for name in asdict(Thresholds())]
    except ImportError:
        pass
    assert sorted(set(wanted) - set(en)) == []


def test_js_has_no_external_resources():
    for name in ("app.js", "app.css"):
        text = (WEB / "static" / name).read_text(encoding="utf-8")
        assert not re.search(r"(?:src|href|url\()\s*=?\s*[\"']?https?://", text)
    for template in (WEB / "templates").glob("*.html"):
        assert "http://" not in template.read_text(encoding="utf-8").replace("http://www.w3.org", "")
