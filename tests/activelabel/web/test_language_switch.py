from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from activelabel.web import pages


def test_language_switch_keeps_the_page_and_step():
    app = FastAPI()
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=pages.STATIC_DIR), name="static")
    page = TestClient(app).get("/projects/1?lang=en").text
    # the link is rebuilt on click with the current path and step (#annotations, #curation...)
    assert "location.pathname + '?lang=pt' + location.hash" in page
