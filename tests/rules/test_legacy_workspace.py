"""A workspace made by an older version opens with the current code, and opening it changes nothing.

``tests/fixtures/make_workspace_v0.py`` builds a small workspace (see its docstring for what it holds). It is
copied to a temporary folder, the whole application is started on it (lifespan included, as the real server
does) and every main GET route is requested. Each must answer 200, and no file may be changed or removed:
reading old data never rewrites it. Only the derived caches in ``DERIVED`` may appear.

When a change to the stored format makes this test fail, the change needs a migration that runs on purpose
(and an entry in the format version), not a silent rewrite.
"""

import hashlib
import importlib.util
import shutil
from fnmatch import fnmatch
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from activelabel.config import Settings
from activelabel.web.app import create_app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "make_workspace_v0.py"

# What opening may create: caches that can be rebuilt (thumbnails, the probe of the training environment).
# Nothing that existed before may change, and nothing else may appear.
DERIVED = ["projects/*/thumbs", "projects/*/thumbs/*", "training", "training/env.json"]

# the project ids are 1..4 in the order of make_workspace_v0.TYPES
PROJECTS = {"classification": 1, "detection": 2, "segmentation": 3, "keypoints": 4}

PAGES = ["/", "/projects/new", "/projects/import", "/static/app.js", "/static/app.css"]
GLOBAL_API = ["/api/projects", "/api/annotation-tools", "/api/training/catalog", "/api/training/environment"]
PER_PROJECT = [
    "/projects/{p}",
    "/api/projects/{p}",
    "/api/projects/{p}/images",
    "/api/projects/{p}/jobs",
    "/api/projects/{p}/annotations",
    "/api/projects/{p}/curation",
    "/api/projects/{p}/curation/suggested",
    "/api/projects/{p}/annotate",
    "/api/projects/{p}/tasks",
    "/api/projects/{p}/datasets",
    "/api/projects/{p}/datasets/1",
    "/api/projects/{p}/models",
    "/api/projects/{p}/models/public-catalog",
    "/api/projects/{p}/preannotations",
    "/api/projects/{p}/links",
    "/api/projects/{p}/runs",
    "/api/projects/{p}/videos",
]
# only the projects that have them: the finished run and its saved model (1), the public model card (2)
CLASSIFICATION_ONLY = [
    "/api/projects/1/runs/1",
    "/api/projects/1/runs/1/log",
    "/api/projects/1/models/1",
    "/api/projects/1/models/1/download/pytorch",
]
DETECTION_ONLY = ["/api/projects/2/models/1"]
ROUTES = [
    *PAGES,
    *GLOBAL_API,
    *(route.format(p=p) for p in PROJECTS.values() for route in PER_PROJECT),
    *CLASSIFICATION_ONLY,
    *DETECTION_ONLY,
]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """The workspace as the generator leaves it (built once; every test works on a copy)."""
    spec = importlib.util.spec_from_file_location("make_workspace_v0", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(tmp_path_factory.mktemp("legacy"))


@pytest.fixture
def workspace(built, tmp_path) -> Path:
    copy = tmp_path / "workspace"
    shutil.copytree(built, copy)
    return copy


def snapshot(root: Path) -> dict[str, tuple[str, int] | None]:
    """Every file (hash and modification time) and every folder (None) under a folder."""
    state: dict[str, tuple[str, int] | None] = {}
    for path in sorted(root.rglob("*")):
        key = path.relative_to(root).as_posix()
        if path.is_dir():
            state[key] = None
        else:
            state[key] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return state


def test_the_generated_workspace_has_what_the_routes_need(built):
    names = sorted(p.name for p in (built / "projects").iterdir())
    assert names == ["1", "2", "3", "4", "next_id.json"]
    for project in PROJECTS.values():
        for name in ("project.json", "images.json", "annotations.json", "tasks.json", "curation.json"):
            assert (built / "projects" / str(project) / name).exists(), (project, name)
    detection = built / "projects" / "2"
    assert (detection / "links.json").exists() and (detection / "fiftyone.json").exists()
    assert (detection / "annotate.json").exists() and (detection / "models" / "m1" / "model.json").exists()
    assert (built / "projects" / "1" / "models" / "m1" / "best.pt").exists()


def test_every_main_route_answers_200_and_nothing_is_rewritten(workspace):
    before = snapshot(workspace)
    failures = []
    with TestClient(create_app(Settings(workspace=workspace))) as client:
        projects = client.get("/api/projects").json()
        assert [p["name"] for p in projects] == [
            "Pontos da mão — teste",
            "Segmentação de céu e chão",
            "Detecção de pessoas e carros",
            "Classificação de flores",
        ]
        assert not any(p.get("damaged") for p in projects)
        first = client.get("/api/projects/1/images").json()["items"][0]["id"]
        routes = [*ROUTES, f"/api/projects/1/images/{first}/thumb", f"/api/projects/1/images/{first}/file"]
        for route in routes:
            answer = client.get(route)
            if answer.status_code != 200:
                failures.append(f"{route} -> {answer.status_code}")
    assert not failures, failures
    after = snapshot(workspace)
    assert {k: v for k, v in after.items() if k in before} == before, "an existing file was changed"
    created = sorted(k for k in after if k not in before and not any(fnmatch(k, g) for g in DERIVED))
    assert not created, f"files created that are not derived caches: {created}"


def test_the_snapshot_notices_a_change(workspace):
    before = snapshot(workspace)
    (workspace / "projects" / "1" / "project.json").write_text("{}", encoding="utf-8")
    assert snapshot(workspace) != before
