# ActiveLabel — assessment

An engineering review of the architecture and of the development process: what is solid, what is weak,
what can go wrong, and a prioritised proposal. It states facts that can be checked in the repository and
marks opinions as such. Cost: S (hours), M (a day or two), L (a week or more). Risk is the risk of the
change itself.

## Strengths

- **Clear layering.** Domain code (`labels`, `datasets`, `tasks_store`, `sync`, `training`) does not import
  the web layer; the routers in `web/` are thin and share one `AppState`. Remote tools sit behind small
  protocols (`sync/remote.py`, `annotate/base.py`).
- **Data you can inspect.** Everything is JSON/YAML on disk, written atomically (`storage.write_json`),
  with a lock per folder and numbers that are never reused. A project can be copied, diffed and repaired
  by hand.
- **Safe by default for people's work.** Reversible exclusion, confirmations, backups before destructive
  annotation changes, a three-way sync with a review step, pre-annotations that never overwrite a person.
- **Reproducibility.** Frozen dataset versions with a fingerprint, run configuration as YAML, seeds.
- **Heavy work is isolated.** Training, export and pre-annotation run in a separate environment and
  process, behind one queue, with temporary folders removed at the end and crash recovery at start-up.
- **Tooling.** ruff (lint, docstrings, annotations), mypy with `disallow_untyped_defs`, pre-commit, CI on
  Linux and Windows (Python 3.10 and 3.12), about 600 tests in 58 files, a separate FiftyOne job.
- **Documentation per feature**, with the HTTP contract next to the behaviour.

## Weaknesses

1. **Frontend monolith.** `web/static/app.js` is one file of more than 5,000 lines with no build step, no
   modules, no lint and no tests. Every screen shares its scope. It is the largest maintainability risk.
2. **Large modules.** `sync/engine.py` (about 1,300 lines), `training/runner.py` (about 1,100), `datasets.py`
   (about 850) and `web/api.py` (about 800) mix several concerns. The runner holds training, export,
   pre-annotation, recovery and process handling in one class.
3. **Duplication.** `SYNC_TOOLS` is defined in both `sync_api.py` and `annotate_api.py`; each router has its
   own `_store`, `_find`, `_settings` and `_tool` helpers with small differences; the "stage a file, check
   the token, find the project folder" pattern is repeated. The per-folder lock registry is re-implemented in eight modules
   (`manifest`, `labels`, `datasets`, `tasks_store`, `video`, `preannotate`, `models`, `runs`).
4. **Two job systems.** `AppState.jobs` (light jobs, one worker for the whole app) and the runner's own
   `JobManager` (heavy jobs) are separate. A long analysis of one project delays the sync of another, and
   `busy()` has to consult both. Reasonable today; confusing to extend.
5. **Whole-file reads.** `annotations.json` and `images.json` are read and rewritten as a whole on every
   change, under a lock. Fine for thousands of images; slow for hundreds of thousands, and every writer
   pays for the largest file. `GET /images` loads the whole manifest and the whole ground truth for each page.
6. **Stringly-typed boundaries.** Job kinds, scopes, actions and statuses are strings repeated in the
   Python code, in `app.js` and in the docs. A typo is a runtime error.
7. **Docs and code can drift.** Endpoint tables and the file layout are written by hand; the audit that led
   to this document found many differences (names, endpoints, files on disk).

## Risks

### Security of a local server without login

- There is **no authentication, no origin check and no CORS policy**. The default bind (`127.0.0.1`) keeps
  other machines out, but any web page open in the same browser can send requests to
  `http://127.0.0.1:8090` (cross-site request forgery) and a hostile DNS name that resolves to
  `127.0.0.1` can read answers (DNS rebinding). The API can delete projects, read images and use the
  stored CVAT login to contact the configured server. With `--host 0.0.0.0` the whole API is open to the
  network.
- `ACTIVELABEL_IMPORT_ROOT` limits path imports to a folder (good), but it is a second trust boundary to
  keep checked (symlinks, `..`).
- Secrets are in the system keyring (good); `tools.json` holds only addresses and users. Log output must
  never include them (a test for this is cheap).
- Uploaded archives and annotation files are untrusted input: size and entry limits, path traversal in
  zips, XML (use a hardened parser) and decompression bombs deserve a review and fuzz-style tests.
- The training environment installs packages from public indexes on demand: supply-chain exposure, with no
  version pinning or hash checking.

### Upgrade path of stored data

- **No schema version** is written in `project.json`, `images.json`, `annotations.json`, `links.json`, run
  or model files. Today compatibility is kept by reading old shapes (for example link entries that only
  have `base`, or a project without `tasks.json`, migrated on first read). That works until two changes
  conflict, and a downgrade silently misreads newer files.
- Dataset versions copy the labels (good for reproducibility) but there is no tool to verify a workspace
  or repair one (missing image file, orphan thumbnail, run without config).
- `annotations.<timestamp>.bak.json` backups accumulate with no clean-up.

### Packaging and installation

- A first install pulls FiftyOne (MongoDB binary), OpenCV, NumPy and more: large and slow, and FiftyOne
  pins force `graphql-core<3.3` and `importlib-metadata<8`. There is no lock file, so a new release of a
  dependency can break a fresh install without a code change.
- No wheel is published and no release is tagged; `__version__` is a single string in `__init__.py`,
  duplicated in `pyproject.toml`. There is no changelog.
- The launchers `activelabel.cmd` and `activelabel.sh` assume `.venv` next to the file.
- Training packages are installed at the first run: a first-time user waits for a multi-gigabyte download
  with only a progress message (documented, but it is the biggest surprise for a new user).

### Process

- CI has no coverage measure, no dependency or licence audit and no check that the docs match the API.
- The pre-commit hook runs the full unit-test suite on every commit, which gets slower with the suite.
- Frontend behaviour is checked by hand (screenshots), not by tests.

## Roadmap

| # | Proposal | Cost | Risk | Why now |
| --- | --- | --- | --- | --- |
| 1 | **Origin and host checks** for the API: reject requests whose `Host` is not the bound address and whose `Origin` (when present) is not the same origin; require a custom header or `application/json` on state-changing calls. Add a test per verb. | S | Low | Closes the CSRF and rebinding exposure of a server without login |
| 2 | **`schema` field in every stored root file** plus a migration step run at start-up (`storage.migrate(project_dir)`), with a test that loads a fixture of each old version. Keep reading old shapes until then. | M | Medium | Makes upgrades and downgrades explicit before more formats appear |
| 3 | **Generate the endpoint tables** from `/openapi.json` (a script in `tools/` run in CI that fails when `docs/` lists a route that does not exist or misses one), and check Markdown links in CI. | S | Low | Stops the doc drift found in the audit |
| 4 | **Lock file for installs** (`requirements.lock` or `uv.lock`) generated in CI per Python version, and a "fresh install" CI job on Linux and Windows that runs `activelabel --help` and starts the server. | M | Low | A new user's first install should not depend on today's releases |
| 5 | **Release process**: one version source (`importlib.metadata` or `setuptools-scm`), tags, a changelog, a wheel built in CI. | S | Low | Lets people pin and report a version |
| 6 | **Split `app.js` into ES modules** (one per screen: images, annotations, curation, datasets, training, models, jobs) loaded with `<script type="module">`, no bundler; add a handful of browser smoke tests (Playwright) for the main flows. | L | Medium | Largest maintainability gain; the module split can be done screen by screen in small PRs |
| 7 | **Split `TrainingRunner`** into a queue/recovery part, a training runner, an export runner and a pre-annotation runner sharing a process helper; split `sync/engine.py` into plan and apply. | L | Medium | Easier tests and review; needs the existing runner tests as a safety net |
| 8 | **One shared helper module for routers** (`web/deps.py`: tool names, store lookups, error mapping) and one `FolderLock` used by every store. | S | Low | Removes the duplication listed above |
| 9 | **Paged reads** for large projects: keep a light index (id, status, issues) next to `images.json`, and read annotations by image for `GET /images`; measure with a synthetic project of 100,000 images before and after. | L | Medium | Only needed if projects grow past what the whole-file reads handle; measure first |
| 10 | **`activelabel doctor` command**: checks a workspace (missing files, orphans, schema versions, free disk, keyring available, training environment state) and can clean old backups. | M | Low | Gives support and upgrades a safe first step |
| 11 | **Untrusted input review**: zip entry limits and path checks, hardened XML parsing, size limits per upload, tests with hostile files (some exist in `test_data_safety.py`; extend them to every importer). | M | Low | Uploads come from outside |
| 12 | **Pin and hash** the packages the training environment installs (a versions table in `training/catalog.py`), with a visible "what will be installed" step before the first download. | M | Medium | Reproducible runs and supply-chain control |

Suggested order: 1, 3, 5, 8 (cheap, low risk), then 2 and 4, then 11 and 10, and the large refactors
(6, 7, 9) one screen or one class at a time, each as a small pull request with the existing tests
unchanged.
