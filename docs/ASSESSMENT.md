# ActiveLabel — assessment

An engineering review of the architecture and of the development process: what is solid, what is weak,
what can go wrong, and a prioritised proposal. It states facts that can be checked in the repository and
marks opinions as such. Cost: S (hours), M (a day or two), L (a week or more). Risk is the risk of the
change itself.

## Status since the review

Done since this review was written: stable error codes next to the English `detail` on every API error
and on failed jobs (`web/errors.py`, `errors.py`; docs/DESIGN.md lists them), limits on names, classes, job
size, archives and video frames, one FiftyOne dataset name per project, a shared check of label content
against the task, and a faster manifest; tests that mirror the source tree with a coverage floor of 80 %; CI
on Linux, Windows and macOS from a path with accents and spaces, a check of the built wheel, the training
workers tested with a real CPU PyTorch and a nightly run against a real CVAT; JavaScript unit tests and
Playwright smoke tests of the interface; the interface split into native ES modules under `static/js`
(`shell`, `integrations`, `cv`, `pages`) loaded from `main.js` with `<script type="module">`; training
packages installed at pinned, tested versions; one version source (the git tag) and a release process
([RELEASING.md](../RELEASING.md)); the **origin and host guard** `LocalOnly` (`web/guard.py`: a `Host` that
is not this server gets 403 `host_not_allowed`, a change from another site gets 403 `origin_not_allowed`,
`ACTIVELABEL_ALLOWED_HOSTS` adds names); a **`format_version` in every stored JSON object** with a
start-up migration that only adds data and backs files up first (`migrations.py`); lint rules that keep
imports at the top of the module and `Args`/`Returns` in docstrings. Still open: the split of the runner
and of the largest modules, shared router helpers, generated endpoint tables, a lock file for the Python
dependencies of the app and a workspace check command.

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
- **Tooling.** ruff (lint, docstrings, annotations, imports at the top), mypy with `disallow_untyped_defs`,
  pre-commit, CI on Linux and Windows (Python 3.12; macOS after a merge), tests that mirror
  the source tree, JavaScript and browser tests, a separate FiftyOne job.
- **Documentation per feature**, with the HTTP contract next to the behaviour.

## Weaknesses

1. **Large interface modules.** The interface is split into ES modules without a build step, but some are
   still big (`cv/training.js` has more than 1,200 lines, `cv/images.js` about 900). The modules have unit
   and browser tests; there is no lint for JavaScript.
2. **Large modules.** `sync/engine.py` (more than 1,800 lines), `training/runner.py` (about 1,750), `datasets.py`
   (about 1,800) and `web/api.py` (about 1,350) mix several concerns. The runner holds training, export,
   pre-annotation, recovery and process handling in one class.
3. **Duplication.** Each router has its
   own `_store`, `_find`, `_settings` and `_tool` helpers with small differences; the "stage a file, check
   the token, find the project folder" pattern is repeated. The per-folder lock registry is re-implemented in several modules
   (`manifest`, `labels`, `datasets`, `tasks_store`, `video`, `preannotate`, `models`, `runs`, `links`).
4. **Two job systems.** `AppState.jobs` (light jobs, one worker for the whole app) and the runner's own
   `JobManager` (heavy jobs) are separate. A long analysis of one project delays the sync of another, and
   `busy()` has to consult both. Reasonable today; confusing to extend.
5. **Whole-file reads.** `annotations.json` and `images.json` are read and rewritten as a whole on every
   change, under a lock. Fine for thousands of images; slow for hundreds of thousands, and every writer
   pays for the largest file. `GET /images` loads the whole manifest and the whole ground truth for each page.
6. **Stringly-typed boundaries.** Job kinds, scopes, actions and statuses are strings repeated in the
   Python code, in the interface modules (`static/js`) and in the docs. A typo is a runtime error.
7. **Docs and code can drift.** Endpoint tables and the file layout are written by hand; the audit that led
   to this document found many differences (names, endpoints, files on disk).

## Risks

### Security of a local server without login

- There is **no authentication**. The origin and host guard (`web/guard.py`) closes the browser attacks: a page
  of another site cannot change data (cross-site request forgery) and a domain that points at `127.0.0.1`
  cannot read answers (DNS rebinding). Anything that can reach the port from the machine, such as another
  program or another user of a shared computer, is still trusted, and with `--host 0.0.0.0` anyone on the
  network who may use the allowed names can use the API, which can delete projects, read images and use
  the stored CVAT login to contact the configured server.
- `ACTIVELABEL_IMPORT_ROOT` limits path imports to a folder (good), but it is a second trust boundary to
  keep checked (symlinks, `..`).
- Secrets are in the system keyring (good); `tools.json` holds only addresses and users. Log output must
  never include them (a test for this is cheap).
- Uploaded archives and annotation files are untrusted input: size and entry limits, path traversal in
  zips, XML (use a hardened parser) and decompression bombs deserve a review and fuzz-style tests.
- The training environment installs packages from public indexes on demand: supply-chain exposure (the
  versions are pinned in `training/environment.py`, but the files are not checked against hashes).

### Upgrade path of stored data

- Stored objects carry `format_version` and the server migrates older data at start-up (`migrations.py`).
  A migration may only add data, so the release before it still reads the files; a workspace newer than the
  code is read as it is, with the newer fields ignored. A change that removes or renames a field still needs
  a deliberate, documented step.
- Dataset versions copy the labels (good for reproducibility) but there is no tool to verify a workspace
  or repair one (missing image file, orphan thumbnail, run without config).
- `annotations.<timestamp>.bak.json` backups: only the newest 20 per project are kept (activelabel/backups.py).

### Packaging and installation

- A first install pulls FiftyOne (MongoDB binary), OpenCV, NumPy and more: large and slow, and FiftyOne
  pins force `graphql-core<3.3` and `importlib-metadata<8`. There is no lock file, so a new release of a
  dependency can break a fresh install without a code change.
- The version comes from the git tag through setuptools-scm, there is a changelog, and a `release` workflow
  builds the wheel and a draft release (RELEASING.md).
- The launchers `activelabel.cmd` and `activelabel.sh` assume `.venv` next to the file.
- Training packages are installed at the first run, at pinned versions: a first-time user waits for a
  large download with only a progress message (documented, but it is the biggest surprise for a new user).

### Process

- CI measures coverage (floor of 80 %), but has no dependency or licence audit and no check that the docs match the API.
- The pre-commit hook runs the full unit-test suite on every commit, which gets slower with the suite.
- Frontend behaviour is covered by JavaScript unit tests and Playwright smoke tests (job `ui`); visual
  quality is still judged by hand.

## Roadmap

| # | Proposal | Cost | Risk | Why now |
| --- | --- | --- | --- | --- |
| 1 | **Generate the endpoint tables** from `/openapi.json` (a script in `tools/` run in CI that fails when `docs/` lists a route that does not exist or misses one), and check Markdown links in CI. | S | Low | Stops the doc drift found in the audit |
| 2 | **Lock file for installs** (`requirements.lock` or `uv.lock`) generated in CI per Python version, and a "fresh install" CI job on Linux and Windows that runs `activelabel --help` and starts the server. | M | Low | A new user's first install should not depend on today's releases |
| 3 | **Split `TrainingRunner`** into a queue/recovery part, a training runner, an export runner and a pre-annotation runner sharing a process helper; split `sync/engine.py` into plan and apply. | L | Medium | Easier tests and review; needs the existing runner tests as a safety net |
| 4 | **One shared helper module for routers** (`web/deps.py`: tool names, store lookups, error mapping) and one `FolderLock` used by every store. | S | Low | Removes the duplication listed above |
| 5 | **Paged reads** for large projects: keep a light index (id, status, issues) next to `images.json`, and read annotations by image for `GET /images`; measure with a synthetic project of 100,000 images before and after. | L | Medium | Only needed if projects grow past what the whole-file reads handle; measure first |
| 6 | **`activelabel doctor` command**: checks a workspace (missing files, orphans, schema versions, free disk, keyring available, training environment state) and can clean old backups. | M | Low | Gives support and upgrades a safe first step |
| 7 | **Untrusted input review**: zip entry limits and path checks, hardened XML parsing, size limits per upload, tests with hostile files (some exist in `test_data_safety.py`; extend them to every importer). | M | Low | Uploads come from outside |
| 8 | **Hash** the packages the training environment installs (the versions are already pinned in `training/environment.py`), with a visible "what will be installed" step before the first download. | M | Medium | Reproducible runs and supply-chain control |

Suggested order: the cheap, low-risk items first (the endpoint tables and the shared router helpers), then
the lock file, the input review and the workspace check, and the large refactors (the runner, the sync
engine, paged reads) one class at a time, each as a small pull request with the existing tests unchanged.
