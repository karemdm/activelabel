# Contributing to ActiveLabel

Thanks for helping. The short version:

1. Open an issue (or comment on one) before a large change.
2. Work on a branch; keep each pull request small and focused, with commits that each do one thing
   (`fix:`, `feat:`, `docs:`, `refactor:`, `test:`), written in English.
3. Set up and run the checks as described in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): `ruff check`,
   `ruff format`, `mypy` and `pytest -m "not integration"` must pass, and for a change to the interface
   also `npm run test:js` and `npm run test:e2e`. The CI jobs that must pass before a merge are listed there.
4. A bug fix comes with a test that fails without the fix. A change to behaviour, an endpoint, a file on
   disk or a button updates the documents of `docs/`.
5. Texts of the interface go through `i18n/en.json` and `i18n/pt.json` (both, same placeholders, correct
   Portuguese accents).
6. Do not add attribution lines of any tool or assistant to commits, code, docs or pull requests, and do
   not commit secrets, workspaces or personal data.

By contributing you agree that your contribution is licensed under the Apache License 2.0 (see
[LICENSE](LICENSE)).
