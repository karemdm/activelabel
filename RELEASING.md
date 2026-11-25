# Releasing

How the owner publishes a release, for example the first one, **v0.1.0**. The version is never typed in the
code: it is read from the git tag (setuptools-scm, `[tool.setuptools_scm]` in pyproject.toml). Until the
first tag exists a build carries version `0.0`, which the CI accepts only while there is no `v*` tag.

## Once, before the first release

- In **Settings > Environments**, create the environment `release` and add yourself as a required
  reviewer, so the draft release waits for your approval.

## Each release

1. **Check `main`.** The last CI run of `main` is green, and the issues labelled `bloqueia-release` are
   closed.
2. **Update CHANGELOG.md** in a pull request: rename `## Unreleased` to `## 0.1.0 - YYYY-MM-DD`, add an
   empty `## Unreleased` above it and remove the line "The project has no tagged release yet". Merge it.
3. **Tag the merge commit of `main`** (only the owner decides when; the team never tags on its own):

   ```bash
   git switch main
   git pull
   git tag -a v0.1.0 -m "ActiveLabel 0.1.0"
   git push origin v0.1.0
   ```

4. **Wait for the `release` workflow** (Actions tab). It builds the wheel and the sdist, checks that their
   version is exactly `0.1.0`, installs the wheel on Linux, Windows and macOS and starts the server, then
   waits for your approval of the `release` environment and creates a **draft** release with both files
   and the notes grouped by label (`.github/release.yml`).
5. **Review the draft** in **Releases**: read the notes, download the wheel and try it
   (`pip install activelabel-0.1.0-py3-none-any.whl`, then `activelabel`). Publish the draft.

## When something goes wrong

- **The workflow failed before the draft:** nothing was published. Fix the problem on `main`, then move the
  tag to the new commit: `git tag -d v0.1.0`, `git push origin :refs/tags/v0.1.0`, and tag again (step 3).
- **The draft is wrong:** delete the draft (not published, so nobody saw it) and the tag, then start again.
- **A published release is wrong:** do not move its tag. Fix it on `main` and release the next patch
  version (v0.1.1).

The version numbers follow [Semantic Versioning](https://semver.org/): while it is 0.x, a new feature or a
change in the workspace format raises the minor number (0.2.0) and a fix raises the patch number (0.1.1).
