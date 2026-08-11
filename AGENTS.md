# AGENTS.md

Guidance for coding agents (and humans) contributing to `statsmodels`. This
file is a practical supplement to [CONTRIBUTING.md](CONTRIBUTING.md) and
the [developer docs](https://www.statsmodels.org/devel/dev/index.html) —
read those too, but treat this as the checklist for "will this actually get
merged."

## Project snapshot

- Pure-Python core with a few Cython extensions (Kalman filter code in
  `statsmodels/tsa/statespace/_*.pyx`, etc.). Most new statistical code is
  pure Python.
- Cython extensions are built via `meson`/`ninja` (see
  `pyproject.toml` `[tool.meson-python]` section). Cython extensions
  are accepted only when the performance gain over pure Python is significant.
- Build backend is `meson-python`; the package is normally developed via an
  **editable install** so pure-Python edits are picked up immediately.
- Target Python: `py310`+ (see `[tool.ruff] target-version` /
  `[tool.black] target-version` in `pyproject.toml`).
- Upstream repo is `statsmodels/statsmodels`; when linking issues/PRs in
  commit messages or docs, link that repo, not a personal fork.
- Ensure features taken from upstream packages, `numpy`, `scipy`, `pandas`, etc.,
  are compatible with the minimum versions in `pyproject.toml` and `requirements.txt`.

## Environment setup

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-build-isolation
```

Using an editable install mean changes in both Python and Cython are live
immediately; Cython extensions are rebuilt automatically on import
if the source `.pyx` file is newer than the compiled `.so` or `.dll` file.

## Code style & linting

- Formatting: `ruff` (line-length 88, `py310` target, isort-equivalent import sorting via
  `[tool.ruff.lint.isort]`). Import sections: `future`, `compat`,
  `standard-library`, `third-party`, `first-party`, `local-folder` — with
  `statsmodels.compat` treated as its own `compat` section.
- `ruff` and `flake8` must pass in the directories `statsmodels`, `docs`, and `tools`.
- `black` (line-length 88) is acceptable for auto-formatting, but not required.
  Only use `black` to format changes to files, or in new files. Do not use `black` to
  wholesale reformat existing files.

## Docstrings & documentation

- Every public function, class, method, and attribute needs a numpydoc
  docstring (numpy docstring standard, not Google/Sphinx style).
- New public functionality must be:
  1. Exported from the relevant `api.py` if the module has one (e.g. new
     `tsa` models go in `statsmodels/tsa/api.py`, matching how `VAR`,
     `SVAR`, `VARMAX` are exposed there) — burying a new class only inside
     an internal module path is a common reason reviewers bounce a PR.
  2. Added to the Sphinx docs — find the relevant `docs/source/*.rst` file
     (e.g. `docs/source/vector_ar.rst` for VAR-family models) and add an
     `autosummary` entry.
  3. Noted in the current release notes file under `docs/source/release/`
     (currently `version0.15.0.rst` — check for the highest-numbered
     unreleased `versionX.Y.Z.rst`).
- If it's a substantial new feature, an example notebook is strongly encouraged.
  (see the "Add an example" step in `CONTRIBUTING.md`).

## Testing conventions

- Tests live in a `tests/` subpackage next to the code
  (`statsmodels/<area>/tests/test_*.py`). Reference values from other
  packages (R, Stata, SAS, a prior statsmodels version) go in a sibling
  `tests/results/` directory — document *where* each reference value came
  from and why it might not match exactly.
- The preferred style is pure modern `pytest` with `pytest.mark.parametrize`
  for multiple cases, and `pytest` fixtures for shared setup.
- Avoid `unittest.TestCase` style for new tests.
- Statistical/numerical claims should be validated against an existing
  package or known closed-form result wherever possible (this project
  follows a test-driven-development norm for new models/statistical
  functions) — an internal-consistency-only test suite (e.g. "shape is
  right", "output is finite") is necessary but not sufficient.
- Two sharp edges that will make a locally-green test suite fail once
  merged — **verify against the actual `pytest` run in this repo, not just
  by reading the code**:
  - **Global RNG state is protected.** `statsmodels/conftest.py` has an
    autouse fixture (`check_global_randomstate_usage`) that asserts the
    legacy global `np.random` singleton state is unchanged after every
    test. Use a seeded `np.random.default_rng(seed)` `Generator` in new
    tests, not bare `np.random.randn(...)`/`np.random.seed(...)`. If a test
    genuinely must touch the global singleton, mark it
    `@pytest.mark.singleton_randomstate`.
  - **Many warnings are promoted to errors.** `pyproject.toml`'s
    `[tool.pytest.ini_options] filterwarnings` turns a long list of
    `DeprecationWarning`/`FutureWarning` patterns into hard test failures.
    New code must not exercise deprecated pandas/numpy/scipy code paths,
    and should assume that list grows over time — a warning that's merely
    "ignored" today may be promoted to `error` later.
  - Other markers worth knowing: `slow`, `example`, `matplotlib`,
    `high_memory`, `low_precision`, `polars`, `joblib`, `todo`, `smoke`.
    `xfail_strict = true` and `empty_parameter_set_mark = "fail_at_collect"`
    are both set — an `xfail` that unexpectedly passes, or a parametrize
    call that resolves to zero cases, both fail collection/CI.
- Run the suite locally before claiming it passes:
  `pytest -n auto statsmodels/<area>/tests/test_new_thing.py`, and at least
  the containing subpackage's full test dir if the change touches shared
  base classes. Full-suite runs are slow; use `--skip-slow` /
  `-m "not slow"` and `pytest-xdist` (`-n auto`) for iteration. Recommended
  to use `--skip-examples` in full test runs to not test the example notebooks.
  When running the full test suite in parallel using pytest-xdist, it is recommended
  ``export OPENBLAS_NUM_THREADS=1`` to avoid oversubscription of CPU cores
  in multithreaded BLAS code (or ``$env:OPENBLAS_NUM_THREADS=1`` on Windows PowerShell).
- The standard fast full test command is:

```bash
export OPENBLAS_NUM_THREADS=1
pytest -n auto -m "(not slow and not example)" statsmodels
```

## New estimator / model architecture

New models and results classes are expected to fit the existing framework,
not reimplement it standalone:

- Subclass `statsmodels.base.model.Model` (or a more specific base such as
  `statsmodels.tsa.base.tsa_model.TimeSeriesModel`) for the model, and the
  matching `*Results` base for the results object, rather than writing a
  bespoke class hierarchy from scratch.
- Follow existing naming for fitted quantities: `params`, `bse`, `tvalues`,
  `pvalues`, `conf_int()`, and provide a `summary()` — don't invent
  differently-named equivalents unless the base API genuinely doesn't fit.
- Check for prior art before designing a new results API. E.g. IRF-style
  output already has a convention in
  `statsmodels.tsa.vector_ar.irf.IRAnalysis` (`irfs`, `cum_effects`,
  `plot()`); a new estimator producing IRFs should look like that rather
  than inventing parallel naming.
- If the method rests on a nontrivial identification/statistical assumption
  (e.g. a particular ordering, exogeneity, or orthogonality requirement),
  say so explicitly in the class docstring — "the math is correct given
  assumption X" is not enough if X is never stated for the caller.

## Git / commit conventions

- **One branch, one feature.** Don't bundle unrelated changes.
- Commit subject line under ~80 chars; prefix with the informal type tag
  used throughout `git log` (see `docs/source/dev/maintainer_notes.rst`):
  - `ENH:` new feature
  - `BUG:` bug fix
  - `STY:` style-only change (formatting, no logic change)
  - `DOC:` docs/docstring/comment changes
  - `CMP:` compiled-code issues (Cython/C regeneration, etc.)
  - `TST:` test-only change, unrelated to a specific bug fix
  - `REF:` refactoring
  - `REL:` release-related
- Reference/close issues via `#XXXX`, `GH-XXXX`, or `gh-XXXX` in the commit
  message where applicable.
- Every non-trivial PR needs: tests, numpydoc docstrings, a docs entry if
  it's new public API, and a release-notes line — a PR description claiming
  "N/N tests pass" should be independently verified by actually running
  those tests in this repo (its `conftest.py` enforces rules — random-state
  hygiene, warnings-as-errors — that a standalone/ad hoc test run elsewhere
  won't catch).

## Working in this repo as an agent

- When editing large existing files, prefer targeted, minimal diffs over
  regenerating the whole file (e.g. via an AST round-trip) — whole-file
  regeneration silently drops comments and can reformat unrelated code,
  producing a noisy, hard-to-review diff.
- Don't run destructive git operations (`reset --hard`, force-push,
  branch deletion) against branches you didn't create in the current
  session without checking `git status`/`git log` first — this checkout
  may be shared with other in-progress work.
