# Continuous Integration Tools

These scripts are used to implement Continuous Integration on Azure and GitHub.
They should not be removed without careful consideration of the consequences.

## Slow tests

Tests marked `slow` run in only three Linux jobs of the Azure pipeline, the
oldest, a middle and the newest combination of Python and dependencies:

* `python_310`, Python 3.10 with the minimum versions of the dependencies
* `python_313_copy_on_write`, Python 3.13
* `python_315_no_formulaic`, Python 3.15

These jobs set `PYTEST_PATTERN` in `azure/azure_template_posix.yml` to select
all tests. Every other job skips the slow tests with
`-m "not slow and not joblib"` or `--skip-slow`. A new job skips the slow tests
unless it sets `PYTEST_PATTERN`.
