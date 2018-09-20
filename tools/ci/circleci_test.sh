#!/usr/bin/env bash

source activate statsmodels-test

python -c 'import statsmodels.api as sm; sm.show_versions();'

echo pytest -r a ${COVERAGE_OPTS} statsmodels --skip-examples
pytest -r a ${COVERAGE_OPTS} statsmodels --skip-examples
