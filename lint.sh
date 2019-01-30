#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" == true ]; then
    echo "Linting all files with limited rules"
    flake8 statsmodels
    if [ $? -ne "0" ]; then
        echo "Changed files failed linting using the required set of rules."
        echo "Additions and changes must conform to Python code style rules."
        RET=1
    fi

    # Run with --isolated to ignore config files, the files included here
    # pass _all_ flake8 checks
    echo "Linting known clean files with strict rules"
    flake8 --isolated \
        statsmodels/resampling/ \
        statsmodels/interface/ \
        statsmodels/duration/__init__.py \
        statsmodels/gam/gam_penalties.py \
        statsmodels/gam/generalized_additive_model.py \
        statsmodels/gam/gam_cross/validation/ \
        statsmodels/gam/smooth_basis.py \
        statsmodels/graphics/tsaplots.py \
        statsmodels/examples/tests/ \
        statsmodels/iolib/smpickle.py \
        statsmodels/regression/tests/test_lme.py \
        statsmodels/tools/web.py \
        statsmodels/tools/tests/test_linalg.py \
        statsmodels/tools/decorators.py \
        statsmodels/tools/tests/test_decorators.py \
        statsmodels/tsa/base/tests/test_datetools.py \
        statsmodels/tsa/vector_ar/dynamic.py \
        statsmodels/tsa/statespace/tests/results/results_var_R.py \
        statsmodels/tsa/statespace/tests/test_var.py \
        statsmodels/conftest.py \
        setup.py
    if [ $? -ne "0" ]; then
        echo "Previously passing files failed linting."
        RET=1
    fi

    # Tests any new python files
    git fetch --unshallow --quiet
    git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    git fetch origin --quiet
    NEW_FILES=$(git diff origin/master --name-status -u -- "*.py" | grep ^A | cut -c 3- | paste -sd " " -)
    if [ -n "$NEW_FILES" ]; then
        echo "Linting newly added files with strict rules"
        flake8 --isolated $(eval echo $NEW_FILES)
        if [ $? -ne "0" ]; then
            echo "New files failed linting."
            RET=1
        fi
    fi
fi

exit $RET
