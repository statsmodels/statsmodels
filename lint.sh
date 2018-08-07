#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" ]; then
    flake8 statsmodels
    if [ $? -ne "0" ]; then
        RET=1
    fi

    # Run with --isolated to ignore config files, the files included here
    # pass _all_ flake8 checks
    flake8 --isolated \
        statsmodels/info.py \
        statsmodels/resampling/ \
        statsmodels/interface/ \
        statsmodels/tsa/regime_switching \
        statsmodels/regression/mixed_linear_model.py \
        statsmodels/duration/__init__.py
    if [ $? -ne "0" ]; then
        RET=1
    fi

fi

exit $RET
