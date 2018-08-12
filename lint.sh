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
        statsmodels/duration/__init__.py \
        statsmodels/regression/recursive_ls.py
    if [ $? -ne "0" ]; then
        RET=1
    fi

    # Until these checks get merged into the setup.cfg checks or this module
    # gets checked in the strict check above, individual checks for
    # partially-fixed files
    flake8 --isolated --select=E127,E128,E203,E301,E302,E303,E305 \
        statsmodels/discrete/discrete_model.py
    if [ $? -ne "0" ]; then
        RET=1
    fi

fi

exit $RET
