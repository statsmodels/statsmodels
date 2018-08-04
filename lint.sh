#!/bin/bash

echo "inside $0"

RET=0

if [ "$LINT" ]; then
    flake8 statsmodels
    if [ $? -ne "0" ]; then
        RET=1
    fi
fi

exit $RET
