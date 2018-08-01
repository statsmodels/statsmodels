#!/bin/bash

echo "inside $0"

RET=0

flake8 statsmodels --select=E901,E999,F821,F822,F823,E111,E114
if [ $? -ne "0" ]; then
    RET=1
fi

exit $RET
