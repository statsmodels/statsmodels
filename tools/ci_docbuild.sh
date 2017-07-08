#!/usr/bin/env bash
# Script to build docs

# Change to doc directory
cd ${SRCDIR}/docs

# Run notebooks as tests
nosetests -v ../statsmodels/examples/tests

# Clean up
echo '================================= Clean ================================='
make clean
git clean -xdf

# Build documentation
echo '========================================================================'
echo '=                        Building documentation                        ='
echo '========================================================================'
echo 'make html > doc_build.log 2>&1'
make html 2>&1 | tee doc_build.log

# Info
echo '========================================================================'
echo '=                 Opportunities To Improve (Warnings)                  ='
echo '========================================================================'
cat doc_build.log | grep -E '(WARNING)' | grep -v '(noindex|toctree)'

# Check log
echo '========================================================================'
echo '=          Broken Behavior (Errors and Warnings to be Fixed)           ='
echo '========================================================================'
cat doc_build.log | grep -E '(SEVERE|ERROR|WARNING)' | grep -Ev '(noindex|toctree)'
