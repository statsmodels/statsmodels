#!/usr/bin/env bash

apt-get update -y
apt-get install -y build-essential

conda config --set always_yes yes
conda update --quiet conda
# Build package list to avoid empty package=versions; only needed for versioned packages
PKGS="numpy"; if [ ${NUMPY} ]; then PKGS="${PKGS}=${NUMPY}"; fi
PKGS="${PKGS} scipy"; if [ ${SCIPY} ]; then PKGS="${PKGS}=${SCIPY}"; fi
PKGS="${PKGS} patsy"; if [ ${PATSY} ]; then PKGS="${PKGS}=${PATSY}"; fi
PKGS="${PKGS} pandas"; if [ ${PANDAS} ]; then PKGS="${PKGS}=${PANDAS}"; fi
PKGS="${PKGS} Cython"; if [ ${CYTHON} ]; then PKGS="${PKGS}=${CYTHON}"; fi
if [ ${USEMPL} = true ]; then PKGS="${PKGS} matplotlib"; if [ ${MATPLOTLIB} ]; then PKGS="${PKGS}=${MATPLOTLIB}"; fi; fi
if [ ${COVERAGE} = true ]; then export COVERAGE_OPTS=" --cov-config=.coveragerc_travis --cov=statsmodels "; else export COVERAGE_OPTS=""; fi
echo conda create --yes --quiet -n statsmodels-test python=${PYTHON} ${BLAS} ${PKGS} ${OPTIONAL} ${DEPEND_ALWAYS}
conda create --yes --quiet -n statsmodels-test python=${PYTHON} ${BLAS} ${PKGS} ${OPTIONAL} ${DEPEND_ALWAYS}
source activate statsmodels-test
python -m pip install --upgrade pip

pip install pytest pytest-xdist pip pytest-randomly nose flake8

python setup.py develop
