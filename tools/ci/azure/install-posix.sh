#!/usr/bin/env bash

if [[ ${USE_CONDA} == "true" ]]; then
  conda config --set always_yes true
  conda update --all --quiet
  conda create -n statsmodels-test python=${PYTHON_VERSION} -y
  conda init
  echo ${PATH}
  source activate statsmodels-test
  echo ${PATH}
  which python
  CMD="conda install numpy"
else
  CMD="python -m pip install numpy"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install cython pytest pytest-xdist coverage pytest-cov ipython jupyter notebook nbconvert "property_cached>=1.6.3" black==20.8b1 isort flake8 nbconvert==5.6.1 coveralls

if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD==${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD==${PANDAS}"; fi;

if [[ ${USE_MATPLOTLIB} == true ]]; then
  CMD="$CMD matplotlib"
  if [[ -n ${MATPLOTLIB} ]]; then
    CMD="$CMD==${MATPLOTLIB}";
  fi
fi

CMD="${CMD} patsy ${BLAS}"
echo $CMD
eval $CMD

if [[ ${USE_CVXOPT} = true ]]; then pip install cvxopt; fi

if [ "${PIP_PRE}" = true ]; then
  pip install -i https://pypi.anaconda.org/scipy-wheels-nightly/simple numpy pandas scipy --upgrade --use-deprecated=legacy-resolver
fi
