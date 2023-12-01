#!/usr/bin/env bash


python -m pip install --upgrade pip setuptools wheel build
python -m pip install -r requirements-dev.txt
pip uninstall numpy scipy pandas cython -y

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


if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD==${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD==${PANDAS}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON} ]]; then CMD="$CMD==${CYTHON}"; fi;

if [[ ${USE_MATPLOTLIB} == true ]]; then
  CMD="$CMD matplotlib"
  if [[ -n ${MATPLOTLIB} ]]; then
    CMD="$CMD==${MATPLOTLIB}";
  fi
else
  # Uninstall if not needed
  python -m pip uninstall matplotlib -y || true
fi

CMD="${CMD} patsy ${BLAS}"
echo $CMD
eval $CMD

if [[ ${USE_CVXOPT} = true ]]; then python -m pip install cvxopt; fi

if [ "${PIP_PRE}" = true ]; then
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple "numpy<2" pandas scipy --upgrade --use-deprecated=legacy-resolver
fi
