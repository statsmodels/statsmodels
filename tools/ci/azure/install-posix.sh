#!/usr/bin/env bash

if [[ ${USE_CONDA} == "true" ]]; then
  # Miniforge (conda-forge only) is installed by the pipeline and the env's
  # bin directory is on PATH, so no activation is needed.
  mamba create -n statsmodels-test python=${PYTHON_VERSION} -y
  echo ${PATH}
  which python
  # Fail here rather than silently using a different Python
  if [[ "$(which python)" != */envs/statsmodels-test/bin/python ]]; then
    echo "python is not from the statsmodels-test environment"
    exit 1
  fi
  CMD="mamba install -n statsmodels-test -y -c conda-forge numpy"
else
  CMD="python -m pip install numpy"
fi

echo "Python location: $(where python)"
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -r requirements-dev.txt

if [[ ${USE_CONDA} != "true" ]]; then
  python -m pip uninstall numpy scipy pandas cython -y
else
  # Clean up packages indirectly installed by pip when using conda
  python -m pip uninstall numpy cython -y
fi

if [[ -n ${NUMPY_VERSION} ]]; then CMD="$CMD==${NUMPY_VERSION}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY_VERSION} ]]; then CMD="$CMD==${SCIPY_VERSION}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS_VERSION} ]]; then CMD="$CMD==${PANDAS_VERSION}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON_VERSION} ]]; then CMD="$CMD==${CYTHON_VERSION}"; fi;

if [[ ${USE_MATPLOTLIB} == true ]]; then
  CMD="$CMD matplotlib"
  if [[ -n ${MATPLOTLIB_VERSION} ]]; then
    CMD="$CMD==${MATPLOTLIB_VERSION}";
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
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy pandas scipy --upgrade --use-deprecated=legacy-resolver
  if [[ ${USE_MATPLOTLIB} == true ]]; then
    python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple matplotlib --upgrade --use-deprecated=legacy-resolver
  fi
fi
