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
  CMD="conda install -c conda-forge numpy"
else
  CMD="python -m pip install numpy"
fi

echo "Python location: $(where python)"
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -r requirements-dev.txt

if [[ ${USE_CONDA} != "true" ]]; then
  python -m pip uninstall numpy scipy pandas cython -y
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
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy pandas scipy --upgrade --use-deprecated=legacy-resolver
  if [[ ${USE_MATPLOTLIB} == true ]]; then
    python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple matplotlib --upgrade --use-deprecated=legacy-resolver
  fi
fi

# Special for formulaic install
# TODO: Remove after formulaic is standard
python -m pip install hatchling hatch-vcs
python -m pip install -r requirements.txt
