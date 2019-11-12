# Source to configure pip CI builds
# Wheelhouse for various packages missing from pypi.
EXTRA_WHEELS="https://5cf40426d9f06eb7461d-6fe47d9331aba7cd62fc36c7196769e4.ssl.cf2.rackcdn.com"
# Wheelhouse for daily builds of some packages.
PRE_WHEELS="https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
EXTRA_PIP_FLAGS="--find-links=$EXTRA_WHEELS"
# Build package list to avoid empty package=versions; only needed for versioned packages
PKGS="${PKGS} numpy"; if [ ${NUMPY} ]; then PKGS="${PKGS}==${NUMPY}"; fi
PKGS="${PKGS} scipy"; if [ ${SCIPY} ]; then PKGS="${PKGS}==${SCIPY}"; fi
PKGS="${PKGS} patsy"; if [ ${PATSY} ]; then PKGS="${PKGS}==${PATSY}"; fi
PKGS="${PKGS} pandas"; if [ ${PANDAS} ]; then PKGS="${PKGS}==${PANDAS}"; fi
PKGS="${PKGS} Cython"; if [ ${CYTHON} ]; then PKGS="${PKGS}==${CYTHON}"; fi
if [ ${USE_MATPLOTLIB} = true ]; then
    PKGS="${PKGS} matplotlib"
    if [ ${MATPLOTLIB} ]; then
        PKGS="${PKGS}==${MATPLOTLIB}"
    fi
fi
if [ "${PIP_PRE}" = true ]; then
    EXTRA_PIP_FLAGS="--pre $EXTRA_PIP_FLAGS --find-links $PRE_WHEELS"
fi

# travis osx python support is limited. Use homebrew/pyenv to install python.
if [ "$TRAVIS_OS_NAME" = "osx" ]; then

  brew update && brew upgrade pyenv

  eval "$(pyenv init -)"

  pyenv install "$PYTHON"
  pyenv shell "$PYTHON"
fi

# Install in our own virtualenv
python -m pip install --upgrade pip
pip install --upgrade virtualenv
virtualenv --python=python venv
source venv/bin/activate
python --version # just to check
python -m pip install --upgrade pip
pip install ${EXTRA_PIP_FLAGS} ${PKGS} ${DEPEND_ALWAYS}
