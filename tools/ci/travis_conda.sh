# Source to configure conda CI builds
if [ "$TRAVIS_OS_NAME" = "linux" ]; then
  CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh";
fi
if [ "$TRAVIS_OS_NAME" = "osx" ]; then
  CONDA_FILE="Miniconda3-latest-MacOSX-x86_64.sh";
fi
wget https://repo.continuum.io/miniconda/"$CONDA_FILE" -O miniconda.sh -nv
chmod +x miniconda.sh
./miniconda.sh -b -p "$HOME"/miniconda
export PATH="$HOME"/miniconda/bin:$PATH
conda config --set always_yes yes
conda update --quiet conda
conda config --set restore_free_channel true
# Build package list to avoid empty package=versions; only needed for versioned packages
PKGS="numpy"; if [ ${NUMPY} ]; then PKGS="${PKGS}=${NUMPY}"; fi
PKGS="${PKGS} scipy"; if [ ${SCIPY} ]; then PKGS="${PKGS}=${SCIPY}"; fi
PKGS="${PKGS} patsy"; if [ ${PATSY} ]; then PKGS="${PKGS}=${PATSY}"; fi
PKGS="${PKGS} pandas"; if [ ${PANDAS} ]; then PKGS="${PKGS}=${PANDAS}"; fi
PKGS="${PKGS} Cython"; if [ ${CYTHON} ]; then PKGS="${PKGS}=${CYTHON}"; fi
if [ ${USEMPL} = true ]; then PKGS="${PKGS} matplotlib"; if [ ${MATPLOTLIB} ]; then PKGS="${PKGS}=${MATPLOTLIB}"; fi; fi
if [ ${COVERAGE} = true ]; then export COVERAGE_OPTS=" --cov=statsmodels "; else export COVERAGE_OPTS=""; fi
echo conda create --yes --quiet -n statsmodels-test python=${PYTHON} ${BLAS} ${PKGS} ${OPTIONAL} ${DEPEND_ALWAYS}
conda create --yes --quiet -n statsmodels-test python=${PYTHON} ${BLAS} ${PKGS} ${OPTIONAL} ${DEPEND_ALWAYS}
source activate statsmodels-test
python -m pip install --upgrade pip
