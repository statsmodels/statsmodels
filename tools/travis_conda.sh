# Source to configure conda CI builds
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p /home/travis/miniconda
export PATH=/home/travis/miniconda/bin:$PATH
conda config --set always_yes yes
conda update --quiet conda
# Build package list to avoid empty package=versions; only needed for versioned packages
PKGS="python=${PYTHON}"
PKGS="${PKGS} numpy"; if [ ${NUMPY} ]; then PKGS="${PKGS}=${NUMPY}"; fi
PKGS="${PKGS} scipy"; if [ ${SCIPY} ]; then PKGS="${PKGS}=${SCIPY}"; fi
PKGS="${PKGS} patsy"; if [ ${PATSY} ]; then PKGS="${PKGS}=${PATSY}"; fi
PKGS="${PKGS} pandas"; if [ ${PANDAS} ]; then PKGS="${PKGS}=${PANDAS}"; fi
PKGS="${PKGS} Cython"; if [ ${CYTHON} ]; then PKGS="${PKGS}=${CYTHON}"; fi
if [ ${USEMPL} = true ]; then PKGS="${PKGS} matplotlib"; if [ ${MATPLOTLIB} ]; then PKGS="${PKGS}=${MATPLOTLIB}"; fi; fi
echo conda create --yes --quiet -n statsmodels-test ${PKGS} ${OPTIONAL} pyyaml
conda create --yes --quiet -n statsmodels-test ${PKGS} ${OPTIONAL} ${DEPEND_ALWAYS}
source activate statsmodels-test
