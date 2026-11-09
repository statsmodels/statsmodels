#!/usr/bin/env bash
# No-sudo setup: creates the conda env, builds ci-opt, runs all tests.
# Works on ASU Sol (after `module load mamba/latest`) and on any x86-64 Linux/WSL
# with mamba, micromamba or conda on PATH. Run from the repo root:
#   bash compiler/scripts/setup_conda.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/causal-ci}"
BUILD_DIR="${BUILD_DIR:-$HOME/ci-build}"

for tool in mamba micromamba conda; do
  if command -v "$tool" >/dev/null 2>&1; then MAMBA="$tool"; break; fi
done
: "${MAMBA:?no mamba/micromamba/conda found (on Sol: module load mamba/latest)}"

echo "[1/4] Conda env at $ENV_PREFIX (using $MAMBA; ~5-10 min first time)"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  "$MAMBA" env create -y -p "$ENV_PREFIX" -f "$REPO_ROOT/compiler/environment.yml" \
    || "$MAMBA" env create -p "$ENV_PREFIX" -f "$REPO_ROOT/compiler/environment.yml"
else
  echo "  exists, skipping"
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "[2/4] Build ci-opt into $BUILD_DIR"
cmake -S "$REPO_ROOT/compiler" -B "$BUILD_DIR" -G Ninja \
  -DMLIR_DIR="$ENV_PREFIX/lib/cmake/mlir" \
  -DCMAKE_C_COMPILER="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
cmake --build "$BUILD_DIR"

echo "[3/4] Native tests"
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "[4/4] Python tests + IHDP download (needs internet; do this on a login node)"
cd "$REPO_ROOT/compiler/python"
python -c "from causal_compiler.datasets import load_ihdp; print('IHDP', load_ihdp(1).shape)"
CI_OPT_COMMAND="$BUILD_DIR/tools/ci-opt/ci-opt" \
  python -m pytest -q -c /dev/null --rootdir . tests

echo
echo "Setup OK. ci-opt: $BUILD_DIR/tools/ci-opt/ci-opt"
