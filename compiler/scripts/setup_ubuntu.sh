#!/usr/bin/env bash
# One-shot setup for Ubuntu 24.04 (native, WSL2, or Sol). Run from repo root:
#   bash compiler/scripts/setup_ubuntu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/ci-build}"
VENV_DIR="${VENV_DIR:-$HOME/ci-venv}"

echo "[1/4] System packages (MLIR 18, clang 18, cmake, ninja)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libmlir-18-dev mlir-18-tools llvm-18-dev clang-18 cmake ninja-build \
  python3-venv

echo "[2/4] Build ci-opt into $BUILD_DIR"
cmake -S "$REPO_ROOT/compiler" -B "$BUILD_DIR" -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
cmake --build "$BUILD_DIR"
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo "[3/4] Python venv in $VENV_DIR"
# The fork's own statsmodels needs a meson/Cython build; the compiler only
# uses public statsmodels APIs, so the PyPI release is sufficient.
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q numpy scipy pandas patsy statsmodels pytest \
  iree-base-compiler iree-base-runtime

echo "[4/4] Python tests (with ci-opt integration)"
export PATH="$VENV_DIR/bin:$PATH"
cd "$REPO_ROOT/compiler/python"
# -c /dev/null: ignore the repo pyproject's pytest addopts (pytest-randomly).
CI_OPT_COMMAND="$BUILD_DIR/tools/ci-opt/ci-opt" \
  "$VENV_DIR/bin/python" -m pytest -q -c /dev/null --rootdir . tests

echo
echo "Done. Run the demo with:"
echo "  source $VENV_DIR/bin/activate && cd compiler/python"
echo "  OPENBLAS_NUM_THREADS=1 python -m causal_compiler.demo --ci-opt $BUILD_DIR/tools/ci-opt/ci-opt"
