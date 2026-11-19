#!/bin/bash
# EconML LinearDML baseline on ASU Sol. Submit from the repo root:
#   sbatch compiler/scripts/sol_econml.sbatch
#SBATCH -J causal-ci-econml
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_jzou22            # change to your own Slurm account (sacctmgr show assoc user=$USER)
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --export=NONE

set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/causal-ci}"
CI_OPT="${BUILD_DIR:-$HOME/ci-build}/tools/ci-opt/ci-opt"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}/compiler/python"
OUT="results/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p')"

python -m causal_compiler.econml_baseline --ci-opt "$CI_OPT" \
  --sizes "${SIZES:-747,10000,100000,1000000}" --csv "$OUT/econml.csv"
