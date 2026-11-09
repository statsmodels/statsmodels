#!/bin/bash
# Covariate sweep on ASU Sol. Submit from the repo root:
#   sbatch compiler/scripts/sol_sweep.sbatch
#SBATCH -J causal-ci-sweep
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

python -m causal_compiler.sweep --ci-opt "$CI_OPT" \
  --rows "${ROWS:-1000000}" --covariates "${COVS:-4,8,16,25,32,64}" \
  --csv "$OUT/sweep.csv"
