#!/bin/bash
# Benchmark job for ASU Sol. Submit from the repo root:
#   sbatch compiler/scripts/sol_job.sbatch
# Prerequisite: compiler/scripts/setup_conda.sh has completed (env + build + IHDP cached).
#SBATCH -J causal-ci-bench
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_jzou22            # change to your own Slurm account (sacctmgr show assoc user=$USER)
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --export=NONE

set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/causal-ci}"
BUILD_DIR="${BUILD_DIR:-$HOME/ci-build}"
CI_OPT="$BUILD_DIR/tools/ci-opt/ci-opt"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

cd "${SLURM_SUBMIT_DIR:-$PWD}/compiler/python"
OUT="results/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p')"

# Fair single-thread comparison (NumPy BLAS and IREE both on 1 thread).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m causal_compiler.demo --ci-opt "$CI_OPT" --target-cpu host \
  --sizes "${SIZES:-747,10000,100000,1000000,10000000}" --repeats 11 \
  --csv "$OUT/ihdp_single_thread.csv"

# Team's original synthetic benchmark (includes iree-benchmark-module timings).
python -m causal_compiler.synthetic --rows "${SYN_ROWS:-1000000}" --covariates 8 --repeats 11 \
  --ci-opt-command "$CI_OPT" --target-cpu host | tee "$OUT/synthetic_${SYN_ROWS:-1000000}.txt"

echo "Results in compiler/python/$OUT"
