#!/bin/bash
# K-intervention logistic benchmark on ASU Sol. Submit from the repo root:
#   sbatch compiler/scripts/sol_kinterv.sbatch
#SBATCH -J causal-ci-kinterv
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_jzou22            # change to your own Slurm account (sacctmgr show assoc user=$USER)
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0-02:00:00
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

python -m causal_compiler.kinterv \
  --rows "${ROWS:-100000,1000000}" --k "${KS:-1,2,4,8,16,32,64}" \
  --csv "$OUT/kinterv.csv"
