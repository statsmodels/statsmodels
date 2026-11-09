# Running the causal compiler: local machine or ASU Sol

Everything below was tested from a clean start (MLIR 18.1, IREE 3.13).
Pick **one** of Part A (your computer) or Part B (Sol). Part C covers running the code once it's set up.

What gets built/installed:
- `ci-opt` — our MLIR tool (the `ci` dialect + lowering passes), built from `compiler/`
- a Python env with statsmodels, IREE (compiles lowered MLIR to CPU code), pytest
- the IHDP dataset (747 rows, 25 covariates; Yao et al. 2021, Table 2), cached in `compiler/python/data/`

---

## Part A — Your own computer

### A1. Windows → use WSL2 (required; the build is Linux-only)
1. Open PowerShell **as Administrator**:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
2. Reboot, open **Ubuntu 24.04** from the Start menu, create a username/password.
3. Do every remaining step **inside the Ubuntu terminal**, in your Linux home
   (`cd ~`). Don't work under `/mnt/c/...` — it builds several times slower.
4. (Optional) VS Code: install the "WSL" extension, then run `code .` from the Ubuntu terminal.

### A2. Linux / WSL — clone the branch
```bash
sudo apt-get update && sudo apt-get install -y git
cd ~
git clone -b 598_compiler_project https://github.com/Ian-Schultz-ASU/statsmodels.git
cd statsmodels
```

### A3. Install + build + test — choose ONE

**Option 1: apt (Ubuntu 24.04 with sudo; fastest)**
```bash
bash compiler/scripts/setup_ubuntu.sh
source ~/ci-venv/bin/activate
export CI_OPT=~/ci-build/tools/ci-opt/ci-opt
```

**Option 2: conda (any x86-64 Linux, no sudo; same as Sol)**
```bash
# install Miniforge once if you have no conda/mamba:
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p ~/miniforge3 && source ~/miniforge3/bin/activate

bash compiler/scripts/setup_conda.sh
export PATH=~/envs/causal-ci/bin:$PATH
export CI_OPT=~/ci-build/tools/ci-opt/ci-opt
```

**Expected end of output (either option):**
```
100% tests passed ... out of 2
12 passed in ~10s
```
Stop and fix anything that fails before moving on (see Troubleshooting).

### A4. macOS
Not tested. Apple Silicon also breaks `setup_conda.sh`, which hard-codes
the `x86_64-conda-linux-gnu-*` compilers. Use Sol (Part B) or a Linux VM.

---

## Part B — ASU Sol supercomputer

### B1. Connect
- Connect to the ASU **Cisco VPN** if you are off campus.
- Either `ssh <asurite>@sol.asu.edu`, or open https://sol.asu.edu → System → *Sol Shell Access*.

### B2. Clone (on the login node)
```bash
cd ~
git clone -b 598_compiler_project https://github.com/Ian-Schultz-ASU/statsmodels.git
cd statsmodels
```

### B3. Build on a lightwork node (Sol's partition for env creation and compiling)
```bash
salloc -p lightwork -q public -t 0-2 -c 4      # interactive shell on a lightwork node
module load mamba/latest
cd ~/statsmodels
bash compiler/scripts/setup_conda.sh           # ~10 min first time
exit                                           # release the node
```
- Creates the env at `~/envs/causal-ci` and builds into `~/ci-build`.
- The last step downloads IHDP. If the node has no internet, that step fails.
  Download it on the login node instead, then re-run the script: it skips the
  env it already made.
  ```bash
  module load mamba/latest && export PATH=~/envs/causal-ci/bin:$PATH
  cd ~/statsmodels/compiler/python
  python -c "from causal_compiler.datasets import load_ihdp; load_ihdp(1)"
  ```

### B4. Run benchmarks as a batch job
```bash
cd ~/statsmodels
sbatch compiler/scripts/sol_job.sbatch          # prints: Submitted batch job <JOBID>
squeue -u $USER                                 # PD = pending, R = running
tail -f slurm.<JOBID>.out                       # live output; Ctrl-C to stop watching
```
Results: `compiler/python/results/<JOBID>/ihdp_single_thread.csv` and `synthetic_1000000.txt`.

What the job runs (edit `compiler/scripts/sol_job.sbatch` to change it):
- IHDP at 747 / 10k / 100k / 1M / 10M rows, single-threaded (fair vs NumPy)
- the team's synthetic benchmark at 1M rows
- 8 cores, 64 GB, 1 hour on `public/public`

Quick test run first, with smaller sizes:
```bash
SIZES=747,100000 SYN_ROWS=100000 sbatch --export=ALL -t 0-00:15:00 compiler/scripts/sol_job.sbatch
```

Copy results to your laptop (run this on the laptop):
```bash
scp -r <asurite>@sol.asu.edu:~/statsmodels/compiler/python/results ./sol_results
```

---

## Part C — Running the code (local or inside a Sol `salloc` session)

Activate first:
- apt setup: `source ~/ci-venv/bin/activate`
- conda setup: `export PATH=~/envs/causal-ci/bin:$PATH`

Then `cd ~/statsmodels/compiler/python`.

| Task | Command |
|---|---|
| All Python tests | `CI_OPT_COMMAND=$CI_OPT python -m pytest -q -c /dev/null --rootdir . tests` |
| Native (MLIR) tests | `cd ~/ci-build && ctest --output-on-failure` |
| See the `ci` dialect lowered | `$CI_OPT --ci-lower-treatment-effect ../test/treatment-effect-lowering.mlir` |
| IHDP end-to-end + scaling | `OPENBLAS_NUM_THREADS=1 python -m causal_compiler.demo --ci-opt $CI_OPT` |
| Faster run | add `--sizes 747,100000` |
| Synthetic benchmark | `OPENBLAS_NUM_THREADS=1 python -m causal_compiler.synthetic --rows 100000 --covariates 8 --ci-opt-command $CI_OPT --target-cpu host` |
| Rebuild after editing C++/TableGen | `cmake --build ~/ci-build && (cd ~/ci-build && ctest)` |

Python edits under `causal_compiler/` take effect right away; there's nothing to reinstall.

### Reading the IHDP output
- **Section 4:** `max |compiled - reference|` should be about 1e-15, ending in `PASS`.
  Outputs are not bitwise identical, because the compiled code does the
  arithmetic in a different order.
- **Section 5 columns:**
  - `statsmodels_ms` — the eager pandas + patsy + predict path
  - `two_gemv_numpy_ms` — the naive kernel
  - `hand_fused_numpy_ms` — the best hand-written NumPy
  - `compiled_iree_ms` — our pipeline
  - `vs_*` — speedups (>1 means compiled is faster)
- Reference numbers (2-core VM, 1M rows): compiled is ~180× faster than
  statsmodels, ~3× faster than two-GEMV, and ~1.1× faster than hand-fused
  NumPy. Sol numbers will differ. Report the Sol numbers.

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| `find_package(MLIR)` fails | Check that `MLIRConfig.cmake` exists under `-DMLIR_DIR` (`/usr/lib/llvm-18/lib/cmake/mlir` for apt, `~/envs/causal-ci/lib/cmake/mlir` for conda). |
| Hundreds of `incomplete type ... __normal_iterator` errors | A clang + new libstdc++ mismatch. Use the GCC 13 compilers from `environment.yml` (the script already does). |
| `unrecognized arguments: --randomly-dont-reset-seed` | Add `-c /dev/null --rootdir .` to pytest. |
| `iree-benchmark-module is required` | The env's `bin/` isn't on `PATH`. Activate the env (Part C). |
| `pip install -e .` fails with `mesonpy` | Not needed. The compiler uses the PyPI statsmodels. |
| Sol job: `Invalid qos or partition` | Check https://docs.rc.asu.edu/partitions-and-qos, then edit the `#SBATCH -p/-q` lines. |
| Sol job killed with `OUT_OF_MEMORY` | Drop `10000000` from `SIZES`, or raise `--mem`. |
| Compiled path much slower than NumPy at large N | Inputs are being copied on every call. The demo keeps them resident (`load_resident_kernel`); `synthetic.py`'s `_load_iree_cpu_kernel` still copies. |
