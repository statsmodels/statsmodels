# Progress update — Mohak (datasets, benchmarking, Sol)

Branch: `demo-setup` (based on `598_compiler_project`)

## TL;DR
- The pipeline now runs **end to end on a Table 2 dataset (IHDP)**. This answers the professor's feedback. The steps are: fitted statsmodels OLS → `ci` dialect → `ci-opt` lowering → IREE CPU code → output checked against statsmodels.
- It **builds and runs on ASU Sol** without sudo (conda toolchain + Slurm job scripts) and on WSL/Ubuntu.
- **On Sol (AMD EPYC 7713, single core):**
  - 200–2400× faster than the eager statsmodels path;
  - 2.5–5× faster than a two-GEMV NumPy kernel;
  - about 4× less memory;
  - outputs agree to 3.6e-15.
- **Against hand-written fused NumPy we are roughly even**: 1.07× faster with 8 covariates, 0.83–0.91× (slower) with 25. Details and likely cause below.
- **Things we need to discuss:**
  - The speedup comes from an algebraic identity built into `ci.treatment_effect`, not from the compiler detecting redundancy (no CSE yet).
  - Graph capture and DML/EconML from the proposal are not done.

---

## 1. What I added

| File | Purpose |
|---|---|
| `compiler/RUNNING.md` | Step-by-step setup and run instructions for WSL/Ubuntu and ASU Sol |
| `compiler/scripts/setup_ubuntu.sh` | apt-based setup (MLIR 18, build, all tests) |
| `compiler/scripts/setup_conda.sh` + `compiler/environment.yml` | No-sudo setup (used on Sol). GCC 13 is pinned because conda-forge clang 18 fails to compile MLIR 18 headers against newer libstdc++ |
| `compiler/scripts/sol_job.sbatch` | Slurm job: IHDP scaling + the existing synthetic benchmark |
| `compiler/scripts/sol_sweep.sbatch` | Slurm job: covariate sweep (below) |
| `causal_compiler/datasets.py` | IHDP loader (CEVAE copy, 10 replications, cached) + bootstrap row resampling |
| `causal_compiler/demo.py` | End-to-end run: IHDP → IR at each stage → correctness check → scaling table → CSV |
| `causal_compiler/sweep.py` | Covariate sweep: current lowering vs a `linalg.matvec` formulation vs NumPy |
| `tests/test_datasets.py` | Loader tests (Python suite is now 12 tests, all passing) |

### Quick start
```bash
bash compiler/scripts/setup_conda.sh            # or setup_ubuntu.sh with sudo
cd compiler/python
OPENBLAS_NUM_THREADS=1 python -m causal_compiler.demo --ci-opt ~/ci-build/tools/ci-opt/ci-opt
```
On Sol: `sbatch compiler/scripts/sol_job.sbatch` (account `grp_jzou22`). See `RUNNING.md`.

### Sol gotchas (already handled in the scripts)
- Pass the account explicitly: `-A grp_jzou22`.
- `PYTHONNOUSERSITE=1` is required. Otherwise Python picks up old packages from `~/.local`, and pandas fails to import.
- Download IHDP on a login node in case compute nodes have no internet.

---

## 2. Results — Sol job 778782 (AMD EPYC 7713, 1 core, medians of 11 runs)

**Workload:** IHDP replication 1, `y ~ A * (x1 + … + x25)` (52 params). Rows above 747 are bootstrap-resampled from IHDP, for systems scaling only.

| Rows | statsmodels | 2× GEMV NumPy | Hand-fused NumPy | **Compiled (IREE)** | vs statsmodels | vs 2× GEMV | vs hand-fused |
|---|---|---|---|---|---|---|---|
| 747 | 41.7 ms | 0.019 ms | 0.008 ms | **0.017 ms** | 2423× | 1.09× | 0.45× |
| 10k | 57.4 ms | 0.184 ms | 0.062 ms | **0.077 ms** | 741× | 2.37× | 0.80× |
| 100k | 215 ms | 3.39 ms | 0.61 ms | **0.67 ms** | 319× | 5.04× | 0.91× |
| 1M | 2884 ms | 36.5 ms | 11.9 ms | **14.3 ms** | 201× | 2.55× | 0.83× |

- **Correctness:**
  - max |compiled − statsmodels| = 3.6e-15 (relative 6e-16);
  - the ATE matches statsmodels exactly (3.9611);
  - the model's ATE vs IHDP ground truth is 3.96 vs 4.02, which is a model sanity check, not a compiler check.
- **Memory at 1M rows:** 191 MiB of compiled input vs 793 MiB of counterfactual design matrices.
- **Synthetic benchmark, 1M rows, 8 covariates (`iree-benchmark-module`):**
  - compiled 3.42 ms vs hand-fused NumPy 3.65 ms, **1.07×**;
  - vs full-design GEMV, **5.7×**.

How each column is timed:
- **statsmodels** is end to end (pandas + patsy + predict).
- **The NumPy columns** are kernel-only, with inputs already built.
- **IREE** includes the Python→runtime call, with inputs already resident on the device.

---

## 3. Findings that affect everyone

1. **`synthetic.py` measures the input copy, not the kernel.**
   - `_load_iree_cpu_kernel` passes NumPy arrays on every call, so IREE copies the whole input each time.
   - On Sol at 1M rows, the Python path takes 63.5 ms vs 3.4 ms native. On my VM it made compiled code look 5× *slower* than NumPy.
   - `demo.py` and `sweep.py` keep inputs resident with `iree.runtime.asdevicearray`. **Please don't quote the Python-path number.**
2. **The current lowering likely scales poorly with covariate count (→ Chris).**
   - `--ci-lower-treatment-effect` emits one `linalg.generic` with **one 1-D operand per covariate** (`(d0) -> (d0, k)` maps), which means strided column reads.
   - Preliminary sweep (2-core VM, 200k rows; ratio > 1 means compiled is faster than NumPy):

     | Covariates | Current lowering vs NumPy | `linalg.matvec` version vs NumPy |
     |---|---|---|
     | 4 | 1.16× | 1.28× |
     | 25 | 0.83× | 1.12× |
     | 64 | 0.57× | 0.71× |

   - Suggested change: gather the interaction coefficients, then `linalg.matvec` + an elementwise add/scale.
   - The Sol run of `sol_sweep.sbatch` (1M rows, 4–64 covariates) is **pending** and will confirm or refute this.
3. **The kernel is memory-bound.**
   - At 1M × 25, every variant reads ~200 MB at 14–17 GB/s, which is close to single-core bandwidth. Better codegen alone has little headroom.
   - Larger wins require **reading the data fewer times**: K interventions per pass, f32, or sparse inputs.
4. **Outputs are not bitwise identical** (about 20–23% of rows match bit for bit), because the contrast is computed in a different order.
   - The proposal says "identical". I suggest we say **"agrees to f64 round-off, verified at 1e-10"**.

---

## 4. Where we are vs the proposal (≈ week 12–13)

| Weeks | Proposal milestone | Status |
|---|---|---|
| 4–6 | Baseline models with EconML | ⚠️ statsmodels OLS only, no EconML |
| 6–8 | Traceable DML path, validated vs EconML | ❌ Not done (OLS = Risk 1 fallback) |
| 8–9 | Graph capture (torch.fx-style) | ❌ Not done (front end parses the formula string) |
| 9–11 | Dialect + CSE redundancy detection | ⚠️ Dialect done (2 ops, 2 passes); no CSE. The redundancy is removed by a hand-derived identity inside `ci.treatment_effect` |
| 11–13 | Lowering to LLVM/Triton; GPU | ✅ CPU via linalg → IREE/LLVM. ❌ GPU |
| 13–14 | Benchmark harness | ✅ Done (local + Sol) |

**Risk for the final report:** a reviewer can say "the speedup is just
(t₁−t₀)·(β_T + X·β_int), which you could write by hand in NumPy". Our numbers
above confirm we're about even with that hand-written version.

---

## 5. Proposed plan for the remaining weeks

| # | Task | Why | Suggested owner |
|---|---|---|---|
| 1 | **K interventions per pass** (N×K output) | Exercises the proposal's core "repeated computation" claim. Memory-bound, so reading X once for K values is a real win over K NumPy passes | Mohak (harness) + Ian (op/API) |
| 2 | **Logistic/GLM + a CSE/hoisting pass** that computes the shared `X·β_nonT` once across interventions | The compiler *detects* redundancy rather than having it hand-coded, which is the actual hypothesis | Chris |
| 3 | **Matvec lowering fix** (if the Sol sweep confirms) | Removes the 10–20% gap vs NumPy at 25 covariates | Chris |
| 4 | **EconML `LinearDML` as an external baseline.** `effect(X, T0, T1)` is linear in X, which matches `ci.treatment_effect`, so feed its fitted coefficients through our pipeline and compare with `est.effect()` | Reconnects to the proposal's EconML baseline cheaply (need to confirm the exact form against EconML's docs) | Mohak |
| 5 | More Table 2 datasets: Twins (11,984 × 46), News (5,000, sparse, 2–16 treatments) | Breadth; News tests the sparse-data challenge | Mohak |
| 6 | State in the report: Risk 1 fallback (OLS), GPU and graph capture descoped or minimal | Honest scope | All |

### Decisions needed from you two
- Do we formally drop GPU and graph capture, and put the time into items 1–2?
- Chris: are you OK changing the lowering to `linalg.matvec` once the Sol sweep is in?
- Ian: should `treatment_values` become a list (K values), and should `ci.treatment_effect` gain an N×K form or a new op?

---

## References
- Yao et al., *A Survey on Causal Inference*, ACM TKDD 2021, Table 2 (datasets): https://dl.acm.org/doi/pdf/10.1145/3444944
- IHDP CSVs (CEVAE repo): https://github.com/AMLab-Amsterdam/CEVAE
- ASU RC partitions/QoS: https://docs.rc.asu.edu/partitions-and-qos
