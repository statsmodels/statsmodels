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

**Update, Nov 17 (exclusive Sol node, §2b and §3):**
- **K interventions on a logistic model:** a fused kernel reads X once for all K doses. Against the same kernel compiled once per dose, it is **up to 3.0× faster** (1.00× at K=1, as it should be). It's **2.1–3.5× faster than expert NumPy** that also computes the shared part once, and 83–178× faster than statsmodels. It matches statsmodels to 4.4e-16. This is the first measured support for the proposal's hypothesis. The kernel is still hand-written MLIR; we need a `ci` op to produce it (§6).
- **Covariate sweep:** a `linalg.matvec` lowering is **1.08–2.48× faster than the current lowering at every covariate count**, so I'm recommending the change.
- **Fixed a use-after-free** in how results were read back from IREE (§3.5). Earlier results are unaffected.

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
| `causal_compiler/kinterv.py` + `scripts/sol_kinterv.sbatch` | K-intervention logistic benchmark (§2b), with a hand-written fused MLIR kernel |
| `tests/test_kinterv_bench.py` | Fused kernel matches statsmodels `Logit.predict` |
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

## 2b. K interventions on a logistic model — Sol job 1848167 (`--exclusive`, AMD EPYC 7713, 1 core)

**Question:** this is the proposal's core claim. When a workflow asks for P(y=1 | do(T=t_k)) at K doses, can whole-workflow compilation avoid repeating the shared work?

**Workload** (`causal_compiler/kinterv.py`, job `scripts/sol_kinterv.sbatch`):
- IHDP covariates x1..x25, standardized and bootstrap-resampled;
- a simulated continuous dose `T` and a simulated binary outcome;
- statsmodels `Logit` with `y ~ T * (x1 + … + x25)` (52 params);
- doses spread evenly over [0, 4].

Unlike the OLS contrast, nothing cancels algebraically here. What's shared across doses is `X·[β_x, β_int]`.

**Variants compared:**

| Variant | What it does |
|---|---|
| statsmodels | K × `results.predict(frame.assign(T=t_k))`, end to end |
| numpy_naive | K × (build the full design matrix, GEMV, sigmoid) |
| numpy_hoisted | expert NumPy: `X @ W` once, then the N×K sigmoid in NumPy (several passes, with temporaries) |
| iree_per_k | the kernel compiled for K = 1 and called K times (re-reads X each time) |
| **iree_fused** | one kernel: `linalg.matmul` X·W (N×2), then a single fused N×K sigmoid `linalg.generic` (reads X once) |

**Results (medians; ratios > 1 mean fused is faster):**

| Rows | K | Fused | vs per-dose IREE | vs expert NumPy | vs naive NumPy | vs statsmodels |
|---|---|---|---|---|---|---|
| 100k | 1 | 1.19 ms | 1.00× | 2.91× | 22× | 83× |
| 100k | 4 | 2.80 ms | 1.74× | 3.16× | 39× | 140× |
| 100k | 16 | 9.17 ms | 2.18× | 2.70× | 48× | 173× |
| 100k | 32 | 17.7 ms | 2.23× | 2.63× | 50× | 178× |
| 100k | 64 | 52.5 ms | 1.53× | 2.13× | 24× | 118× |
| 1M | 1 | 20.0 ms | 1.00× | 1.95× | 18× | — |
| 1M | 4 | 36.8 ms | 2.22× | 2.56× | 38× | — |
| 1M | 16 | 145 ms | 2.48× | 2.15× | 39× | — |
| 1M | 64 | 537 ms | **3.04×** | 2.12× | 43× | — |

- **Correctness:** all variants match statsmodels (≤ 100k rows) or naive NumPy (1M rows) to ≤ 4.4e-16.
- **Reproducibility:** a separate shared-node run (job 1847191) agreed within about 10%.

**How to read it:**
- **Hoisting across interventions works.** It gives up to 3.0× over compiling each dose separately, and exactly 1.00× at K = 1, as it should.
- **The gain levels off at about 3×.** At 1M rows each extra dose still costs about 8 ms in the fused kernel (against about 25 ms per dose in the per-dose version). That's the `exp`/sigmoid over N × K values, which reading X once can't remove. The next lever is a faster (vectorized) f64 `exp` or an f32 path; that's untested.
- **Fused beats expert NumPy by 2.1–3.5×.** The gain comes from fusing the epilogue into one pass with no temporaries, not from a faster matvec.
- **The drop at 100k rows, K = 64** is likely the output (51 MB) spilling out of L3 (32 MB per CCD). NumPy slows down at the same point.
- **Caveat:** `iree_fused` is hand-written MLIR showing what a `ci` lowering should emit. No `ci` op produces it yet (§6).

---

## 3. Findings that affect everyone

1. **`synthetic.py` measures the input copy, not the kernel.**
   - `_load_iree_cpu_kernel` passes NumPy arrays on every call, so IREE copies the whole input each time.
   - On Sol at 1M rows, the Python path takes 63.5 ms vs 3.4 ms native. On my VM it made compiled code look 5× *slower* than NumPy.
   - `demo.py` and `sweep.py` keep inputs resident with `iree.runtime.asdevicearray`. **Please don't quote the Python-path number.**
2. **The current lowering likely scales poorly with covariate count (→ Chris).**
   - `--ci-lower-treatment-effect` emits one `linalg.generic` with **one 1-D operand per covariate** (`(d0) -> (d0, k)` maps), which means strided column reads.
   - **Sol sweep:** job 1847445, `--exclusive`, AMD EPYC 7713, 1M rows, single-threaded. A ratio above 1 means the left-hand version is faster.

     | Covariates | NumPy | Current lowering | `linalg.matvec` | Current vs NumPy | matvec vs NumPy | matvec vs current |
     |---|---|---|---|---|---|---|
     | 4 | 2.2 ms | 1.7 ms | 1.5 ms | 1.34× | 1.46× | 1.08× |
     | 8 | 4.4 | 3.7 | 3.2 | 1.18× | 1.39× | 1.18× |
     | 16 | 6.9 | 7.5 | 6.5 | 0.93× | 1.06× | 1.15× |
     | 25 | 11.6 | 14.5 | 11.3 | 0.80× | 1.02× | 1.28× |
     | 32 | 11.8 | 20.0 | 14.2 | 0.59× | 0.83× | 1.40× |
     | 64 | 52.3 | 93.0 | 37.5 | 0.56× | 1.40× | 2.48× |

   - **Matvec is faster than the current lowering at every covariate count.** The current lowering falls behind NumPy from 16 covariates on; matvec keeps pace with NumPy up to 25.
   - **Don't read the 64-covariate win as a matvec advantage.** NumPy/OpenBLAS itself slows down at d = 64 (52 ms, against 12 ms at 32), and it did so in both runs.
   - **A shared-node (non-exclusive) run was noisy.** NumPy took 31.8 ms at 25 covariates there. Use exclusive nodes for any number we report.
   - **Recommended change:** gather the interaction coefficients, then `linalg.matvec` + an elementwise add/scale. A further idea, untested: store X column-major (d × N) so each covariate is read contiguously.
3. **The kernel is memory-bound.**
   - At 1M × 25, every variant reads ~200 MB at 14–17 GB/s, which is close to single-core bandwidth. Better codegen alone has little headroom.
   - Larger wins require **reading the data fewer times**: K interventions per pass, f32, or sparse inputs.
4. **Outputs are not bitwise identical** (about 20–23% of rows match bit for bit), because the contrast is computed in a different order.
   - The proposal says "identical". I suggest we say **"agrees to f64 round-off, verified at 1e-10"**.
5. **Reading IREE results with `np.asarray(result)` is a use-after-free.**
   - It returns a view that doesn't keep the IREE buffer alive. If `result` is a temporary, the memory can be freed and reused before it's read.
   - This made the K-intervention correctness check fail on Sol, although the kernels were correct.
   - **Fixed** in `kinterv.py`, `demo.py`, `sweep.py` and `synthetic.py`: copy the result while the device array is still referenced, e.g. `np.array(np.asarray(r), copy=True)` with `r` held in a variable.
   - Earlier results are unaffected: a stale read could only cause false failures, and the timings never read results back.
   - **Use the same pattern in any new code.**

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
| 1 | **K interventions per pass** (N×K output) | The proposal's core "repeated computation" claim. **Benchmark done (§2b): up to 3.0× vs per-dose, 2.1–3.5× vs expert NumPy, using a hand-written kernel** | Mohak ✅ harness; Ian: op/API (§6) |
| 2 | **`ci` op + lowering that emits the fused kernel**, ideally via a hoisting pass that moves the shared `X·W` out of the per-dose computation | Turns §2b from a hand-written kernel into a compiler result, which is the actual hypothesis | Chris / Ian |
| 3 | **Matvec lowering fix**: confirmed by the Sol sweep (§3.2) | 1.08–2.48× over the current lowering; on par with NumPy up to 25 covariates | Chris |
| 4 | **EconML `LinearDML` as an external baseline.** `effect(X, T0, T1)` is linear in X, which matches `ci.treatment_effect`, so feed its fitted coefficients through our pipeline and compare with `est.effect()` | Reconnects to the proposal's EconML baseline cheaply (need to confirm the exact form against EconML's docs) | Mohak |
| 5 | More Table 2 datasets: Twins (11,984 × 46), News (5,000, sparse, 2–16 treatments) | Breadth; News tests the sparse-data challenge | Mohak |
| 6 | State in the report: Risk 1 fallback (OLS), GPU and graph capture descoped or minimal | Honest scope | All |

### Decisions needed from you two
- Do we formally drop GPU and graph capture, and put the time into items 2–3?
- Chris: the sweep confirms the `linalg.matvec` lowering is faster. Can you take it?
- Ian: is the op sketch in §6 the right shape? Should `treatment_values` in the Python API become a list of K values?

---

## 6. Sketch: a `ci` op for K interventions (for discussion)

This is a sketch, not an implementation. The idea is one op that carries the causal meaning, and a lowering that produces exactly the §2b fused kernel.

```mlir
%p = "ci.intervention_predict"(%data, %coefficients, %doses) {
       treatment_name = "T",
       link = "logit",                     // or "identity" for OLS
       base_coefficients = array<i64: ...>, // β_x indices
       interaction_coefficients = array<i64: ...>,
       intercept_index = 0 : i64, treatment_index = 1 : i64
     } : (tensor<NxDxf64>, tensor<Pxf64>, tensor<Kxf64>) -> tensor<NxKxf64>
```

**Lowering** (what `kinterv.FUSED_MLIR` already does by hand):
1. Gather W = [β_x, β_int] (D × 2) and c = [β_0, β_T].
2. `linalg.matmul` X · W → N × 2. This is the only pass over X, shared by every dose.
3. One `linalg.generic` over (N, K): `η = m₀ + c₀ + t_k·(m₁ + c₁)`, then apply the link's inverse (sigmoid for logit).

**Why an op and not only a pass:**
- The op makes the "K interventions of the same model" structure explicit, so the lowering doesn't have to rediscover it.
- A later CSE/hoisting pass could produce this op from K separate prediction ops (`ci.predict` with T replaced by t_k). That is the "compiler detects redundancy" version of the claim, and it's worth doing if time allows.

**How `ci.treatment_effect` fits in:** it's the special case `link = identity`, with 2 doses and the difference taken, so the two ops can share most of their lowering.

---

## References
- Yao et al., *A Survey on Causal Inference*, ACM TKDD 2021, Table 2 (datasets): https://dl.acm.org/doi/pdf/10.1145/3444944
- IHDP CSVs (CEVAE repo): https://github.com/AMLab-Amsterdam/CEVAE
- ASU RC partitions/QoS: https://docs.rc.asu.edu/partitions-and-qos
