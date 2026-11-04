# Causal inference compiler prototype

The compiler implementation and its tests live under this directory. The
current Python front end is in `python/causal_compiler`; native MLIR sources,
passes, and CTest fixtures are in `include/`, `lib/`, `tools/`, and `test/`.

The Python API currently accepts fitted formula-based OLS results and an
in-memory pandas DataFrame. Its supported first use case is a numeric treatment
with a main effect and direct two-factor interactions with raw numeric columns.
`Workflow.run()` uses Statsmodels predictions as the reference result;
`Workflow.compile()` emits and lowers MLIR. The `ci.load_dataset` operation is
currently a descriptor and does not read files.

## Python tests

Run the focused Python API tests from the repository root with:

```powershell
Push-Location compiler/python
python -m pytest tests/test_compiler_api.py tests/test_synthetic.py
Pop-Location
```

The Python environment needs the repository's Statsmodels dependencies,
including NumPy, pandas, Patsy, and pytest. The native compiler uses the
project's CMake configuration and CTest tests. To include the optional
WSL/IREE integration test, set `CI_OPT_COMMAND` before pytest:

```powershell
Push-Location compiler/python
$env:CI_OPT_COMMAND = "wsl -d Ubuntu -- /home/mrdoc/statsmodels-ci-build/tools/ci-opt/ci-opt"
python -m pytest tests/test_compiler_api.py tests/test_synthetic.py
Remove-Item Env:CI_OPT_COMMAND
Pop-Location
```

## Synthetic CPU experiment

The benchmark separates three costs:

- The Statsmodels formula path, including pandas counterfactual-frame creation,
  formula design construction, and prediction.
- Kernel-only NumPy paths using preconstructed full-design matrices (two
  matrix-vector products) or the compact interaction matrix (one product).
  Inputs and model coefficients are prepared before timing.
- The compiled MLIR path measured by `iree-benchmark-module`, with the module
  already compiled and inputs loaded before its timed iterations. This native
  harness excludes Python call and compilation overhead, but still includes
  IREE dispatch/runtime costs and output allocation. It reports both IREE's
  synchronous `local-sync` driver and worker-pool `local-task` driver.

To run the NumPy-only measurements:

```powershell
Push-Location compiler/python
$env:OPENBLAS_NUM_THREADS = "1"
python -m causal_compiler.synthetic --rows 100000 --covariates 8
Pop-Location
```

For IREE CPU execution, build `ci-opt` in Ubuntu WSL and install the optional
`iree-compiler` and `iree-runtime` Python packages (the latter supplies
`iree-benchmark-module`):

```powershell
wsl -d Ubuntu -- bash -lc 'cmake -S /mnt/c/Users/mrdoc/OneDrive/Documents/Classes/CSE598/statsmodels_fork/statsmodels/compiler -B /home/mrdoc/statsmodels-ci-build -G Ninja -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir && cmake --build /home/mrdoc/statsmodels-ci-build'
Push-Location compiler/python
$env:OPENBLAS_NUM_THREADS = "1"
python -m causal_compiler.synthetic --rows 100000 --covariates 8 --ci-opt-command 'wsl -d Ubuntu -- /home/mrdoc/statsmodels-ci-build/tools/ci-opt/ci-opt' --target-cpu host
Pop-Location
```

For the Ryzen 7 PRO 8840U (Zen 4), compare against a Zen 4-specialized
build by selecting `znver4`:

```powershell
Push-Location compiler/python
$env:OPENBLAS_NUM_THREADS = "1"
python -m causal_compiler.synthetic --rows 100000 --covariates 8 --repeats 11 --ci-opt-command 'wsl -d Ubuntu -- /home/mrdoc/statsmodels-ci-build/tools/ci-opt/ci-opt' --target-cpu znver4
Pop-Location
```

IREE's LLVM loop and SLP vectorization options are enabled by default for
compiled runs. Use `--no-loop-vectorization` and/or
`--no-slp-vectorization` to isolate their effects. Compilation stays outside
the timed iterations. `local-sync` measures the single-worker path, while
`local-task` uses IREE's worker pool; record the selected target and device
when comparing results.

Compilation by both `ci-opt` and IREE occurs before timed benchmark iterations;
neither compile time nor process startup/input-file loading is included in the
IREE native benchmark's timed iterations. The benchmark validates formula,
NumPy, and compiled outputs. The Python runtime call is reported separately as
a diagnostic, not as the compiled-kernel comparison.

On the initial implementation, generated optimized LLVM for the
`linalg.generic` dispatch contained scalar `load double`, `fmul double`, and
`fadd double` instructions. Enabling LLVM loop and SLP vectorization produces
vector-typed instructions, confirming the CPU optimizer is no longer limited
to scalar-only code generation. The inspected double-precision vector values
were 2 lanes wide, however, so this is not yet evidence that the kernel fully
uses Zen 4's wider vector registers. NumPy uses OpenBLAS GEMV with its own
architecture-specific optimized implementation.

The revised 100,000-row/8-interaction runs showed:

- Statsmodels formula prediction, including DataFrame and design construction:
  about 60--82 ms across runs.
- Two full-design NumPy GEMVs with matrices prebuilt: about 1.5--2.2 ms.
- One direct NumPy interaction GEMV: about 0.175--0.211 ms in the higher-sample
  runs.
- IREE `local-sync` native benchmark: about 0.259--0.270 ms, including IREE
  dispatch and result allocation but excluding Python invocation and compile.
- IREE `local-task` worker-pool benchmark: about 0.153--0.165 ms in observed
  runs. This uses parallel CPU workers, unlike the one-thread BLAS comparison.
- IREE empty-output invocation floor: about 0.008--0.009 ms in the same native
  harness. Treat this as a dispatch/allocation baseline, not a value to subtract
  exactly from kernel time.

A separate Python-runtime no-op call cost roughly 0.9--1.1 ms, explaining why
the prior Python-call benchmark looked much slower. That Python-call figure is
now reported only as a diagnostic and should not be used as kernel latency.
When comparing like-for-like single-worker compute, the current compiled path
is around 1.2--1.5x slower than direct NumPy; the task-pool run is similar to
one observed 16-thread NumPy result, but is not an equal thread-count
comparison. These are exploratory timings, not a stable performance claim.

An additional 100,000-row/8-interaction run used `threadpoolctl` to compare
NumPy with one and 16 OpenBLAS threads, and tested IREE with and without LLVM
vectorization on `host` and `znver4` targets. Direct NumPy GEMV measured about
0.175 ms at one BLAS thread and 0.131 ms at 16 threads. With loop and SLP
vectorization disabled, IREE `local-sync` was about 0.241 ms on both targets;
enabling them reduced it to 0.184 ms on `host` and 0.189 ms on `znver4`.
`local-task` changed from 0.118/0.119 ms to 0.114/0.114 ms, respectively. A
subsequent full CLI run on `znver4` measured 0.185 ms for direct NumPy, 0.189
ms for IREE `local-sync`, and 0.114 ms for IREE `local-task`. These results
suggest vectorization helps the single-worker IREE path, while the task-pool
kernel is competitive with this NumPy configuration. Worker counts are not
matched, and the single-worker compiled path remains slightly slower than
one-thread NumPy.
The host-versus-Zen-4 difference is too small and variable to claim a benefit
from the target override alone. In this environment, OpenBLAS reports its
dispatch architecture as `SkylakeX`, so its results also should not be
interpreted as a Zen-4-tuned NumPy baseline.

The current lowerer uses row-parallel `linalg.generic`, avoiding the former
repeated tensor insertion. The next optimization experiment should expose the
feature reduction as a contiguous reduction rather than loading each
interaction column separately, then inspect generated vector width and compare
with NumPy at matched thread counts. Reduce dispatch/allocation overhead only
after preserving the current native benchmark's separation of compilation,
input setup, and timed execution. Matrix-size and FLOP figures are analytical
estimates, not measured peak memory or hardware counters.

## Dataset requirements for the next milestone

Choose a legally reusable dataset with documentation and no personally
identifying information. For the initial end-to-end loader and scoring path,
prefer:

- One numeric outcome column and one numeric treatment column, ideally binary
  with values `0` (control) and `1` (treated).
- At least one continuous numeric covariate whose effect may vary by
  treatment, so a formula such as `outcome ~ treatment * x` has a nonzero
  interaction to exercise.
- Several additional numeric columns not needed by the formula. These make
  it possible to check whether projection avoids loading irrelevant features.
- Complete, finite values in the initial columns. Missing values, categorical
  fields, weights, and clustered observations can be excluded for the first
  loader milestone.
- A stable row identifier or row ordering, if available, so Python-reference
  and compiled row-level effects can be compared reliably.
- A columnar on-disk copy such as Parquet for testing column selection during
  reads. Keep a small sample (thousands of rows) for correctness tests and, if
  practical, a larger copy for performance and memory comparisons. CSV can be
  retained as a fallback input format, but it does not offer efficient
  column projection during file reads.

The dataset does not need to establish that the estimated effect is causal:
the immediate compiler goal is to match the fitted OLS model's
treated-minus-control predictions. A known or independently validated
data-generating example is still useful as a separate check that the chosen
formula and interaction have the expected meaning.
