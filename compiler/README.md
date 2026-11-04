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

For the NumPy comparison only, from the repository root in PowerShell, run:

```powershell
Push-Location compiler/python
python -m causal_compiler.synthetic --rows 100000 --covariates 8
Pop-Location
```

This creates seeded synthetic data, fits formula OLS once, verifies the direct
contrast against two formula-based predictions, and reports median CPU latency,
estimated operation counts, and dense feature-matrix sizes. Model fitting is
excluded from timed sections. The reported matrix sizes are estimates, not
measured peak memory.

For execution of lowered MLIR through IREE's CPU backend, first build `ci-opt`
in Ubuntu WSL and install the optional `iree-compiler` and `iree-runtime`
Python packages. Then run:

```powershell
wsl -d Ubuntu -- bash -lc 'cmake -S /mnt/c/Users/mrdoc/OneDrive/Documents/Classes/CSE598/statsmodels_fork/statsmodels/compiler -B /home/mrdoc/statsmodels-ci-build -G Ninja -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir && cmake --build /home/mrdoc/statsmodels-ci-build'
Push-Location compiler/python
python -m causal_compiler.synthetic --rows 100000 --covariates 8 --ci-opt-command 'wsl -d Ubuntu -- /home/mrdoc/statsmodels-ci-build/tools/ci-opt/ci-opt' --target-cpu host
Pop-Location
```

The compiled path validates its output against Statsmodels and reports runtime
latency separately. The current workflow specializes the emitted function to
the selected row count. Results include Python/IREE dispatch and output
handling; they are not a pure hardware-counter measurement of the kernel.
`--target-cpu host` specializes to the CPU running IREE (the Ryzen in this
setup). The lowerer now produces row-parallel `linalg.generic` instead of
repeatedly inserting each row into a tensor. The older `scf.for` tensor-insert
version took about 15.6 seconds for 100,000 rows in an initial probe, so it was
replaced before the current benchmark path was validated.

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
