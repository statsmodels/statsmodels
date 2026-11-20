"""K interventions on a logistic model: per-intervention vs fused (one pass over X).

Workload (semi-synthetic, IHDP-style): IHDP covariates x1..x25 (standardized,
bootstrap-resampled to N rows), a simulated continuous dose ``T`` and a
simulated binary outcome. A statsmodels ``Logit`` is fitted with
``y ~ T * (x1 + ... + x25)``, then we ask for P(y=1 | do(T=t_k)) for K doses.

Why this matters for the proposal: unlike the OLS contrast, the logistic
prediction does not cancel algebraically, but the expensive part, X @ [b_x,
b_int] is identical across all K interventions. Evaluating each intervention
separately re-reads X K times; a whole-workflow compiler can hoist it and
read X once, then fuse the N x K sigmoid epilogue.

Variants (single-threaded, inputs resident):
- statsmodels:     K x results.predict(frame with T=t_k)       (end to end)
- numpy_naive:     K x (build [1, t, X, t*X] design, GEMV, sigmoid)
- numpy_hoisted:   X @ W once, then N x K epilogue in NumPy  (expert baseline)
- iree_per_k:      the fused kernel compiled for K=1, called K times
- iree_fused:      one kernel: matmul X @ W (N x 2) + fused N x K sigmoid

The IREE kernels are hand-written MLIR showing what a ci lowering should
emit; no ci op produces them yet.

Run from ``compiler/python``::

    OPENBLAS_NUM_THREADS=1 python -m causal_compiler.kinterv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import pandas as pd

from .datasets import load_ihdp, resample_rows
from .synthetic import _iree_cpu_codegen_args

COVARIATES = [f"x{i}" for i in range(1, 26)]

FUSED_MLIR = """
#c0 = affine_map<(i, j) -> (i, 0)>
#c1 = affine_map<(i, j) -> (i, 1)>
#b0 = affine_map<(i, j) -> (0)>
#b1 = affine_map<(i, j) -> (1)>
#kk = affine_map<(i, j) -> (j)>
#id = affine_map<(i, j) -> (i, j)>
func.func public @predict(%x: tensor<{n}x{d}xf64>, %w: tensor<{d}x2xf64>,
                          %c: tensor<2xf64>, %t: tensor<{k}xf64>)
    -> tensor<{n}x{k}xf64> {{
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %e2 = tensor.empty() : tensor<{n}x2xf64>
  %acc = linalg.fill ins(%zero : f64) outs(%e2 : tensor<{n}x2xf64>) -> tensor<{n}x2xf64>
  %m = linalg.matmul ins(%x, %w : tensor<{n}x{d}xf64>, tensor<{d}x2xf64>)
                     outs(%acc : tensor<{n}x2xf64>) -> tensor<{n}x2xf64>
  %e = tensor.empty() : tensor<{n}x{k}xf64>
  %p = linalg.generic {{indexing_maps = [#c0, #c1, #b0, #b1, #kk, #id],
                        iterator_types = ["parallel", "parallel"]}}
      ins(%m, %m, %c, %c, %t : tensor<{n}x2xf64>, tensor<{n}x2xf64>,
          tensor<2xf64>, tensor<2xf64>, tensor<{k}xf64>)
      outs(%e : tensor<{n}x{k}xf64>) {{
    ^bb0(%xb: f64, %xg: f64, %b0: f64, %bt: f64, %tk: f64, %o: f64):
      // eta = b0 + X.b_x + t * (b_T + X.b_int)
      %base = arith.addf %xb, %b0 : f64
      %slope = arith.addf %xg, %bt : f64
      %ts = arith.mulf %tk, %slope : f64
      %eta = arith.addf %base, %ts : f64
      %neg = arith.negf %eta : f64
      %ex = math.exp %neg : f64
      %den = arith.addf %one, %ex : f64
      %prob = arith.divf %one, %den : f64
      linalg.yield %prob : f64
  }} -> tensor<{n}x{k}xf64>
  return %p : tensor<{n}x{k}xf64>
}}
"""


def make_data(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """IHDP covariates (standardized, resampled) + simulated dose and outcome."""
    base = load_ihdp(1)[COVARIATES]
    base = (base - base.mean()) / base.std(ddof=0)
    frame = resample_rows(base, n_rows, seed=seed)
    rng = np.random.default_rng(1234)
    alpha = rng.normal(0.0, 0.2, len(COVARIATES))
    gamma = rng.normal(0.0, 0.05, len(COVARIATES))
    z = frame[COVARIATES].to_numpy()
    rng = np.random.default_rng(seed + 1)
    dose = np.clip(2.0 + 0.5 * z[:, 0] + rng.normal(size=n_rows), 0.0, 4.0)
    logit = -0.5 + 0.3 * dose + z @ alpha + dose * (z @ gamma)
    frame["T"] = dose
    frame["y"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(float)
    return frame


def fit_logit(n_fit: int):
    from statsmodels.formula.api import logit

    formula = "y ~ T * (" + " + ".join(COVARIATES) + ")"
    return logit(formula, data=make_data(n_fit, seed=7)).fit(disp=0)


def unpack(results):
    """Map fitted Logit params to (W [d x 2], c [b0, b_T])."""
    p = results.params
    w = np.column_stack([
        p[COVARIATES].to_numpy(),
        p[[f"T:{c}" for c in COVARIATES]].to_numpy(),
    ])
    return np.ascontiguousarray(w), np.array([p["Intercept"], p["T"]])


def compile_fused(n: int, d: int, k: int, target_cpu: str) -> bytes:
    from iree.compiler import compile_str

    return compile_str(
        FUSED_MLIR.format(n=n, d=d, k=k), target_backends=["llvm-cpu"],
        extra_args=_iree_cpu_codegen_args(target_cpu))


_CONFIG = None


def load(module: bytes):
    """Load a module on a single shared local-sync device.

    All modules must share one device: device arrays created for one
    device are not valid inputs to a function loaded on another.
    """
    import iree.runtime as rt

    global _CONFIG
    if _CONFIG is None:
        _CONFIG = rt.Config("local-sync")
    config = _CONFIG
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.from_flatbuffer(
        ctx.instance, module, warn_if_copy=False))
    return config.device, ctx.modules.module.predict


def to_numpy(device_array) -> np.ndarray:
    """Copy an IREE result to a NumPy array that owns its memory.

    np.asarray(result) is a view that does not keep ``result`` alive; if the
    device array is released (e.g. a temporary), the view reads freed memory.
    Holding ``device_array`` here keeps it alive until the copy is made.
    """
    return np.array(np.asarray(device_array), copy=True)


def timed(fn, repeats: int) -> float:
    """Median seconds; fewer repeats for slow (>0.5 s) operations."""
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    n = 2 if first > 0.5 else repeats
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    import iree.runtime as rt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="100000,1000000")
    ap.add_argument("--k", default="1,2,4,8,16,32,64")
    ap.add_argument("--fit-rows", type=int, default=100_000)
    ap.add_argument("--statsmodels-max-rows", type=int, default=100_000)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--target-cpu", default="host")
    ap.add_argument("--csv", default="kinterv_results.csv")
    a = ap.parse_args()

    results = fit_logit(a.fit_rows)
    w, c = unpack(results)
    d = w.shape[0]
    print(f"Logit fitted on {a.fit_rows} rows, {results.params.size} params; "
          f"converged={results.mle_retvals['converged']}")

    header = ["rows", "K", "statsmodels_ms", "numpy_naive_ms",
              "numpy_hoisted_ms", "iree_per_k_ms", "iree_fused_ms",
              "fused_vs_per_k", "fused_vs_numpy_hoisted",
              "fused_vs_numpy_naive", "fused_vs_statsmodels",
              "max_abs_err_vs_reference"]
    print("  ".join(f"{h:>14}" for h in header[:7])
          + "   fused/perK fused/hoist fused/naive  fused/sm   max_err")
    rows_out = []
    per_k_cache: dict[int, tuple] = {}
    for n in [int(s) for s in a.rows.split(",")]:
        frame = make_data(n, seed=n)
        x = np.ascontiguousarray(frame[COVARIATES].to_numpy())
        device, single = load(compile_fused(n, d, 1, a.target_cpu))
        x_dev = rt.asdevicearray(device, x)
        w_dev = rt.asdevicearray(device, w)
        c_dev = rt.asdevicearray(device, c)
        for k in [int(s) for s in a.k.split(",")]:
            doses = np.linspace(0.0, 4.0, k) if k > 1 else np.array([2.0])

            # Reference: statsmodels when affordable, else naive NumPy.
            def statsmodels_all():
                return np.column_stack([
                    results.predict(frame.assign(T=t)).to_numpy() for t in doses])

            def numpy_naive():
                out = np.empty((n, k))
                ones = np.ones(n)
                beta = np.concatenate([c, w[:, 0], w[:, 1]])
                for j, t in enumerate(doses):
                    tcol = np.full(n, t)
                    design = np.column_stack([ones, tcol, x, x * t])
                    out[:, j] = 1.0 / (1.0 + np.exp(-(design @ beta)))
                return out

            def numpy_hoisted():
                m = x @ w
                eta = np.multiply.outer(m[:, 1] + c[1], doses)
                eta += (m[:, 0] + c[0])[:, None]
                np.negative(eta, out=eta)
                np.exp(eta, out=eta)
                eta += 1.0
                return np.reciprocal(eta, out=eta)

            _, fused = load(compile_fused(n, d, k, a.target_cpu))
            t_dev = rt.asdevicearray(device, doses)
            singles = [rt.asdevicearray(device, np.array([t])) for t in doses]

            def iree_fused():
                return fused(x_dev, w_dev, c_dev, t_dev)

            def iree_per_k():
                return [single(x_dev, w_dev, c_dev, s) for s in singles]

            use_sm = n <= a.statsmodels_max_rows
            ref = statsmodels_all() if use_sm else numpy_naive()
            got_fused = to_numpy(iree_fused())
            got_per_k = np.column_stack([
                to_numpy(single(x_dev, w_dev, c_dev, s))[:, 0] for s in singles])
            errors = {
                name: float(np.abs(v - ref).max()) for name, v in (
                    ("iree_fused", got_fused), ("iree_per_k", got_per_k),
                    ("numpy_hoisted", numpy_hoisted()),
                    ("numpy_naive", numpy_naive()))}
            err = max(errors.values())
            if err > 1e-10:
                raise AssertionError(
                    f"mismatch at rows={n} K={k} (reference="
                    f"{'statsmodels' if use_sm else 'numpy_naive'}): {errors}")

            t_sm = timed(statsmodels_all, 3) if use_sm else float("nan")
            t_nv = timed(numpy_naive, a.repeats)
            t_ho = timed(numpy_hoisted, a.repeats)
            t_pk = timed(iree_per_k, a.repeats)
            t_fu = timed(iree_fused, a.repeats)
            row = [n, k, t_sm * 1e3, t_nv * 1e3, t_ho * 1e3, t_pk * 1e3,
                   t_fu * 1e3, t_pk / t_fu, t_ho / t_fu, t_nv / t_fu,
                   t_sm / t_fu, err]
            rows_out.append(row)
            print(f"{n:>14}  {k:>14}  " + "  ".join(
                f"{v:>14.3f}" for v in row[2:7])
                + f"   {row[7]:9.2f}x {row[8]:9.2f}x {row[9]:9.1f}x"
                + f" {row[10]:9.1f}x  {err:.1e}", flush=True)

    with open(a.csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows_out)
    print(f"wrote {a.csv}. Ratios > 1 mean iree_fused is faster. Reference = "
          f"statsmodels for rows <= {a.statsmodels_max_rows}, else naive NumPy.")


if __name__ == "__main__":
    main()
