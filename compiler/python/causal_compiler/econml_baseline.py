"""EconML LinearDML on IHDP, with its CATE stage run through ci.treatment_effect.

LinearDML's final stage is linear in X: for scalar outcome and treatment,

    effect(X, T0, T1) = (T1 - T0) * (intercept_ + X @ coef_)

which is exactly what ``ci.treatment_effect`` computes, as
(t1 - t0) * (beta_T + sum_j beta_int_j * x_j). So we fit LinearDML in
EconML, pack ``[intercept_, coef_...]`` as the coefficient vector, lower with
``ci-opt --ci-lower-treatment-effect``, compile with IREE and compare against
``est.effect(X)``. This brings EconML into the pipeline (proposal baseline)
without reimplementing DML: fitting stays in EconML, inference is compiled.

Run from ``compiler/python``::

    OPENBLAS_NUM_THREADS=1 python -m causal_compiler.econml_baseline --ci-opt PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time

import numpy as np

from .datasets import load_ihdp, resample_rows
from .synthetic import _iree_cpu_codegen_args

COVARIATES = [f"x{i}" for i in range(1, 26)]


def fit_linear_dml(data, seed: int = 0):
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    est = LinearDML(
        model_y=LassoCV(cv=3, random_state=seed),
        model_t=LogisticRegressionCV(cv=3, max_iter=2000),
        discrete_treatment=True,
        fit_cate_intercept=True,
        cv=3,
        random_state=seed,
    )
    est.fit(data["y"].to_numpy(), data["A"].to_numpy().astype(int),
            X=data[COVARIATES].to_numpy())
    return est


def coefficient_vector(est) -> np.ndarray:
    """[intercept_, coef_...] for scalar Y and binary T."""
    coef = np.asarray(est.coef_, dtype=float).reshape(-1)
    return np.concatenate([[float(np.asarray(est.intercept_).reshape(-1)[0])], coef])


def emit_mlir(n_rows: int, n_covs: int) -> str:
    """ci.treatment_effect with beta_T = coefficients[0], beta_int = [1..d]."""
    cols = ", ".join(json.dumps(c) for c in COVARIATES[:n_covs])
    idx = ", ".join(str(i) for i in range(1, n_covs + 1))
    return f"""module {{
  func.func public @counterfactual(
      %data: tensor<{n_rows}x{n_covs}xf64>,
      %coefficients: tensor<{n_covs + 1}xf64>
  ) -> tensor<{n_rows}xf64> {{
    %effects = "ci.treatment_effect"(%data, %coefficients) {{
      control_value = 0.0 : f64,
      interaction_columns = [{cols}],
      interaction_coefficients = array<i64: {idx}>,
      main_effect_index = 0 : i64,
      treated_value = 1.0 : f64,
      treatment_name = "A"
    }} : (tensor<{n_rows}x{n_covs}xf64>, tensor<{n_covs + 1}xf64>) -> tensor<{n_rows}xf64>
    return %effects : tensor<{n_rows}xf64>
  }}
}}"""


def compile_kernel(ci_opt, n_rows, n_covs, target_cpu, x, coeffs):
    """Lower with ci-opt, compile with IREE, return a resident-input callable."""
    import iree.runtime as rt
    from iree.compiler import compile_str

    lowered = subprocess.run(
        [*ci_opt, "--ci-lower-treatment-effect"], input=emit_mlir(n_rows, n_covs),
        text=True, capture_output=True)
    if lowered.returncode:
        raise RuntimeError(lowered.stderr)
    vmfb = compile_str(lowered.stdout, target_backends=["llvm-cpu"],
                       extra_args=_iree_cpu_codegen_args(target_cpu))
    config = rt.Config("local-sync")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.from_flatbuffer(ctx.instance, vmfb, warn_if_copy=False))
    fn = ctx.modules.module.counterfactual
    x_dev = rt.asdevicearray(config.device, x)
    c_dev = rt.asdevicearray(config.device, coeffs)

    def run():
        return fn(x_dev, c_dev)

    def run_numpy():
        result = fn(x_dev, c_dev)            # keep alive while copying
        return np.array(np.asarray(result), copy=True)

    return run, run_numpy


def median_ms(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci-opt", required=True)
    ap.add_argument("--sizes", default="747,10000,100000,1000000")
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--target-cpu", default="host")
    ap.add_argument("--csv", default="econml_results.csv")
    a = ap.parse_args()
    ci_opt = shlex.split(a.ci_opt)

    data = load_ihdp(1)
    est = fit_linear_dml(data)
    coeffs = coefficient_vector(est)
    x747 = np.ascontiguousarray(data[COVARIATES].to_numpy())

    # 1. The linear form holds (otherwise mapping onto ci.treatment_effect is invalid).
    form_err = np.abs(est.effect(x747) - (coeffs[0] + x747 @ coeffs[1:])).max()
    print(f"LinearDML fitted on IHDP (747 rows, 25 covariates)")
    print(f"  effect(X) == intercept_ + X @ coef_ : max |diff| = {form_err:.2e}")
    print(f"  ATE: EconML {est.ate(x747):.4f} | IHDP simulated truth "
          f"{(data.mu1 - data.mu0).mean():.4f}   (model check, not compiler check)")
    if form_err > 1e-10:
        raise SystemExit("LinearDML effect is not intercept + X @ coef; check "
                         "featurizer / treatment settings before compiling.")

    header = ["rows", "econml_effect_ms", "numpy_ms", "compiled_ms",
              "compiled_vs_econml", "compiled_vs_numpy", "max_abs_err_vs_econml"]
    print("\n" + "  ".join(f"{h:>21}" for h in header))
    rows_out = []
    for n in [int(s) for s in a.sizes.split(",")]:
        frame = data if n == len(data) else resample_rows(data, n, seed=n)
        x = np.ascontiguousarray(frame[COVARIATES].to_numpy())
        run, run_np = compile_kernel(ci_opt, n, x.shape[1], a.target_cpu, x, coeffs)
        reference = est.effect(x)
        err = float(np.abs(run_np() - reference).max())
        if err > 1e-10:
            raise AssertionError(f"compiled != est.effect at rows={n}: {err:.3e}")
        t_ec = median_ms(lambda: est.effect(x), max(3, a.repeats // 3))
        t_np = median_ms(lambda: coeffs[0] + x @ coeffs[1:], a.repeats)
        t_cc = median_ms(run, a.repeats)
        row = [n, t_ec, t_np, t_cc, t_ec / t_cc, t_np / t_cc, err]
        rows_out.append(row)
        print("  ".join(f"{v:>21}" if isinstance(v, int) else
                        f"{v:>21.3e}" if i == 6 else f"{v:>21.3f}"
                        for i, v in enumerate(row)), flush=True)

    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows_out)
    print(f"\nwrote {a.csv}. Inference only: EconML still does the fitting.")


if __name__ == "__main__":
    main()
